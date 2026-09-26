r"""
DepthWizard web app - pipeline runner.

Composes the existing city3d.py / spike.py functions (unchanged) into one job with progress reporting:
  imagery (Esri) or uploaded image -> fine-tuned height model -> OSM outlines -> LOD1 blocks
  -> city calibration -> city3d.write_outputs (viewer.html, dsm_cm.png, ...) -> extra web layers
Mirrors city3d.main() with --use-calib, so results match the command line.
"""
import collections, contextlib, hashlib, io, json, math, os, sys, threading, time, traceback

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import city3d  # noqa: E402  (also imports spike)
import spike   # noqa: E402

JOBS_DIR = os.path.join(ROOT, "app", "jobs")
CACHE_DIR = os.path.join(JOBS_DIR, "_cache")              # model output + OSM per area, shared by calib on/off
WEIGHTS = os.path.join(ROOT, "runs", "run1", "best.pt")
SCALE_FILE = os.path.join(ROOT, "runs", "height_scale.json")
CALIB_FILE = os.path.join(ROOT, "runs", "city_calib.json")
TEMPLATE = os.path.join(ROOT, "viewer_city_template.html")
TILE_CACHE = os.path.join(ROOT, "cache", "tiles")
ZOOM = 19
GRID = 360               # 3D mesh resolution (city3d default)
TREE_MAX = 15.0
LAYER_MAX_SIDE = 1024    # height.bin resolution for the 2D views / hover
UPLOAD_MAX_SIDE = 3000
TRAIN_GSD = (0.10, 0.30) # prepare_tiles.py --gsd range the model was trained on
TILE = 518               # spike.run_finetuned tile size (50 % overlap)
OSM_WAIT_S = 90          # city3d.fetch_osm can retry for minutes when Overpass is down

_sec_per_tile = {"cuda": 0.15, "cpu": 2.5}   # refined after every run
_MODEL_LOAD_S = 6.0
_div = None


class PipelineError(Exception):
    """Error with a message meant for the user."""


class Job:
    def __init__(self, job_id, kind, params):
        self.id, self.kind, self.params = job_id, kind, params
        self.state, self.pct, self.stage = "queued", 0.0, "Waiting in queue"
        self.log = collections.deque(maxlen=200)
        self.error, self.meta, self.created = None, None, time.time()
        self.osm_timeout = False

    def update(self, pct=None, stage=None):
        if pct is not None:
            self.pct = max(self.pct, float(pct))
        if stage is not None:
            self.stage = stage

    def to_dict(self):
        return {"id": self.id, "kind": self.kind, "state": self.state, "pct": round(self.pct, 1),
                "stage": self.stage, "log": list(self.log)[-6:], "error": self.error, "meta": self.meta}

    @classmethod
    def from_disk(cls, job_id):
        path = os.path.join(JOBS_DIR, job_id, "meta.json")
        if not os.path.exists(path):
            return None
        meta = json.load(open(path, encoding="utf-8"))
        j = cls(job_id, meta.get("kind", "?"), meta.get("params", {}))
        j.state, j.pct, j.stage, j.meta, j.created = "done", 100.0, "Done", meta, meta.get("created", 0)
        return j


class _Tee(io.TextIOBase):
    """Copies the pipeline's print() output into the job log (shown under the progress bar)."""
    def __init__(self, job):
        self.job, self.buf = job, ""

    def write(self, s):
        sys.__stdout__.write(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.job.log.append(line.rstrip())
        return len(s)


def job_dir(job_id):
    return os.path.join(JOBS_DIR, job_id)


def sha(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
        h.update(b"|")
    return h.hexdigest()


def calib_info():
    if not os.path.exists(CALIB_FILE):
        return None
    c = json.load(open(CALIB_FILE))
    return {"k": c.get("k"), "fit_on": c.get("fit_on")}


# ------------------------------------------------------------------ entry point (runs in the worker thread)
def run(job):
    job.state = "running"
    os.makedirs(job_dir(job.id), exist_ok=True)
    try:
        with contextlib.redirect_stdout(_Tee(job)):
            if job.kind == "location":
                _run_location(job)
            else:
                _run_upload(job)
        job.state, job.pct, job.stage = "done", 100.0, "Done"
    except PipelineError as e:
        job.state, job.error = "error", str(e)
    except Exception as e:  # noqa: BLE001 - report anything to the UI
        traceback.print_exc()
        job.state, job.error = "error", _friendly(e)


def _friendly(e):
    name = type(e).__name__
    if "OutOfMemory" in name:
        return "The GPU ran out of memory. Try a smaller area size."
    if name in ("URLError", "HTTPError", "TimeoutError", "RemoteDisconnected", "ConnectionResetError") or "urlopen" in str(e):
        return f"Could not download satellite tiles - check the internet connection ({e})."
    return f"{name}: {e}"


# ------------------------------------------------------------------ sources
def _run_location(job):
    p = job.params
    job.update(2, "Downloading satellite imagery")
    img, geo = city3d.fetch_imagery(p["lat"], p["lon"], p["size_m"], ZOOM, TILE_CACHE)
    if _blank_frac(img) > 0.2:
        print(f"[!] many empty tiles at zoom {ZOOM} - retrying at zoom {ZOOM - 1}")
        img, geo = city3d.fetch_imagery(p["lat"], p["lon"], p["size_m"], ZOOM - 1, TILE_CACHE)
    if _blank_frac(img) > 0.5:
        raise PipelineError("No satellite imagery is available for this area. Try another location.")
    job.update(15)
    key = sha("loc", round(p["lat"], 6), round(p["lon"], 6), p["size_m"], geo["z"])[:16]
    pred = _predict(job, img, key)
    hm = _to_metres(pred)
    polys = city3d.osm_polygons(_fetch_osm(job, geo, os.path.join(CACHE_DIR, key + ".osm.json")), geo)
    print(f"    OSM outlines: {len(polys)}")
    _finish(job, img, geo, hm, polys)


def _fetch_osm(job, geo, cache_file):
    """city3d.fetch_osm, but the UI waits at most OSM_WAIT_S. A slow fetch keeps running in the background
    and fills the cache, so the next run of this area gets the outlines."""
    if os.path.exists(cache_file):
        job.update(77, "Using saved OpenStreetMap outlines")
        return city3d.fetch_osm(geo, cache_file)
    job.update(70, f"Fetching OpenStreetMap outlines (up to {OSM_WAIT_S} s)")
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("data", city3d.fetch_osm(geo, cache_file)), daemon=True)
    t.start()
    with _creep(job, 70, 77, OSM_WAIT_S / 3):
        t.join(OSM_WAIT_S)
    if "data" not in box:
        print("[!] OpenStreetMap is not answering - continuing with auto-detected buildings only")
        job.osm_timeout = True
        return {"elements": []}
    return box["data"]


def _run_upload(job):
    p = job.params
    job.update(2, "Preparing uploaded image")
    img = Image.open(p["path"]).convert("RGB")
    W, H = img.size
    if min(W, H) < 64:
        raise PipelineError("The image is too small (need at least 64 x 64 pixels).")
    gsd = float(p["gsd"])
    f = gsd / min(max(gsd, TRAIN_GSD[0]), TRAIN_GSD[1])       # resample into the model's training resolution
    f = min(f, UPLOAD_MAX_SIDE / max(W, H))
    if abs(f - 1) > 1e-3:
        img = img.resize((max(1, round(W * f)), max(1, round(H * f))), Image.LANCZOS)
        print(f"[1] resampled {W}x{H} @ {gsd:.2f} m/px -> {img.size[0]}x{img.size[1]} @ {gsd * W / img.size[0]:.2f} m/px")
    else:
        print(f"[1] image {W}x{H} @ {gsd:.2f} m/px")
    geo = {"z": ZOOM, "x0": 0, "y0": 0, "gsd": gsd * W / img.size[0], "W": img.size[0], "H": img.size[1]}
    job.update(15)
    pred = _predict(job, img, sha("up", p["file_sha"], round(gsd, 4))[:16])
    hm = _to_metres(pred)
    print("    uploaded image: no map position -> auto-detected buildings only")
    _finish(job, img, geo, hm, [])
    for ext in ("dsm_cm.pgw", "image.jgw"):                   # world files are meaningless without a location
        f = os.path.join(job_dir(job.id), ext)
        if os.path.exists(f):
            os.remove(f)


def _blank_frac(img):
    return float((np.asarray(img).max(axis=2) < 4).mean())


@contextlib.contextmanager
def _creep(job, start, end, est_s):
    """For blocking calls without their own progress: move the bar from start towards end by elapsed/estimated
    time (eases out, never reaches end until the call returns)."""
    stop, t0 = threading.Event(), time.time()

    def tick():
        while not stop.wait(0.25):
            f = (time.time() - t0) / max(est_s, 0.1)
            job.update(start + (end - start) * (f if f < 0.8 else 0.8 + 0.18 * (1 - math.exp(-(f - 0.8) * 2))))

    threading.Thread(target=tick, daemon=True).start()
    try:
        yield
    finally:
        stop.set()


# ------------------------------------------------------------------ model
def _count_tiles(W, H):
    step = TILE // 2
    Wp, Hp = max(W, TILE), max(H, TILE)
    nx = len(range(0, Wp - TILE + 1, step)) + ((Wp - TILE) % step != 0)
    ny = len(range(0, Hp - TILE + 1, step)) + ((Hp - TILE) % step != 0)
    return nx * ny


def _predict(job, img, key):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, key + ".npy")
    W, H = img.size
    if os.path.exists(cache):
        pred = np.load(cache)
        if pred.shape == (H, W):
            print("[2] reusing saved model output")
            job.update(69, "Reusing saved model output")
            return pred
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = _count_tiles(W, H)
    est = _MODEL_LOAD_S + n * _sec_per_tile[dev]
    job.update(15, f"Running height model ({n} tiles, {'GPU' if dev == 'cuda' else 'CPU - slow'})")
    print("[2] fine-tuned model -> height")
    t0 = time.time()
    try:
        with _creep(job, 15, 69, est):
            pred = spike.run_finetuned(img, WEIGHTS)
    finally:
        if dev == "cuda":
            torch.cuda.empty_cache()
    _sec_per_tile[dev] = max(0.02, (time.time() - t0 - _MODEL_LOAD_S) / n)
    np.save(cache, pred)
    return pred


def _to_metres(pred):
    """Same conversion as city3d.main(): raw model units -> metres (before city calibration)."""
    global _div
    if _div is None:
        import torch
        _div = torch.load(WEIGHTS, map_location="cpu").get("div", 25.5)
    scale = json.load(open(SCALE_FILE))["height_scale_m_per_unit"]
    hm = np.clip(pred * _div * scale, 0, None).astype(np.float32)
    print(f"    height range {np.percentile(hm, 50):.1f} m (median) .. {np.percentile(hm, 99.5):.1f} m (top)")
    return hm


# ------------------------------------------------------------------ buildings, calibration, outputs
def _finish(job, img, geo, hm, polys):
    p, outdir = job.params, job_dir(job.id)
    job.update(78, "Building 3D blocks")
    scale = json.load(open(SCALE_FILE))["height_scale_m_per_unit"]
    cap_m = 255 * scale
    print("[4] building blocks")
    blds, checks, veg = city3d.build_buildings(img, hm, polys, geo["gsd"], cap_m)

    # city calibration - same as city3d.main() with --use-calib
    calib = {"mode": "none", "k": 1.0}
    pairs = np.array([(a, t) for a, t in checks if 3 <= t <= 60], np.float32).reshape(-1, 2)
    if len(pairs) >= 3:
        pred_b, tag_b = pairs[:, 0], pairs[:, 1]
        calib["spearman"] = round(city3d.spearman(pred_b, tag_b), 2)
        calib["n"] = int(len(pairs))
    if p.get("calibrate") and os.path.exists(CALIB_FILE):
        saved = json.load(open(CALIB_FILE))
        calib.update({"mode": "load", "k": saved["k"], "fit_on": saved.get("fit_on", "?")})
        if len(pairs) >= 3:
            calib["test_mae_before"] = round(float(np.abs(pred_b - tag_b).mean()), 2)
            calib["test_mae_after"] = round(float(np.abs(pred_b * saved["k"] - tag_b).mean()), 2)
        print(f"    city factor k = {saved['k']} (fitted on '{calib['fit_on']}')")
    k = calib["k"]
    if len(pairs) >= 3:
        city3d.scatter_png(pred_b, tag_b, k, os.path.join(outdir, "osm_check.png"))
    if k != 1.0:
        hm = hm * k
        for b in blds:
            b["h"] *= k
        checks = [(a * k, t) for a, t in checks]
    json.dump([{"h": round(float(b["h"]), 2), "osm_h": b["tag_h"], "osm_src": b["tag_src"], "src": b["src"],
                "name": b["name"], "area_m2": round(float(b["area"]), 1)} for b in blds],
              open(os.path.join(outdir, "buildings.json"), "w"), indent=1)

    job.update(88, "Writing 3D viewer and layers")
    st = city3d.write_outputs(img, hm, veg, blds, checks, geo, cap_m * k, outdir, p["title"], GRID,
                              TEMPLATE, calib, TREE_MAX)
    job.meta = _write_layers(job, hm, geo, st)
    print(f"[ok] {st['osm']} OSM + {st['auto']} auto buildings")


def _write_layers(job, hm, geo, stats):
    """height.bin (uint16 cm, <= 1024 px) + meta.json for the 2D views and hover."""
    outdir = job_dir(job.id)
    H, W = hm.shape
    s = min(1.0, LAYER_MAX_SIDE / max(H, W))
    gw, gh = max(2, round(W * s)), max(2, round(H * s))
    small = cv2.resize(hm, (gw, gh), interpolation=cv2.INTER_AREA) if s < 1 else hm
    with open(os.path.join(outdir, "height.bin"), "wb") as f:
        f.write(np.clip(np.round(small * 100), 0, 65535).astype("<u2").tobytes())
    top = float(np.percentile(hm, 99.5))
    meta = {"id": job.id, "kind": job.kind, "title": job.params["title"], "created": job.created,
            "params": {k: v for k, v in job.params.items() if k != "path"},
            "W": W, "H": H, "gw": gw, "gh": gh, "gsd": geo["gsd"], "geo": geo,
            "size_m": [round(W * geo["gsd"]), round(H * geo["gsd"])],
            "hmax": max(5, math.ceil(top / 5) * 5), "hpeak": round(float(hm.max()), 1),
            "world_file": job.kind == "location", "osm_timeout": job.osm_timeout, "stats": stats}
    json.dump(meta, open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8"), indent=1)
    return meta
