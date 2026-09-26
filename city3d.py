r"""
DepthWizard - Stage 5: real-world 3D city from ONE satellite image (LOD1 blocks, heights in metres)

  python city3d.py --lat 12.9766 --lon 77.5993 --size 600 --name cubbon
  -> out\city_cubbon\viewer.html  (open in Chrome)

What it does:
  1. Downloads a georeferenced satellite image (Esri World Imagery, zoom 19 ~0.29 m/pixel)
  2. Fine-tuned model -> height in METRES per pixel (scale from Stage 4)
  3. Building outlines from OpenStreetMap; where OSM has none -> auto-detect from height + non-green
  4. Each building = ONE height (median inside outline) -> flat roof, straight walls (LOD1 city model)
  5. Checks our heights against OSM floor counts (real Bengaluru validation)
  6. Writes 3D viewer + DSM export (16-bit PNG in cm + world file, opens in QGIS as EPSG:3857)
"""
import argparse, base64, io, json, math, os, sys, time, urllib.parse, urllib.request

import cv2
import numpy as np
from PIL import Image

import spike

TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
UA = {"User-Agent": "DepthWizard-SIH2026-student-prototype"}
EARTH = 6378137.0
FLOOR_M = 3.2          # typical Indian floor-to-floor height (for OSM building:levels)
MIN_OBJ_M = 2.5        # taller than this = raised object


# ------------------------------------------------------------------ web-mercator helpers
def ll_to_px(lon, lat, z):
    n = 256 * 2 ** z
    s = min(max(math.sin(math.radians(lat)), -0.9999), 0.9999)
    return (lon + 180) / 360 * n, (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n


def px_to_ll(x, y, z):
    n = 256 * 2 ** z
    lon = x / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def gsd_at(lat, z):
    return 2 * math.pi * EARTH * math.cos(math.radians(lat)) / (256 * 2 ** z)


def http_get(url, data=None, timeout=60, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if k == tries - 1:
                raise
            print(f"   retry ({e})"); time.sleep(2 + 2 * k)


# ------------------------------------------------------------------ 1. imagery
def fetch_imagery(lat, lon, size_m, z, cache):
    gsd = gsd_at(lat, z)
    cx, cy = ll_to_px(lon, lat, z)
    half = size_m / 2 / gsd
    x0, y0, x1, y1 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
    tx0, ty0, tx1, ty1 = x0 // 256, y0 // 256, x1 // 256, y1 // 256
    ntiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    print(f"[1] imagery: zoom {z}, {gsd:.3f} m/px, {x1 - x0}x{y1 - y0} px, {ntiles} tiles")
    canvas = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), np.uint8)
    blank = 0
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            f = os.path.join(cache, str(z), str(tx), f"{ty}.jpg")
            if not os.path.exists(f):
                os.makedirs(os.path.dirname(f), exist_ok=True)
                open(f, "wb").write(http_get(TILE_URL.format(z=z, x=tx, y=ty)))
            t = cv2.imread(f)
            if t is None or t.std() < 4:
                blank += 1
                continue
            canvas[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = t[:256, :256]
    if blank > ntiles * 0.2:
        print(f"[!] {blank}/{ntiles} tiles empty - try --zoom {z - 1}")
    ox, oy = x0 - tx0 * 256, y0 - ty0 * 256
    img = canvas[oy:oy + (y1 - y0), ox:ox + (x1 - x0)][:, :, ::-1].copy()  # RGB
    geo = {"z": z, "x0": x0, "y0": y0, "gsd": gsd, "W": img.shape[1], "H": img.shape[0]}
    return Image.fromarray(img), geo


# ------------------------------------------------------------------ 2. OSM footprints
def fetch_osm(geo, cache_file):
    if os.path.exists(cache_file):
        return json.load(open(cache_file, encoding="utf-8"))
    z, x0, y0, W, H = geo["z"], geo["x0"], geo["y0"], geo["W"], geo["H"]
    w, n = px_to_ll(x0, y0, z)
    e, s = px_to_ll(x0 + W, y0 + H, z)
    q = f'[out:json][timeout:90];(way["building"]({s},{w},{n},{e});relation["building"]({s},{w},{n},{e}););out geom tags;'
    for url in OVERPASS:
        try:
            print(f"[3] OSM buildings from {url.split('/')[2]} ...")
            data = json.loads(http_get(url, data=urllib.parse.urlencode({"data": q}).encode(), timeout=120))
            json.dump(data, open(cache_file, "w", encoding="utf-8"))
            return data
        except Exception as ex:
            print(f"   failed: {ex}")
    print("[!] OSM unavailable - using auto-detected buildings only")
    return {"elements": []}


def parse_height_tags(tags):
    for k in ("height", "building:height"):
        v = tags.get(k)
        if v:
            try:
                return float(v.lower().replace("m", "").strip()), "height tag"
            except ValueError:
                pass
    v = tags.get("building:levels")
    if v:
        try:
            return float(v.split(";")[0]) * FLOOR_M, f"{v} floors"
        except ValueError:
            pass
    return None, None


def osm_polygons(data, geo):
    z, x0, y0 = geo["z"], geo["x0"], geo["y0"]
    polys = []
    for el in data.get("elements", []):
        rings = []
        if el.get("type") == "way" and el.get("geometry"):
            rings = [el["geometry"]]
        elif el.get("type") == "relation":
            rings = [m["geometry"] for m in el.get("members", []) if m.get("role") == "outer" and m.get("geometry")]
        tags = el.get("tags", {})
        for ring in rings:
            pts = []
            for p in ring:
                gx, gy = ll_to_px(p["lon"], p["lat"], z)
                pts.append([gx - x0, gy - y0])
            if len(pts) >= 3:
                polys.append({"pts": np.array(pts, np.float32), "tags": tags})
    return polys


# ------------------------------------------------------------------ 3. buildings
def vegetation_mask(img):
    a = np.asarray(img).astype(np.float32) + 1
    s = a.sum(axis=2)
    r, g, b = a[..., 0] / s, a[..., 1] / s, a[..., 2] / s
    return (2 * g - r - b) > 0.06          # excess-green index


def building_height(hm, mask, erode_px):
    m = mask
    if erode_px > 0:
        e = cv2.erode(mask.astype(np.uint8), np.ones((2 * erode_px + 1,) * 2, np.uint8)) > 0
        if e.sum() >= 10:
            m = e
    vals = hm[m]
    return float(np.median(vals)) if vals.size else 0.0


def build_buildings(img, hm, polys, gsd, cap_m):
    H, W = hm.shape
    veg = vegetation_mask(img)
    taken = np.zeros((H, W), np.uint8)
    out, checks = [], []
    erode = max(1, int(round(0.6 / gsd)))
    for p in polys:
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [np.round(p["pts"]).astype(np.int32)], 1)
        area_m2 = mask.sum() * gsd * gsd
        if area_m2 < 12:
            continue
        m = mask > 0
        h = building_height(hm, m, erode)
        taken |= mask
        tag_h, tag_src = parse_height_tags(p["tags"])
        b = {"pts": p["pts"], "h": max(h, 2.5), "src": "osm", "name": p["tags"].get("name", ""),
             "tag_h": tag_h, "tag_src": tag_src, "area": area_m2}
        if tag_h:
            checks.append((h, tag_h))
        out.append(b)
    # auto-detect: tall, not green, not already an OSM building
    cand = (hm > MIN_OBJ_M) & ~veg & (cv2.dilate(taken, np.ones((5, 5), np.uint8)) == 0)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=4)
    min_px = 25 / (gsd * gsd)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_px:
            continue
        m = lab == i
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, max(1.0, 0.8 / gsd), True)[:, 0, :].astype(np.float32)
        if len(c) < 3:
            continue
        out.append({"pts": c, "h": max(building_height(hm, m, erode), 2.5), "src": "auto", "name": "",
                    "tag_h": None, "tag_src": None, "area": float(m.sum() * gsd * gsd)})
    for b in out:
        b["capped"] = b["h"] >= 0.93 * cap_m
    return out, checks, veg


# ------------------------------------------------------------------ 4. viewer + exports
def tree_canopy(hm, veg, bmask, gsd, max_tree_m=15.0):
    """Turn the model's noisy per-pixel tree heights into smooth, rounded crowns."""
    H, W = hm.shape
    m = (veg & (bmask == 0) & (hm > 1.5)).astype(np.uint8)
    k = max(3, int(round(1.5 / gsd)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] * gsd * gsd >= 12       # drop tiny green specks
    m = keep[lab]
    if not m.any():
        return np.zeros_like(hm)
    cap = float(min(max_tree_m, np.percentile(hm[m], 95)))           # realistic max tree height
    h = np.where(m, np.minimum(hm, cap), 0).astype(np.float32)
    s = min(1.0, gsd / 1.0)                                          # work at ~1 m per pixel
    Wd, Hd = max(8, int(W * s)), max(8, int(H * s))
    hd = cv2.resize(h, (Wd, Hd), interpolation=cv2.INTER_AREA)
    md = cv2.resize(m.astype(np.float32), (Wd, Hd), interpolation=cv2.INTER_AREA)
    hd = np.where(md > 0.05, hd / np.maximum(md, 1e-3), 0).astype(np.float32)
    hd = cv2.medianBlur(hd, 5)                                       # kill spikes
    hd = cv2.GaussianBlur(hd, (0, 0), 3.0)                           # smooth crowns
    taper = np.clip(cv2.GaussianBlur(md, (0, 0), 3.0) * 1.4, 0, 1)   # gentle, rounded edges (less smear)
    out = np.clip(hd, 0, cap) * taper
    # slope limit: height can grow at most ~1 m per metre in from the canopy edge -> dome-like, no cliffs
    dist = cv2.distanceTransform((md > 0.35).astype(np.uint8), cv2.DIST_L2, 5)   # metres (1 m grid)
    out = np.minimum(out, dist * 1.0 + 0.5) * (md > 0.05)
    out = cv2.GaussianBlur(out.astype(np.float32), (0, 0), 1.0)
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def pack16(a, scale=100.0):
    return base64.b64encode(np.clip(np.round(a * scale), 0, 65535).astype("<u2").tobytes()).decode()


def write_outputs(img, hm, veg, buildings, checks, geo, cap_m, outdir, name, grid, template, calib=None,
                  tree_max=15.0):
    os.makedirs(outdir, exist_ok=True)
    H, W, gsd = geo["H"], geo["W"], geo["gsd"]
    SX, SZ = W * gsd, H * gsd
    # layers: full surface (model DSM) and trees-only canopy
    bmask = np.zeros((H, W), np.uint8)
    for b in buildings:
        cv2.fillPoly(bmask, [np.round(b["pts"]).astype(np.int32)], 1)
    canopy = tree_canopy(hm, veg, bmask, gsd, tree_max)
    gw = grid if W >= H else max(2, int(grid * W / H))
    gh = grid if H >= W else max(2, int(grid * H / W))
    rs = lambda a: cv2.resize(a, (gw, gh), interpolation=cv2.INTER_AREA)
    tex = img.resize((min(2048, W), int(min(2048, W) * H / W)), Image.LANCZOS)
    buf = io.BytesIO(); tex.save(buf, "JPEG", quality=88)
    blds = []
    for b in buildings:
        pts = [[round(float(np.clip(x, 0, W) - W / 2) * gsd, 2), round(float(H / 2 - np.clip(y, 0, H)) * gsd, 2)]
               for x, y in b["pts"]]                                  # clip to image edge (no stretched roofs)
        blds.append({"p": pts, "h": round(float(b["h"]), 1), "s": b["src"], "n": b["name"][:40],
                     "t": round(float(b["tag_h"]), 1) if b["tag_h"] else None, "ts": b["tag_src"],
                     "c": bool(b["capped"])})
    val = None
    if checks:
        pr, tg = np.array(checks).T
        ok = tg <= cap_m                      # only compare where truth is inside the model's range
        val = {"n": int(len(checks)), "n_in_range": int(ok.sum()),
               "mae_m": round(float(np.abs(pr[ok] - tg[ok]).mean()), 2) if ok.any() else None,
               "median_ratio": round(float(np.median(pr[ok] / np.maximum(tg[ok], 0.1))), 2) if ok.any() else None}
    stats = {"size_m": [round(SX), round(SZ)], "gsd": round(gsd, 3), "osm": sum(b["src"] == "osm" for b in buildings),
             "auto": sum(b["src"] == "auto" for b in buildings), "capped": int(sum(bool(b["capped"]) for b in buildings)),
             "cap_m": round(cap_m, 1), "validation": val, "calibration": calib or {"mode": "none", "k": 1.0}}
    data = {"title": name, "SX": SX, "SZ": SZ, "gw": gw, "gh": gh, "surface": pack16(rs(hm)),
            "canopy": pack16(rs(canopy)), "photo": base64.b64encode(buf.getvalue()).decode(),
            "buildings": blds, "stats": stats}
    html = open(template, encoding="utf-8").read().replace("/*__DATA__*/null", json.dumps(data))
    open(os.path.join(outdir, "viewer.html"), "w", encoding="utf-8").write(html)
    # exports
    img.save(os.path.join(outdir, "image.jpg"), quality=92)
    cv2.imwrite(os.path.join(outdir, "dsm_cm.png"), np.clip(np.round(hm * 100), 0, 65535).astype(np.uint16))
    res = 2 * math.pi * EARTH / (256 * 2 ** geo["z"])           # web-mercator metres per pixel
    mx = geo["x0"] * res - math.pi * EARTH + res / 2
    my = math.pi * EARTH - geo["y0"] * res - res / 2
    for ext in ("dsm_cm.pgw", "image.jgw"):
        open(os.path.join(outdir, ext), "w").write(f"{res}\n0\n0\n{-res}\n{mx}\n{my}\n")
    col = cv2.applyColorMap((np.clip(hm / max(cap_m, 1), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(outdir, "side_by_side.png"), np.hstack([np.asarray(img)[:, :, ::-1], col]))
    json.dump({"geo": geo, **stats}, open(os.path.join(outdir, "report.json"), "w"), indent=2)
    return stats


# ------------------------------------------------------------------ 5. city calibration (relative -> true metres)
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def fit_k(pred, tag):
    return float(np.median(tag / np.maximum(pred, 0.3)))   # robust: median of ratios


def cross_validate(pred, tag, splits=200, seed=0):
    """Fit k on half the buildings, measure error on the OTHER half. Repeat. Honest numbers."""
    rng = np.random.default_rng(seed)
    n = len(pred); before, after, ks = [], [], []
    for _ in range(splits):
        idx = rng.permutation(n); tr, te = idx[: n // 2], idx[n // 2:]
        k = fit_k(pred[tr], tag[tr]); ks.append(k)
        before.append(np.abs(pred[te] - tag[te]).mean()); after.append(np.abs(pred[te] * k - tag[te]).mean())
    return float(np.mean(before)), float(np.mean(after)), float(np.std(ks))


def scatter_png(pred, tag, k, path):
    S, pad = 520, 50
    top = max(float(tag.max()), float((pred * k).max()), 10) * 1.05
    img = np.full((S, S, 3), 245, np.uint8)
    P = lambda x, y: (int(pad + x / top * (S - 2 * pad)), int(S - pad - y / top * (S - 2 * pad)))
    cv2.line(img, P(0, 0), P(top, top), (160, 160, 160), 1)
    cv2.rectangle(img, P(0, 0), P(top, top), (90, 90, 90), 1)
    for p, t in zip(pred, tag):
        cv2.circle(img, P(t, p), 4, (150, 150, 150), -1)          # raw model (grey)
        cv2.circle(img, P(t, p * k), 4, (42, 96, 224), -1)       # calibrated (orange, BGR)
    for v in range(0, int(top) + 1, 10):
        cv2.putText(img, str(v), (P(v, 0)[0] - 8, S - pad + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
        cv2.putText(img, str(v), (pad - 30, P(0, v)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
    cv2.putText(img, "x: OSM height (m)   y: predicted (m)", (pad, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    cv2.putText(img, f"grey = raw   orange = calibrated (k={k:.2f})   line = perfect", (pad, S - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1)
    cv2.imwrite(path, img)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--size", type=float, default=600, help="area width in metres (300-1000)")
    ap.add_argument("--zoom", type=int, default=19, help="19 ~0.29 m/px (matches training). 18 if tiles are blank")
    ap.add_argument("--name", default="city")
    ap.add_argument("--weights", default="runs/run1/best.pt")
    ap.add_argument("--scale-file", default="runs/height_scale.json")
    ap.add_argument("--grid", type=int, default=360)
    ap.add_argument("--no-osm", action="store_true", help="auto-detected buildings only")
    ap.add_argument("--fit-calib", action="store_true",
                    help="fit the city height factor from OSM floors here, save to runs/city_calib.json")
    ap.add_argument("--use-calib", action="store_true",
                    help="apply the saved city factor (fitted on ANOTHER area = honest test)")
    ap.add_argument("--calib-file", default="runs/city_calib.json")
    ap.add_argument("--tree-max", type=float, default=15.0, help="max tree height in metres (visual layer)")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join("out", f"city_{a.name}")
    os.makedirs(outdir, exist_ok=True)
    scale = json.load(open(a.scale_file))["height_scale_m_per_unit"]
    cap_m = 255 * scale

    img, geo = fetch_imagery(a.lat, a.lon, a.size, a.zoom, os.path.join("cache", "tiles"))
    cache = os.path.join(outdir, "pred_ft.npy")
    if os.path.exists(cache) and np.load(cache).shape == (geo["H"], geo["W"]):
        print("[2] reusing saved model output"); pred = np.load(cache)
    else:
        print("[2] fine-tuned model -> height")
        pred = spike.run_finetuned(img, a.weights); np.save(cache, pred)
    import torch
    div = torch.load(a.weights, map_location="cpu").get("div", 25.5)
    hm = np.clip(pred * div * scale, 0, None).astype(np.float32)        # METRES
    print(f"    height range {np.percentile(hm, 50):.1f} m (median) .. {np.percentile(hm, 99.5):.1f} m (top)")

    polys = [] if a.no_osm else osm_polygons(fetch_osm(geo, os.path.join(outdir, "osm.json")), geo)
    print(f"    OSM outlines: {len(polys)}")
    print("[4] building blocks")
    blds, checks, veg = build_buildings(img, hm, polys, geo["gsd"], cap_m)
    calib = {"mode": "none", "k": 1.0}
    pairs = np.array([(p, t) for p, t in checks if 3 <= t <= 60], np.float32).reshape(-1, 2)
    if len(pairs) >= 3:
        pred_b, tag_b = pairs[:, 0], pairs[:, 1]
        rho = spearman(pred_b, tag_b)
        print(f"[5] {len(pairs)} OSM-tagged buildings | rank agreement (Spearman) {rho:.2f}  "
              f"(>0.5 = model ranks heights well -> calibration will work)")
        calib["spearman"] = round(rho, 2); calib["n"] = int(len(pairs))
    if a.fit_calib:
        if len(pairs) < 8:
            sys.exit("[x] need >= 8 OSM buildings with height/floor tags to fit - try another area")
        b_mae, a_mae, k_sd = cross_validate(pred_b, tag_b)
        k = fit_k(pred_b, tag_b)
        calib.update({"mode": "fit", "k": round(k, 3), "fit_on": a.name, "heldout_mae_before": round(b_mae, 2),
                      "heldout_mae_after": round(a_mae, 2), "k_spread": round(k_sd, 3)})
        os.makedirs(os.path.dirname(a.calib_file) or ".", exist_ok=True)
        json.dump(calib, open(a.calib_file, "w"), indent=2)
        print(f"    city factor k = {k:.2f} (+-{k_sd:.2f})  | held-out MAE {b_mae:.2f} m -> {a_mae:.2f} m  "
              f"| saved {a.calib_file}")
    elif a.use_calib:
        saved = json.load(open(a.calib_file))
        calib.update({"mode": "load", "k": saved["k"], "fit_on": saved.get("fit_on", "?")})
        if len(pairs) >= 3:
            calib["test_mae_before"] = round(float(np.abs(pred_b - tag_b).mean()), 2)
            calib["test_mae_after"] = round(float(np.abs(pred_b * saved["k"] - tag_b).mean()), 2)
            print(f"    using k = {saved['k']} fitted on '{calib['fit_on']}' -> this area (never seen): "
                  f"MAE {calib['test_mae_before']} m -> {calib['test_mae_after']} m")
    k = calib["k"]
    if len(pairs) >= 3:
        scatter_png(pred_b, tag_b, k, os.path.join(outdir, "osm_check.png"))
    if k != 1.0:
        hm = hm * k
        for b in blds:
            b["h"] *= k
        checks = [(p * k, t) for p, t in checks]
    json.dump([{"h": round(float(b["h"]), 2), "osm_h": b["tag_h"], "osm_src": b["tag_src"], "src": b["src"],
                "name": b["name"], "area_m2": round(float(b["area"]), 1)} for b in blds],
              open(os.path.join(outdir, "buildings.json"), "w"), indent=1)
    st = write_outputs(img, hm, veg, blds, checks, geo, cap_m * k, outdir, a.name, a.grid,
                       os.path.join(here, "viewer_city_template.html"), calib, a.tree_max)
    print(f"[ok] {st['osm']} OSM + {st['auto']} auto buildings | {st['capped']} at the {st['cap_m']} m training cap")
    v = st["validation"]
    if v and v["mae_m"] is not None:
        print(f"[v] Bengaluru check vs OSM heights/floors: {v['n_in_range']} buildings, MAE {v['mae_m']} m, "
              f"ours/OSM median ratio {v['median_ratio']}")
    else:
        print("[i] no OSM buildings with height/floor tags here - no validation for this area")
    print(f"     open {os.path.join(outdir, 'viewer.html')}")


if __name__ == "__main__":
    main()
