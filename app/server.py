r"""
DepthWizard web app - FastAPI backend.

  app\run_app.bat      (or: .venv\Scripts\python -m uvicorn app.server:app --port 8000)
  -> http://127.0.0.1:8000
"""
import io, json, os, re, threading, zipfile
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import pipeline

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
ID_RE = re.compile(r"^[0-9a-f]{12}$")
FILES = {"image.jpg": "image/jpeg", "height.bin": "application/octet-stream", "meta.json": "application/json",
         "dsm_cm.png": "image/png", "report.json": "application/json", "buildings.json": "application/json",
         "osm_check.png": "image/png"}
MAX_UPLOAD = 60 * 2**20
BRIDGE = '<script src="/static/viewer_bridge.js"></script>\n'
NO_STORE = {"Cache-Control": "no-store"}

app = FastAPI(title="DepthWizard")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
EXEC = ThreadPoolExecutor(max_workers=1)   # one GPU -> one job at a time, others queue
JOBS: dict[str, pipeline.Job] = {}
LOCK = threading.Lock()


class LocationReq(BaseModel):
    lat: float = Field(ge=-85, le=85)
    lon: float = Field(ge=-180, le=180)
    size_m: float = Field(ge=300, le=1000)
    calibrate: bool = True


def _job(job_id):
    if not ID_RE.match(job_id):
        raise HTTPException(404, "Unknown job")
    with LOCK:
        j = JOBS.get(job_id) or pipeline.Job.from_disk(job_id)
        if j:
            JOBS[job_id] = j
    if not j:
        raise HTTPException(404, "Unknown job")
    return j


def _submit(job_id, kind, params):
    with LOCK:
        j = JOBS.get(job_id) or pipeline.Job.from_disk(job_id)
        retry = j and (j.state == "error" or (j.state == "done" and (j.meta or {}).get("osm_timeout")))
        if j and not retry:                 # already done, running or queued -> reuse
            JOBS[job_id] = j
            return j
        j = JOBS[job_id] = pipeline.Job(job_id, kind, params)
    EXEC.submit(pipeline.run, j)
    return j


# ------------------------------------------------------------------ pages
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(STATIC, "index.html"), headers=NO_STORE)


@app.get("/api/config")
def config():
    return {"calibration": pipeline.calib_info(), "weights": os.path.relpath(pipeline.WEIGHTS, pipeline.ROOT)}


# ------------------------------------------------------------------ jobs
@app.post("/api/jobs/location")
def new_location(req: LocationReq):
    lat, lon, size = round(req.lat, 6), round(req.lon, 6), round(req.size_m)
    job_id = pipeline.sha("loc", lat, lon, size, req.calibrate)[:12]
    j = _submit(job_id, "location", {"lat": lat, "lon": lon, "size_m": size, "calibrate": req.calibrate,
                                     "title": f"{lat:.4f}, {lon:.4f}"})
    return {"job_id": j.id, "state": j.state}


@app.post("/api/jobs/upload")
async def new_upload(file: UploadFile = File(...), gsd: float = Form(0.30), calibrate: bool = Form(True)):
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image is larger than 60 MB")
    if not 0.02 <= gsd <= 5:
        raise HTTPException(422, "Ground resolution must be between 0.02 and 5 m per pixel")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = (im.format or "png").lower()
            im.verify()
    except Exception:
        raise HTTPException(400, "That file is not an image we can read (use JPG or PNG)")
    file_sha = pipeline.sha(data)
    job_id = pipeline.sha("up", file_sha, round(gsd, 4), calibrate)[:12]
    os.makedirs(pipeline.job_dir(job_id), exist_ok=True)
    path = os.path.join(pipeline.job_dir(job_id), "upload." + ("jpg" if fmt == "jpeg" else fmt))
    with open(path, "wb") as f:
        f.write(data)
    title = os.path.splitext(os.path.basename(file.filename or "upload"))[0][:60] or "upload"
    j = _submit(job_id, "upload", {"path": path, "gsd": gsd, "calibrate": calibrate, "title": title,
                                   "file_sha": file_sha})
    return {"job_id": j.id, "state": j.state}


@app.get("/api/jobs")
def recent(limit: int = 20):
    out = []
    if os.path.isdir(pipeline.JOBS_DIR):
        for name in os.listdir(pipeline.JOBS_DIR):
            p = os.path.join(pipeline.JOBS_DIR, name, "meta.json")
            if ID_RE.match(name) and os.path.exists(p):
                m = json.load(open(p, encoding="utf-8"))
                s = m.get("stats", {})
                out.append({"id": name, "title": m.get("title"), "kind": m.get("kind"), "created": m.get("created", 0),
                            "size_m": m.get("size_m"), "buildings": s.get("osm", 0) + s.get("auto", 0),
                            "calibrated": s.get("calibration", {}).get("mode") == "load"})
    out.sort(key=lambda r: -r["created"])
    return out[:limit]


@app.get("/api/jobs/{job_id}")
def status(job_id: str):
    return Response(json.dumps(_job(job_id).to_dict()), media_type="application/json", headers=NO_STORE)


def _done(job_id):
    j = _job(job_id)
    if j.state != "done":
        raise HTTPException(409, "Job is not finished yet")
    return pipeline.job_dir(job_id)


@app.get("/api/jobs/{job_id}/viewer", response_class=HTMLResponse)
def viewer(job_id: str):
    """The unchanged city3d viewer.html, plus one script tag that connects it to the app."""
    html = open(os.path.join(_done(job_id), "viewer.html"), encoding="utf-8").read()
    head, sep, tail = html.rpartition("</body>")
    return HTMLResponse(head + BRIDGE + sep + tail if sep else html + BRIDGE)


@app.get("/api/jobs/{job_id}/files/{name}")
def files(job_id: str, name: str):
    if name not in FILES:
        raise HTTPException(404, "Unknown file")
    path = os.path.join(_done(job_id), name)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type=FILES[name])


@app.get("/api/jobs/{job_id}/heightmap.zip")
def heightmap_zip(job_id: str):
    d = _done(job_id)
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", meta.get("title", job_id)).strip("_") or job_id
    readme = (f"DepthWizard height map - {meta.get('title')}\n\n"
              f"dsm_cm.png  16-bit grayscale, value = height above ground in CENTIMETRES (divide by 100 for metres)\n"
              f"            {meta['W']} x {meta['H']} px, {meta['gsd']:.3f} m per pixel, "
              f"area {meta['size_m'][0]} x {meta['size_m'][1]} m\n")
    readme += ("dsm_cm.pgw  world file - open dsm_cm.png in QGIS and set the CRS to EPSG:3857 (Web Mercator)\n"
               if meta.get("world_file") else "(uploaded image - no map position, so no world file)\n")
    cal = meta.get("stats", {}).get("calibration", {})
    readme += (f"\nHeights calibrated with city factor k={cal.get('k')} (fitted on {cal.get('fit_on')}).\n"
               if cal.get("mode") == "load" else "\nRaw model heights (no city calibration).\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n in ("dsm_cm.png", "dsm_cm.pgw"):
            if os.path.exists(os.path.join(d, n)):
                z.write(os.path.join(d, n), f"{base}_{n}")
        z.writestr("README.txt", readme)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{base}_heightmap.zip"'})
