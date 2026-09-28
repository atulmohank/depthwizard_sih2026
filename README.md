# DepthWizard – SIH 2026

Satellite image → building height map → 3D city

## Folder structure

```
.
├── spike.py                    # one image -> height map -> 3D viewer (out/<name>/viewer.html)
├── city3d.py                   # lat/lon -> satellite tiles -> height map -> 3D city
├── prepare_tiles.py            # cut Potsdam RGB + nDSM rasters into training tiles
├── train.py                    # fine-tune the depth model (baseline / quick / full run)
├── calibrate_scale.py          # fit the depth -> metres height scale
├── evaluate.py                 # stage-4 metrics (MAE, RMSE, Pearson, ...)
├── viewer_template.html        # three.js viewer for spike.py
├── viewer_city_template.html   # three.js viewer for city3d.py
├── run.bat                     # shortcut: run.bat image.jpg [flags]
├── requirements.txt            # pipeline deps (torch, transformers, ...)
├── runs/                       # training configs, logs, result tables (weights not committed)
└── app/                        # FastAPI web app
    ├── server.py, pipeline.py
    ├── static/                 # index.html, app.js, app.css, viewer_bridge.js
    ├── requirements.txt        # web app extras
    └── run_app.bat
```

Not in the repo (git-ignored): `data/` (raw satellite / DSM tiles), `*.pt` model weights, `out/`, `images/`, `cache/`, `.venv/`.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install a CUDA build of PyTorch from https://pytorch.org if you have an NVIDIA GPU.

**Single image → 3D**

```bash
python spike.py --image city.jpg
```

Open `out/city/viewer.html` in a browser. Useful flags: `--model base`, `--grid 384`, `--invert`, `--detrend plane|off`.

**Location → 3D city**

```bash
python city3d.py --lat 12.9766 --lon 77.5993 --size 600 --name cubbon
```

**Web app**

```bash
pip install -r app/requirements.txt
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000. On Windows, `app\run_app.bat` does all of this.

**Training (optional)**

```bash
python prepare_tiles.py --rgb data/raw/2_Ortho_RGB --ndsm data/raw/1_DSM_normalisation
python train.py --name run1
```
