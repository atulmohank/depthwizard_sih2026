"""
DepthWizard spike (SIH26175) - Step 1
One aerial/satellite image -> Depth Anything V2 -> relative height map -> 3D flythrough (viewer.html)

Usage:
  python spike.py --image city.jpg
  python spike.py --image city.jpg --model base --grid 384
Outputs (in ./out/<image-name>/):
  height_rel.npy     relative height, 0..1, full image size
  height_gray.png    16-bit grayscale height map
  height_color.png   colored height map (for the deck)
  side_by_side.png   image | height map (for the deck)
  viewer.html        open in Chrome -> 3D flythrough
"""
import argparse, base64, io, json, os, sys, time

import numpy as np
import cv2
from PIL import Image

MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
}


def load_image(path, max_side):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return img


def run_depth(img, model_key):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[i] device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else "  <- slow; check your CUDA torch install"))
    name = MODELS[model_key]
    print(f"[i] loading {name} (first run downloads weights)...")
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForDepthEstimation.from_pretrained(name).to(device).eval()

    inputs = proc(images=img, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        pred = model(**inputs).predicted_depth  # (1, h, w) relative inverse depth: bigger = closer
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(img.size[1], img.size[0]), mode="bicubic", align_corners=False
    ).squeeze().float().cpu().numpy()
    print(f"[i] inference {time.time() - t0:.2f}s")
    return pred


def run_finetuned(img, weights, tile=518, overlap=0.5, batch=4):
    """Fine-tuned model: predict tile-by-tile at native resolution (same as training), blend overlaps."""
    import torch
    from transformers import AutoModelForDepthEstimation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(weights, map_location="cpu")
    model = AutoModelForDepthEstimation.from_pretrained(MODELS[ck.get("model", "small")])
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    print(f"[i] fine-tuned weights: {weights} (epoch {ck.get('epoch')}) on {device}")
    mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
    a = np.array(img).astype(np.float32) / 255.0
    H, W = a.shape[:2]
    ph, pw = max(0, tile - H), max(0, tile - W)
    a = np.pad(a, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    Hp, Wp = a.shape[:2]
    step = max(1, int(tile * (1 - overlap)))
    ys = list(range(0, Hp - tile + 1, step)); xs = list(range(0, Wp - tile + 1, step))
    if ys[-1] != Hp - tile: ys.append(Hp - tile)
    if xs[-1] != Wp - tile: xs.append(Wp - tile)
    win = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32) + 1e-3
    acc = np.zeros((Hp, Wp), np.float32); wsum = np.zeros((Hp, Wp), np.float32)
    coords = [(y, x) for y in ys for x in xs]
    t0 = time.time()
    for i in range(0, len(coords), batch):
        cb = coords[i:i + batch]
        xb = np.stack([((a[y:y + tile, x:x + tile] - mean) / std).transpose(2, 0, 1) for y, x in cb])
        with torch.no_grad():
            p = model(pixel_values=torch.from_numpy(xb).to(device)).predicted_depth.float().cpu().numpy()
        for (y, x), pt in zip(cb, p):
            if pt.shape != (tile, tile):
                pt = cv2.resize(pt, (tile, tile))
            acc[y:y + tile, x:x + tile] += pt * win; wsum[y:y + tile, x:x + tile] += win
    print(f"[i] {len(coords)} tiles in {time.time() - t0:.1f}s")
    return (acc / wsum)[:H, :W]


def to_height(pred, invert):
    # Top-down view: "closer to camera" == "taller". So inverse depth ~ height (no flip by default).
    h = -pred if invert else pred
    lo, hi = np.percentile(h, 1), np.percentile(h, 99)  # clip outliers
    h = np.clip((h - lo) / max(hi - lo, 1e-6), 0, 1)
    return h.astype(np.float32)


def detrend(h, mode, ground_frac):
    """Remove the fake tilt/dome the model adds (it thinks it's a side-view photo).
    plane  : subtract best-fit tilted plane (fixes simple ramps)
    ground : estimate the 'ground surface' with a big min-filter, subtract it -> only
             things sticking up above local ground remain (classic DSM -> nDSM trick)"""
    if mode == "off":
        return h
    H, W = h.shape
    s = min(1.0, 512 / max(H, W))
    small = cv2.resize(h, (max(2, int(W * s)), max(2, int(H * s))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    if mode == "plane":
        yy, xx = np.mgrid[0:sh, 0:sw]
        A = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)]
        coef, *_ = np.linalg.lstsq(A, small.ravel(), rcond=None)
        base = (A @ coef).reshape(sh, sw).astype(np.float32)
    else:  # ground
        k = max(3, int(max(sh, sw) * ground_frac)) | 1  # must be bigger than the largest building
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        base = cv2.morphologyEx(small, cv2.MORPH_OPEN, ker)       # lower envelope = ground
        base = cv2.GaussianBlur(base, (0, 0), sigmaX=k / 2)        # smooth it
    base = cv2.resize(base, (W, H), interpolation=cv2.INTER_LINEAR)
    out = h - base
    if mode == "ground":
        out = np.clip(out, 0, None)
    lo, hi = np.percentile(out, 1), np.percentile(out, 99.5)
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def b64_png(arr_or_img, fmt="PNG", quality=90):
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def colorize(height):
    return cv2.applyColorMap((height * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def save_outputs(img, raw, fixed, outdir):
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "height_rel.npy"), fixed)
    cv2.imwrite(os.path.join(outdir, "height_gray.png"), (fixed * 65535).astype(np.uint16))
    c_raw, c_fix = colorize(raw), colorize(fixed)
    cv2.imwrite(os.path.join(outdir, "height_color_raw.png"), c_raw)
    cv2.imwrite(os.path.join(outdir, "height_color.png"), c_fix)
    rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(outdir, "side_by_side.png"), np.hstack([rgb, c_raw, c_fix]))  # image | raw | fixed
    return cv2.cvtColor(c_raw, cv2.COLOR_BGR2RGB), cv2.cvtColor(c_fix, cv2.COLOR_BGR2RGB)


def build_viewer(img, raw, fixed, col_raw, col_fix, grid, title, mode, outpath, template_path):
    W, H = img.size
    gw = grid if W >= H else max(2, int(grid * W / H))
    gh = grid if H >= W else max(2, int(grid * H / W))
    def pack(hm):
        small = cv2.resize(hm, (gw, gh), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (3, 3), 0)  # soften per-pixel noise spikes
        return base64.b64encode((small * 65535).astype("<u2").tobytes()).decode()

    tex_side = 2048
    s = min(1.0, tex_side / max(W, H))
    tex = img.resize((int(W * s), int(H * s)), Image.LANCZOS) if s < 1 else img

    data = {
        "title": title,
        "gw": gw, "gh": gh, "aspect": W / H,
        "mode": mode,
        "heightsRaw": pack(raw),
        "heights": pack(fixed),
        "photo": b64_png(tex, "JPEG", 88),
        "heatRaw": b64_png(Image.fromarray(col_raw).resize(tex.size, Image.LANCZOS), "JPEG", 88),
        "heat": b64_png(Image.fromarray(col_fix).resize(tex.size, Image.LANCZOS), "JPEG", 88),
    }
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("/*__DATA__*/null", json.dumps(data))
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", choices=MODELS.keys(), default="small")
    ap.add_argument("--max-side", type=int, default=1536, help="resize input so longest side <= this")
    ap.add_argument("--grid", type=int, default=320, help="3D mesh resolution (vertices per side)")
    ap.add_argument("--invert", action="store_true", help="flip height if buildings come out as pits")
    ap.add_argument("--weights", default=None, help="fine-tuned runs/<name>/best.pt (Stage 3)")
    ap.add_argument("--detrend", choices=["auto", "ground", "plane", "off"], default="auto",
                    help="remove fake tilt/dome. auto = ground for raw model, off for fine-tuned")
    ap.add_argument("--ground-size", type=float, default=0.12,
                    help="ground filter size as fraction of image; must exceed biggest building (try 0.08-0.25)")
    ap.add_argument("--reuse", action="store_true", help="reuse saved model output (skip model, fast re-tuning)")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    if not os.path.exists(a.image):
        sys.exit(f"[x] image not found: {a.image}")
    here = os.path.dirname(os.path.abspath(__file__))
    name = os.path.splitext(os.path.basename(a.image))[0]
    outdir = os.path.join(a.out, name)

    img = load_image(a.image, a.max_side)
    print(f"[i] image {img.size[0]}x{img.size[1]}")
    if a.detrend == "auto":
        a.detrend = "off" if a.weights else "ground"
    tag = "ft" if a.weights else a.model
    if a.weights:
        outdir = outdir + "_finetuned"
    cache = os.path.join(outdir, f"pred_{tag}.npy")
    if a.reuse and os.path.exists(cache):
        print("[i] reusing saved model output")
        pred = np.load(cache)
    else:
        pred = run_finetuned(img, a.weights) if a.weights else run_depth(img, a.model)
        os.makedirs(outdir, exist_ok=True)
        np.save(cache, pred)
    raw = to_height(pred, a.invert)
    fixed = detrend(raw, a.detrend, a.ground_size)
    print(f"[i] detrend: {a.detrend}" + (f" (ground-size {a.ground_size})" if a.detrend == "ground" else ""))
    c_raw, c_fix = save_outputs(img, raw, fixed, outdir)
    build_viewer(img, raw, fixed, c_raw, c_fix, a.grid, name, a.detrend,
                 os.path.join(outdir, "viewer.html"), os.path.join(here, "viewer_template.html"))
    print(f"[ok] done -> {outdir}")
    print(f"     open {os.path.join(outdir, 'viewer.html')} in Chrome")


if __name__ == "__main__":
    main()
