"""
DepthWizard - Stage 2: prepare training tiles from ISPRS Potsdam
RGB orthophoto + nDSM (height above ground)  ->  518x518 tile pairs at satellite-like resolutions

STEP A (check files, 10 seconds):
  python prepare_tiles.py --rgb data/raw/2_Ortho_RGB --ndsm data/raw/1_DSM_normalisation --inspect
STEP B (make tiles, ~15-30 min):
  python prepare_tiles.py --rgb data/raw/2_Ortho_RGB --ndsm data/raw/1_DSM_normalisation

Output:
  data/tiles/train/img/*.jpg      RGB tiles
  data/tiles/train/height/*.png   16-bit height tiles (value = height in cm, or raw 8-bit value, see meta.json)
  data/tiles/val/...              held-out patches (never trained on -> honest evaluation later)
  data/tiles/meta.json            settings + counts
  data/tiles/preview.jpg          check this! image | height must line up
"""
import argparse, glob, json, os, re, sys

import cv2
import numpy as np

cv2.setNumThreads(4)
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**40))

NATIVE_GSD = 0.05  # Potsdam = 5 cm per pixel
VAL_PATCHES = {"2_11", "4_10", "5_11", "7_8"}  # held out, commonly used as validation in papers


def patch_id(path):
    """'top_potsdam_2_10_RGB.tif' and 'dsm_potsdam_02_10_normalized_lastools.jpg' -> '2_10'"""
    m = re.search(r"potsdam_0*(\d+)_0*(\d+)", os.path.basename(path).lower())
    return f"{m.group(1)}_{m.group(2)}" if m else None


def find(folder, exts=(".tif", ".tiff", ".jpg", ".png"), prefer=None):
    """Map patch id -> file. Skips empty (0-byte) files. If several versions exist
    (e.g. nDSM '_lastools' and '_ownapproach'), always takes the one containing `prefer`."""
    files = [f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
             if f.lower().endswith(exts)]
    out, skipped = {}, []
    for f in sorted(files):
        pid = patch_id(f)
        if not pid:
            continue
        if os.path.getsize(f) == 0:
            skipped.append(os.path.basename(f))
            continue
        if prefer:
            has = prefer in os.path.basename(f).lower()
            if pid in out and not has:
                continue          # keep the preferred version already found
            if pid in out and prefer in os.path.basename(out[pid]).lower() and not has:
                continue
            if not has and any(prefer in os.path.basename(g).lower() and patch_id(g) == pid
                               and os.path.getsize(g) > 0 for g in files):
                continue          # a preferred version exists for this patch -> skip this one
        out[pid] = f
    if skipped:
        print(f"[!] skipped {len(skipped)} empty file(s): {skipped}")
    return out


def read_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        sys.exit(f"[x] cannot read {path}")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:  # RGBIR -> drop IR
        im = im[:, :, :3]
    if im.dtype != np.uint8:
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return im  # BGR (OpenCV order)


def read_height(path):
    h = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if h is None:
        sys.exit(f"[x] cannot read {path}")
    if h.ndim == 3:
        h = h[:, :, 0]
    return h


def inspect(pairs, rgb_files, ndsm_files):
    print(f"[i] RGB files found : {len(rgb_files)}")
    print(f"[i] nDSM files found: {len(ndsm_files)}")
    print(f"[i] matched pairs   : {len(pairs)}  (val: {sorted(VAL_PATCHES & set(pairs))})")
    missing = sorted(set(rgb_files) ^ set(ndsm_files))
    if missing:
        print(f"[!] unmatched patch ids: {missing}")
    if not pairs:
        return
    versions = {}
    for p in pairs:
        v = "lastools" if "lastools" in pairs[p][1].lower() else ("ownapproach" if "ownapproach" in pairs[p][1].lower() else "other")
        versions[v] = versions.get(v, 0) + 1
    print(f"[i] nDSM versions used: {versions}   <- should be ONE version only")
    pid = sorted(pairs)[0]
    r, h = read_rgb(pairs[pid][0]), read_height(pairs[pid][1])
    print(f"[i] sample {pid}: RGB {r.shape} {r.dtype} | height {h.shape} {h.dtype} "
          f"min {float(h.min()):.2f} max {float(h.max()):.2f} mean {float(h.mean()):.2f}")
    if r.shape[:2] != h.shape[:2]:
        print("[!] RGB and height sizes differ -> height will be resized to match RGB")
    if h.dtype == np.uint8:
        print("[i] height is 8-bit (0-255): stored as-is; real metres come from the readme scale / Stage 5 calibration")
    else:
        print("[i] height is float: treated as METRES, stored as centimetres")


def tile_starts(n, tile, stride):
    if n <= tile:
        return [0]
    s = list(range(0, n - tile + 1, stride))
    if s[-1] != n - tile:
        s.append(n - tile)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True, help="folder with top_potsdam_*_RGB.tif")
    ap.add_argument("--ndsm", required=True, help="folder with normalised DSM files")
    ap.add_argument("--ndsm-version", default="lastools",
                    help="which nDSM version to use when several exist (default: lastools)")
    ap.add_argument("--out", default="data/tiles")
    ap.add_argument("--tile", type=int, default=518, help="Depth Anything native size")
    ap.add_argument("--gsd", default="0.10,0.20,0.30",
                    help="target resolutions in metres/pixel (satellite-like). 5cm native is too sharp")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="only first N patches (quick test)")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()

    rgb_files, ndsm_files = find(a.rgb), find(a.ndsm, prefer=a.ndsm_version)
    pairs = {p: (rgb_files[p], ndsm_files[p]) for p in rgb_files if p in ndsm_files}
    if a.inspect or not pairs:
        inspect(pairs, rgb_files, ndsm_files)
        if not pairs:
            sys.exit("[x] no matching pairs - check the folder paths")
        return

    gsds = [float(x) for x in a.gsd.split(",")]
    stride = max(1, int(a.tile * (1 - a.overlap)))
    ids = sorted(pairs)[: a.limit] if a.limit else sorted(pairs)
    for split in ("train", "val"):
        for sub in ("img", "height"):
            os.makedirs(os.path.join(a.out, split, sub), exist_ok=True)

    counts = {"train": 0, "val": 0}
    height_mode = None
    previews = []
    for i, pid in enumerate(ids, 1):
        split = "val" if pid in VAL_PATCHES else "train"
        rgb = read_rgb(pairs[pid][0])
        h = read_height(pairs[pid][1])
        if h.shape[:2] != rgb.shape[:2]:
            h = cv2.resize(h, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        mode = "raw_8bit" if h.dtype == np.uint8 else "centimetres"
        if height_mode and mode != height_mode:
            sys.exit(f"[x] {pid}: height format changed ({height_mode} -> {mode}); keep only one nDSM version in the folder")
        height_mode = mode
        if h.dtype == np.uint8:
            hf = h.astype(np.float32)
        else:
            hf = np.nan_to_num(h.astype(np.float32), nan=0.0)
            hf = np.clip(hf, 0, 400) * 100.0  # metres -> cm, clip junk (>400 m)

        n_patch = 0
        for g in gsds:
            f = NATIVE_GSD / g
            W, H = int(rgb.shape[1] * f), int(rgb.shape[0] * f)
            r_s = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
            h_s = cv2.resize(hf, (W, H), interpolation=cv2.INTER_AREA)
            for y in tile_starts(H, a.tile, stride):
                for x in tile_starts(W, a.tile, stride):
                    rt = r_s[y:y + a.tile, x:x + a.tile]
                    ht = h_s[y:y + a.tile, x:x + a.tile]
                    if rt.shape[0] < a.tile or rt.shape[1] < a.tile:
                        continue
                    if rt.std() < 3:  # blank tile
                        continue
                    name = f"p{pid}_g{int(g * 100):02d}_y{y}_x{x}"
                    cv2.imwrite(os.path.join(a.out, split, "img", name + ".jpg"), rt,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cv2.imwrite(os.path.join(a.out, split, "height", name + ".png"),
                                np.clip(ht, 0, 65535).astype(np.uint16))
                    counts[split] += 1
                    n_patch += 1
                    if len(previews) < 8 and n_patch == 3:
                        previews.append((rt, ht))
        print(f"[{i}/{len(ids)}] {pid} ({split}) -> {n_patch} tiles")

    # preview: image | height, 8 rows
    rows = []
    for rt, ht in previews:
        s = cv2.resize(rt, (256, 256))
        hn = ht / max(float(np.percentile(ht, 99)), 1e-6)
        c = cv2.applyColorMap((np.clip(hn, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        rows.append(np.hstack([s, cv2.resize(c, (256, 256))]))
    if rows:
        cv2.imwrite(os.path.join(a.out, "preview.jpg"), np.vstack(rows))

    meta = {"tile": a.tile, "gsd_m": gsds, "overlap": a.overlap, "height_units": height_mode,
            "val_patches": sorted(VAL_PATCHES), "counts": counts, "patches_used": len(ids)}
    with open(os.path.join(a.out, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)
    print(f"[ok] train {counts['train']} | val {counts['val']} tiles -> {a.out}")
    print(f"     CHECK {os.path.join(a.out, 'preview.jpg')}: buildings in the image must line up with hot colours")


if __name__ == "__main__":
    main()