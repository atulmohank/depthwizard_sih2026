r"""
DepthWizard - Stage 4a: find how many METRES one raw nDSM unit is.

Idea: nDSM = DSM - ground. Ground changes slowly, so between two nearby pixels:
      change in DSM (metres)  ~=  scale x  change in nDSM (raw units)
We measure thousands of such pairs (building edges give strong signal) and fit the scale robustly.

  python calibrate_scale.py --dsm data\raw\1_DSM\1_DSM --ndsm data\raw\1_DSM_normalisation
Output: runs\height_scale.json  (the number to use as --height-scale)
"""
import argparse, glob, json, os, re, sys

import cv2
import numpy as np

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**40))


def pid_of(path):
    m = re.search(r"potsdam_0*(\d+)_0*(\d+)", os.path.basename(path).lower())
    return f"{m.group(1)}_{m.group(2)}" if m else None


def index(folder, must=None):
    out = {}
    for f in sorted(glob.glob(os.path.join(folder, "**", "*"), recursive=True)):
        if not f.lower().endswith((".tif", ".tiff", ".jpg", ".png")) or os.path.getsize(f) == 0:
            continue
        if must and must not in os.path.basename(f).lower():
            continue
        p = pid_of(f)
        if p:
            out[p] = f
    return out


def read(path):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        try:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            a = np.array(Image.open(path))
        except Exception as e:
            sys.exit(f"[x] cannot read {path}: {e}")
    return a[:, :, 0] if a.ndim == 3 else a


def pairs_for_patch(dsm, n8, down, offsets, min_dn, max_samples, rng):
    dsm = dsm.astype(np.float32)
    bad = ~np.isfinite(dsm) | (dsm < -500) | (dsm > 9000)            # nodata
    sat = n8 >= 250                                                   # clipped at top of 8-bit range
    invalid = (bad | sat).astype(np.uint8)
    dsm = np.where(bad, 0, dsm)
    H, W = dsm.shape
    sz = (W // down, H // down)
    D = cv2.resize(dsm, sz, interpolation=cv2.INTER_AREA)
    N = cv2.resize(n8.astype(np.float32), sz, interpolation=cv2.INTER_AREA)
    inv = cv2.dilate(cv2.resize(invalid, sz, interpolation=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8)) > 0
    dD, dN = [], []
    for d in offsets:
        for ax in (0, 1):
            if ax == 1:
                a, b, ia, ib = D[:, d:], D[:, :-d], inv[:, d:], inv[:, :-d]
                na, nb = N[:, d:], N[:, :-d]
            else:
                a, b, ia, ib = D[d:], D[:-d], inv[d:], inv[:-d]
                na, nb = N[d:], N[:-d]
            dn = na - nb
            m = (~ia) & (~ib) & (np.abs(dn) >= min_dn)
            dD.append((a - b)[m]); dN.append(dn[m])
    dD, dN = np.concatenate(dD), np.concatenate(dN)
    if len(dD) > max_samples:
        k = rng.choice(len(dD), max_samples, replace=False)
        dD, dN = dD[k], dN[k]
    return dD, dN


def robust_scale(dD, dN, iters=5):
    s = float(np.median(dD / dN))
    keep = np.ones(len(dD), bool)
    for _ in range(iters):
        r = dD - s * dN
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-6
        keep = np.abs(r) < 3 * 1.4826 * mad
        s = float((dD[keep] * dN[keep]).sum() / (dN[keep] ** 2).sum())
    return s, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsm", required=True, help="folder with dsm_potsdam_XX_YY.tif (float, metres)")
    ap.add_argument("--ndsm", required=True, help="folder with the 8-bit normalised DSMs")
    ap.add_argument("--ndsm-version", default="lastools")
    ap.add_argument("--patches", type=int, default=12, help="how many patches to use (more = slower, steadier)")
    ap.add_argument("--down", type=int, default=4, help="work at 1/down resolution for speed")
    ap.add_argument("--out", default="runs/height_scale.json")
    a = ap.parse_args()

    dsm_f = index(a.dsm)
    n_f = index(a.ndsm, must=a.ndsm_version)
    ids = sorted(set(dsm_f) & set(n_f))
    print(f"[i] DSM files {len(dsm_f)} | nDSM ({a.ndsm_version}) files {len(n_f)} | matched {len(ids)}")
    if not ids:
        sys.exit("[x] no matching patches - check paths")
    rng = np.random.default_rng(0)
    ids = sorted(rng.choice(ids, min(a.patches, len(ids)), replace=False).tolist())

    all_D, all_N, per = [], [], {}
    for i, p in enumerate(ids, 1):
        dsm, n8 = read(dsm_f[p]), read(n_f[p])
        if n8.shape != dsm.shape:
            n8 = cv2.resize(n8, (dsm.shape[1], dsm.shape[0]), interpolation=cv2.INTER_NEAREST)
        if i == 1:
            fin = dsm[np.isfinite(dsm)]
            print(f"[i] sample {p}: DSM {dsm.dtype} {float(np.percentile(fin, 1)):.1f}..{float(np.percentile(fin, 99)):.1f} m | "
                  f"nDSM {n8.dtype} {int(n8.min())}..{int(n8.max())}")
        dD, dN = pairs_for_patch(dsm, n8, a.down, offsets=(3, 6), min_dn=15, max_samples=300_000, rng=rng)
        if len(dD) < 1000:
            print(f"  {p}: too few pairs ({len(dD)}), skipped"); continue
        s, _ = robust_scale(dD, dN)
        per[p] = s
        all_D.append(dD); all_N.append(dN)
        print(f"  [{i}/{len(ids)}] {p}: scale {s:.4f} m/unit  ({len(dD)} pairs)")

    dD, dN = np.concatenate(all_D), np.concatenate(all_N)
    s, keep = robust_scale(dD, dN)
    vals = np.array(list(per.values()))
    spread = float(vals.std() / max(abs(vals.mean()), 1e-9) * 100)

    # linearity check: is the scale the same for small and large height changes?
    bins = [(15, 40), (40, 80), (80, 140), (140, 250)]
    lin = {}
    for lo, hi in bins:
        m = keep & (np.abs(dN) >= lo) & (np.abs(dN) < hi)
        if m.sum() > 500:
            lin[f"{lo}-{hi}"] = round(float(np.median(dD[m] / dN[m])), 4)
    lin_vals = np.array(list(lin.values())) if lin else np.array([s])
    lin_dev = float((lin_vals.max() - lin_vals.min()) / max(abs(s), 1e-9) * 100)

    print("\n=========== RESULT ===========")
    print(f"scale = {s:.4f} metres per raw unit   ->  raw 255 = {255 * s:.1f} m (max representable height)")
    print(f"per-patch spread: {spread:.1f}%   (<5% = consistent)")
    print(f"linearity by height-change size: {lin}   deviation {lin_dev:.1f}%  (<10% = linear)")
    ok = spread < 5 and lin_dev < 10 and 0.01 < s < 2
    print("VERDICT:", "OK - use this scale" if ok else "CHECK - send this output to Claude before using it")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"height_scale_m_per_unit": round(s, 5), "max_height_m": round(255 * s, 2),
               "per_patch": {k: round(v, 5) for k, v in per.items()}, "spread_pct": round(spread, 2),
               "linearity": lin, "linearity_dev_pct": round(lin_dev, 2), "verdict_ok": bool(ok)},
              open(a.out, "w"), indent=2)
    print(f"[ok] saved {a.out}")
    print(f"NEXT: python evaluate.py --height-scale {s:.4f}")


if __name__ == "__main__":
    main()
