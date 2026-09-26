r"""
DepthWizard - Stage 4b: final before/after table in METRES (for the deck)

  python evaluate.py --height-scale 0.XXXX
  (optional) --weights runs\run1\best.pt

Output:
  runs\stage4_results.json
  runs\stage4_table.md     <- paste into PPT
"""
import argparse, json, math, os

import numpy as np
import torch
from torch.utils.data import DataLoader

from train import Tiles, align_scale_shift, forward, load_model

OBJ_M = 2.5  # pixels taller than 2.5 m = buildings/trees (standard nDSM threshold)


@torch.no_grad()
def run(model, loader, device, amp, div, scale, align):
    model.eval()
    s = {"abs": 0.0, "sq": 0.0, "n": 0, "obj_abs": 0.0, "obj_n": 0, "w1": 0, "w2": 0, "gnd_abs": 0.0, "gnd_n": 0}
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        gt_t = y.to(device).float() / div
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            p = forward(model, x)
        p = p.float()
        if align:
            p = align_scale_shift(p, gt_t)
        pm, gm = p * div * scale, gt_t * div * scale          # -> metres
        e = (pm - gm).abs()
        s["abs"] += e.sum().item(); s["sq"] += (e ** 2).sum().item(); s["n"] += e.numel()
        s["w1"] += int((e <= 1).sum()); s["w2"] += int((e <= 2).sum())
        o = gm > OBJ_M
        s["obj_abs"] += e[o].sum().item(); s["obj_n"] += int(o.sum())
        s["gnd_abs"] += e[~o].sum().item(); s["gnd_n"] += int((~o).sum())
    n = max(s["n"], 1)
    return {"MAE_m": s["abs"] / n, "RMSE_m": math.sqrt(s["sq"] / n),
            "buildings_trees_MAE_m": s["obj_abs"] / max(s["obj_n"], 1),
            "ground_MAE_m": s["gnd_abs"] / max(s["gnd_n"], 1),
            "within_1m_pct": 100 * s["w1"] / n, "within_2m_pct": 100 * s["w2"] / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height-scale", type=float, required=True, help="from runs\\height_scale.json")
    ap.add_argument("--weights", default="runs/run1/best.pt")
    ap.add_argument("--data", default="data/tiles")
    ap.add_argument("--model", default="small")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else None
    ds = Tiles(a.data, "val")
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    div = 25.5
    print(f"[i] {len(ds)} validation tiles (patches never seen in training) | scale {a.height_scale} m/unit")

    res = {}
    m, _ = load_model(a.model, device)
    print("[1/3] raw pretrained model (given best-case scale fit using the true answer)...")
    res["raw_model_aligned"] = run(m, dl, device, amp, div, a.height_scale, align=True)
    del m; torch.cuda.empty_cache() if device.type == "cuda" else None

    m, meta = load_model(a.model, device, a.weights)
    div = meta.get("div", div)
    print("[2/3] fine-tuned model, NO help (true real-world use)...")
    res["finetuned"] = run(m, dl, device, amp, div, a.height_scale, align=False)
    print("[3/3] fine-tuned model, same scale fit as raw (like-for-like)...")
    res["finetuned_aligned"] = run(m, dl, device, amp, div, a.height_scale, align=True)

    raw, ft, fta = res["raw_model_aligned"], res["finetuned"], res["finetuned_aligned"]
    def imp(a_, b_, lower_better=True):
        return f"{(a_ - b_) / a_ * 100:.0f}% better" if lower_better else f"+{b_ - a_:.0f} pts"
    rows = [
        ("Overall height error (MAE)", "MAE_m", "m", True),
        ("RMSE", "RMSE_m", "m", True),
        ("Buildings/trees error", "buildings_trees_MAE_m", "m", True),
        ("Ground error", "ground_MAE_m", "m", True),
        ("Pixels within 1 m", "within_1m_pct", "%", False),
        ("Pixels within 2 m", "within_2m_pct", "%", False),
    ]
    lines = ["| Metric | Raw model* | Fine-tuned (ours, no help) | Change |", "|---|---|---|---|"]
    for name, k, u, lb in rows:
        lines.append(f"| {name} | {raw[k]:.2f} {u} | **{ft[k]:.2f} {u}** | {imp(raw[k], ft[k], lb)} |")
    lines.append("")
    lines.append(f"*Raw model got a per-tile scale fit using the true heights (best case for it). "
                 f"Like-for-like, fine-tuned with the same fit: MAE {fta['MAE_m']:.2f} m.")
    lines.append(f"Validation: {len(ds)} tiles from 4 ISPRS Potsdam patches never used in training. "
                 f"Scale {a.height_scale} m per nDSM unit, derived from the official DSM.")
    table = "\n".join(lines)
    print("\n" + table)
    os.makedirs("runs", exist_ok=True)
    json.dump({"height_scale": a.height_scale, **{k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res.items()}},
              open("runs/stage4_results.json", "w"), indent=2)
    open("runs/stage4_table.md", "w", encoding="utf-8").write(table + "\n")
    print("\n[ok] saved runs\\stage4_results.json and runs\\stage4_table.md")


if __name__ == "__main__":
    main()
