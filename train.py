"""
DepthWizard - Stage 3: fine-tune Depth Anything V2 (Small) on aerial tiles -> predicts height above ground

  0) BASELINE (raw model, ~2 min):   python train.py --baseline
  1) QUICK TEST (catch bugs, ~5 min): python train.py --quick
  2) FULL RUN (~1-3 hrs on RTX 4050): python train.py --name run1

Outputs in runs/<name>/:
  best.pt           best weights (lowest val error)   <- use with spike.py --weights
  log.csv           per-epoch metrics (for the deck table)
  preview_eXX.jpg   image | true height | predicted height, every epoch
"""
import argparse, csv, glob, json, math, os, random, time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ----------------------------------------------------------------------------- data
class Tiles(Dataset):
    def __init__(self, root, split, limit=0, augment=False, seed=0):
        self.root, self.split, self.aug = root, split, augment
        self.imgs = sorted(glob.glob(os.path.join(root, split, "img", "*.jpg")))
        if not self.imgs:
            raise SystemExit(f"[x] no tiles in {os.path.join(root, split, 'img')} - run Stage 2 first")
        if limit:
            random.Random(seed).shuffle(self.imgs)
            self.imgs = sorted(self.imgs[:limit])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        name = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p)[:, :, ::-1]  # BGR -> RGB
        h = cv2.imread(os.path.join(self.root, self.split, "height", name + ".png"), cv2.IMREAD_UNCHANGED)
        h = h.astype(np.float32)
        if self.aug:
            k = random.randint(0, 3)                 # top-down view: any rotation is valid
            img, h = np.rot90(img, k), np.rot90(h, k)
            if random.random() < 0.5:
                img, h = img[:, ::-1], h[:, ::-1]
            img = img.astype(np.float32) * random.uniform(0.8, 1.2) + random.uniform(-15, 15)
            img = np.clip(img, 0, 255)
        x = (np.ascontiguousarray(img).astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(np.ascontiguousarray(h).copy()), name


# ----------------------------------------------------------------------------- model
def load_model(key, device, weights=None):
    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(MODELS[key])
    meta = {}
    if weights:
        ck = torch.load(weights, map_location="cpu")
        model.load_state_dict(ck["state_dict"])
        meta = {k: v for k, v in ck.items() if k != "state_dict"}
        print(f"[i] loaded fine-tuned weights {weights} (epoch {meta.get('epoch')})")
    return model.to(device), meta


def forward(model, x):
    pred = model(pixel_values=x).predicted_depth  # (B, h, w)
    if pred.shape[-2:] != x.shape[-2:]:
        pred = F.interpolate(pred[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return pred


# ----------------------------------------------------------------------------- loss
def grad_loss(pred, gt, scales=4):
    """Penalise wrong edges -> sharp building walls instead of blobs."""
    total = 0.0
    p, g = pred[:, None], gt[:, None]
    for s in range(scales):
        if s:
            p, g = F.avg_pool2d(p, 2), F.avg_pool2d(g, 2)
        total = total + (p[..., :, 1:] - p[..., :, :-1] - (g[..., :, 1:] - g[..., :, :-1])).abs().mean()
        total = total + (p[..., 1:, :] - p[..., :-1, :] - (g[..., 1:, :] - g[..., :-1, :])).abs().mean()
    return total / scales


# ----------------------------------------------------------------------------- eval
def align_scale_shift(p, g):
    """Best s,b so that s*p+b ~= g (per image). Fair treatment for the raw model, which only knows RELATIVE depth."""
    P = torch.stack([p.flatten(1), torch.ones_like(p.flatten(1))], dim=-1)  # (B, N, 2)
    sol = torch.linalg.lstsq(P, g.flatten(1)[..., None]).solution            # (B, 2, 1)
    return (P @ sol)[..., 0].view_as(p)


@torch.no_grad()
def evaluate(model, loader, div, device, amp, align, unit_mult, max_batches=0):
    model.eval()
    abs_sum = sq_sum = obj_abs = 0.0
    n = n_obj = 0
    for b, (x, y, _) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x, gt = x.to(device, non_blocking=True), y.to(device, non_blocking=True).float() / div
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            pred = forward(model, x)
        pred = pred.float()
        if align:
            pred = align_scale_shift(pred, gt)
        err = (pred - gt).abs() * unit_mult
        abs_sum += err.sum().item(); sq_sum += (err ** 2).sum().item(); n += err.numel()
        obj = gt > 1.0  # raised things (buildings/trees)
        obj_abs += err[obj].sum().item(); n_obj += int(obj.sum().item())
    model.train()
    return {"mae": abs_sum / max(n, 1), "rmse": math.sqrt(sq_sum / max(n, 1)),
            "mae_objects": obj_abs / max(n_obj, 1)}


@torch.no_grad()
def save_preview(model, ds, div, device, amp, path, n=4):
    model.eval()
    rows = []
    for i in range(min(n, len(ds))):
        x, y, _ = ds[i * max(1, len(ds) // n)]
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            p = forward(model, x[None].to(device))[0].float().cpu().numpy()
        g = y.numpy() / div
        top = max(float(np.percentile(g, 99.5)), 1e-3)
        img = ((x.numpy().transpose(1, 2, 0) * STD + MEAN) * 255).clip(0, 255).astype(np.uint8)[:, :, ::-1]
        col = lambda a: cv2.applyColorMap((np.clip(a / top, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        rows.append(np.hstack([cv2.resize(im, (256, 256)) for im in (img, col(g), col(p))]))
    cv2.imwrite(path, np.vstack(rows))
    model.train()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tiles")
    ap.add_argument("--model", choices=MODELS, default="small")
    ap.add_argument("--name", default="run1")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation (effective batch = batch x accum)")
    ap.add_argument("--lr-enc", type=float, default=5e-6, help="encoder (pretrained backbone) - keep small")
    ap.add_argument("--lr-dec", type=float, default=5e-5, help="decoder/head - learns the new task")
    ap.add_argument("--grad-w", type=float, default=0.5, help="edge loss weight")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="500 train / 100 val tiles, 2 epochs")
    ap.add_argument("--baseline", action="store_true", help="only evaluate the RAW pretrained model")
    ap.add_argument("--init", default=None, help="start from a previous best.pt")
    ap.add_argument("--height-scale", type=float, default=None,
                    help="metres per raw nDSM unit (from Potsdam nDSM readme). Without it errors are in raw units")
    ap.add_argument("--grad-ckpt", action="store_true", help="save GPU memory (slower) if you get out-of-memory")
    a = ap.parse_args()
    if a.quick:
        a.limit_train, a.limit_val, a.epochs = a.limit_train or 500, a.limit_val or 100, 2
        a.name = a.name if a.name != "run1" else "quick"

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[i] GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)")
        amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.backends.cudnn.benchmark = True
    else:
        print("[!] no GPU found - this will be VERY slow. Check: python -c \"import torch;print(torch.cuda.is_available())\"")
        amp = None

    meta_path = os.path.join(a.data, "meta.json")
    units = json.load(open(meta_path))["height_units"] if os.path.exists(meta_path) else "raw_8bit"
    div = 25.5 if units == "raw_8bit" else 100.0     # model learns values ~0-10
    if units == "centimetres":
        unit_mult, unit_name = 1.0, "m"
    elif a.height_scale:
        unit_mult, unit_name = div * a.height_scale, "m"
    else:
        unit_mult, unit_name = div, "raw units"
    print(f"[i] height units: {units} -> errors reported in {unit_name}")

    val_ds = Tiles(a.data, "val", a.limit_val)
    val_dl = DataLoader(val_ds, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                        pin_memory=True, persistent_workers=a.workers > 0)
    model, _ = load_model(a.model, device, a.init)

    if a.baseline:
        m = evaluate(model, val_dl, div, device, amp, align=True, unit_mult=unit_mult)
        print(f"[BASELINE raw model, best-case scale+shift fit per tile]  MAE {m['mae']:.2f} {unit_name} | "
              f"RMSE {m['rmse']:.2f} | MAE on buildings/trees {m['mae_objects']:.2f}")
        os.makedirs("runs", exist_ok=True)
        json.dump({"model": a.model, "units": unit_name, **m}, open("runs/baseline.json", "w"), indent=2)
        print("[ok] saved runs/baseline.json  (this is the 'before' number for the deck)")
        return

    train_ds = Tiles(a.data, "train", a.limit_train, augment=True)
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                          pin_memory=True, persistent_workers=a.workers > 0)
    out = os.path.join("runs", a.name)
    os.makedirs(out, exist_ok=True)
    print(f"[i] train {len(train_ds)} | val {len(val_ds)} tiles | batch {a.batch}x{a.accum} | {a.epochs} epochs -> {out}")

    if a.grad_ckpt:
        try:
            model.gradient_checkpointing_enable(); print("[i] gradient checkpointing ON")
        except Exception as e:
            print(f"[!] gradient checkpointing not available: {e}")

    enc = [p for n, p in model.named_parameters() if n.startswith("backbone")]
    dec = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr_enc}, {"params": dec, "lr": a.lr_dec}], weight_decay=0.01)
    steps_total = a.epochs * (len(train_dl) // a.accum)
    warm = min(200, max(1, steps_total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps_total - warm))))
    scaler = torch.cuda.amp.GradScaler(enabled=(amp == torch.float16))

    with open(os.path.join(out, "config.json"), "w") as fp:
        json.dump(vars(a) | {"units": units, "div": div}, fp, indent=2)
    log_path = os.path.join(out, "log.csv")
    with open(log_path, "w", newline="") as fp:
        csv.writer(fp).writerow(["epoch", "train_loss", f"val_mae_{unit_name}", "val_rmse", "val_mae_objects",
                                 "val_mae_aligned", "minutes"])

    best = float("inf")
    t_start = time.time()
    model.train()
    for ep in range(1, a.epochs + 1):
        t_ep, run_loss, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for it, (x, y, _) in enumerate(train_dl, 1):
            x, gt = x.to(device, non_blocking=True), y.to(device, non_blocking=True).float() / div
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                pred = forward(model, x)
            pred = pred.float()
            loss = F.l1_loss(pred, gt) + a.grad_w * grad_loss(pred, gt)
            if not torch.isfinite(loss):
                print("[!] non-finite loss, skipping batch"); opt.zero_grad(set_to_none=True); continue
            scaler.scale(loss / a.accum).backward()
            if it % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
            run_loss += loss.item(); nb += 1
            if it % 50 == 0 or it == len(train_dl):
                el = time.time() - t_ep
                eta = el / it * (len(train_dl) - it)
                mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
                print(f"  ep {ep} [{it}/{len(train_dl)}] loss {run_loss / nb:.4f} | {it / el:.1f} it/s | "
                      f"epoch ETA {eta / 60:.1f} min | GPU mem {mem:.1f} GB")

        m = evaluate(model, val_dl, div, device, amp, align=False, unit_mult=unit_mult)
        ma = evaluate(model, val_dl, div, device, amp, align=True, unit_mult=unit_mult)
        save_preview(model, val_ds, div, device, amp, os.path.join(out, f"preview_e{ep:02d}.jpg"))
        mins = (time.time() - t_start) / 60
        print(f"[epoch {ep}] val MAE {m['mae']:.2f} {unit_name} | RMSE {m['rmse']:.2f} | "
              f"buildings/trees MAE {m['mae_objects']:.2f} | aligned MAE {ma['mae']:.2f} | {mins:.1f} min total")
        with open(log_path, "a", newline="") as fp:
            csv.writer(fp).writerow([ep, round(run_loss / max(nb, 1), 5), round(m["mae"], 4), round(m["rmse"], 4),
                                     round(m["mae_objects"], 4), round(ma["mae"], 4), round(mins, 1)])
        ck = {"state_dict": model.state_dict(), "model": a.model, "div": div, "units": units,
              "epoch": ep, "val": m}
        torch.save(ck, os.path.join(out, "last.pt"))
        if m["mae"] < best:
            best = m["mae"]
            torch.save(ck, os.path.join(out, "best.pt"))
            print(f"  -> new best, saved {os.path.join(out, 'best.pt')}")
    print(f"[ok] done in {(time.time() - t_start) / 60:.1f} min. Best val MAE {best:.2f} {unit_name}")


if __name__ == "__main__":
    main()
