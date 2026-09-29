"""Train the 3D U-Net on preprocessed data.

Writes to --run-dir: last.pt (every epoch, resumable), best.pt (lowest val
loss), history.csv, history.png, config.json.

Usage:
  python train.py --work-dir D:/data/brats_work --run-dir runs/unet_e10 --epochs 10
  python train.py --work-dir D:/data/brats_work --run-dir runs/unet_e10 --epochs 20 --resume
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from brats3d.config import MODALITIES, NUM_CLASSES, REGIONS, SEED, WorkPaths
from brats3d.dataset import BratsNpy
from brats3d.losses import DiceFocalLoss
from brats3d.model import UNet3D


def build_model(cfg):
    return UNet3D(len(MODALITIES), NUM_CLASSES, cfg["base_filters"], cfg["norm"])


def hard_region_dice(pred, target):
    """Batch-pooled hard Dice per region in crop space (training monitor only)."""
    out = {}
    for r, labels in REGIONS.items():
        lab = torch.tensor(labels, device=pred.device)
        p, g = torch.isin(pred, lab), torch.isin(target, lab)
        denom = p.sum() + g.sum()
        out[r] = 1.0 if denom == 0 else (2 * (p & g).sum() / denom).item()
    return out


def run_epoch(model, loader, loss_fn, device, amp, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    losses, dices = [], {r: [] for r in REGIONS}
    with torch.set_grad_enabled(train):
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            losses.append(loss.item())
            for r, d in hard_region_dice(logits.argmax(1), y).items():
                dices[r].append(d)
    return float(np.mean(losses)), {r: float(np.mean(v)) for r, v in dices.items()}


def plot_history(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist.epoch, hist.train_loss, label="train")
    ax[0].plot(hist.epoch, hist.val_loss, label="val")
    ax[0].set(xlabel="epoch", ylabel="Dice + focal loss", title="Loss")
    for r in REGIONS:
        ax[1].plot(hist.epoch, hist[f"val_dice_{r}"], label=f"val {r}")
    ax[1].set(xlabel="epoch", ylabel="hard Dice (crop space)", title="Val Dice", ylim=(0, 1))
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--epochs", type=int, default=10, help="total epochs (including resumed ones)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base-filters", type=int, default=16)
    ap.add_argument("--norm", choices=["instance", "none"], default="instance",
                    help="'none' reproduces the original notebook architecture")
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 autocast")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    paths = WorkPaths(args.work_dir)
    split = json.loads(paths.splits.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    print(f"device {device}{' (' + torch.cuda.get_device_name() + ')' if device.type == 'cuda' else ''}, amp={amp}")
    print(f"train {len(split['train'])} cases, val {len(split['val'])} cases")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = {k: v for k, v in vars(args).items() if k not in ("resume",)}
    cfg["modalities"] = list(MODALITIES)

    train_dl = DataLoader(BratsNpy(paths, split["train"], augment=True), batch_size=args.batch_size,
                          shuffle=True, num_workers=args.workers, pin_memory=True,
                          persistent_workers=args.workers > 0, drop_last=True)
    val_dl = DataLoader(BratsNpy(paths, split["val"]), batch_size=1, shuffle=False,
                        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

    model = build_model(cfg).to(device)
    loss_fn = DiceFocalLoss(NUM_CLASSES, class_weights=[0.25] * NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    start, best, rows = 0, float("inf"), []
    if args.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"], ck["best_val_loss"]
        rows = pd.read_csv(run / "history.csv").to_dict("records")[:start]
        print(f"resumed from epoch {start}")
    (run / "config.json").write_text(json.dumps(cfg, indent=2))

    for epoch in range(start + 1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_dice = run_epoch(model, train_dl, loss_fn, device, amp, optimizer, scaler)
        va_loss, va_dice = run_epoch(model, val_dl, loss_fn, device, amp)
        row = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
               **{f"train_dice_{r}": v for r, v in tr_dice.items()},
               **{f"val_dice_{r}": v for r, v in va_dice.items()},
               "minutes": (time.time() - t0) / 60}
        rows.append(row)
        print(f"epoch {epoch:3d} | loss {tr_loss:.4f} / {va_loss:.4f} | val Dice "
              + " ".join(f"{r} {v:.3f}" for r, v in va_dice.items()) + f" | {row['minutes']:.1f} min")

        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": epoch, "config": cfg,
                 "best_val_loss": min(best, va_loss)}
        if va_loss < best:
            best = va_loss
            torch.save(state, run / "best.pt")
        torch.save(state, run / "last.pt")
        hist = pd.DataFrame(rows)
        hist.to_csv(run / "history.csv", index=False)
        plot_history(hist, run / "history.png")

    print(f"done. checkpoints in {run}")


if __name__ == "__main__":
    main()
