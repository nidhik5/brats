"""Evaluate a trained checkpoint on the (unfiltered) validation split.

Predictions are made on the 128^3 crop, then pasted back into the original
240x240x155 grid and scored against the FULL ground truth, so tumor clipped by
the crop counts as missed.

Writes to <run-dir>/eval_<weights>/:
  per_case.csv        one row per (case, region): Dice, HD95, volumes, outcome
  summary.md          region tables, volume agreement, Dice by volume bin, worst cases
  *.png               Dice-vs-volume, Bland-Altman, volume scatter, boxplots, crop loss
  gallery/*.png       worst cases (FLAIR + GT + prediction on the max-tumor slice)
  preds/<case>.npz    cropped predicted label maps (reused by --skip-predict)

Usage:
  python evaluate.py --work-dir D:/data/brats_work --run-dir runs/unet_e10
  python evaluate.py --work-dir D:/data/brats_work --run-dir runs/unet_e10 --skip-predict
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from brats3d.config import MODALITIES, REGIONS, WorkPaths
from brats3d.data import load_seg, region_mask, uncrop
from brats3d.metrics import region_metrics
from brats3d.model import UNet3D
from brats3d import report


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    if tuple(cfg["modalities"]) != tuple(MODALITIES):
        raise ValueError(f"checkpoint modalities {cfg['modalities']} != config {MODALITIES}")
    model = UNet3D(len(MODALITIES), 4, cfg["base_filters"], cfg["norm"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["epoch"]


@torch.no_grad()
def predict(model, paths, case_id, device, amp):
    x = np.load(paths.image(case_id)).astype(np.float32).transpose(3, 0, 1, 2)[None]
    x = torch.from_numpy(x).to(device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        logits = model(x)
    return logits.argmax(1)[0].to(torch.uint8).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--weights", choices=["last", "best"], default="last",
                    help="'last' (default) avoids selecting the checkpoint on the set we report on")
    ap.add_argument("--skip-predict", action="store_true", help="reuse saved preds/ and recompute metrics+report")
    ap.add_argument("--save-nifti", action="store_true", help="also write full-size prediction .nii.gz")
    ap.add_argument("--gallery", type=int, default=6, help="worst-N cases per region to render")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    paths = WorkPaths(args.work_dir)
    split = json.loads(paths.splits.read_text())
    cases = pd.read_csv(paths.cases_csv).set_index("case_id")
    out = Path(args.run_dir) / f"eval_{args.weights}"
    (out / "preds").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    epoch = None
    if not args.skip_predict:
        model, epoch = load_model(Path(args.run_dir) / f"{args.weights}.pt", device)
        print(f"loaded {args.weights}.pt (epoch {epoch}) on {device}")

    rows = []
    for i, cid in enumerate(split["val"], 1):
        pred_file = out / "preds" / f"{cid}.npz"
        if args.skip_predict:
            pred_crop = np.load(pred_file)["pred"]
        else:
            pred_crop = predict(model, paths, cid, device, amp)
            np.savez_compressed(pred_file, pred=pred_crop)
        pred = uncrop(pred_crop)
        gt, spacing, affine = load_seg(cases.loc[cid, "case_dir"])
        if args.save_nifti:
            nib.save(nib.Nifti1Image(np.where(pred == 3, 4, pred).astype(np.uint8), affine),
                     str(out / "preds" / f"{cid}_pred.nii.gz"))  # back to BraTS labels
        for r in REGIONS:
            m = region_metrics(region_mask(pred, r), region_mask(gt, r), spacing)
            rows.append({"case_id": cid, "region": r, "grade": cases.loc[cid, "grade"],
                         "crop_loss": cases.loc[cid, f"crop_loss_{r}"], **m})
        if i % 10 == 0 or i == len(split["val"]):
            print(f"  {i}/{len(split['val'])}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_case.csv", index=False)
    meta = {"weights": args.weights, "epoch": epoch, "n_val": len(split["val"]),
            "n_train": len(split["train"])}
    report.write_all(df, cases, out, meta, n_gallery=args.gallery)
    print(f"outputs in {out}  (start with summary.md)")


if __name__ == "__main__":
    main()
