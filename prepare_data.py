"""Preprocess BraTS 2020 training data, measure crop loss, write a case-level split.

Outputs in --work-dir:
  images/<case>.npy   float16 (128,128,128,C), per-channel brain z-score, cropped
  masks/<case>.npy    uint8   (128,128,128), labels {0,1,2,3}, cropped
  cases.csv           per-case metadata, crop loss per region, train eligibility
  splits.json         {"train": [...filtered], "val": [...all], ...}
  crop_report.md      how much tumor the fixed crop clips

Usage:
  python prepare_data.py --data-root D:/data/brats2020 --work-dir D:/data/brats_work
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from brats3d.config import CROP, MIN_FG_FRACTION, MODALITIES, REGIONS, SEED, WorkPaths
from brats3d.data import (crop_loss, find_cases, find_name_mapping, fix_missing_seg,
                          load_image, load_seg, nifti_path, load_nifti)


def process_case(args):
    case_id, case_dir, work_dir, overwrite = args
    paths = WorkPaths(work_dir)
    seg, spacing, _ = load_seg(case_dir)
    if overwrite or not (paths.image(case_id).exists() and paths.mask(case_id).exists()):
        img = load_image(case_dir, MODALITIES)
        np.save(paths.image(case_id), img[CROP].astype(np.float16))
        np.save(paths.mask(case_id), seg[CROP])
    flair = load_nifti(nifti_path(case_dir, "flair"))[0] > 0
    inside = np.zeros(flair.shape, bool)
    inside[CROP] = True
    seg_crop = seg[CROP]
    row = {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "spacing": "x".join(f"{s:g}" for s in spacing),
        "fg_frac_cropped": float((seg_crop > 0).mean()),
        "brain_loss": float((flair & ~inside).sum() / max(flair.sum(), 1)),
    }
    for r, labels in REGIONS.items():
        row[f"{r}_ml"] = float(np.isin(seg, labels).sum()) * float(np.prod(spacing)) / 1000
    for r, v in crop_loss(seg).items():
        row[f"crop_loss_{r}"] = v
    return row


def stratified_split(df, val_frac, seed):
    rng = np.random.default_rng(seed)
    val = []
    for _, g in df.groupby("grade", dropna=False):
        ids = sorted(g.case_id)
        rng.shuffle(ids)
        val += ids[: int(round(len(ids) * val_frac))]
    return set(val)


def crop_report(df, split):
    lines = ["# Fixed-crop tumor loss", "",
             f"Crop `[56:184, 56:184, 13:141]` of 240x240x155. "
             f"Loss = fraction of a region's GT voxels outside the crop.", ""]
    for name, sub in [("All cases", df), ("Val cases", df[df.case_id.isin(split["val"])])]:
        lines += [f"## {name} (n={len(sub)})", "",
                  "| Region | cases with region | cases with any loss | mean loss | max loss | worst case |",
                  "|---|---|---|---|---|---|"]
        for r in REGIONS:
            c = sub[f"crop_loss_{r}"].dropna()
            worst = sub.loc[c.idxmax(), "case_id"] if len(c) and c.max() > 0 else "-"
            lines.append(f"| {r} | {len(c)} | {(c > 0).sum()} | {c.mean():.2e} | {c.max():.2e} | {worst} |")
        lines += ["", f"Brain (FLAIR>0) voxels outside crop: mean {sub.brain_loss.mean():.2%}, "
                      f"max {sub.brain_loss.max():.2%}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="folder containing the unzipped Kaggle dataset")
    ap.add_argument("--work-dir", required=True, help="output folder for preprocessed data")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    paths = WorkPaths(args.work_dir)
    paths.images.mkdir(parents=True, exist_ok=True)
    paths.masks.mkdir(parents=True, exist_ok=True)

    cases = find_cases(args.data_root)
    print(f"found {len(cases)} cases")
    for d in cases.values():
        msg = fix_missing_seg(d)  # case 355 in the Kaggle copy
        if msg:
            print(msg)

    jobs = [(cid, d, args.work_dir, args.overwrite) for cid, d in cases.items()]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, row in enumerate(ex.map(process_case, jobs), 1):
            rows.append(row)
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")
    df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)

    nm = find_name_mapping(args.data_root)
    if nm is not None:
        m = pd.read_csv(nm)[["BraTS_2020_subject_ID", "Grade"]]
        df = df.merge(m.rename(columns={"BraTS_2020_subject_ID": "case_id", "Grade": "grade"}),
                      on="case_id", how="left")
    else:
        df["grade"] = "unknown"

    # Split ALL cases at the patient level first; the background filter applies to train only.
    val_ids = stratified_split(df, args.val_frac, args.seed)
    df["split"] = np.where(df.case_id.isin(val_ids), "val", "train")
    df["train_eligible"] = df.fg_frac_cropped > MIN_FG_FRACTION
    train_all = df[df.split == "train"]
    split = {
        "seed": args.seed,
        "val_frac": args.val_frac,
        "stratified_by": "grade" if nm is not None else None,
        "min_fg_fraction_train": MIN_FG_FRACTION,
        "train": sorted(train_all[train_all.train_eligible].case_id),
        "train_excluded_by_bg_filter": sorted(train_all[~train_all.train_eligible].case_id),
        "val": sorted(val_ids),
    }
    df.to_csv(paths.cases_csv, index=False)
    paths.splits.write_text(json.dumps(split, indent=2))
    report = crop_report(df, split)
    paths.crop_report.write_text(report, encoding="utf-8")

    print(f"train {len(split['train'])} (excluded by bg filter: {len(split['train_excluded_by_bg_filter'])}), "
          f"val {len(split['val'])} (unfiltered)")
    print(report)


if __name__ == "__main__":
    main()
