"""Tables and figures for the evaluation (reads the per-case DataFrame only)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy import stats

from .config import CROP, HD95_EMPTY_PENALTY_MM, REGIONS
from .data import load_nifti, load_seg, nifti_path, region_mask, uncrop

VOLUME_BINS_ML = [0, 1, 5, 20, 50, np.inf]
OUTCOMES = ["both_present", "fn_empty_pred", "fp_empty_gt", "both_empty"]
LABEL_CMAP = ListedColormap(["#e41a1c", "#4daf4a", "#ffd92f"])  # NCR, ED, ET
REGION_COLORS = {"WT": "#1f77b4", "TC": "#d62728", "ET": "#ff7f0e"}


def _mean_sd(s):
    return f"{s.mean():.3f} ± {s.std():.3f}" if len(s) else "-"


def _med_iqr(s, p=3):
    if not len(s):
        return "-"
    q1, q2, q3 = np.percentile(s, [25, 50, 75])
    return f"{q2:.{p}f} [{q1:.{p}f}, {q3:.{p}f}]"


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _bin_labels():
    e = VOLUME_BINS_ML
    return [f"({e[i]:g}, {e[i + 1]:g}] mL" if np.isfinite(e[i + 1]) else f"> {e[i]:g} mL"
            for i in range(len(e) - 1)]


def add_volume_bins(df):
    df = df.copy()
    df["gt_bin"] = pd.cut(df.gt_ml, VOLUME_BINS_ML, labels=_bin_labels(), right=True)
    return df


# ---------------------------------------------------------------- tables
def summary_markdown(df, cases, meta):
    df = add_volume_bins(df)
    s = [f"# Validation results",
         "",
         f"Checkpoint: `{meta['weights']}.pt` (epoch {meta['epoch']}). "
         f"Val n = {meta['n_val']} cases (no background filter); train n = {meta['n_train']}.",
         "Metrics computed in the original 240x240x155 grid against full ground truth.",
         f"Empty-mask convention: both empty -> Dice 1, HD95 0; exactly one empty -> Dice 0, "
         f"HD95 {HD95_EMPTY_PENALTY_MM:.2f} mm.",
         ""]

    s += ["## Region overlap (all val cases, BraTS convention)", ""]
    rows = []
    for r in REGIONS:
        d = df[df.region == r]
        rows.append([r, len(d), _mean_sd(d.dice), _med_iqr(d.dice), _mean_sd(d.hd95), _med_iqr(d.hd95, 2)])
    s += [_md_table(["Region", "n", "Dice mean ± SD", "Dice median [IQR]",
                     "HD95 mm mean ± SD", "HD95 mm median [IQR]"], rows), ""]

    s += ["## Region overlap excluding convention-defined cases", "",
          "Dice over cases with GT present; HD95 over cases with both GT and prediction present "
          "(penalty values excluded, since 373 mm dominates any mean).", ""]
    rows = []
    for r in REGIONS:
        d = df[df.region == r]
        g = d[d.gt_ml > 0]
        b = d[d.outcome == "both_present"]
        rows.append([r, len(g), _mean_sd(g.dice), _med_iqr(g.dice), len(b), _med_iqr(b.hd95, 2)])
    s += [_md_table(["Region", "n GT>0", "Dice mean ± SD", "Dice median [IQR]",
                     "n both present", "HD95 mm median [IQR]"], rows), ""]

    s += ["## Empty-mask outcomes", ""]
    rows = [[r] + [int((df[df.region == r].outcome == o).sum()) for o in OUTCOMES] for r in REGIONS]
    s += [_md_table(["Region", "both present", "FN: GT>0, pred empty",
                     "FP: GT empty, pred>0", "both empty"], rows), ""]

    s += ["## Volume agreement (mL; 1 mm³ voxel = 0.001 mL)", "",
          "Bias and limits of agreement are pred − GT (Bland–Altman). "
          "Relative error only over GT>0.", ""]
    rows = []
    for r in REGIONS:
        d = df[df.region == r]
        diff = d.vol_err_ml
        bias, sd = diff.mean(), diff.std()
        rel = d.rel_vol_err.dropna().abs()
        rho = stats.pearsonr(d.gt_ml, d.pred_ml)[0] if d.gt_ml.std() > 0 else np.nan
        rows.append([r, len(d), f"{bias:+.2f}", f"[{bias - 1.96 * sd:+.2f}, {bias + 1.96 * sd:+.2f}]",
                     f"{d.abs_vol_err_ml.mean():.2f}", _med_iqr(d.abs_vol_err_ml, 2),
                     _med_iqr(rel * 100, 1), f"{rho:.3f}"])
    s += [_md_table(["Region", "n", "bias", "95% LoA", "MAE", "abs err median [IQR]",
                     "|rel err| % median [IQR]", "Pearson r"], rows), ""]

    s += ["## Dice by ground-truth volume (GT>0 cases)", "",
          "Spearman ρ between GT volume and Dice quantifies size dependence.", ""]
    rows = []
    for r in REGIONS:
        g = df[(df.region == r) & (df.gt_ml > 0)]
        rho, p = stats.spearmanr(g.gt_ml, g.dice) if len(g) > 2 else (np.nan, np.nan)
        for b, gb in g.groupby("gt_bin", observed=False):
            if not len(gb):
                continue
            rows.append([r, b, len(gb), _med_iqr(gb.dice), f"{(gb.dice < 0.5).mean():.0%}",
                         f"{(gb.outcome == 'fn_empty_pred').mean():.0%}"])
        rows.append([r, f"**Spearman ρ = {rho:.2f} (p = {p:.1e})**", len(g), "", "", ""])
    s += [_md_table(["Region", "GT volume", "n", "Dice median [IQR]", "Dice < 0.5",
                     "missed entirely"], rows), ""]

    if df.grade.notna().any() and set(df.grade.dropna()) != {"unknown"}:
        s += ["## By grade (GT>0 cases)", ""]
        rows = []
        for (r, gr), g in df[df.gt_ml > 0].groupby(["region", "grade"]):
            rows.append([r, gr, len(g), _med_iqr(g.dice), _med_iqr(g.gt_ml, 1)])
        s += [_md_table(["Region", "Grade", "n", "Dice median [IQR]", "GT mL median [IQR]"], rows), ""]

    s += ["## Crop loss on val cases", ""]
    rows = []
    for r in REGIONS:
        c = df[(df.region == r) & (df.gt_ml > 0)]
        lost = c[c.crop_loss > 0]
        # Max achievable Dice if a fraction f of GT is unreachable: 2(1-f)/(2-f).
        ceiling = (2 * (1 - lost.crop_loss) / (2 - lost.crop_loss)).min() if len(lost) else 1.0
        rows.append([r, len(c), len(lost), f"{c.crop_loss.max():.2e}", f"{ceiling:.4f}"])
    s += [_md_table(["Region", "n GT>0", "cases with clipped tumor", "max fraction clipped",
                     "worst-case Dice ceiling"], rows), ""]

    s += ["## Worst 10 cases by WT Dice", ""]
    wide = df.pivot(index="case_id", columns="region", values=["dice", "hd95", "gt_ml", "pred_ml"])
    worst = wide["dice"]["WT"].sort_values().index[:10]
    rows = []
    for cid in worst:
        rows.append([cid, cases.loc[cid, "grade"],
                     f"{wide.loc[cid, ('gt_ml', 'WT')]:.1f}", f"{wide.loc[cid, ('pred_ml', 'WT')]:.1f}",
                     *(f"{wide.loc[cid, ('dice', r)]:.3f}" for r in REGIONS),
                     f"{wide.loc[cid, ('hd95', 'WT')]:.1f}",
                     f"{wide.loc[cid, ('gt_ml', 'ET')]:.2f}"])
    s += [_md_table(["Case", "Grade", "WT GT mL", "WT pred mL", "WT Dice", "TC Dice", "ET Dice",
                     "WT HD95", "ET GT mL"], rows), ""]
    return "\n".join(s)


# ---------------------------------------------------------------- figures
def fig_dice_hd_vs_volume(df, path):
    df = add_volume_bins(df)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), sharex="col")
    for j, r in enumerate(REGIONS):
        d = df[(df.region == r) & (df.gt_ml > 0)]
        n_empty = int(((df.region == r) & (df.gt_ml == 0)).sum())
        fn = d.outcome == "fn_empty_pred"
        a = ax[0, j]
        a.scatter(d.gt_ml[~fn], d.dice[~fn], s=18, alpha=0.7, color=REGION_COLORS[r])
        a.scatter(d.gt_ml[fn], d.dice[fn], s=40, marker="x", color="k", label="pred empty")
        med = d.groupby("gt_bin", observed=True).agg(v=("gt_ml", "median"), dice=("dice", "median"))
        a.plot(med.v, med.dice, "k-o", ms=4, lw=1.5, label="bin median")
        for e in VOLUME_BINS_ML[1:-1]:
            a.axvline(e, color="0.85", lw=0.8, zorder=0)
        a.set(xscale="log", ylim=(-0.03, 1.03), ylabel="Dice",
              title=f"{r}: Dice vs GT volume (n={len(d)}; {n_empty} empty-GT excluded)")
        a.legend(loc="lower right", fontsize=8)
        a.grid(alpha=0.3)
        b = df[(df.region == r) & (df.outcome == "both_present")]
        a = ax[1, j]
        a.scatter(b.gt_ml, b.hd95, s=18, alpha=0.7, color=REGION_COLORS[r])
        a.set(xscale="log", yscale="symlog", xlabel="GT volume (mL, log)", ylabel="HD95 (mm)",
              title=f"{r}: HD95 vs GT volume (both present, n={len(b)})")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_bland_altman(df, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, r in zip(ax, REGIONS):
        d = df[df.region == r]
        mean = (d.pred_ml + d.gt_ml) / 2
        diff = d.pred_ml - d.gt_ml
        bias, sd = diff.mean(), diff.std()
        a.scatter(mean, diff, s=18, alpha=0.7, color=REGION_COLORS[r])
        a.axhline(bias, color="k", lw=1.2, label=f"bias {bias:+.2f}")
        for k in (-1.96, 1.96):
            a.axhline(bias + k * sd, color="k", ls="--", lw=1)
        a.axhline(0, color="0.6", lw=0.8)
        a.set(xlabel="(pred + GT) / 2  [mL]", ylabel="pred − GT  [mL]",
              title=f"{r}: LoA [{bias - 1.96 * sd:+.1f}, {bias + 1.96 * sd:+.1f}] mL")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_volume_scatter(df, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    for a, r in zip(ax, REGIONS):
        d = df[df.region == r]
        pos = np.concatenate([d.gt_ml[d.gt_ml > 0], d.pred_ml[d.pred_ml > 0]])
        lo = max(pos.min() / 2, 1e-3) if len(pos) else 1e-3
        hi = max(pos.max() * 1.5, lo * 10) if len(pos) else 1.0
        # Zero volumes can't sit on a log axis; clamp them to `lo` and mark them.
        x, y = d.gt_ml.clip(lower=lo), d.pred_ml.clip(lower=lo)
        zero = (d.gt_ml == 0) | (d.pred_ml == 0)
        a.scatter(x[~zero], y[~zero], s=18, alpha=0.7, color=REGION_COLORS[r])
        a.scatter(x[zero], y[zero], s=40, marker="x", color="k", label="a volume is 0 (clamped)")
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        a.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
              xlabel="GT volume (mL)", ylabel="predicted volume (mL)", title=f"{r}: volume")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_boxplots(df, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    rng = np.random.default_rng(0)
    for a, col, sub, title in [
        (ax[0], "dice", df, "Dice (all val cases)"),
        (ax[1], "hd95", df[df.outcome == "both_present"], "HD95 mm (both present)"),
    ]:
        data = [sub[sub.region == r][col].values for r in REGIONS]
        a.boxplot(data, tick_labels=list(REGIONS), showfliers=False)
        for i, (r, v) in enumerate(zip(REGIONS, data), 1):
            a.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=10, alpha=0.5, color=REGION_COLORS[r])
        a.set_title(title)
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_gallery(df, cases, out_dir, n):
    gdir = Path(out_dir) / "gallery"
    gdir.mkdir(exist_ok=True)
    picks = []
    for r in ("WT", "ET"):
        d = df[(df.region == r) & (df.outcome != "both_empty")].sort_values(["dice", "gt_ml"])
        picks += [(r, row) for _, row in d.head(n).iterrows()]
    for r, row in picks:
        cid = row.case_id
        case_dir = cases.loc[cid, "case_dir"]
        flair = load_nifti(nifti_path(case_dir, "flair"))[0]
        gt = load_seg(case_dir)[0]
        pred = uncrop(np.load(Path(out_dir) / "preds" / f"{cid}.npz")["pred"])
        ref = region_mask(gt, r) if row.gt_ml > 0 else region_mask(pred, r)
        z = int(ref.sum((0, 1)).argmax())
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
        for a, lab, title in [(ax[0], None, "FLAIR"), (ax[1], gt, "ground truth"), (ax[2], pred, "prediction")]:
            a.imshow(flair[:, :, z].T, cmap="gray", origin="lower")
            if lab is not None:
                sl = lab[:, :, z].T
                a.imshow(np.ma.masked_where(sl == 0, sl), cmap=LABEL_CMAP, vmin=1, vmax=3,
                         alpha=0.55, origin="lower", interpolation="nearest")
            a.add_patch(Rectangle((CROP[0].start, CROP[1].start), CROP[0].stop - CROP[0].start,
                                  CROP[1].stop - CROP[1].start, fill=False, ec="cyan", lw=0.8, ls="--"))
            a.set_title(title)
            a.axis("off")
        in_z = CROP[2].start <= z < CROP[2].stop
        fig.suptitle(f"{cid} ({cases.loc[cid, 'grade']}) — worst {r}: Dice {row.dice:.3f}, "
                     f"HD95 {row.hd95:.1f} mm, GT {row.gt_ml:.2f} mL, pred {row.pred_ml:.2f} mL, "
                     f"{row.outcome}; slice z={z}{'' if in_z else ' (OUTSIDE crop z-range)'}\n"
                     "red NCR/NET · green edema · yellow ET · cyan box = crop", fontsize=9)
        fig.tight_layout()
        fig.savefig(gdir / f"{r}_{cid}.png", dpi=110)
        plt.close(fig)


def write_all(df, cases, out_dir, meta, n_gallery=6):
    out_dir = Path(out_dir)
    (out_dir / "summary.md").write_text(summary_markdown(df, cases, meta), encoding="utf-8")
    fig_dice_hd_vs_volume(df, out_dir / "dice_hd95_vs_volume.png")
    fig_bland_altman(df, out_dir / "bland_altman.png")
    fig_volume_scatter(df, out_dir / "volume_scatter.png")
    fig_boxplots(df, out_dir / "boxplots.png")
    if n_gallery > 0:
        fig_gallery(df, cases, out_dir, n_gallery)
