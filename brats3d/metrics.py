"""Evaluation metrics with explicit empty-mask handling.

Conventions (BraTS challenge):
  outcome        GT      pred    Dice   HD95
  both_empty     empty   empty   1.0    0.0
  fp_empty_gt    empty   >0      0.0    HD95_EMPTY_PENALTY_MM (373.13)
  fn_empty_pred  >0      empty   0.0    HD95_EMPTY_PENALTY_MM
  both_present   >0      >0      2|P&G|/(|P|+|G|)   95th pct of symmetric surface distances

HD95 follows medpy.metric.hd95: surface voxels are the mask minus its
6-connected erosion; distances in mm from each surface to the other are pooled
and the 95th percentile is taken.
"""
import numpy as np
from scipy import ndimage

from .config import HD95_EMPTY_PENALTY_MM

_STRUCT = ndimage.generate_binary_structure(3, 1)


def outcome(pred, gt):
    p, g = bool(pred.any()), bool(gt.any())
    if p and g:
        return "both_present"
    if g:
        return "fn_empty_pred"
    if p:
        return "fp_empty_gt"
    return "both_empty"


def dice(pred, gt):
    p, g = int(pred.sum()), int(gt.sum())
    if p + g == 0:
        return 1.0
    return 2.0 * int((pred & gt).sum()) / (p + g)


def _surface(mask):
    return mask & ~ndimage.binary_erosion(mask, structure=_STRUCT, border_value=0)


def surface_distances(pred, gt, spacing=(1.0, 1.0, 1.0)):
    """Pooled pred->gt and gt->pred surface distances (mm). Both masks non-empty."""
    # Restrict to the union's bounding box (+margin) for speed. All surface
    # voxels lie inside it, so EDT distances are exact.
    idx = np.argwhere(pred | gt)
    lo = np.maximum(idx.min(0) - 2, 0)
    hi = idx.max(0) + 3
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    p, g = pred[box], gt[box]
    # Masks touching the volume edge: pad so erosion treats outside as empty.
    p = np.pad(p, 1)
    g = np.pad(g, 1)
    ps, gs = _surface(p), _surface(g)
    dt_to_g = ndimage.distance_transform_edt(~gs, sampling=spacing)
    dt_to_p = ndimage.distance_transform_edt(~ps, sampling=spacing)
    return np.concatenate([dt_to_g[ps], dt_to_p[gs]])


def hd95(pred, gt, spacing=(1.0, 1.0, 1.0)):
    o = outcome(pred, gt)
    if o == "both_empty":
        return 0.0
    if o != "both_present":
        return HD95_EMPTY_PENALTY_MM
    return float(np.percentile(surface_distances(pred, gt, spacing), 95))


def region_metrics(pred, gt, spacing=(1.0, 1.0, 1.0)):
    """All per-region numbers for one case. pred/gt are boolean masks."""
    voxel_ml = float(np.prod(spacing)) / 1000.0
    gt_ml = float(gt.sum()) * voxel_ml
    pred_ml = float(pred.sum()) * voxel_ml
    return {
        "outcome": outcome(pred, gt),
        "dice": dice(pred, gt),
        "hd95": hd95(pred, gt, spacing),
        "gt_ml": gt_ml,
        "pred_ml": pred_ml,
        "vol_err_ml": pred_ml - gt_ml,
        "abs_vol_err_ml": abs(pred_ml - gt_ml),
        "rel_vol_err": (pred_ml - gt_ml) / gt_ml if gt_ml > 0 else float("nan"),
    }
