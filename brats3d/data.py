"""Raw BraTS 2020 I/O and preprocessing (no TensorFlow dependency)."""
import re
from pathlib import Path

import nibabel as nib
import numpy as np

from .config import CROP, FULL_SHAPE, MODALITIES, REGIONS

CASE_RE = re.compile(r"^BraTS20_Training_\d{3}$")
NIFTI_EXTS = (".nii", ".nii.gz")
ALL_MODALITIES = ("t1", "t1ce", "t2", "flair")


def find_cases(data_root):
    """Map case id -> case directory for every BraTS20_Training_XXX folder."""
    cases = {}
    for p in sorted(Path(data_root).rglob("BraTS20_Training_*")):
        if p.is_dir() and CASE_RE.match(p.name):
            cases.setdefault(p.name, p)
    if not cases:
        raise FileNotFoundError(f"No BraTS20_Training_XXX folders under {data_root}")
    return cases


def find_name_mapping(data_root):
    hits = sorted(Path(data_root).rglob("name_mapping.csv"))
    return hits[0] if hits else None


def nifti_path(case_dir, suffix):
    case_dir = Path(case_dir)
    for ext in NIFTI_EXTS:
        p = case_dir / f"{case_dir.name}_{suffix}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{case_dir.name}: missing _{suffix}.nii[.gz]")


def fix_missing_seg(case_dir):
    """Rename a misnamed segmentation file to <case>_seg.nii.

    The Kaggle copy of BraTS 2020 ships case 355's label map as
    'W39_1998.09.19_Segm.nii'. Any case with no _seg file and exactly one
    unrecognised NIfTI gets that file renamed. Returns a log string or None.
    """
    case_dir = Path(case_dir)
    try:
        nifti_path(case_dir, "seg")
        return None
    except FileNotFoundError:
        pass
    known = {f"{case_dir.name}_{m}" for m in ALL_MODALITIES}
    candidates = [
        p for p in case_dir.iterdir()
        if p.name.endswith(NIFTI_EXTS) and p.name.split(".nii")[0] not in known
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"{case_dir.name}: no _seg file and {len(candidates)} candidates")
    src = candidates[0]
    ext = ".nii.gz" if src.name.endswith(".gz") else ".nii"
    dst = case_dir / f"{case_dir.name}_seg{ext}"
    src.rename(dst)
    return f"{case_dir.name}: renamed {src.name} -> {dst.name}"


def load_nifti(path):
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if data.shape != FULL_SHAPE:
        raise ValueError(f"{path}: shape {data.shape}, expected {FULL_SHAPE}")
    return data, tuple(float(z) for z in img.header.get_zooms()[:3]), img.affine


def load_seg(case_dir):
    """Label map with BraTS label 4 remapped to 3. Returns (seg, spacing, affine)."""
    data, spacing, affine = load_nifti(nifti_path(case_dir, "seg"))
    seg = np.rint(data).astype(np.uint8)
    labels = set(np.unique(seg).tolist())
    if not labels <= {0, 1, 2, 4}:
        raise ValueError(f"{Path(case_dir).name}: unexpected labels {sorted(labels)}")
    seg[seg == 4] = 3
    return seg, spacing, affine


def zscore_brain(vol):
    """Z-score over nonzero (brain) voxels; background stays 0.

    Replaces the original per-slice MinMaxScaler, which (via reshape(-1, 155))
    normalised every axial slice independently.
    """
    vol = vol.astype(np.float32)
    brain = vol > 0
    out = np.zeros_like(vol)
    if brain.any():
        v = vol[brain]
        out[brain] = (v - v.mean()) / max(float(v.std()), 1e-8)
    return out


def load_image(case_dir, modalities=MODALITIES):
    """Normalised (240,240,155,C) float32 stack in `modalities` order."""
    chans = [zscore_brain(load_nifti(nifti_path(case_dir, m))[0]) for m in modalities]
    return np.stack(chans, axis=-1)


def region_mask(seg, region):
    return np.isin(seg, REGIONS[region])


def crop_loss(seg):
    """Fraction of each region's voxels that fall outside CROP (NaN if region empty)."""
    inside = np.zeros(seg.shape, bool)
    inside[CROP] = True
    out = {}
    for r in REGIONS:
        m = region_mask(seg, r)
        n = int(m.sum())
        out[r] = float((m & ~inside).sum() / n) if n else float("nan")
    return out


def uncrop(pred_crop):
    """Paste a CROP_SHAPE label map into a zero FULL_SHAPE volume."""
    full = np.zeros(FULL_SHAPE, np.uint8)
    full[CROP] = pred_crop
    return full
