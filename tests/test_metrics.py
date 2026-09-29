"""Metric sanity checks. Run: python -m pytest tests -q"""
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from brats3d.config import CROP, FULL_SHAPE, HD95_EMPTY_PENALTY_MM
from brats3d.data import crop_loss, region_mask
from brats3d.metrics import _surface, dice, hd95, region_metrics


def box(shape, lo, hi):
    m = np.zeros(shape, bool)
    m[tuple(slice(a, b) for a, b in zip(lo, hi))] = True
    return m


def brute_hd95(p, g, spacing):
    ps = np.argwhere(_surface(np.pad(p, 1))) * spacing
    gs = np.argwhere(_surface(np.pad(g, 1))) * spacing
    d = cdist(ps, gs)
    return np.percentile(np.concatenate([d.min(1), d.min(0)]), 95)


def test_identical():
    m = box((30, 30, 30), (5, 5, 5), (15, 15, 15))
    assert dice(m, m) == 1.0
    assert hd95(m, m) == 0.0


@pytest.mark.parametrize("spacing", [(1, 1, 1), (1, 1, 2.5)])
def test_hd95_matches_brute_force(spacing):
    rng = np.random.default_rng(0)
    p = box((40, 40, 40), (5, 6, 7), (20, 18, 25))
    g = box((40, 40, 40), (9, 4, 10), (28, 22, 30))
    p |= rng.random(p.shape) < 0.002  # stray false-positive voxels
    assert hd95(p, g, spacing) == pytest.approx(brute_hd95(p, g, np.array(spacing)))


def test_mask_touching_border():
    g = box((20, 20, 20), (0, 0, 0), (10, 10, 10))
    p = box((20, 20, 20), (0, 0, 0), (12, 10, 10))
    assert hd95(p, g) == pytest.approx(brute_hd95(p, g, np.ones(3)))


def test_empty_conventions():
    e = np.zeros((10, 10, 10), bool)
    m = box((10, 10, 10), (2, 2, 2), (4, 4, 4))
    assert (dice(e, e), hd95(e, e)) == (1.0, 0.0)
    assert (dice(m, e), hd95(m, e)) == (0.0, HD95_EMPTY_PENALTY_MM)
    assert (dice(e, m), hd95(e, m)) == (0.0, HD95_EMPTY_PENALTY_MM)
    r = region_metrics(e, m)
    assert r["outcome"] == "fn_empty_pred" and r["gt_ml"] == pytest.approx(0.008)
    assert np.isnan(region_metrics(m, e)["rel_vol_err"])


def test_regions_and_crop_loss():
    seg = np.zeros(FULL_SHAPE, np.uint8)
    seg[100:110, 100:110, 100:110] = 1          # inside crop
    seg[100:110, 100:110, 5:15] = 3             # z 5..14; crop starts at z=13 -> 8/10 outside
    assert region_mask(seg, "ET").sum() == 1000
    assert region_mask(seg, "WT").sum() == 2000
    loss = crop_loss(seg)
    assert loss["ET"] == pytest.approx(0.8)
    assert loss["WT"] == pytest.approx(0.4)
    assert CROP[2].start == 13
