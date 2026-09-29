"""Constants shared by preprocessing, training and evaluation."""
from pathlib import Path

import numpy as np

# Raw BraTS 2020 volumes are 240x240x155 at 1 mm isotropic spacing.
FULL_SHAPE = (240, 240, 155)
# Fixed crop inherited from the original notebook. prepare_data.py measures how
# much tumor it clips; evaluation pastes predictions back into FULL_SHAPE so any
# clipped tumor counts as a miss.
CROP = (slice(56, 184), slice(56, 184), slice(13, 141))
CROP_SHAPE = (128, 128, 128)

# Channel order of the network input. T1 is not used (see README limitations).
MODALITIES = ("t1ce", "t2", "flair")

# Labels after remapping BraTS label 4 -> 3:
#   0 background, 1 necrotic/non-enhancing core, 2 edema, 3 enhancing tumor
NUM_CLASSES = 4
REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}

# Training-only filter: drop cases whose cropped mask is >99% background.
MIN_FG_FRACTION = 0.01

# BraTS challenge convention when exactly one of GT/prediction is empty:
# HD95 = diagonal of the 240x240x155 mm volume.
HD95_EMPTY_PENALTY_MM = float(np.sqrt(240**2 + 240**2 + 155**2))  # 373.13

SEED = 42


class WorkPaths:
    """Layout of the preprocessed-data directory (--work-dir)."""

    def __init__(self, work_dir):
        self.root = Path(work_dir)
        self.images = self.root / "images"   # <case>.npy float16 (128,128,128,C)
        self.masks = self.root / "masks"     # <case>.npy uint8 (128,128,128)
        self.cases_csv = self.root / "cases.csv"
        self.splits = self.root / "splits.json"
        self.crop_report = self.root / "crop_report.md"

    def image(self, case_id):
        return self.images / f"{case_id}.npy"

    def mask(self, case_id):
        return self.masks / f"{case_id}.npy"
