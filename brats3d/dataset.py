"""PyTorch dataset over the preprocessed .npy volumes."""
import numpy as np
import torch
from torch.utils.data import Dataset


class BratsNpy(Dataset):
    """Yields (image[C,D,H,W] float32, mask[D,H,W] int64, case_id).

    augment=True applies independent random flips on each spatial axis, which
    keeps labels valid. Replaces the old img_loader generator, which never
    shuffled and paired images/masks by os.listdir order.
    """

    def __init__(self, paths, case_ids, augment=False):
        self.paths = paths
        self.case_ids = list(case_ids)
        self.augment = augment

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, i):
        cid = self.case_ids[i]
        x = np.load(self.paths.image(cid)).astype(np.float32).transpose(3, 0, 1, 2)
        y = np.load(self.paths.mask(cid)).astype(np.int64)
        if self.augment:
            for ax in range(3):
                if torch.rand(()) < 0.5:  # torch RNG: seeded per DataLoader worker
                    x = np.flip(x, axis=ax + 1)
                    y = np.flip(y, axis=ax)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)), cid
