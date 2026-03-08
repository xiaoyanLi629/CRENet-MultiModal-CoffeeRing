"""
experiments/augmented_dataset.py
=================================
Augmented dataset wrapper for CRENet training on small datasets.

Coffee-ring patterns are approximately radially symmetric, so the
following transforms are physically valid:
  - Horizontal / vertical flip
  - 90°, 180°, 270° rotation
  - Mild Gaussian noise (simulates sensor noise)
  - Mild brightness/contrast jitter for optical images only

IMPORTANT: The same spatial transform (flip, rotation) is applied
consistently across all three modalities of the same sample, preserving
the physical correspondence between optical, SEM, and EDS data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import random


class AugmentedCoffeeRingDataset(Dataset):
    """
    Wraps a CoffeeRingDataset and applies consistent random augmentation
    across all three modalities during training.

    Args:
        base_dataset : a CoffeeRingDataset instance
        augment      : whether to apply augmentation (set False for val/test)
        noise_std    : std of additive Gaussian noise (applied to all modalities)
        color_jitter : range of brightness/contrast jitter for optical only
                       (tuple: (brightness, contrast), each in [0, max_delta])
    """

    def __init__(
        self,
        base_dataset,
        augment: bool = True,
        noise_std: float = 0.02,
        color_jitter: tuple = (0.1, 0.1),
    ):
        self.base         = base_dataset
        self.augment      = augment
        self.noise_std    = noise_std
        self.color_jitter = color_jitter

        # Available spatial transforms (consistent across modalities)
        # k = number of 90-degree rotations; flip_h/flip_v = axis flip
        self._spatial_choices = [
            (0, False, False),  # identity
            (1, False, False),  # 90°
            (2, False, False),  # 180°
            (3, False, False),  # 270°
            (0, True,  False),  # flip-H
            (0, False, True),   # flip-V
            (1, True,  False),  # 90° + flip-H
            (1, False, True),   # 90° + flip-V
        ]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]

        if not self.augment:
            return sample

        optical = sample['optical'].numpy()   # (3, H, W)
        sem     = sample['sem'].numpy()       # (1, H, W)
        eds     = sample['eds'].numpy()       # (7, H, W)
        label   = sample['label']

        # ── Pick one random spatial transform ──────────────────────────────
        k, flip_h, flip_v = random.choice(self._spatial_choices)

        optical = self._apply_spatial(optical, k, flip_h, flip_v)
        sem     = self._apply_spatial(sem,     k, flip_h, flip_v)
        eds     = self._apply_spatial(eds,     k, flip_h, flip_v)

        # ── Gaussian noise (all modalities) ───────────────────────────────
        if self.noise_std > 0:
            optical = optical + np.random.normal(0, self.noise_std, optical.shape)
            sem     = sem     + np.random.normal(0, self.noise_std, sem.shape)
            eds     = eds     + np.random.normal(0, self.noise_std / 2, eds.shape)

        # ── Colour jitter (optical only) ──────────────────────────────────
        b_range, c_range = self.color_jitter
        brightness = 1.0 + random.uniform(-b_range, b_range)
        contrast   = 1.0 + random.uniform(-c_range, c_range)
        optical    = optical * brightness
        optical    = (optical - optical.mean()) * contrast + optical.mean()

        # ── Clip to valid range ───────────────────────────────────────────
        optical = np.clip(optical, -3, 3).astype(np.float32)
        sem     = np.clip(sem,     -3, 3).astype(np.float32)
        eds     = np.clip(eds,      0, 1).astype(np.float32)

        return {
            'optical': torch.FloatTensor(optical),
            'sem':     torch.FloatTensor(sem),
            'eds':     torch.FloatTensor(eds),
            'label':   label,
        }

    @staticmethod
    def _apply_spatial(arr, k, flip_h, flip_v):
        """Apply k×90° rotation and optional flips to a (C, H, W) array."""
        # Rotate in H-W plane
        if k > 0:
            arr = np.rot90(arr, k=k, axes=(1, 2)).copy()
        if flip_h:
            arr = arr[:, :, ::-1].copy()
        if flip_v:
            arr = arr[:, ::-1, :].copy()
        return arr
