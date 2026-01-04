"""Datasets and patch extraction utilities."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def crop_lon_wrap(arr: np.ndarray, top: int, left: int, ph: int, pw: int) -> np.ndarray:
    C, H, W = arr.shape
    patch_lat = arr[:, top:top + ph, :]
    if left + pw <= W:
        return patch_lat[:, :, left:left + pw]
    r = (left + pw) - W
    return np.concatenate([patch_lat[:, :, left:W], patch_lat[:, :, 0:r]], axis=-1)


def _prepare_sample(
    x: np.ndarray,
    m: np.ndarray,
    *,
    fill_invalid: float,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    norm_clamp: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float32, copy=False)
    m = m.astype(np.float32, copy=False)

    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    m = (m > 0.5).astype(np.float32, copy=False)

    finite = np.isfinite(x)
    if not finite.all():
        m = m * finite.astype(np.float32)
        x = np.nan_to_num(x, nan=fill_invalid, posinf=fill_invalid, neginf=fill_invalid)

    if fill_invalid != 0.0 or not np.all(m > 0.5):
        x = np.where(m > 0.5, x, fill_invalid).astype(np.float32, copy=False)

    if mean is not None and std is not None:
        x = (x - mean[:, None, None]) / (std[:, None, None] + 1e-6)
        if norm_clamp > 0:
            x = np.clip(x, -norm_clamp, norm_clamp)
        x = np.where(m > 0.5, x, 0.0).astype(np.float32, copy=False)

    return x, m


class AMSUARandomPatchDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        files: List[str],
        patch_h: int,
        patch_w: int,
        samples: int,
        in_channels: int = 40,
        fill_invalid: float = 0.0,
        min_valid_frac: float = 0.01,
        max_resample: int = 30,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        norm_clamp: float = 10.0,
    ):
        self.data_dir = data_dir
        self.files = files
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.samples = samples
        self.in_channels = in_channels
        self.fill_invalid = float(fill_invalid)
        self.min_valid_frac = float(min_valid_frac)
        self.max_resample = int(max_resample)
        self.mean = mean
        self.std = std
        self.norm_clamp = float(norm_clamp)

        self._cache_path = None
        self._cache_X = None
        self._cache_M = None

    def __len__(self) -> int:
        return self.samples

    def _load(self, rel: str) -> Tuple[np.ndarray, np.ndarray]:
        import os

        path = os.path.join(self.data_dir, rel)
        if self._cache_path == path and self._cache_X is not None:
            return self._cache_X, self._cache_M
        d = np.load(path, allow_pickle=True)
        X = d["X"]
        M = d["M"]
        self._cache_path = path
        self._cache_X = X
        self._cache_M = M
        return X, M

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_patch = m_patch = None
        for _ in range(self.max_resample):
            rel = self.files[np.random.randint(0, len(self.files))]
            X, M = self._load(rel)

            if X.ndim != 4:
                raise ValueError(f"{rel}: X must be (T,F,H,W), got {X.shape}")
            T, F, H, W = X.shape
            if F != self.in_channels:
                raise ValueError(f"{rel}: feature={F} != in_channels={self.in_channels}")

            t = np.random.randint(0, T)
            x, m = _prepare_sample(
                X[t],
                M[t],
                fill_invalid=self.fill_invalid,
                mean=self.mean,
                std=self.std,
                norm_clamp=self.norm_clamp,
            )

            top = np.random.randint(0, H - self.patch_h + 1)
            left = np.random.randint(0, W)
            x_patch = crop_lon_wrap(x, top, left, self.patch_h, self.patch_w)
            m_patch = crop_lon_wrap(m, top, left, self.patch_h, self.patch_w)

            if float(m_patch.mean()) >= self.min_valid_frac:
                return torch.from_numpy(x_patch), torch.from_numpy(m_patch)

        return torch.from_numpy(x_patch), torch.from_numpy(m_patch)


class AMSUAEvalPatchDataset(Dataset):
    """
    用固定 seed 预生成 (file_idx, t, top, left)，保证评估可复现。
    """

    def __init__(
        self,
        data_dir: str,
        files: List[str],
        patch_h: int,
        patch_w: int,
        num_samples: int,
        seed: int,
        in_channels: int,
        mean: Optional[np.ndarray],
        std: Optional[np.ndarray],
        norm_clamp: float = 10.0,
        fill_invalid: float = 0.0,
    ):
        import os

        self.data_dir = data_dir
        self.files = files
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.num_samples = num_samples
        self.seed = seed
        self.in_channels = in_channels
        self.mean = mean
        self.std = std
        self.norm_clamp = float(norm_clamp)
        self.fill_invalid = float(fill_invalid)

        f0 = os.path.join(data_dir, files[0])
        d0 = np.load(f0, allow_pickle=True)
        X0 = d0["X"]
        T, F, H, W = X0.shape
        if F != in_channels:
            raise ValueError(f"feature mismatch: {F} vs {in_channels}")
        if H < patch_h or W < patch_w:
            raise ValueError("patch larger than data")

        rng = np.random.RandomState(seed)
        self.index = []
        for _ in range(num_samples):
            file_idx = int(rng.randint(0, len(files)))
            t = int(rng.randint(0, T))
            top = int(rng.randint(0, H - patch_h + 1))
            left = int(rng.randint(0, W))
            self.index.append((file_idx, t, top, left))

        self._cache_path = None
        self._cache_X = None
        self._cache_M = None

    def __len__(self) -> int:
        return self.num_samples

    def _load(self, rel: str) -> Tuple[np.ndarray, np.ndarray]:
        import os

        path = os.path.join(self.data_dir, rel)
        if self._cache_path == path and self._cache_X is not None:
            return self._cache_X, self._cache_M
        d = np.load(path, allow_pickle=True)
        X = d["X"]
        M = d["M"]
        self._cache_path = path
        self._cache_X = X
        self._cache_M = M
        return X, M

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx, t, top, left = self.index[i]
        rel = self.files[file_idx]
        X, M = self._load(rel)

        x, m = _prepare_sample(
            X[t],
            M[t],
            fill_invalid=self.fill_invalid,
            mean=self.mean,
            std=self.std,
            norm_clamp=self.norm_clamp,
        )

        x_patch = crop_lon_wrap(x, top, left, self.patch_h, self.patch_w)
        m_patch = crop_lon_wrap(m, top, left, self.patch_h, self.patch_w)

        return torch.from_numpy(x_patch), torch.from_numpy(m_patch)
