from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
import powerbox as pbox


def pk_powerlaw(k):
    return 0.1 * k ** -2.0


def make_powerbox_bg(N: int, boxlength: float, pk, seed: int) -> np.ndarray:
    pb = pbox.PowerBox(N=N, dim=2, pk=pk, boxlength=boxlength, seed=seed)
    return pb.delta_x().astype(np.float32)


def inject_rectangle(
    bg: np.ndarray,
    snr: float,
    rect_w: int,
    rect_h: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Inject a constant rectangle.

    SNR definition (per-pixel, matching your earlier toy):
      amplitude = snr * std(bg)
      so amplitude/std(bg) = snr

    Returns:
      img, mask, meta (true box + snr)
    """
    N = bg.shape[0]
    noise = float(bg.std() + 1e-12)
    amp = float(snr * noise)

    x0 = int(rng.integers(0, N - rect_w + 1))
    y0 = int(rng.integers(0, N - rect_h + 1))

    img = bg.copy()
    img[y0:y0 + rect_h, x0:x0 + rect_w] += amp

    mask = np.zeros_like(bg, dtype=np.float32)
    mask[y0:y0 + rect_h, x0:x0 + rect_w] = 1.0

    meta = {
        "has_signal": 1,
        "snr": float(snr),
        "x0": x0,
        "y0": y0,
        "w": rect_w,
        "h": rect_h,
    }
    return img, mask, meta


class RectHeatmapDataset(Dataset):
    """
    On-the-fly dataset that returns a segmentation target.

    Output:
      x:  [1, N, N] normalized image (float32)
      y:  [1, N, N] mask (float32 0/1)
      meta: dict with truth (for debugging/metrics)
    """

    def __init__(
        self,
        n_samples: int,
        N: int = 256,
        boxlength: float = 1.0,
        p_signal: float = 0.5,
        snr_range: tuple[float, float] = (3.0, 20.0),
        rect_w: int = 20,
        rect_h: int = 14,
        base_seed: int = 123,
        deterministic: bool = False,
    ):
        self.n_samples = n_samples
        self.N = N
        self.boxlength = boxlength
        self.p_signal = p_signal
        self.snr_range = snr_range
        self.rect_w = rect_w
        self.rect_h = rect_h
        self.base_seed = base_seed
        self.pk = pk_powerlaw

        # For validation: deterministic=True gives a stable, repeatable set
        self.deterministic = deterministic
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_samples

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        # Deterministic stream for val; epoch-varying stream for train
        if self.deterministic:
            ss = np.random.SeedSequence([self.base_seed, int(idx)])
        else:
            ss = np.random.SeedSequence([self.base_seed, self.epoch, int(idx)])
        return np.random.default_rng(ss)

    def __getitem__(self, idx: int):
        rng = self._rng_for_index(idx)

        pb_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32).item())
        bg = make_powerbox_bg(self.N, self.boxlength, self.pk, pb_seed)

        if rng.random() < self.p_signal:
            snr = float(rng.uniform(*self.snr_range))
            img, mask, meta = inject_rectangle(bg, snr, self.rect_w, self.rect_h, rng)
        else:
            img = bg
            mask = np.zeros_like(bg, dtype=np.float32)
            meta = {"has_signal": 0, "snr": 0.0, "x0": 0, "y0": 0, "w": 0, "h": 0}

        # normalize image for training stability
        img = (img - img.mean()) / (img.std() + 1e-6)

        x = torch.from_numpy(img[None, :, :])         # [1,N,N]
        y = torch.from_numpy(mask[None, :, :])        # [1,N,N]

        return x, y, meta
