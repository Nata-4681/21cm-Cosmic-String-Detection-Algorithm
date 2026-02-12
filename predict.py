from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from data import RectHeatmapDataset
from model import TinyUNet
from postprocess import detect_from_probmap, centroid_from_mask, estimate_snr_from_blob


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/checkpoint.pt")
    p.add_argument("--num", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min_area", type=int, default=80)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(Path(args.ckpt), map_location="cpu")
    train_args = ckpt.get("args", {})

    N = int(train_args.get("N", 256))
    rect_w = int(train_args.get("rect_w", 20))
    rect_h = int(train_args.get("rect_h", 14))
    p_signal = float(train_args.get("p_signal", 0.5))
    snr_min = float(train_args.get("snr_min", 3.0))
    snr_max = float(train_args.get("snr_max", 20.0))
    base_channels = int(train_args.get("base_channels", 16))

    model = TinyUNet(base=base_channels).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Use the same generator logic, but deterministic=False so it’s fresh each call
    ds = RectHeatmapDataset(
        n_samples=args.num,
        N=N,
        p_signal=p_signal,
        snr_range=(snr_min, snr_max),
        rect_w=rect_w,
        rect_h=rect_h,
        base_seed=np.random.randint(0, 1_000_000),
        deterministic=False,
    )
    ds.set_epoch(np.random.randint(0, 1_000_000))

    for i in range(args.num):
        x, y_true, meta = ds[i]
        x_b = x[None].to(device)

        with torch.no_grad():
            logits = model(x_b)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        img = x[0].cpu().numpy()  # normalized image (consistent with training)

        detected, blob = detect_from_probmap(prob, threshold=args.threshold, min_area=args.min_area)
        cx, cy = centroid_from_mask(blob)
        snr_hat = estimate_snr_from_blob(img, blob)

        print(f"\nExample {i+1}")
        print(f"  TRUE has_signal: {meta['has_signal']} | TRUE snr: {meta['snr']:.2f}")
        print(f"  DETECTED: {int(detected)}")
        if detected:
            print(f"  centroid (px): ({cx:.1f}, {cy:.1f})")
            print(f"  estimated SNR: {snr_hat:.2f}")

        # plots
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].imshow(img, origin="lower")
        axes[0].set_title("Input (normalized)")
        axes[0].axis("off")

        axes[1].imshow(prob, origin="lower", vmin=0, vmax=1)
        axes[1].set_title("Pred prob")
        axes[1].axis("off")

        axes[2].imshow(img, origin="lower")
        # overlay blob in red-ish via alpha
        overlay = np.zeros((*blob.shape, 4), dtype=float)
        overlay[blob] = [1, 0, 0, 0.35]
        axes[2].imshow(overlay, origin="lower")
        axes[2].set_title(f"Blob overlay (det={int(detected)})")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
