from __future__ import annotations

import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from data import RectHeatmapDataset
from model import TinyUNet
from postprocess import detect_from_probmap, centroid_from_mask, estimate_snr_from_blob, iou


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--rect_w", type=int, default=20)
    p.add_argument("--rect_h", type=int, default=14)
    p.add_argument("--p_signal", type=float, default=0.5)
    p.add_argument("--snr_min", type=float, default=3.0)
    p.add_argument("--snr_max", type=float, default=20.0)

    p.add_argument("--train_samples", type=int, default=1000)
    p.add_argument("--val_samples", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=123)

    # postprocess params for tracking
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min_area", type=int, default=80)

    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--base_channels", type=int, default=16)
    return p.parse_args()


@torch.no_grad()
def save_viz_grid(model, val_ds, outdir: Path, epoch: int, device: str, threshold: float, min_area: int, k: int = 6):
    model.eval()
    k = min(k, len(val_ds))
    fig, axes = plt.subplots(k, 4, figsize=(12, 2.2 * k))

    for i in range(k):
        x, y, meta = val_ds[i]
        x_b = x[None].to(device)  # [1,1,N,N]
        logits = model(x_b)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        img = x[0].cpu().numpy()         # normalized image (for consistent viz)
        true = y[0].cpu().numpy() > 0.5

        detected, blob = detect_from_probmap(prob, threshold=threshold, min_area=min_area)

        axes[i, 0].imshow(img, origin="lower")
        axes[i, 0].set_title(f"Input (has={meta['has_signal']})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(prob, origin="lower", vmin=0, vmax=1)
        axes[i, 1].set_title("Pred prob")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(blob, origin="lower", vmin=0, vmax=1)
        axes[i, 2].set_title(f"Blob (det={int(detected)})")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(true, origin="lower", vmin=0, vmax=1)
        axes[i, 3].set_title("True mask")
        axes[i, 3].axis("off")

    plt.tight_layout()
    path = outdir / f"viz_epoch_{epoch:02d}.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir / "checkpoint.pt"

    train_ds = RectHeatmapDataset(
        n_samples=args.train_samples,
        N=args.N,
        p_signal=args.p_signal,
        snr_range=(args.snr_min, args.snr_max),
        rect_w=args.rect_w,
        rect_h=args.rect_h,
        base_seed=args.seed,
        deterministic=False,
    )
    val_ds = RectHeatmapDataset(
        n_samples=args.val_samples,
        N=args.N,
        p_signal=args.p_signal,
        snr_range=(args.snr_min, args.snr_max),
        rect_w=args.rect_w,
        rect_h=args.rect_h,
        base_seed=args.seed + 999,
        deterministic=True,
    )

    num_workers = 2
    pin_memory = torch.cuda.is_available()

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=max(16, args.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


    model = TinyUNet(base=args.base_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Handle class imbalance: positives are rare pixels.
    rect_area = max(1, args.rect_w * args.rect_h)
    total_area = args.N * args.N
    pos_weight = torch.tensor([(total_area - rect_area) / rect_area], dtype=torch.float32, device=device)

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)

        # ---- train ----
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [train]", dynamic_ncols=True)
        for x, y, meta in pbar:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = bce(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_dl)

        # ---- validate (detection + IoU + centroid + SNR metrics) ----
        model.eval()
        det_correct = 0
        det_total = 0
        ious = []
        loc_errs = []
        snr_errs = []

        with torch.no_grad():
            pbar = tqdm(val_dl, desc=f"Epoch {epoch+1}/{args.epochs} [val]", dynamic_ncols=True)
            for x, y, meta_batch in pbar:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                prob = torch.sigmoid(logits).cpu().numpy()     # [B,1,N,N]
                x_np = x.cpu().numpy()                         # normalized images
                y_np = y.cpu().numpy() > 0.5                   # true masks

                B = prob.shape[0]
                for i in range(B):
                    pmap = prob[i, 0]
                    true_mask = y_np[i, 0]
                    img = x_np[i, 0]

                    true_has = int(true_mask.sum() > 0)
                    detected, blob = detect_from_probmap(pmap, threshold=args.threshold, min_area=args.min_area)

                    det_correct += int(int(detected) == true_has)
                    det_total += 1

                    if true_has == 1:
                        ious.append(iou(blob, true_mask))

                        # centroid error (pixels). If missed detection, penalize with NaN -> we skip.
                        if detected:
                            cx_p, cy_p = centroid_from_mask(blob)
                            # true centroid from true mask
                            cx_t, cy_t = centroid_from_mask(true_mask)
                            loc_errs.append(np.hypot(cx_p - cx_t, cy_p - cy_t))

                            # SNR estimate from predicted blob vs "truth" SNR stored in meta
                            snr_hat = estimate_snr_from_blob(img, blob)
                            # meta_batch is a list-like of dicts from DataLoader; best effort:
                            # if this fails due to collate behavior, skip snr_err
                            try:
                                snr_true = float(meta_batch["snr"][i])  # if collated into dict of lists/tensors
                            except Exception:
                                snr_true = None
                            if snr_true is not None:
                                snr_errs.append(abs(snr_hat - snr_true))

        val_acc = det_correct / max(1, det_total)
        mean_iou = float(np.mean(ious)) if len(ious) else 0.0
        mean_loc = float(np.mean(loc_errs)) if len(loc_errs) else float("nan")
        mean_snr = float(np.mean(snr_errs)) if len(snr_errs) else float("nan")

        print(
            f"Epoch {epoch+1:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | val det acc {val_acc:.3f} | "
            f"val IoU {mean_iou:.3f} | val loc err(px) {mean_loc:.2f} | val |Δsnr| {mean_snr:.3f}"
        )

        # save a small visual panel each epoch
        save_viz_grid(model, val_ds, outdir, epoch + 1, device, args.threshold, args.min_area, k=6)

    # Save checkpoint
    torch.save(
        {"model_state": model.state_dict(), "args": vars(args)},
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
