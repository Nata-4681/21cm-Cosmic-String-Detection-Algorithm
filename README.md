Will update later.

1. To run, install requirements.txt

2. Run "python train.py"

3. Run "python predict.py --num N" where N (int) is the number of images to predict the signal for.

## Parameters (train.py)

### Geometry & signal distribution
- `--N` : image size (N×N pixels)
- `--rect_w` : rectangle width (pixels)
- `--rect_h` : rectangle height (pixels)
- `--p_signal` : probability an image contains a rectangle
- `--snr_min` : minimum injected SNR for signal images
- `--snr_max` : maximum injected SNR for signal images

### Dataset size & optimization
- `--train_samples` : number of training images **per epoch**
- `--val_samples` : number of validation images **per epoch**
- `--batch_size` : images per optimization step
- `--epochs` : total number of epochs
- `--lr` : learning rate
- `--seed` : random seed (controls reproducibility of generation)

### Postprocessing (used for validation metrics / viz)
- `--threshold` : probability cutoff to binarize the heatmap
- `--min_area` : minimum blob size (pixels) to count as a detection

### Output & model size
- `--outdir` : output folder (default: `outputs`)
- `--base_channels` : model width (bigger = stronger/slower)

### Resume
- `--resume` : path to a checkpoint (usually `.../latest.pt`) to continue training
- `--run_name` : optional run folder name (otherwise timestamped)

---

## Useful commands

### Fast “sanity check” run (quick results)
```bash
python train.py --N 128 --train_samples 1000 --val_samples 200 --epochs 3
```