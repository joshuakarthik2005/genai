# Paddy Leaf Disease Augmentation using DCGAN

Implementation of the case study *"Paddy Leaf Disease Augmentation Using DCGAN"*.
Rice-disease image datasets are small and imbalanced, so classifiers overfit and miss rare
diseases. This app trains one **DCGAN per minority disease class**, generates realistic synthetic
leaf images, **balances the dataset**, and measures the gain on a **ResNet-18 classifier**.

```
data/raw/<class>/*.jpg ──► stratified split ──► data/split/{train,test}
                                   │
              train (real) ────────┼──► DCGAN per minority class ──► runs/gan/<class>/generator.pt
                                   │                                         │
                                   └──► real + synthetic = data/balanced ◄───┘
                                                   │
             ResNet-18 baseline (real only)   vs   ResNet-18 augmented (balanced)
                                                   │
                         evaluated on the same REAL-only test split ──► accuracy, macro-F1, per-class F1
```

## Model (as in the case study)

| Generator | Discriminator |
|---|---|
| Input: 100-d noise vector | Input: 64×64×3 real / synthetic image |
| Transposed convolutions (4→8→16→32→64) | Strided convolutions (64→32→16→8→4) |
| BatchNorm + ReLU | BatchNorm + LeakyReLU(0.2) |
| Output: Tanh | Output: Sigmoid |

Preprocessing: resize to 64×64, normalise pixels to [-1, 1]. Training: Adam (lr 2e-4, β1 0.5),
BCE loss, N(0, 0.02) weight init, one-sided label smoothing (real = 0.9).

**DiffAugment** (Zhao et al., 2020) is applied to real and fake images before the discriminator.
Without it, the discriminator memorises tiny classes (e.g. 20 images) and the generator collapses
to noise; with it, the generator learns diverse leaves. Disable with `--no-augment`.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Put images in `data/raw/<class_name>/`, one folder per disease, e.g. from:
- Kaggle *Rice Leaf Diseases* (Bacterial blight, Brown spot, Leaf smut)
- Kaggle *Rice Diseases Image Dataset* (BrownSpot, Healthy, Hispa, LeafBlast)
- Mendeley *Rice Leaf Disease Image Samples* (Bacterial blight, Blast, Brown spot, Tungro)

Or upload a ZIP from the app. For a quick test without real data:

```bash
python make_demo_data.py   # procedural, deliberately imbalanced 4-class dataset
```

## Web app

```bash
streamlit run app.py
```

Pages: **Overview** · **1 Dataset** (upload, class distribution, real-only train/test split) ·
**2 Train DCGAN** (live loss curves, sample snapshots) · **3 Generate & balance** (real vs synthetic
grid, ZIP download, balanced dataset) · **4 Classifier** (baseline vs augmented: accuracy, macro-F1,
per-class F1/recall, confusion matrices) · **5 Predict** (upload a leaf, get the disease).

## Command line

```bash
python pipeline.py split
python pipeline.py train-gan --classes minority --iterations 5000   # or --classes all / "Blast,Tungro"
python pipeline.py balance
python pipeline.py classify --train data/split/train --name baseline  --epochs 15 --pretrained
python pipeline.py classify --train data/balanced    --name augmented --epochs 15 --pretrained
python pipeline.py compare
# or everything at once:
python pipeline.py all --iterations 5000 --epochs 15 --pretrained
```

Useful flags: `--width 32` (≈4× faster GAN on CPU), `--target N` (images per class after
balancing), `--img-size` (classifier input, default 128), `--pretrained` (ImageNet ResNet-18).

## Notes

- The test split contains only real images and is never seen by the GANs, so baseline vs augmented
  is a fair comparison. Watch macro-F1 and the minority-class F1 rather than plain accuracy.
- CPU works but is slow; on real photos use a GPU and 5k–20k GAN iterations per class.
- Synthetic images are 64×64 (DCGAN resolution) and are upscaled by the classifier's resize.

## Layout

```
app.py               Streamlit UI
pipeline.py          CLI
make_demo_data.py    procedural demo dataset
paddy_gan/models.py  Generator, Discriminator, ResNet-18 classifier
paddy_gan/data.py    preprocessing, split
paddy_gan/gan.py     DCGAN training (+ DiffAugment), generation
paddy_gan/augment.py dataset balancing
paddy_gan/classifier.py classifier training, metrics, prediction
```
