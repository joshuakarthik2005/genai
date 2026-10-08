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

## 🔬 Research Novelty & Scientific Value

### Why ChatGPT / DALL-E Cannot Replace This
1. **Domain Fidelity**: Foundation models hallucinate generic symptoms from internet scrape data. Our DCGAN is mathematically constrained via adversarial optimization to match the precise lesion and color distributions of genuine rice diseases.
2. **Offline & Edge Deployable**: Rural smallholder farms lack internet connectivity. Our trained generator and ResNet classifiers run fully offline at zero inference API cost.
3. **Downstream Empirical Validation**: Synthetic samples are scientifically benchmarked on a strictly **held-out real-only test set** and evaluated with **FID (Fréchet Inception Distance)** and **Grad-CAM**.

---

## 3-Way Head-to-Head Experimental Comparison

| Method | Training Data | Description |
|---|---|---|
| **Baseline** | Real images only | Standard imbalanced training |
| **Traditional Augment** | Real + flip/rotate/jitter | Classical geometric & color transforms |
| **GAN Augment (Ours)** | Real + DCGAN synthetic | **Generative AI minority balancing** |

---

## Running the Complete Experiment

### 1. Locally via CLI (`run_experiment.py`)

Run the full end-to-end pipeline with FID calculation and Grad-CAM generation:

```bash
# Standard run (using raw Rice Leaf Disease Images)
python run_experiment.py --raw "Rice Leaf Disease Images/Rice Leaf Disease Images" --iterations 3000 --epochs 15 --fid

# Fast CPU test
python run_experiment.py --raw "Rice Leaf Disease Images/Rice Leaf Disease Images" --iterations 500 --epochs 5 --width 32 --img-size 64
```

Outputs generated in `runs/`:
- `runs/plots/overall_comparison.png`: 3-way Accuracy and Macro-F1 comparison bar chart
- `runs/plots/perclass_f1.png`: Per-class F1 score across all three approaches
- `runs/plots/confusion_*.png`: Confusion matrices for each training condition
- `runs/plots/training_curves.png`: Training loss & test metrics across epochs
- `runs/plots/gan_curves.png`: Discriminator and Generator loss dynamics
- `runs/plots/fid_scores.png`: Fréchet Inception Distance per disease class
- `runs/plots/gradcam_grid.png`: Visual explanation of classifier attention regions
- `runs/plots/summary_table.png`: Publication-ready results comparison table
- `runs/report/experiment_summary.json`: Complete quantitative JSON metrics

### 2. On Kaggle GPU (`paddy_dcgan_experiment.ipynb`)

A fully self-contained, 30-cell research notebook is ready to upload to Kaggle or Google Colab:
- Generated by `python build_notebook.py`
- Works with 2× T4 GPUs on Kaggle or local environments
- Contains automatic dataset path resolution, 3-way training, FID evaluation, Grad-CAM visualizations, and exportable figures

---

## Project Layout

```
├── paddy_dcgan_experiment.ipynb   # Complete 30-cell Kaggle/Colab research notebook
├── build_notebook.py              # Script to build and maintain the notebook
├── run_experiment.py              # End-to-end automated 3-way experiment runner
├── app.py                         # Interactive Streamlit dashboard
├── pipeline.py                    # Modular CLI pipeline
├── requirements.txt               # Dependencies
├── Rice Leaf Disease Images/      # 4-class paddy disease dataset (5,932 images)
│   └── Rice Leaf Disease Images/
│       ├── Bacterialblight/
│       ├── Blast/
│       ├── Brownspot/
│       └── Tungro/
└── paddy_gan/                     # Core library
    ├── models.py                  # DCGAN Generator, Discriminator, ResNet-18
    ├── gan.py                     # DCGAN training with DiffAugment
    ├── augment.py                 # GAN dataset balancing
    ├── trad_augment.py            # Traditional augmentation balancing
    ├── classifier.py              # ResNet classifier training & evaluation
    ├── fid.py                     # Batched Fréchet Inception Distance (torchmetrics)
    ├── gradcam.py                 # Layer-4 Grad-CAM explainability
    ├── visualize.py               # Dark-mode publication comparison plots
    └── data.py                    # Stratified splits & loader utilities
```

## Team

| Roll No. | Name |
|---|---|
| CB.SC.U4CSE23501 | Joshua Karthik A |
| CB.SC.U4CSE23519 | Venkatesh K |
| CB.SC.U4CSE23544 | Bhuvanesh S |
