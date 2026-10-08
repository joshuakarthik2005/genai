"""
Full experiment runner for the Paddy DCGAN novelty study.

Runs the complete pipeline:
  1. Copy dataset to data/raw
  2. Stratified train/test split
  3. Train DCGAN per minority class
  4. Build 3 balanced datasets:
       - data/split/train      (baseline: real only)
       - data/balanced_trad    (traditional augmentation: flips, jitter, rotate)
       - data/balanced_gan     (GAN augmentation)
  5. Train 3 ResNet-18 classifiers (one per dataset)
  6. Compute FID for GAN synthetic images
  7. Generate Grad-CAM heatmaps on test images
  8. Generate all comparison plots + summary table
  9. Print final comparison report

Usage:
  python run_experiment.py --raw "Rice Leaf Disease Images/Rice Leaf Disease Images" --iterations 3000 --epochs 15

Quick CPU test (faster):
  python run_experiment.py --raw "Rice Leaf Disease Images/Rice Leaf Disease Images" --iterations 1000 --epochs 8 --width 32 --img-size 64
"""
import argparse
import json
import shutil
from pathlib import Path

from paddy_gan.augment import balance_dataset
from paddy_gan.classifier import load_classifier, predict, train_classifier
from paddy_gan.data import class_counts, list_images, make_split
from paddy_gan.fid import compute_fid_all_classes
from paddy_gan.gan import train_dcgan
from paddy_gan.gradcam import explain_prediction
from paddy_gan.trad_augment import balance_dataset_traditional
from paddy_gan.visualize import (
    plot_confusion_matrix,
    plot_fid_scores,
    plot_gan_curves,
    plot_gradcam_grid,
    plot_image_grid,
    plot_overall_comparison,
    plot_perclass_f1,
    plot_summary_table,
    plot_training_curves,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW       = Path("data/raw")
SPLIT          = Path("data/split")
BAL_TRAD       = Path("data/balanced_trad")
BAL_GAN        = Path("data/balanced_gan")
GAN_DIR        = Path("runs/gan")
CLS_DIR        = Path("runs/classifier")
PLOTS_DIR      = Path("runs/plots")
REPORT_DIR     = Path("runs/report")


def banner(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def cb_cls(e, total, s):
    print(f"    epoch {e:3d}/{total} | loss={s['train_loss']:.4f} "
          f"acc={s['test_acc']:.3f} macroF1={s['test_macro_f1']:.3f}")


def cb_gan(step, total, s):
    if step % 200 == 0 or step == total:
        print(f"    [{step:5d}/{total}] D={s['loss_D']:.3f} G={s['loss_G']:.3f} "
              f"D(x)={s['D_x']:.2f} D(G(z))={s['D_G_z']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw",        default="Rice Leaf Disease Images/Rice Leaf Disease Images",
                    help="Path to raw dataset (one subfolder per class)")
    ap.add_argument("--iterations", type=int, default=3000, help="GAN training steps per class")
    ap.add_argument("--epochs",     type=int, default=15,   help="Classifier training epochs")
    ap.add_argument("--width",      type=int, default=64,   help="DCGAN base channels (32 = ~4x faster)")
    ap.add_argument("--img-size",   type=int, default=128,  help="Classifier input resolution")
    ap.add_argument("--batch-size", type=int, default=32,   help="Classifier batch size")
    ap.add_argument("--target",     type=int, default=None, help="Images per class after balancing")
    ap.add_argument("--pretrained", action="store_true",    help="Use ImageNet pretrained ResNet-18")
    ap.add_argument("--no-augment", action="store_true",    help="Disable DiffAugment in GAN")
    ap.add_argument("--skip-gan",   action="store_true",    help="Skip GAN training (use existing checkpoints)")
    ap.add_argument("--skip-train", action="store_true",    help="Skip classifier training (use existing models)")
    ap.add_argument("--fid",        action="store_true",    help="Compute FID scores (slow on CPU)")
    ap.add_argument("--gradcam-n",  type=int, default=4,    help="Number of Grad-CAM examples per class")
    ap.add_argument("--seed",       type=int, default=42)
    a = ap.parse_args()

    # ── Step 1: Copy raw data ─────────────────────────────────────────────────
    banner("Step 1: Preparing dataset")
    raw_src = Path(a.raw)
    if not raw_src.exists():
        raise FileNotFoundError(f"Dataset not found at: {raw_src}")

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for cls_dir in raw_src.iterdir():
        if cls_dir.is_dir():
            dst = DATA_RAW / cls_dir.name
            if not dst.exists():
                shutil.copytree(cls_dir, dst)

    counts = class_counts(DATA_RAW)
    print("\nClass counts in raw dataset:")
    for cls, n in sorted(counts.items()):
        print(f"  {cls:25s}: {n:5d} images")
    majority = max(counts.values())
    minority_classes = [c for c, n in counts.items() if n < majority]
    print(f"\nMajority class size: {majority}")
    print(f"Minority classes: {minority_classes}")

    # ── Step 2: Train/test split ──────────────────────────────────────────────
    banner("Step 2: Stratified train/test split (80/20)")
    split_info = make_split(DATA_RAW, SPLIT, test_frac=0.2, seed=a.seed)
    for cls, s in split_info.items():
        print(f"  {cls:25s} train={s['train']:5d} test={s['test']:5d}")

    # ── Step 3: Train GANs ────────────────────────────────────────────────────
    if not a.skip_gan:
        banner("Step 3: Training DCGAN per minority class")
        print(f"  Iterations per class: {a.iterations}  |  Width: {a.width}  |  DiffAugment: {not a.no_augment}")
        for cls in minority_classes:
            print(f"\n  ── DCGAN for class: {cls}")
            train_dcgan(
                SPLIT / "train" / cls,
                GAN_DIR / cls,
                iterations=a.iterations,
                width=a.width,
                augment=not a.no_augment,
                progress_cb=cb_gan,
                seed=a.seed,
            )
    else:
        print("  [Skipped] Using existing GAN checkpoints.")

    # ── Step 4: Build 3 balanced datasets ────────────────────────────────────
    banner("Step 4: Building balanced datasets")

    target = a.target or majority
    print(f"\n  Target images per class: {target}")

    print("\n  [A] Traditional augmentation dataset...")
    trad_report = balance_dataset_traditional(
        SPLIT / "train", BAL_TRAD, target=target, seed=a.seed
    )

    print("\n  [B] GAN augmentation dataset...")
    gan_report = balance_dataset(
        SPLIT / "train", GAN_DIR, BAL_GAN, target=target, seed=a.seed
    )

    # ── Step 5: Train 3 classifiers ───────────────────────────────────────────
    banner("Step 5: Training classifiers")
    experiments = {
        "baseline": SPLIT / "train",
        "trad_aug": BAL_TRAD,
        "gan_aug":  BAL_GAN,
    }
    all_metrics = {}

    for name, train_dir in experiments.items():
        if a.skip_train:
            metrics_path = CLS_DIR / name / "metrics.json"
            if metrics_path.exists():
                print(f"\n  [Skipped] Loading existing metrics for {name}")
                all_metrics[name] = json.loads(metrics_path.read_text())
                continue
        print(f"\n  ── Classifier: {name}")
        m = train_classifier(
            train_dir, SPLIT / "test", CLS_DIR / name,
            epochs=a.epochs,
            img_size=a.img_size,
            batch_size=a.batch_size,
            pretrained=a.pretrained,
            seed=a.seed,
            progress_cb=cb_cls,
        )
        all_metrics[name] = m
        print(f"    RESULT → accuracy={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}")

    # ── Step 6: FID scores ────────────────────────────────────────────────────
    fid_scores = {}
    if a.fid:
        banner("Step 6: Computing FID scores (GAN synthetic vs. real)")
        fid_scores = compute_fid_all_classes(SPLIT / "train", BAL_GAN)
    else:
        print("\n  [Skipped] FID computation. Re-run with --fid to enable.")

    # ── Step 7: Grad-CAM on test images ───────────────────────────────────────
    banner("Step 7: Generating Grad-CAM heatmaps")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    net, classes, img_sz = load_classifier(CLS_DIR / "gan_aug" / "model.pt")
    gradcam_results = []
    test_dir = SPLIT / "test"

    for cls in classes[:a.gradcam_n]:
        cls_test_dir = test_dir / cls
        if not cls_test_dir.exists():
            continue
        img_paths = list_images(cls_test_dir)[:2]
        for img_path in img_paths:
            from PIL import Image
            orig = Image.open(img_path).convert("RGB")
            result = explain_prediction(net, classes, img_sz, orig)
            result["original"]   = orig.resize((img_sz, img_sz))
            result["true_class"] = cls
            gradcam_results.append(result)

    if gradcam_results:
        plot_gradcam_grid(gradcam_results, PLOTS_DIR / "gradcam_grid.png")

    # ── Step 8: Generate all plots ────────────────────────────────────────────
    banner("Step 8: Generating comparison plots")

    print("\n  Overall comparison chart...")
    plot_overall_comparison(all_metrics, PLOTS_DIR / "overall_comparison.png")

    print("  Per-class F1 chart...")
    plot_perclass_f1(all_metrics, PLOTS_DIR / "perclass_f1.png")

    print("  Confusion matrices...")
    label_map = {"baseline": "Baseline (Real only)",
                 "trad_aug": "Traditional Augmentation",
                 "gan_aug":  "GAN Augmentation (Ours)"}
    for name, m in all_metrics.items():
        plot_confusion_matrix(m, label_map.get(name, name),
                               PLOTS_DIR / f"confusion_{name}.png")

    print("  Training curves...")
    plot_training_curves(all_metrics, PLOTS_DIR / "training_curves.png")

    print("  GAN training curves...")
    hist_paths = {cls: GAN_DIR / cls / "history.json"
                  for cls in minority_classes
                  if (GAN_DIR / cls / "history.json").exists()}
    if hist_paths:
        plot_gan_curves(hist_paths, PLOTS_DIR / "gan_curves.png")

    if fid_scores:
        print("  FID score chart...")
        plot_fid_scores(fid_scores, PLOTS_DIR / "fid_scores.png")

    print("  Synthetic image grids...")
    for cls in minority_classes:
        real_paths = list_images(SPLIT / "train" / cls)[:8]
        syn_paths  = sorted((BAL_GAN / cls).glob("syn_*.png"))[:8]
        if real_paths and syn_paths:
            plot_image_grid(real_paths, syn_paths, cls,
                            PLOTS_DIR / f"real_vs_syn_{cls}.png")

    print("  Summary table...")
    plot_summary_table(all_metrics, fid_scores, PLOTS_DIR / "summary_table.png")

    # ── Step 9: Print final report ────────────────────────────────────────────
    banner("FINAL RESULTS")
    classes_list = next(iter(all_metrics.values()))["classes"]
    header = f"{'Method':30s} | {'Accuracy':>10s} | {'Macro F1':>10s}"
    for c in classes_list:
        header += f" | {('F1('+c+')'):>14s}"
    if fid_scores:
        header += f" | {'Avg FID':>10s}"
    print(header)
    print("-" * len(header))

    for name, m in all_metrics.items():
        row = f"{label_map.get(name, name):30s} | {m['accuracy']*100:10.2f}% | {m['macro_f1']*100:10.2f}%"
        for c in classes_list:
            row += f" | {m['per_class'][c]['f1']*100:14.2f}%"
        if fid_scores:
            if name == "gan_aug":
                vals = [v for v in fid_scores.values() if v is not None]
                avg_fid = f"{sum(vals)/len(vals):.1f}" if vals else "—"
            else:
                avg_fid = "-"
            row += f" | {avg_fid:>10s}"
        print(row)

    print(f"\n[SUCCESS] All plots saved to: {PLOTS_DIR.resolve()}")
    print(f"[SUCCESS] Models saved to:    {CLS_DIR.resolve()}")
    if hist_paths:
        print(f"[SUCCESS] GAN checkpoints:    {GAN_DIR.resolve()}")

    # Save experiment summary JSON
    summary = {
        "args": vars(a),
        "class_counts": counts,
        "split_info": split_info,
        "trad_balance_report": trad_report,
        "gan_balance_report": {k: {str(kk): vv for kk, vv in v.items()}
                                for k, v in gan_report.items()},
        "results": {
            name: {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                   "per_class_f1": {c: m["per_class"][c]["f1"] for c in m["classes"]}}
            for name, m in all_metrics.items()
        },
        "fid_scores": fid_scores,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[SUCCESS] Experiment summary:  {(REPORT_DIR / 'experiment_summary.json').resolve()}")


if __name__ == "__main__":
    main()
