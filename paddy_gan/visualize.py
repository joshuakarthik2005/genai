"""Comparison report generator: plots, tables, and HTML/PNG outputs.

Generates:
  1. 3-way F1/Accuracy bar chart (Baseline vs Traditional Augment vs GAN Augment)
  2. Per-class F1 grouped bar chart
  3. Confusion matrices for each method
  4. Training loss curves for each classifier
  5. GAN loss curves (D vs G)
  6. FID score bar chart
  7. Grad-CAM overlay grid
  8. Synthetic image sample grids
  9. Summary HTML report
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


# ── Color palette ────────────────────────────────────────────────────────────
PALETTE = {
    "baseline":   "#E74C3C",   # red
    "trad_aug":   "#F39C12",   # orange
    "gan_aug":    "#27AE60",   # green
}
BG      = "#0F1117"
FG      = "#FFFFFF"
GRID_C  = "#2C2C3A"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   GRID_C,
    "axes.labelcolor":  FG,
    "xtick.color":      FG,
    "ytick.color":      FG,
    "text.color":       FG,
    "grid.color":       GRID_C,
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "sans-serif",
    "legend.facecolor": "#1E1E2E",
    "legend.edgecolor": GRID_C,
})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 1. Overall comparison bar chart ──────────────────────────────────────────
def plot_overall_comparison(metrics_dict: dict, out_path: Path):
    """metrics_dict: {name: metrics_json_dict}"""
    names  = list(metrics_dict.keys())
    colors = [PALETTE.get(n, "#8E44AD") for n in names]
    acc    = [metrics_dict[n]["accuracy"]  * 100 for n in names]
    f1     = [metrics_dict[n]["macro_f1"]  * 100 for n in names]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w/2, acc, w, color=colors, alpha=0.85, label="Accuracy (%)", zorder=3)
    bars2 = ax.bar(x + w/2, f1,  w, color=colors, alpha=0.5,  label="Macro F1 (%)", hatch="//", zorder=3)
    ax.set_xticks(x)
    label_map = {"baseline": "Baseline\n(Real only)",
                 "trad_aug": "Traditional\nAugmentation",
                 "gan_aug":  "GAN\nAugmentation (Ours)"}
    ax.set_xticklabels([label_map.get(n, n) for n in names], fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("3-Way Comparison: Accuracy & Macro F1", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="y", zorder=0)
    ax.legend(fontsize=10)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}",
                ha="center", va="bottom", fontsize=9, color=FG)

    _save(fig, out_path)


# ── 2. Per-class F1 grouped bar chart ────────────────────────────────────────
def plot_perclass_f1(metrics_dict: dict, out_path: Path):
    first = next(iter(metrics_dict.values()))
    classes = first["classes"]
    names   = list(metrics_dict.keys())
    label_map = {"baseline": "Baseline", "trad_aug": "Trad. Aug.", "gan_aug": "GAN Aug."}
    colors  = [PALETTE.get(n, "#8E44AD") for n in names]

    x = np.arange(len(classes))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (name, color) in enumerate(zip(names, colors)):
        f1s = [metrics_dict[name]["per_class"][c]["f1"] * 100 for c in classes]
        offset = (i - len(names)/2 + 0.5) * w
        bars = ax.bar(x + offset, f1s, w, color=color, alpha=0.85,
                      label=label_map.get(name, name), zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7, color=FG)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_ylabel("F1 Score (%)", fontsize=11)
    ax.set_title("Per-Class F1 Score: Baseline vs Trad. Aug. vs GAN Aug.", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", zorder=0)
    ax.legend(fontsize=10)
    _save(fig, out_path)


# ── 3. Confusion matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(metrics: dict, title: str, out_path: Path):
    cm     = np.array(metrics["confusion_matrix"])
    classes = metrics["classes"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1, None)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            color = "white" if cm_norm[i, j] > 0.6 else FG
            ax.text(j, i, str(val), ha="center", va="center", fontsize=9, color=color)
    _save(fig, out_path)


# ── 4. Training curves ────────────────────────────────────────────────────────
def plot_training_curves(metrics_dict: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    label_map = {"baseline": "Baseline", "trad_aug": "Trad. Aug.", "gan_aug": "GAN Aug."}

    for name, m in metrics_dict.items():
        h = m["history"]
        color = PALETTE.get(name, "#8E44AD")
        label = label_map.get(name, name)
        axes[0].plot(h["epoch"], h["train_loss"],   color=color, linewidth=2,   label=label)
        axes[1].plot(h["epoch"], h["test_macro_f1"], color=color, linewidth=2, label=label,
                     linestyle="--")
        axes[1].plot(h["epoch"], h["test_acc"],      color=color, linewidth=1.5,
                     linestyle=":", alpha=0.6)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss", fontsize=12, fontweight="bold")
    axes[0].legend(); axes[0].grid()

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("Test Macro-F1 (solid) & Accuracy (dotted)", fontsize=12, fontweight="bold")
    axes[1].legend(); axes[1].grid()

    fig.suptitle("Classifier Training Curves", fontsize=14, fontweight="bold", y=1.02)
    _save(fig, out_path)


# ── 5. GAN training curves ────────────────────────────────────────────────────
def plot_gan_curves(history_paths: dict, out_path: Path):
    """history_paths: {class_name: Path to history.json}"""
    n = len(history_paths)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    colors = ["#27AE60", "#3498DB", "#9B59B6", "#E67E22"]

    for idx, (cls, hist_path) in enumerate(history_paths.items()):
        ax = axes[0][idx]
        hist = json.loads(Path(hist_path).read_text())
        steps = hist["step"]
        ax.plot(steps, hist["loss_D"], color="#E74C3C", linewidth=1.5, label="Loss D")
        ax.plot(steps, hist["loss_G"], color="#27AE60", linewidth=1.5, label="Loss G")
        ax2 = ax.twinx()
        ax2.plot(steps, hist["D_x"],   color="#3498DB", linewidth=1,   label="D(x)",   linestyle="--", alpha=0.7)
        ax2.plot(steps, hist["D_G_z"], color="#F39C12", linewidth=1,   label="D(G(z))", linestyle=":", alpha=0.7)
        ax2.set_ylabel("Discriminator output", color=FG, fontsize=9)
        ax2.tick_params(colors=FG)
        ax2.set_ylim(0, 1)
        ax.set_title(f"GAN: {cls}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.grid()

    fig.suptitle("DCGAN Training Curves (Loss D vs Loss G)", fontsize=13, fontweight="bold", y=1.02)
    _save(fig, out_path)


# ── 6. FID bar chart ──────────────────────────────────────────────────────────
def plot_fid_scores(fid_scores: dict, out_path: Path):
    """fid_scores: {class_name: fid_value}"""
    classes = [c for c, v in fid_scores.items() if v is not None]
    scores  = [fid_scores[c] for c in classes]
    colors  = ["#27AE60" if s < 100 else "#F39C12" if s < 200 else "#E74C3C" for s in scores]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(classes, scores, color=colors, alpha=0.85, zorder=3)
    ax.axhline(50,  color="#27AE60", linestyle="--", alpha=0.6, label="Good (<50)")
    ax.axhline(150, color="#F39C12", linestyle="--", alpha=0.6, label="Fair (<150)")
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, s + 2, f"{s:.1f}",
                ha="center", va="bottom", fontsize=10, color=FG, fontweight="bold")
    ax.set_ylabel("FID Score (lower = better)", fontsize=11)
    ax.set_title("FID Score per Disease Class\n(GAN synthetic vs. real images)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", zorder=0)
    _save(fig, out_path)


# ── 7. Synthetic image grid ───────────────────────────────────────────────────
def plot_image_grid(real_paths: list, syn_paths: list, cls_name: str, out_path: Path,
                    n: int = 8):
    """Plot a side-by-side grid: n real images | n synthetic images."""
    def load(p, size=128):
        return Image.open(p).convert("RGB").resize((size, size))

    real_imgs = [load(p) for p in real_paths[:n]]
    syn_imgs  = [load(p) for p in syn_paths[:n]]

    cols = n
    fig, axes = plt.subplots(2, cols, figsize=(cols * 1.4, 3.5))
    fig.suptitle(f"{cls_name}: Real (top) vs Synthetic GAN (bottom)", fontsize=12, fontweight="bold")

    for i in range(cols):
        for row, imgs in enumerate([real_imgs, syn_imgs]):
            ax = axes[row][i]
            ax.imshow(imgs[i] if i < len(imgs) else np.zeros((128, 128, 3), dtype=np.uint8))
            ax.axis("off")
        axes[0][0].set_ylabel("Real", fontsize=9, color=FG)
        axes[1][0].set_ylabel("Synthetic", fontsize=9, color=FG)

    plt.tight_layout()
    _save(fig, out_path)


# ── 8. Grad-CAM grid ──────────────────────────────────────────────────────────
def plot_gradcam_grid(results: list, out_path: Path):
    """results: list of dicts from explain_prediction (+ 'original' PIL key)"""
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, r in enumerate(results):
        axes[0][i].imshow(r["original"])
        axes[0][i].axis("off")
        axes[0][i].set_title(f"True: {r['true_class']}", fontsize=9, color=FG)

        axes[1][i].imshow(r["overlay"])
        axes[1][i].axis("off")
        pred_color = "#27AE60" if r["pred_class"] == r["true_class"] else "#E74C3C"
        axes[1][i].set_title(f"Pred: {r['pred_class']}\n{r['pred_prob']:.0%}",
                              fontsize=9, color=pred_color)

    axes[0][0].set_ylabel("Original", fontsize=10, color=FG)
    axes[1][0].set_ylabel("Grad-CAM", fontsize=10, color=FG)
    fig.suptitle("Grad-CAM: Where the model looks on the leaf", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_path)


# ── 9. Summary metrics table image ────────────────────────────────────────────
def plot_summary_table(metrics_dict: dict, fid_scores: dict, out_path: Path):
    label_map = {"baseline": "Baseline (Real only)",
                 "trad_aug": "Traditional Augmentation",
                 "gan_aug":  "GAN Augmentation (Ours)"}
    rows = []
    for name, m in metrics_dict.items():
        avg_fid = None
        if name == "gan_aug" and fid_scores:
            vals = [v for v in fid_scores.values() if v is not None]
            avg_fid = f"{np.mean(vals):.1f}" if vals else "—"
        rows.append({
            "Method": label_map.get(name, name),
            "Accuracy": f"{m['accuracy']*100:.1f}%",
            "Macro F1": f"{m['macro_f1']*100:.1f}%",
            **{f"F1({c})": f"{m['per_class'][c]['f1']*100:.1f}%" for c in m["classes"]},
            "Avg FID": avg_fid or "—",
        })

    columns = list(rows[0].keys())
    cell_data = [[r[c] for c in columns] for r in rows]
    colors_col = [PALETTE.get(n, "#8E44AD") for n in metrics_dict.keys()]

    fig, ax = plt.subplots(figsize=(14, 2.5 + 0.5 * len(rows)))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_data,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2)

    # Style header
    for j in range(len(columns)):
        tbl[0, j].set_facecolor("#1A1A2E")
        tbl[0, j].set_text_props(color=FG, fontweight="bold")

    # Style data rows
    row_colors = ["#1E1E2E", "#252535", "#2A2A3F"]
    for i, rc in enumerate(colors_col):
        for j in range(len(columns)):
            tbl[i+1, j].set_facecolor(row_colors[i % len(row_colors)])
            tbl[i+1, j].set_text_props(color=FG)
        tbl[i+1, 0].set_text_props(color=rc, fontweight="bold")

    ax.set_title("Complete Results Summary", fontsize=14, fontweight="bold", pad=20, color=FG)
    _save(fig, out_path)
