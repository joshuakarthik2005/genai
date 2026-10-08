"""Traditional augmentation dataset builder (no GAN).

Creates a balanced dataset using only classical transforms:
  - Random horizontal/vertical flip
  - Color jitter
  - Random rotation
  - Random crop + resize

This is the 'no-GAN' baseline for the 3-way comparison:
  Baseline (real only) vs Traditional Augment vs GAN Augment
"""
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from .data import class_counts, list_images


def _augment_image(img: Image.Image, seed: int) -> Image.Image:
    """Apply a set of classical transforms to produce a new image variant."""
    rng = random.Random(seed)
    img = img.convert("RGB")

    # Random horizontal flip
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random vertical flip
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Random rotation (-30 to +30 degrees)
    angle = rng.uniform(-30, 30)
    img = img.rotate(angle, expand=False, fillcolor=(0, 0, 0))

    # Color jitter: brightness
    factor = rng.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # Color jitter: saturation
    factor = rng.uniform(0.7, 1.3)
    img = ImageEnhance.Color(img).enhance(factor)

    # Color jitter: contrast
    factor = rng.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # Occasional slight blur
    if rng.random() > 0.8:
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

    # Random crop: take 80–100% of the image then resize back
    w, h = img.size
    crop_frac = rng.uniform(0.8, 1.0)
    cw, ch = int(w * crop_frac), int(h * crop_frac)
    x0 = rng.randint(0, w - cw)
    y0 = rng.randint(0, h - ch)
    img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BILINEAR)

    return img


def balance_dataset_traditional(train_dir, out_dir, target=None, seed=42, log=print):
    """Balance dataset using classical image augmentation (no GAN).

    Copies all real images, then generates augmented variants of minority class
    images until each class reaches `target` count.

    Returns {class: {"real": n, "augmented": m}}
    """
    train_dir, out_dir = Path(train_dir), Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    counts = class_counts(train_dir)
    target = target or max(counts.values())
    report = {}

    for cls, n_real in counts.items():
        dest = out_dir / cls
        dest.mkdir(parents=True, exist_ok=True)

        src_paths = list_images(train_dir / cls)

        # Copy real images
        for f in src_paths:
            shutil.copy2(f, dest / f.name)

        # Generate augmented images to reach target
        need = max(0, target - n_real)
        n_aug = 0
        aug_seed = seed
        for i in range(need):
            src = src_paths[i % len(src_paths)]
            img = Image.open(src).convert("RGB")
            aug_img = _augment_image(img, seed=aug_seed + i)
            aug_img.save(dest / f"aug_{i:05d}.png")
            n_aug += 1

        log(f"{cls}: {n_real} real + {n_aug} traditional-augmented")
        report[cls] = {"real": n_real, "augmented": n_aug}

    return report
