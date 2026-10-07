"""Build a balanced dataset: real training images + DCGAN synthetic images."""
import shutil
from pathlib import Path

from .data import class_counts, list_images
from .gan import generate_images, load_generator


def balance_dataset(train_dir, gan_dir, out_dir, target=None, seed=123, upscale=None, log=print):
    """Copy real images, then top up each class to `target` with synthetic images.

    target defaults to the majority class count. Classes without a trained
    generator in gan_dir/<class>/generator.pt are copied as-is.
    Returns {class: {"real": n, "synthetic": m}}.
    """
    train_dir, gan_dir, out_dir = Path(train_dir), Path(gan_dir), Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    counts = class_counts(train_dir)
    target = target or max(counts.values())
    report = {}
    for i, (cls, n_real) in enumerate(counts.items()):
        dest = out_dir / cls
        dest.mkdir(parents=True, exist_ok=True)
        for f in list_images(train_dir / cls):
            shutil.copy2(f, dest / f.name)

        need = max(0, target - n_real)
        ckpt = gan_dir / cls / "generator.pt"
        n_syn = 0
        if need and ckpt.exists():
            netG = load_generator(ckpt)
            for j, img in enumerate(generate_images(netG, need, seed=seed + i)):
                if upscale:
                    img = img.resize((upscale, upscale), resample=3)  # bicubic
                img.save(dest / f"syn_{j:05d}.png")
            n_syn = need
            log(f"{cls}: {n_real} real + {n_syn} synthetic")
        elif need:
            log(f"{cls}: {n_real} real, no generator found -> left imbalanced")
        else:
            log(f"{cls}: {n_real} real (majority class)")
        report[cls] = {"real": n_real, "synthetic": n_syn}
    return report
