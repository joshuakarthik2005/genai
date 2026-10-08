"""FID (Fréchet Inception Distance) computation for GAN quality evaluation.

FID measures how realistic synthetic images are vs. real images.
Lower FID = better. Human-level ~= 0, random noise ~= 300+.
"""
from pathlib import Path

import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms


_FID_TF = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 255).byte()),  # FrechetInceptionDistance needs uint8
])


def _get_image_paths(target, max_images: int = 2000, pattern: str = "*"):
    if isinstance(target, (list, tuple)):
        return list(target)[:max_images]
    target_path = Path(target)
    if not target_path.exists():
        return []
    paths = sorted([
        p for p in target_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ])
    return paths[:max_images]


def _batch_feed_fid(fid_metric, paths, is_real: bool, device: str = "cpu", batch_size: int = 64):
    batch = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            batch.append(_FID_TF(img))
        except Exception:
            continue
        if len(batch) >= batch_size:
            fid_metric.update(torch.stack(batch).to(device), real=is_real)
            batch = []
    if batch:
        fid_metric.update(torch.stack(batch).to(device), real=is_real)


def compute_fid(real_source, fake_source, device: str = "cpu",
                max_images: int = 2000, batch_size: int = 64) -> float:
    """Compute FID between real and synthetic images with memory-efficient batching.

    Returns:
        FID score (float). Lower is better.
    """
    real_paths = _get_image_paths(real_source, max_images=max_images)
    fake_paths = _get_image_paths(fake_source, max_images=max_images)

    if not real_paths:
        raise ValueError(f"No real images found for FID evaluation in: {real_source}")
    if not fake_paths:
        raise ValueError(f"No fake images found for FID evaluation in: {fake_source}")

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    print(f"  Feeding {len(real_paths)} real images to Inception network...")
    _batch_feed_fid(fid_metric, real_paths, is_real=True, device=device, batch_size=batch_size)

    print(f"  Feeding {len(fake_paths)} synthetic images to Inception network...")
    _batch_feed_fid(fid_metric, fake_paths, is_real=False, device=device, batch_size=batch_size)

    score = fid_metric.compute().item()
    return score


def compute_fid_all_classes(real_train_dir: Path, synthetic_dir: Path,
                             device: str = "cpu") -> dict:
    """Compute per-class FID between real training images and synthetic images.

    Args:
        real_train_dir: Path like data/split/train (subfolders = classes)
        synthetic_dir:  Path like data/balanced_gan (contains real + syn_*.png)

    Returns:
        {class_name: fid_score}
    """
    results = {}
    real_train_dir = Path(real_train_dir)
    synthetic_dir = Path(synthetic_dir)

    for cls_dir in sorted(real_train_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        syn_dir = synthetic_dir / cls
        if not syn_dir.exists():
            print(f"  Skipping FID for {cls}: no directory in {synthetic_dir}")
            continue

        # Isolate synthetic files (prefixed with syn_)
        syn_files = sorted(list(syn_dir.glob("syn_*.png")))
        if len(syn_files) < 10:
            print(f"  Skipping FID for {cls}: only {len(syn_files)} synthetic images (need >= 10).")
            continue

        print(f"\nComputing FID for class: {cls} ({len(syn_files)} synthetic images)")
        try:
            score = compute_fid(cls_dir, syn_files, device=device)
            results[cls] = round(score, 2)
            print(f"  FID({cls}) = {score:.2f}")
        except Exception as e:
            print(f"  FID({cls}) failed: {e}")
            results[cls] = None
    return results
