"""Dataset loading, preprocessing and real-only train/test splitting."""
import random
import shutil
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
GAN_IMG_SIZE = 64


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def list_classes(root):
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and list_images(d))


def class_counts(root):
    """{class_name: number_of_images} for an ImageFolder-style directory."""
    return {c: len(list_images(Path(root) / c)) for c in list_classes(root)}


def gan_transform(size=GAN_IMG_SIZE):
    """Case-study preprocessing: resize to 64x64 and normalize pixels to [-1, 1]."""
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


class ImageListDataset(Dataset):
    """Unlabelled images from a list of paths (one disease class for a per-class GAN)."""

    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


def make_split(raw_dir, split_dir, test_frac=0.2, seed=42):
    """Stratified copy of raw_dir into split_dir/train and split_dir/test.

    The test split holds only real images and is never seen by the GANs, so the
    classifier comparison (real vs real+synthetic) is fair.
    """
    raw_dir, split_dir = Path(raw_dir), Path(split_dir)
    if split_dir.exists():
        shutil.rmtree(split_dir)
    rng = random.Random(seed)
    summary = {}
    for cls in list_classes(raw_dir):
        files = list_images(raw_dir / cls)
        rng.shuffle(files)
        n_test = max(1, round(len(files) * test_frac)) if len(files) > 1 else 0
        parts = {"test": files[:n_test], "train": files[n_test:]}
        for part, part_files in parts.items():
            dest = split_dir / part / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in part_files:
                shutil.copy2(f, dest / f.name)
        summary[cls] = {k: len(v) for k, v in parts.items()}
    return summary
