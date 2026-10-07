"""ResNet disease classifier: training, evaluation and single-image prediction."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .models import build_classifier, get_device

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def train_tf(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_tf(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def compute_metrics(y_true, y_pred, classes):
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(cm[i, :].sum())}
    return {
        "accuracy": float(np.trace(cm) / max(1, cm.sum())),
        "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.extend(net(x.to(device)).argmax(1).cpu().tolist())
        ys.extend(y.tolist())
    return ys, ps


def train_classifier(train_dir, test_dir, out_dir, epochs=10, img_size=128, batch_size=32,
                     lr=1e-3, pretrained=False, seed=0, progress_cb=None):
    """Train ResNet-18 on train_dir, evaluate on the real-only test_dir.

    Saves model.pt and metrics.json to out_dir and returns the metrics dict.
    """
    torch.manual_seed(seed)
    device = get_device()
    train_ds = datasets.ImageFolder(train_dir, train_tf(img_size))
    test_ds = datasets.ImageFolder(test_dir, eval_tf(img_size))
    if train_ds.classes != test_ds.classes:
        raise ValueError(f"Class mismatch: train={train_ds.classes} test={test_ds.classes}")
    classes = train_ds.classes

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=0)

    net = build_classifier(len(classes), pretrained=pretrained).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    history = {"epoch": [], "train_loss": [], "test_acc": [], "test_macro_f1": []}
    start = time.time()
    for epoch in range(1, epochs + 1):
        net.train()
        total, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        ys, ps = evaluate(net, test_dl, device)
        m = compute_metrics(ys, ps, classes)
        history["epoch"].append(epoch)
        history["train_loss"].append(total / max(1, n))
        history["test_acc"].append(m["accuracy"])
        history["test_macro_f1"].append(m["macro_f1"])
        if progress_cb:
            progress_cb(epoch, epochs, {k: v[-1] for k, v in history.items()})

    metrics = compute_metrics(*evaluate(net, test_dl, device), classes)
    metrics["history"] = history
    metrics["train_counts"] = {c: sum(1 for _, y in train_ds.samples if y == i) for i, c in enumerate(classes)}
    metrics["seconds"] = time.time() - start

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "classes": classes, "img_size": img_size}, out_dir / "model.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load_classifier(path, device=None):
    device = device or get_device()
    ckpt = torch.load(path, map_location=device)
    net = build_classifier(len(ckpt["classes"])).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt["classes"], ckpt["img_size"]


@torch.no_grad()
def predict(net, classes, img_size, image: Image.Image):
    """Return [(class, probability), ...] sorted by probability."""
    device = next(net.parameters()).device
    x = eval_tf(img_size)(image.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(net(x), 1)[0].cpu().tolist()
    return sorted(zip(classes, probs), key=lambda t: -t[1])
