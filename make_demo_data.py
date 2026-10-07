"""Create a small, deliberately imbalanced, procedurally drawn paddy-leaf dataset.

Only for smoke-testing the pipeline when no real dataset is at hand. For real
results use a real dataset (see README), laid out as data/raw/<class>/*.jpg.
"""
import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

COUNTS = {"Healthy": 200, "Brown_Spot": 120, "Bacterial_Blight": 40, "Leaf_Blast": 25}


def leaf(draw, rng, size):
    """Draw a long diagonal rice leaf; return its centre line points and half-width."""
    angle = rng.uniform(-0.6, 0.6) + math.pi / 4
    cx, cy = size / 2 + rng.uniform(-10, 10), size / 2 + rng.uniform(-10, 10)
    length, width = size * 1.3, rng.uniform(14, 22)
    dx, dy = math.cos(angle) * length / 2, math.sin(angle) * length / 2
    nx, ny = -math.sin(angle) * width, math.cos(angle) * width
    green = (rng.randint(50, 90), rng.randint(130, 175), rng.randint(30, 60))
    draw.polygon([(cx - dx + nx, cy - dy + ny), (cx + dx + nx, cy + dy + ny),
                  (cx + dx - nx, cy + dy - ny), (cx - dx - nx, cy - dy - ny)], fill=green)
    for k in (-0.5, 0, 0.5):  # veins
        draw.line([(cx - dx + nx * k, cy - dy + ny * k), (cx + dx + nx * k, cy + dy + ny * k)],
                  fill=tuple(max(0, c - 20) for c in green), width=1)
    return (cx, cy, dx, dy, nx, ny), width


def point_on_leaf(rng, geom, spread=0.8):
    cx, cy, dx, dy, nx, ny = geom
    t, s = rng.uniform(-0.8, 0.8), rng.uniform(-spread, spread)
    return cx + dx * t + nx * s, cy + dy * t + ny * s


def draw_sample(cls, rng, size=128):
    bg = rng.choice([(235, 232, 225), (220, 225, 215), (95, 120, 70), (200, 200, 190)])
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    geom, width = leaf(d, rng, size)
    cx, cy, dx, dy, nx, ny = geom

    if cls == "Brown_Spot":
        for _ in range(rng.randint(10, 25)):
            x, y = point_on_leaf(rng, geom)
            r = rng.uniform(1.5, 3.5)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(rng.randint(110, 150), 60, 25))
    elif cls == "Leaf_Blast":
        for _ in range(rng.randint(2, 5)):
            x, y = point_on_leaf(rng, geom, 0.5)
            L, W = rng.uniform(6, 11), rng.uniform(2.5, 4)
            ux, uy = dx / math.hypot(dx, dy), dy / math.hypot(dx, dy)
            vx, vy = -uy, ux
            d.polygon([(x + ux * L, y + uy * L), (x + vx * W, y + vy * W),
                       (x - ux * L, y - uy * L), (x - vx * W, y - vy * W)], fill=(120, 70, 40))
            d.polygon([(x + ux * L * 0.5, y + uy * L * 0.5), (x + vx * W * 0.4, y + vy * W * 0.4),
                       (x - ux * L * 0.5, y - uy * L * 0.5), (x - vx * W * 0.4, y - vy * W * 0.4)],
                      fill=(190, 185, 165))
    elif cls == "Bacterial_Blight":
        side = rng.choice([-1, 1])
        t0, t1 = rng.uniform(-0.9, -0.2), rng.uniform(0.2, 0.9)
        k = side * rng.uniform(0.35, 0.7)
        for w, col in ((7, (215, 200, 90)), (3, (240, 235, 180))):
            d.line([(cx + dx * t0 + nx * k, cy + dy * t0 + ny * k),
                    (cx + dx * t1 + nx * k, cy + dy * t1 + ny * k)], fill=col, width=w)

    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    for cls, n in COUNTS.items():
        dest = Path(args.out) / cls
        dest.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            draw_sample(cls, rng).save(dest / f"{cls.lower()}_{i:04d}.jpg", quality=92)
        print(f"{cls}: {n}")


if __name__ == "__main__":
    main()
