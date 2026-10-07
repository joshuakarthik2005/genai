"""Command-line pipeline for DCGAN-based paddy leaf disease augmentation.

  python pipeline.py split      --raw data/raw
  python pipeline.py train-gan  --classes minority --iterations 3000
  python pipeline.py balance
  python pipeline.py classify   --train data/split/train --name baseline
  python pipeline.py classify   --train data/balanced    --name augmented
  python pipeline.py compare
  python pipeline.py all        # every step above in order
"""
import argparse
import json
from pathlib import Path

from paddy_gan.augment import balance_dataset
from paddy_gan.classifier import train_classifier
from paddy_gan.data import class_counts, make_split
from paddy_gan.gan import train_dcgan

SPLIT = Path("data/split")
BALANCED = Path("data/balanced")
GAN_DIR = Path("runs/gan")
CLS_DIR = Path("runs/classifier")


def pick_classes(spec, train_dir):
    counts = class_counts(train_dir)
    if spec == "all":
        return list(counts)
    if spec == "minority":
        top = max(counts.values())
        return [c for c, n in counts.items() if n < top]
    return [c.strip() for c in spec.split(",")]


def cmd_split(a):
    for cls, s in make_split(a.raw, SPLIT, a.test_frac, a.seed).items():
        print(f"{cls:25s} train={s['train']:5d} test={s['test']:5d}")


def cmd_train_gan(a):
    for cls in pick_classes(a.classes, SPLIT / "train"):
        print(f"== DCGAN for {cls}")

        def cb(step, total, s):
            if step % 100 == 0 or step == total:
                print(f"  [{step}/{total}] loss_D={s['loss_D']:.3f} loss_G={s['loss_G']:.3f} "
                      f"D(x)={s['D_x']:.2f} D(G(z))={s['D_G_z']:.2f}")

        train_dcgan(SPLIT / "train" / cls, GAN_DIR / cls, iterations=a.iterations,
                    batch_size=a.batch_size, width=a.width, augment=not a.no_augment, progress_cb=cb)


def cmd_balance(a):
    balance_dataset(SPLIT / "train", GAN_DIR, BALANCED, target=a.target)


def cmd_classify(a):
    def cb(e, total, s):
        print(f"  epoch {e}/{total} loss={s['train_loss']:.3f} acc={s['test_acc']:.3f} macroF1={s['test_macro_f1']:.3f}")

    m = train_classifier(a.train, SPLIT / "test", CLS_DIR / a.name, epochs=a.epochs,
                         img_size=a.img_size, pretrained=a.pretrained, progress_cb=cb)
    print(f"{a.name}: accuracy={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f}")


def cmd_compare(_):
    rows = {}
    for name in ("baseline", "augmented"):
        p = CLS_DIR / name / "metrics.json"
        if p.exists():
            rows[name] = json.loads(p.read_text())
    if not rows:
        print("No classifier runs found.")
        return
    classes = next(iter(rows.values()))["classes"]
    print(f"{'':25s}" + "".join(f"{n:>12s}" for n in rows))
    print(f"{'accuracy':25s}" + "".join(f"{m['accuracy']:12.3f}" for m in rows.values()))
    print(f"{'macro F1':25s}" + "".join(f"{m['macro_f1']:12.3f}" for m in rows.values()))
    for c in classes:
        print(f"{'F1 ' + c:25s}" + "".join(f"{m['per_class'][c]['f1']:12.3f}" for m in rows.values()))


def cmd_all(a):
    cmd_split(a)
    cmd_train_gan(a)
    cmd_balance(a)
    for name, train in (("baseline", SPLIT / "train"), ("augmented", BALANCED)):
        a.name, a.train = name, train
        cmd_classify(a)
    cmd_compare(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--raw", default="data/raw")
        p.add_argument("--test-frac", type=float, default=0.2)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--classes", default="minority", help="'minority', 'all' or comma list")
        p.add_argument("--iterations", type=int, default=3000)
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--width", type=int, default=64, help="DCGAN base channels (32 is ~4x faster on CPU)")
        p.add_argument("--no-augment", action="store_true", help="disable DiffAugment")
        p.add_argument("--target", type=int, default=None, help="images per class after balancing")
        p.add_argument("--train", default=str(SPLIT / "train"))
        p.add_argument("--name", default="baseline")
        p.add_argument("--epochs", type=int, default=10)
        p.add_argument("--img-size", type=int, default=128)
        p.add_argument("--pretrained", action="store_true", help="ImageNet-pretrained ResNet-18")

    for name, fn in (("split", cmd_split), ("train-gan", cmd_train_gan), ("balance", cmd_balance),
                     ("classify", cmd_classify), ("compare", cmd_compare), ("all", cmd_all)):
        p = sub.add_parser(name)
        common(p)
        p.set_defaults(fn=fn)

    a = ap.parse_args()
    a.train = Path(a.train)
    a.fn(a)


if __name__ == "__main__":
    main()
