"""Streamlit UI for Paddy Leaf Disease Augmentation using DCGAN.

Run:  streamlit run app.py
"""
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from paddy_gan.augment import balance_dataset
from paddy_gan.classifier import load_classifier, predict, train_classifier
from paddy_gan.data import IMG_EXTS, class_counts, list_images, make_split
from paddy_gan.gan import generate_images, image_grid, load_generator, train_dcgan

RAW = Path("data/raw")
SPLIT = Path("data/split")
BALANCED = Path("data/balanced")
GAN_DIR = Path("runs/gan")
CLS_DIR = Path("runs/classifier")

st.set_page_config(page_title="Paddy DCGAN", page_icon="🌾", layout="wide")


def trained_gans():
    return sorted(p.parent.name for p in GAN_DIR.glob("*/generator.pt"))


def counts_chart(counts_by_set):
    df = pd.DataFrame(counts_by_set).fillna(0).astype(int)
    st.bar_chart(df)
    st.dataframe(df, width="stretch")


@st.cache_resource
def cached_generator(path, mtime):
    return load_generator(path)


@st.cache_resource
def cached_classifier(path, mtime):
    return load_classifier(path)


# ---------------------------------------------------------------- pages
def page_overview():
    st.title("🌾 Paddy Leaf Disease Augmentation using DCGAN")
    st.markdown(
        """
Rice diseases (Bacterial Blight, Rice Blast, Brown Spot, Leaf Smut, Tungro) cut yield, and
disease image datasets are **scarce and imbalanced**. Classifiers trained on them overfit and
miss minority classes. Classical augmentation (flip/rotate) only copies existing images.

This app trains a **Deep Convolutional GAN per minority class**, generates realistic synthetic
leaves, **balances the dataset**, and shows the effect on a **ResNet-18 disease classifier**.
"""
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Pipeline")
        st.markdown(
            """
1. **Dataset** – load images (`data/raw/<class>/…`), stratified real-only train/test split
2. **Preprocessing** – resize 64×64, normalise pixels to [-1, 1]
3. **DCGAN** – Generator vs Discriminator, adversarial training per class
4. **Balance** – real + synthetic images up to the majority class count
5. **Classifier** – ResNet-18 trained on *real only* vs *real + synthetic*
6. **Predict** – classify an uploaded leaf image
"""
        )
    with c2:
        st.subheader("DCGAN architecture")
        st.table(pd.DataFrame({
            "Generator": ["Noise z (100-d)", "Transposed convolutions", "BatchNorm + ReLU", "Tanh → 64×64×3"],
            "Discriminator": ["Real / synthetic 64×64×3", "Strided convolutions", "BatchNorm + LeakyReLU(0.2)", "Sigmoid → P(real)"],
        }))
        st.caption("Adam (lr 2e-4, β1 0.5), BCE loss, N(0, 0.02) init, one-sided label smoothing.")


def page_dataset():
    st.header("1 · Dataset & split")
    st.write(f"Raw dataset folder: `{RAW}` — one sub-folder per disease class.")

    with st.expander("Upload a dataset (.zip with one folder per class)"):
        up = st.file_uploader("ZIP file", type=["zip"])
        if up and st.button("Extract to data/raw"):
            with zipfile.ZipFile(up) as z:
                n = 0
                for info in z.infolist():
                    p = Path(info.filename)
                    if info.is_dir() or p.suffix.lower() not in IMG_EXTS or len(p.parts) < 2:
                        continue
                    dest = RAW / p.parts[-2] / p.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(z.read(info))
                    n += 1
            st.success(f"Extracted {n} images.")

    with st.expander("No dataset? Generate a small synthetic demo dataset"):
        st.caption("Procedurally drawn leaves for a quick end-to-end test. Use a real dataset for real results.")
        if st.button("Create demo dataset"):
            import random
            from make_demo_data import COUNTS, draw_sample
            rng = random.Random(7)
            for cls, n in COUNTS.items():
                d = RAW / cls
                d.mkdir(parents=True, exist_ok=True)
                for i in range(n):
                    draw_sample(cls, rng).save(d / f"{cls.lower()}_{i:04d}.jpg", quality=92)
            st.success("Demo dataset created.")

    counts = class_counts(RAW)
    if not counts:
        st.info("No images found yet.")
        return
    st.subheader("Class distribution (raw)")
    counts_chart({"raw": counts})
    ratio = max(counts.values()) / max(1, min(counts.values()))
    st.metric("Imbalance ratio (max / min)", f"{ratio:.1f}×")

    st.subheader("Samples")
    cols = st.columns(len(counts))
    for col, cls in zip(cols, counts):
        files = list_images(RAW / cls)[:3]
        col.markdown(f"**{cls}**")
        for f in files:
            col.image(str(f), width="stretch")

    st.subheader("Train / test split")
    frac = st.slider("Test fraction (real images only, never seen by the GAN)", 0.1, 0.4, 0.2, 0.05)
    if st.button("Create split", type="primary"):
        summary = make_split(RAW, SPLIT, frac)
        st.success("Split created.")
        st.dataframe(pd.DataFrame(summary).T)
    elif (SPLIT / "train").exists():
        counts_chart({"train": class_counts(SPLIT / "train"), "test": class_counts(SPLIT / "test")})


def page_train_gan():
    st.header("2 · Train DCGAN")
    train = SPLIT / "train"
    counts = class_counts(train)
    if not counts:
        st.warning("Create the train/test split first (page 1).")
        return
    top = max(counts.values())
    minority = [c for c, n in counts.items() if n < top]
    classes = st.multiselect("Classes to train a GAN for", list(counts), default=minority,
                             format_func=lambda c: f"{c} ({counts[c]} imgs)")
    c1, c2, c3, c4, c5 = st.columns(5)
    iters = c1.number_input("Iterations", 100, 50000, 1000, 100)
    bs = c2.number_input("Batch size", 8, 256, 32, 8)
    width = c3.selectbox("Network width", [32, 64], help="Base channels. 32 is ~4x faster on CPU; 64 is the original DCGAN.")
    sample_every = c4.number_input("Save samples every", 50, 5000, 250, 50)
    augment = c5.checkbox("DiffAugment", True, help="Differentiable augmentation for D; prevents collapse on small classes.")
    st.caption("CPU, width 32: ~1000 iterations ≈ 5 min per class. Real datasets typically need 5k–20k iterations (GPU recommended).")

    if st.button("Start training", type="primary", disabled=not classes):
        for cls in classes:
            st.subheader(cls)
            bar = st.progress(0.0)
            chart = st.empty()
            status = st.empty()
            rows = []

            def cb(step, total, s):
                bar.progress(step / total)
                rows.append({"step": step, "loss_D": s["loss_D"], "loss_G": s["loss_G"]})
                if step % 50 == 0 or step == total:
                    chart.line_chart(pd.DataFrame(rows).set_index("step"))
                    status.caption(f"step {step}/{total} · D(x)={s['D_x']:.2f} · D(G(z))={s['D_G_z']:.2f}")

            train_dcgan(train / cls, GAN_DIR / cls, iterations=int(iters), batch_size=int(bs),
                        width=int(width), augment=augment, sample_every=int(sample_every), progress_cb=cb)
            last = sorted((GAN_DIR / cls / "samples").glob("*.png"))[-1]
            st.image(str(last), caption=f"{cls} samples after training", width=420)
        st.success("Done.")

    st.divider()
    st.subheader("Training progress of saved GANs")
    for cls in trained_gans():
        with st.expander(cls):
            hist = json.loads((GAN_DIR / cls / "history.json").read_text())
            st.line_chart(pd.DataFrame({k: hist[k] for k in ("loss_D", "loss_G")}, index=hist["step"]))
            samples = sorted((GAN_DIR / cls / "samples").glob("*.png"))
            if samples:
                i = st.select_slider("Snapshot", options=list(range(len(samples))), value=len(samples) - 1,
                                     format_func=lambda k: samples[k].stem, key=f"snap_{cls}")
                st.image(str(samples[i]), width=420)


def page_generate():
    st.header("3 · Generate & balance")
    gans = trained_gans()
    if not gans:
        st.warning("Train at least one DCGAN first (page 2).")
        return

    st.subheader("Generate synthetic leaves")
    c1, c2, c3 = st.columns(3)
    cls = c1.selectbox("Class", gans)
    n = c2.slider("How many", 4, 64, 32, 4)
    seed = c3.number_input("Seed", 0, 10_000, 0)
    ckpt = GAN_DIR / cls / "generator.pt"
    netG = cached_generator(str(ckpt), ckpt.stat().st_mtime)
    imgs = generate_images(netG, n, seed=int(seed))

    real = list_images(SPLIT / "train" / cls)[:n]
    a, b = st.columns(2)
    a.markdown("**Real (64×64)**")
    if real:
        a.image(image_grid([Image.open(p).convert("RGB").resize((64, 64)) for p in real]), width="stretch")
    b.markdown("**Synthetic (DCGAN)**")
    b.image(image_grid(imgs), width="stretch")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, im in enumerate(imgs):
            ib = io.BytesIO()
            im.save(ib, "PNG")
            z.writestr(f"{cls}/syn_{i:04d}.png", ib.getvalue())
    st.download_button("Download as ZIP", buf.getvalue(), f"{cls}_synthetic.zip", "application/zip")

    st.divider()
    st.subheader("Build balanced training set")
    counts = class_counts(SPLIT / "train")
    target = st.number_input("Images per class", 1, 100_000, max(counts.values()))
    if st.button("Balance dataset", type="primary"):
        log = st.empty()
        lines = []

        def logger(msg):
            lines.append(msg)
            log.code("\n".join(lines))

        report = balance_dataset(SPLIT / "train", GAN_DIR, BALANCED, target=int(target), log=logger)
        st.success(f"Balanced dataset written to `{BALANCED}`.")
        st.bar_chart(pd.DataFrame(report).T)
    elif BALANCED.exists():
        counts_chart({"before (real)": counts, "after (real + synthetic)": class_counts(BALANCED)})


def page_classifier():
    st.header("4 · Disease classifier: real vs real + synthetic")
    if not (SPLIT / "test").exists():
        st.warning("Create the split first (page 1).")
        return
    c1, c2, c3 = st.columns(3)
    epochs = c1.number_input("Epochs", 1, 200, 8)
    img_size = c2.selectbox("Input size", [64, 96, 128, 160, 224], index=2)
    pretrained = c3.checkbox("ImageNet-pretrained ResNet-18", False,
                             help="Downloads weights on first use; usually much better on real photos.")

    runs = {"baseline": SPLIT / "train", "augmented": BALANCED}
    pick = st.multiselect("Train", list(runs), default=[k for k, v in runs.items() if v.exists()])
    if st.button("Train classifier(s)", type="primary", disabled=not pick):
        for name in pick:
            st.subheader(f"{name} ({runs[name]})")
            bar = st.progress(0.0)
            status = st.empty()

            def cb(e, total, s):
                bar.progress(e / total)
                status.caption(f"epoch {e}/{total} · loss {s['train_loss']:.3f} · "
                               f"test acc {s['test_acc']:.3f} · macro-F1 {s['test_macro_f1']:.3f}")

            train_classifier(runs[name], SPLIT / "test", CLS_DIR / name, epochs=int(epochs),
                             img_size=int(img_size), pretrained=pretrained, progress_cb=cb)

    results = {n: json.loads((CLS_DIR / n / "metrics.json").read_text())
               for n in runs if (CLS_DIR / n / "metrics.json").exists()}
    if not results:
        return
    st.divider()
    st.subheader("Results on real test images")
    cols = st.columns(len(results))
    base = results.get("baseline")
    for col, (name, m) in zip(cols, results.items()):
        delta = None if name == "baseline" or not base else f"{m['macro_f1'] - base['macro_f1']:+.3f}"
        col.metric(f"{name} · accuracy", f"{m['accuracy']:.3f}")
        col.metric(f"{name} · macro F1", f"{m['macro_f1']:.3f}", delta)

    classes = next(iter(results.values()))["classes"]
    st.markdown("**Per-class F1** (minority classes should improve)")
    st.bar_chart(pd.DataFrame({n: {c: m["per_class"][c]["f1"] for c in classes} for n, m in results.items()}))
    st.markdown("**Per-class recall**")
    st.dataframe(pd.DataFrame({n: {c: m["per_class"][c]["recall"] for c in classes} for n, m in results.items()})
                 .style.format("{:.3f}"))
    for name, m in results.items():
        with st.expander(f"Confusion matrix · {name}"):
            st.dataframe(pd.DataFrame(m["confusion_matrix"], index=[f"true {c}" for c in classes],
                                      columns=[f"pred {c}" for c in classes]))
            st.line_chart(pd.DataFrame({"test_acc": m["history"]["test_acc"],
                                        "test_macro_f1": m["history"]["test_macro_f1"]},
                                       index=m["history"]["epoch"]))


def page_predict():
    st.header("5 · Predict disease")
    models = [n for n in ("augmented", "baseline") if (CLS_DIR / n / "model.pt").exists()]
    if not models:
        st.warning("Train a classifier first (page 4).")
        return
    name = st.selectbox("Model", models)
    path = CLS_DIR / name / "model.pt"
    net, classes, size = cached_classifier(str(path), path.stat().st_mtime)
    up = st.file_uploader("Paddy leaf image", type=[e.lstrip(".") for e in IMG_EXTS])
    if up:
        img = Image.open(up).convert("RGB")
        a, b = st.columns([1, 1])
        a.image(img, width="stretch")
        preds = predict(net, classes, size, img)
        b.metric("Prediction", preds[0][0], f"{preds[0][1] * 100:.1f}% confidence")
        b.bar_chart(pd.DataFrame(preds, columns=["class", "probability"]).set_index("class"))


PAGES = {
    "Overview": page_overview,
    "1 · Dataset": page_dataset,
    "2 · Train DCGAN": page_train_gan,
    "3 · Generate & balance": page_generate,
    "4 · Classifier": page_classifier,
    "5 · Predict": page_predict,
}
choice = st.sidebar.radio("Navigate", list(PAGES))
st.sidebar.divider()
st.sidebar.caption("Paddy Leaf Disease Augmentation using DCGAN")
PAGES[choice]()
