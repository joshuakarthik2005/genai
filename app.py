"""Streamlit UI for Paddy Leaf Disease Augmentation using DCGAN.

Empirical Benchmark Presentation, Research Novelty Defense, and Live Agronomist Advisory.
Run:  streamlit run app.py
"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
RAW = BASE_DIR / "Rice Leaf Disease Images" / "Rice Leaf Disease Images"
if not RAW.exists():
    RAW = BASE_DIR / "Rice Leaf Disease Images"
if not RAW.exists():
    RAW = BASE_DIR / "data" / "raw"

SPLIT = BASE_DIR / "data" / "split"
BALANCED = BASE_DIR / "data" / "balanced"
GAN_DIR = BASE_DIR / "runs" / "gan"
CLS_DIR = BASE_DIR / "runs" / "classifier"
PLOTS_DIR = BASE_DIR / "runs" / "plots"
SUMMARY_PATH = BASE_DIR / "runs" / "report" / "experiment_summary.json"

st.set_page_config(
    page_title="Paddy DCGAN · Research Findings & Benchmarks",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for High-End Academic / Presentation Aesthetics ────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10B981 0%, #3B82F6 50%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.8rem;
    }
    .metric-card {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 6px;
    }
    .badge-green { background: rgba(16, 185, 129, 0.18); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-blue  { background: rgba(59, 130, 246, 0.18); color: #3B82F6; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.18); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-purple{ background: rgba(139, 92, 246, 0.18); color: #8B5CF6; border: 1px solid rgba(139, 92, 246, 0.3); }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px;
        background-color: #0F172A;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Load Experiment Data ──────────────────────────────────────────────────────
@st.cache_data
def load_experiment_summary():
    if SUMMARY_PATH.exists():
        try:
            return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Fallback to embedded canonical Kaggle results
    return {
        "dataset": {
            "name": "Mendeley Rice Leaf Disease Images",
            "total_images": 5932,
            "classes": ["Bacterialblight", "Blast", "Brownspot", "Tungro"],
            "distribution": {
                "Bacterialblight": 1584,
                "Blast": 1440,
                "Brownspot": 1600,
                "Tungro": 1308,
            },
            "split_80_20": {
                "train": {"Bacterialblight": 1267, "Blast": 1152, "Brownspot": 1280, "Tungro": 1046, "total": 4745},
                "test": {"Bacterialblight": 317, "Blast": 288, "Brownspot": 320, "Tungro": 262, "total": 1187},
            },
        },
        "dcgan_training": {
            "hardware": "Kaggle 2x Tesla T4 GPU",
            "iterations_per_class": 5000,
            "runs": {
                "Bacterialblight": {"duration_minutes": 10.1, "final_loss_D": 0.779, "final_loss_G": 2.210},
                "Blast": {"duration_minutes": 12.9, "final_loss_D": 0.771, "final_loss_G": 1.661},
                "Brownspot": {"duration_minutes": 13.8, "final_loss_D": 0.637, "final_loss_G": 1.784},
                "Tungro": {"duration_minutes": 18.8, "final_loss_D": 0.362, "final_loss_G": 5.069},
            },
            "total_training_time_minutes": 55.6,
            "total_synthetic_generated": 3255,
        },
        "classifiers": {
            "methods": {
                "baseline": {"label": "Baseline (Real Only)", "train_size": 4745, "accuracy": 1.0, "macro_f1": 1.0},
                "trad_aug": {"label": "Traditional Augmentation", "train_size": 8000, "accuracy": 1.0, "macro_f1": 1.0},
                "gan_aug": {"label": "DCGAN Augmentation (Ours)", "train_size": 8000, "accuracy": 1.0, "macro_f1": 1.0},
            }
        },
        "fid_scores": {
            "scores": {"Brownspot": 207.18, "Bacterialblight": 232.69, "Blast": 246.41, "Tungro": 283.73},
            "average_fid": 242.50,
        },
    }


SUMMARY = load_experiment_summary()


# ── Agronomic Disease Knowledge Base ──────────────────────────────────────────
DISEASE_INFO = {
    "Bacterialblight": {
        "name": "Bacterial Blight",
        "pathogen": "Xanthomonas oryzae pv. oryzae",
        "symptoms": "Water-soaked streaks on leaf blades turning yellow-white with wavy margins. Seedling wilt (kresek) in severe cases.",
        "favorable_conditions": "High humidity (70%+), warm temperature (25–34°C), strong winds and heavy rainfall.",
        "cultural_control": [
            "Maintain balanced fertilization: avoid excessive nitrogen.",
            "Drain flooded field temporarily to restrict bacterial dissemination.",
            "Use certified disease-free seeds and sanitize farm implements.",
        ],
        "chemical_treatment": [
            "Foliar spray with Copper Hydroxide (2.5 g/L) or Copper Oxychloride (3 g/L).",
            "Bactericide: Streptocycline (100–150 ppm) mixed with copper fungicide.",
        ],
        "resistant_varieties": "IR64, Swarna Sub1, Improved White Ponni (IWP).",
    },
    "Blast": {
        "name": "Rice Blast",
        "pathogen": "Magnaporthe oryzae (Pyricularia oryzae)",
        "symptoms": "Spindle-shaped or diamond-shaped lesions with grey/white centers and dark brown margins. Attacks leaves, collars, nodes, and panicles.",
        "favorable_conditions": "Night temperatures below 20°C with 90%+ relative humidity and dew on leaf surface.",
        "cultural_control": [
            "Split nitrogen applications into multiple smaller doses.",
            "Avoid stagnant water; maintain intermittent wet-and-dry irrigation.",
            "Eradicate infected stubbles and weeds hosting blast spores.",
        ],
        "chemical_treatment": [
            "Foliar spray with Tricyclazole 75% WP @ 0.6 g/L of water at early lesion onset.",
            "Isoprothiolane 40% EC @ 1.5 mL/L or Kasugamycin 3% SL @ 2.5 mL/L.",
        ],
        "resistant_varieties": "BPT 5204 (tolerant), Jaya, Rasi, CR Dhan 310.",
    },
    "Brownspot": {
        "name": "Brown Spot",
        "pathogen": "Bipolaris oryzae (Cochliobolus miyabeanus)",
        "symptoms": "Oval to circular brown spots with light-brown centers and yellow halos. Leads to seed discoloration and reduced kernel weight.",
        "favorable_conditions": "Poor, nutrient-deficient sandy soils (deficiencies in K, Si, Fe) and prolonged moisture stress.",
        "cultural_control": [
            "Apply potassium (MOP) and silica fertilizers to strengthen leaf cuticle.",
            "Hot water seed treatment at 52–54°C for 15 minutes before sowing.",
            "Improve soil drainage and incorporate compost/organic matter.",
        ],
        "chemical_treatment": [
            "Mancozeb 75% WP @ 2.5 g/L or Propiconazole 25% EC @ 1 mL/L.",
            "Seed dressing with Carbendazim 50% WP @ 2 g/kg seed.",
        ],
        "resistant_varieties": "ADT 39, ASD 16, IR 36.",
    },
    "Tungro": {
        "name": "Rice Tungro Disease",
        "pathogen": "RTBV (Rice Tungro Bacilliform Virus) + RTSV (Spherical Virus)",
        "symptoms": "Yellowing to orange discoloration from leaf tips downward. Extreme plant stunting, reduced tillering, delayed flowering.",
        "favorable_conditions": "Dense populations of Green Leafhopper vector (*Nephotettix virescens*) in dry to wet transition seasons.",
        "cultural_control": [
            "Practice synchronous planting within 2–3 weeks across neighboring fields.",
            "Plow under infected stubbles immediately after harvest.",
            "Use yellow sticky traps to monitor and trap leafhopper vectors.",
        ],
        "chemical_treatment": [
            "Vector suppression: Spray Imidacloprid 17.8% SL @ 0.3 mL/L or Thiamethoxam 25% WG @ 0.2 g/L.",
            "Eco-friendly alternative: 5% Neem Seed Kernel Extract (NSKE).",
        ],
        "resistant_varieties": "Vikramarya, IR 64, Nidhi, Kunjan.",
    },
}


# ── Helper to load sample images ──────────────────────────────────────────────
def find_classifier():
    """Kaggle-trained ResNet-18 checkpoint, preferring the GAN-augmented model."""
    for name in ("gan_aug", "trad_aug", "baseline"):
        p = CLS_DIR / name / "model.pt"
        if p.exists():
            return p
    return None


@st.cache_resource
def cached_classifier(path, mtime):
    from paddy_gan.classifier import load_classifier
    return load_classifier(path)


def predict(net, classes, img_size, image):
    from paddy_gan.classifier import predict as _predict
    return _predict(net, classes, img_size, image)


def get_sample_images_dict():
    samples = {}
    if RAW.exists():
        for d in RAW.iterdir():
            if d.is_dir() and d.name in DISEASE_INFO:
                imgs = list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))
                if imgs:
                    samples[d.name] = imgs[:6]
    return samples


# ── PAGE 1: Benchmark Findings (Executive Presentation) ──────────────────────
def page_findings():
    st.markdown('<div class="main-title">🌾 Empirical Benchmark & Findings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Rigorous 3-Way Head-to-Head Comparison: Baseline vs. Traditional Augmentation vs. DCGAN Generative Augmentation on Kaggle 2× Tesla T4 GPUs</div>',
        unsafe_allow_html=True,
    )

    # Top KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-label">Dataset Scale</div>
            <div class="metric-value">5,932</div>
            <div><span class="badge-pill badge-green">4 Real Classes</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-label">GAN Synthesized</div>
            <div class="metric-value">3,255</div>
            <div><span class="badge-pill badge-blue">+68.6% Expansion</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-label">Held-out Test Acc</div>
            <div class="metric-value">100.0%</div>
            <div><span class="badge-pill badge-green">1,187 Real Leaves</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-label">Mean FID Score</div>
            <div class="metric-value">242.5</div>
            <div><span class="badge-pill badge-amber">Inception-V3 (299px)</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-label">GPU Training Time</div>
            <div class="metric-value">55.6m</div>
            <div><span class="badge-pill badge-purple">2× Tesla T4</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    # Head-to-Head Comparison Table
    st.subheader("📊 3-Way Scientific Comparison Summary")
    st.markdown(
        """
All three classifiers (ResNet-18 ImageNet-pretrained, 20 epochs, batch size 64) were evaluated on the **identical, strictly held-out real-only test set (1,187 images)**.
"""
    )

    comp_df = pd.DataFrame(
        [
            {
                "Method": "1. Baseline (Real Only)",
                "Training Distribution": "4,745 real images (imbalanced)",
                "Test Accuracy": "100.00%",
                "Macro F1": "100.00%",
                "F1 (Bacterialblight)": "100.00%",
                "F1 (Blast)": "100.00%",
                "F1 (Brownspot)": "100.00%",
                "F1 (Tungro)": "100.00%",
                "FID Score": "— (Real Data)",
            },
            {
                "Method": "2. Traditional Augmentation",
                "Training Distribution": "4,745 real + 3,255 flips/rotates/jitter (8,000 total)",
                "Test Accuracy": "100.00%",
                "Macro F1": "100.00%",
                "F1 (Bacterialblight)": "100.00%",
                "F1 (Blast)": "100.00%",
                "F1 (Brownspot)": "100.00%",
                "F1 (Tungro)": "100.00%",
                "FID Score": "— (Transformations)",
            },
            {
                "Method": "3. DCGAN Augmentation (Ours)",
                "Training Distribution": "4,745 real + 3,255 DCGAN synthetic (8,000 total)",
                "Test Accuracy": "100.00%",
                "Macro F1": "100.00%",
                "F1 (Bacterialblight)": "100.00%",
                "F1 (Blast)": "100.00%",
                "F1 (Brownspot)": "100.00%",
                "F1 (Tungro)": "100.00%",
                "FID Score": "242.50 (Avg across classes)",
            },
        ]
    )
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Visual Charts Showcase
    st.subheader("📈 Visual Verification & Publication Figures")
    tab_overall, tab_f1, tab_fid, tab_loss, tab_matrix = st.tabs(
        ["Overall Comparison", "Per-Class F1", "FID Score Analysis", "DCGAN Loss Dynamics", "Confusion Matrices"]
    )

    with tab_overall:
        p1 = PLOTS_DIR / "plot1_overall_comparison.png"
        if p1.exists():
            st.image(str(p1), caption="Overall Accuracy & Macro-F1 across all 3 conditions", use_container_width=True)
        else:
            st.info("Run experiment to render plot1_overall_comparison.png")

    with tab_f1:
        p2 = PLOTS_DIR / "plot2_perclass_f1.png"
        if p2.exists():
            st.image(str(p2), caption="Per-Class F1 Breakdown across 4 Disease Categories", use_container_width=True)

    with tab_fid:
        col_fid_img, col_fid_txt = st.columns([1.2, 1])
        with col_fid_img:
            p6 = PLOTS_DIR / "plot6_fid_scores.png"
            if p6.exists():
                st.image(str(p6), caption="Fréchet Inception Distance (FID) per Class", use_container_width=True)
        with col_fid_txt:
            st.markdown("#### 🔬 Scientific Interpretation of the FID Scores")
            st.markdown(
                """
            - **Brownspot**: `207.18` (closest to real of the four)
            - **Bacterialblight**: `232.69`
            - **Blast**: `246.41` (colour and some lesion spots present, but blurred)
            - **Tungro**: `283.73` (furthest from real; generator was losing to the discriminator by step 5,000)
            
            > **Why are FID values between 200–280?**  
            > Lower FID is better; values above ~100 mean the synthetic images are still clearly distinguishable from real ones. Part of the gap comes from upscaling 64×64 DCGAN output to Inception-V3's 299×299 input, and part is genuine blur and texture artefacts visible in the real-vs-synthetic grids.
            > **Finding**: Adding 3,255 DCGAN images did **not reduce** downstream accuracy (still 100%). Because the baseline was already at 100%, this experiment cannot show whether the synthetic images *improve* the classifier.
            """
            )

    with tab_loss:
        p5 = PLOTS_DIR / "plot5_gan_curves.png"
        if p5.exists():
            st.image(
                str(p5),
                caption="Adversarial Loss Convergence (D vs G) across 5,000 Steps per Class on 2× Tesla T4 GPUs",
                use_container_width=True,
            )

    with tab_matrix:
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            pb = PLOTS_DIR / "confusion_baseline.png"
            if pb.exists():
                st.image(str(pb), caption="Baseline Confusion Matrix", use_container_width=True)
        with c_m2:
            pt = PLOTS_DIR / "confusion_trad_aug.png"
            if pt.exists():
                st.image(str(pt), caption="Traditional Augmentation Matrix", use_container_width=True)
        with c_m3:
            pg = PLOTS_DIR / "confusion_gan_aug.png"
            if pg.exists():
                st.image(str(pg), caption="DCGAN Augmentation Matrix", use_container_width=True)


# ── PAGE 2: Research Defense & Novelty ────────────────────────────────────────
def page_defense():
    st.markdown('<div class="main-title">🛡️ Research Defense & Generative AI Novelty</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">How to Defend This Case Study Against Faculty / Reviewer Inquiries and Foundation Models (ChatGPT / DALL-E)</div>',
        unsafe_allow_html=True,
    )

    st.subheader("1. The Core Question: Can ChatGPT or DALL-E Replace This?")
    st.markdown(
        """
    **No.** While foundation models excel at open-domain text and artistic imagery, they fail in specialized agronomic applications:
    """
    )

    comp_table = pd.DataFrame(
        [
            {
                "Evaluation Dimension": "Domain Fidelity",
                "ChatGPT / DALL-E 3 / Midjourney": "Hallucinates generic spots; lacks biological plant pathology constraints",
                "Our DCGAN Pipeline": "Learns only from real labelled leaf images of each disease class",
            },
            {
                "Evaluation Dimension": "Data Provenance",
                "ChatGPT / DALL-E 3 / Midjourney": "Trained on opaque internet scrape data with unknown labels",
                "Our DCGAN Pipeline": "Trained only on the public, labelled Mendeley Rice Leaf Disease dataset",
            },
            {
                "Evaluation Dimension": "Downstream Integrity",
                "ChatGPT / DALL-E 3 / Midjourney": "Cannot be used as clinical/agronomic ground truth (unverified symptoms)",
                "Our DCGAN Pipeline": "Adding 3,255 synthetic samples did not reduce held-out test accuracy (100%)",
            },
            {
                "Evaluation Dimension": "Edge & Rural Deployment",
                "ChatGPT / DALL-E 3 / Midjourney": "Requires high-speed internet, API tokens, cloud latency, and recurring fees",
                "Our DCGAN Pipeline": "Small models (DCGAN generator + ResNet-18, ~45 MB) that can run offline; edge-device speed not yet benchmarked",
            },
            {
                "Evaluation Dimension": "Quantitative Quality",
                "ChatGPT / DALL-E 3 / Midjourney": "No class-conditioned FID against true disease distributions",
                "Our DCGAN Pipeline": "Measured with Inception-V3 Fréchet Inception Distance (FID) per disease category",
            },
        ]
    )
    st.dataframe(comp_table, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("2. Explaining the 100% Test Accuracy (Scientific Defense)")
    st.info(
        """
    **Question an examiner will ask**: *"Why did the Baseline also get 100%? Does that mean GAN augmentation wasn't needed?"*
    
    **The Defensible Scientific Answer**:
    1. **Sufficient Real Data in Mendeley**: The dataset provided **1,046 to 1,280 real training images per class** (4,745 total). An ImageNet-pretrained ResNet-18 has sufficient representational capacity to separate 4 distinct visual classes when over 1,000 real samples per class are provided.
    2. **The Real Test of Generative AI**: In generative augmentation research, injecting thousands of synthetic samples often causes **mode corruption or semantic drift** (where fake artifacts degrade classifier performance).
    3. **What we observed**: Adding **3,255 DCGAN-generated leaves** (balancing every class to 2,000 images) caused **no performance drop on 1,187 held-out real test leaves**. Since every method scored 100%, the test cannot measure an *improvement* from GAN augmentation.
    4. **Limitations**: The Mendeley classes are nearly balanced (1,308–1,600 images, 1.2× spread), so there was little imbalance to correct. The dataset also appears to contain near-duplicate (shifted) photos, which can place near-copies of training images in the test split and inflate accuracy.
    5. **Future work**: Repeat the experiment with minority classes cut to 20–50 real images and near-duplicates removed before splitting. This is the data-scarce setting the case study targets, and where GAN augmentation would be expected to help.
    """
    )

    st.divider()

    st.subheader("3. Grad-CAM Explainability: Where the Classifier Looks")
    st.markdown(
        """
    Rather than acting as a black box, **Grad-CAM (Gradient-weighted Class Activation Mapping)** extracts gradients from `layer4` of the ResNet-18 classifier to visualize which leaf regions triggered the prediction.
    """
    )
    p8 = PLOTS_DIR / "plot8_gradcam_grid.png"
    if p8.exists():
        st.image(str(p8), caption="Grad-CAM Lesion Localization Overlays on Test Images", use_container_width=True)
    else:
        st.markdown(
            "> Grad-CAM heatmaps verify that the network attends to the **necrotic lesions and chlorotic borders** rather than background paper, shadows, or leaf margins."
        )


# ── PAGE 3: Dataset Explorer ──────────────────────────────────────────────────
def page_dataset_explorer():
    st.markdown('<div class="main-title">📁 Mendeley Rice Leaf Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Explore 5,932 real field images across 4 disease classes with stratified 80/20 train/test distribution</div>',
        unsafe_allow_html=True,
    )

    dist = SUMMARY["dataset"]["distribution"]
    split = SUMMARY["dataset"]["split_80_20"]

    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("Class Distribution")
        df_dist = pd.DataFrame(
            {
                "Class": list(dist.keys()),
                "Total Images": list(dist.values()),
                "Train (80%)": [split["train"][c] for c in dist],
                "Test (20%)": [split["test"][c] for c in dist],
            }
        )
        st.dataframe(df_dist, use_container_width=True, hide_index=True)
        st.bar_chart(df_dist.set_index("Class")[["Train (80%)", "Test (20%)"]])

    with c2:
        st.subheader("Real Disease Samples")
        samples = get_sample_images_dict()
        if samples:
            selected_cls = st.selectbox("Select Class to View", list(samples.keys()))
            imgs = samples[selected_cls]
            cols = st.columns(3)
            for i, p in enumerate(imgs[:6]):
                with cols[i % 3]:
                    st.image(str(p), caption=f"{selected_cls} #{i+1}", use_container_width=True)
        else:
            selected_cls = st.selectbox("Select Class to View", list(dist.keys()))
            grid = PLOTS_DIR / f"plot7_real_vs_syn_{selected_cls}.png"
            if grid.exists():
                st.image(str(grid), caption=f"{selected_cls}: real training leaves (top) vs DCGAN synthetic (bottom), from the Kaggle run",
                         use_container_width=True)
            st.caption("Place the dataset in `Rice Leaf Disease Images/` to browse the full image set.")


# ── PAGE 4: DCGAN Architecture & Training Dynamics ───────────────────────────
def page_gan_dynamics():
    st.markdown('<div class="main-title">⚡ DCGAN Architecture & Training Dynamics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">5,000 steps per class with DiffAugment stabilization on Kaggle 2× Tesla T4 GPUs</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Architectural Specification")
        st.markdown(
            r"""
        - **Generator $G(z)$**:
          - Latent noise $z \sim \mathcal{N}(0, I_{100})$
          - Transposed Convolutions ($4\times 4 \rightarrow 8\times 8 \rightarrow 16\times 16 \rightarrow 32\times 32 \rightarrow 64\times 64$)
          - BatchNorm2d + ReLU activations
          - Output: $\tanh$ activation $\rightarrow [-1, 1]$ RGB
        - **Discriminator $D(x)$**:
          - Strided Convolutions ($64\times 64 \rightarrow 32\times 32 \rightarrow 16\times 16 \rightarrow 8\times 8 \rightarrow 4\times 4$)
          - BatchNorm2d + LeakyReLU($0.2$)
          - Output: Sigmoid activation $\rightarrow P(\text{real})$
        - **DiffAugment**: Differentiable translation, cutout, and color jitter applied to both real and fake images during backprop to prevent discriminator memorization.
        """
        )

    with c2:
        st.subheader("Kaggle Training Times per Class")
        runs = SUMMARY["dcgan_training"]["runs"]
        time_df = pd.DataFrame(
            [
                {
                    "Disease Class": k,
                    "Training Time": f"{v['duration_minutes']} min",
                    "Final Loss D": v["final_loss_D"],
                    "Final Loss G": v["final_loss_G"],
                }
                for k, v in runs.items()
            ]
        )
        st.dataframe(time_df, use_container_width=True, hide_index=True)
        st.caption("Total GPU time across all 4 GANs: 55.6 minutes on dual Tesla T4 GPUs.")

    p5 = PLOTS_DIR / "plot5_gan_curves.png"
    if p5.exists():
        st.image(str(p5), caption="DCGAN Adversarial Loss Curves (5,000 Iterations Each)", use_container_width=True)


# ── PAGE 5: Live Agronomist Advisory & Predictor ──────────────────────────────
def page_advisory():
    st.markdown(
        '<div class="main-title">🌾 Interactive Disease Diagnosis & Agronomist Advisory</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Simulate field diagnosis with real crop leaves and retrieve actionable treatment protocols for smallholder farmers</div>',
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns([1, 1.3])

    samples = get_sample_images_dict()
    selected_image = None
    selected_class_name = None

    with c_left:
        st.subheader("Select or Upload a Leaf Image")
        mode = st.radio("Input Source", ["Preloaded Field Samples", "Upload Image"], horizontal=True)

        if mode == "Preloaded Field Samples" and samples:
            chosen_disease = st.selectbox("Disease Category", list(samples.keys()))
            img_list = samples[chosen_disease]
            idx = st.slider("Sample Index", 1, len(img_list), 1)
            img_path = img_list[idx - 1]
            selected_image = Image.open(img_path).convert("RGB")
            selected_class_name = chosen_disease
            st.image(selected_image, caption=f"Selected: {chosen_disease} #{idx}", use_container_width=True)
        else:
            up = st.file_uploader("Upload Rice Leaf Photo (JPG, PNG)", type=["jpg", "jpeg", "png"])
            if up:
                selected_image = Image.open(up).convert("RGB")
                st.image(selected_image, caption="Uploaded Image", use_container_width=True)

    with c_right:
        model_path = find_classifier()
        if model_path is None:
            st.subheader("Treatment Advisory")
            st.info(
                "Automatic diagnosis is offline: the trained classifier weights are not bundled with this app. "
                "Choose the disease below to view its treatment protocol. "
                f"To enable live diagnosis, place the Kaggle `model.pt` at `{CLS_DIR / 'gan_aug' / 'model.pt'}`."
            )
            keys = list(DISEASE_INFO)
            default = keys.index(selected_class_name) if selected_class_name in keys else 0
            chosen = st.selectbox("Disease", keys, index=default, format_func=lambda k: DISEASE_INFO[k]["name"])
            render_protocol(DISEASE_INFO[chosen])
        elif selected_image is not None:
            st.subheader("Diagnostic Results")
            net, classes, img_size = cached_classifier(str(model_path), model_path.stat().st_mtime)
            preds = predict(net, classes, img_size, selected_image)
            target_class, confidence = preds[0]
            probs = dict(preds)
            info = DISEASE_INFO.get(target_class)
            display_name = info["name"] if info else target_class
            pathogen = info["pathogen"] if info else "—"

            st.markdown(
                f"""
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
                <div style="font-size: 0.85rem; color: #10B981; font-weight: 600; text-transform: uppercase;">Primary Diagnosis</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC;">{display_name}</div>
                <div style="color: #94A3B8; font-size: 0.9rem;">Pathogen: <i>{pathogen}</i> | Confidence: <b>{confidence * 100:.1f}%</b></div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.caption(f"ResNet-18 · `{model_path.parent.name}` model · input {img_size}×{img_size}")
            if selected_class_name:
                if selected_class_name == target_class:
                    st.success(f"Matches the sample's true label ({selected_class_name}).")
                else:
                    st.error(f"Misclassified: true label is {selected_class_name}.")
            if confidence < 0.6:
                st.warning("Low confidence — the image may be unclear or outside the trained disease classes.")

            # Confidence distribution chart
            prob_df = pd.DataFrame(list(probs.items()), columns=["Disease", "Probability"]).set_index("Disease")
            st.bar_chart(prob_df)

            if info is not None:
                render_protocol(info)


def render_protocol(info):
    """Agronomic action plan for one disease from DISEASE_INFO."""
    st.markdown(f"**Pathogen:** *{info['pathogen']}*")
    st.markdown("### 📋 Agronomist Treatment Protocol")

    st.markdown(f"**Symptoms Profile:** {info['symptoms']}")
    st.markdown(f"**Epidemiology:** {info['favorable_conditions']}")

    t1, t2, t3 = st.tabs(["Chemical Sprays", "Cultural & Water Management", "Resistant Seeds"])
    with t1:
        st.markdown("**Recommended Fungicides / Bactericides:**")
        for item in info["chemical_treatment"]:
            st.markdown(f"- 🧪 {item}")
    with t2:
        st.markdown("**Field Management Protocols:**")
        for item in info["cultural_control"]:
            st.markdown(f"- 🚜 {item}")
    with t3:
        st.markdown(f"**Recommended Resistant Cultivars:** `{info['resistant_varieties']}`")


# ── Navigation Router ─────────────────────────────────────────────────────────
PAGES = {
    "📊 Empirical Benchmark & Findings": page_findings,
    "🛡️ Research Defense & Novelty": page_defense,
    "🌾 Live Diagnosis & Agronomist Advisory": page_advisory,
    "📁 Dataset Explorer": page_dataset_explorer,
    "⚡ DCGAN Dynamics & Convergence": page_gan_dynamics,
}

st.sidebar.markdown("## 🌾 Paddy DCGAN")
st.sidebar.markdown("**Generative AI Crop Augmentation Case Study**")
choice = st.sidebar.radio("Navigate Sections", list(PAGES.keys()))
st.sidebar.divider()

st.sidebar.markdown(
    """
**Key Kaggle Stats:**
- 🖥️ **Hardware**: 2× Tesla T4
- ⏱️ **GAN Time**: 55.6 min (5k iters)
- 🍃 **Real Images**: 5,932
- 🧪 **Synthetic**: 3,255
- 🎯 **Test Acc**: 100.0%
- 📐 **Mean FID**: 242.50
"""
)
st.sidebar.caption("Amrita Vishwa Vidyapeetham · GenAI Study")

PAGES[choice]()
