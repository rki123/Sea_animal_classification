import os
import sys

# keras backend must be set before any keras import
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass
import keras
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import IMG_SIZE, EFF_MODEL_PATH, VIT_CKPT_PATH, ENSEMBLE_CFG
from src.efficientnet_model import load_efficientnet
from src.vit_model import load_vit
from src.dataset import vit_transforms



HF_REPO = "https://huggingface.co/rki123/sea-animal-classifier"

def _pull_weights():
    if not HF_REPO:
        return
    from huggingface_hub import hf_hub_download
    needed = {
        EFF_MODEL_PATH: "efficientnet_sea.keras",
        VIT_CKPT_PATH:  "vit_b16_sea.pt",
        ENSEMBLE_CFG:   "ensemble_pso_config.json",
    }
    for dst, src in needed.items():
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            hf_hub_download(repo_id=HF_REPO, filename=src,
                            local_dir=os.path.dirname(dst) or ".")

try:
    _pull_weights()
except Exception as e:
    st.error(f"Failed to fetch model weights: {e}")
    st.stop()


st.set_page_config(
    page_title="DeepOcean - Sea Animal Classifier",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background-color: #0b101e;
        color: #e0e6ed;
    }
    h1, h2, h3 {
        color: #00d2ff;
        font-family: 'Inter', sans-serif;
    }
    .card {
        background: rgba(16, 24, 43, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    .stFileUploader > div > div {
        background-color: rgba(0, 210, 255, 0.05);
        border: 2px dashed #00d2ff;
        border-radius: 12px;
    }
    .stFileUploader > div > div:hover {
        background-color: rgba(0, 210, 255, 0.1);
        border-color: #3a86ff;
    }
    .pred-label {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a86ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all():
    try:
        with open(ENSEMBLE_CFG) as f:
            cfg = json.load(f)
        cfg["weights"] = [cfg["eff_weight"], cfg["vit_weight"]]
        n = len(cfg["classes"])
    except FileNotFoundError:
        return None, None, None, None, "ensemble_pso_config.json not found — run the training notebook first."

    try:
        eff = load_efficientnet(n)
        vit = load_vit(n)
        vit.eval()
    except Exception as e:
        return None, None, None, None, str(e)

    _, val_tf = vit_transforms()
    return eff, vit, cfg, val_tf, None


eff_model, vit_model, cfg, vit_tf, err = load_all()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(img: Image.Image):
    classes = cfg["classes"]
    w_eff, w_vit = cfg["weights"]

    # EfficientNet path
    img_rgb = img.convert("RGB").resize(IMG_SIZE)
    arr = keras.utils.img_to_array(img_rgb)
    arr = np.expand_dims(arr, 0)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)

    t_eff = torch.tensor(arr, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        eff_probs = torch.softmax(eff_model(t_eff, training=False), dim=-1).cpu().numpy()[0]

    # ViT path
    t_vit = vit_tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            vit_probs = torch.softmax(vit_model(t_vit), dim=1).cpu().numpy()[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    blended = w_eff * eff_probs + w_vit * vit_probs
    top_idx = np.argsort(blended)[::-1][:5]

    return (
        [classes[i] for i in top_idx],
        blended[top_idx].tolist(),
        arr,
    )


def _find_layer(model, name):
    for layer in model.layers:
        if layer.name == name:
            return layer
        if hasattr(layer, "layers"):
            res = _find_layer(layer, name)
            if res:
                return res
    return None


def gradcam(model, arr, class_idx):
    layer = _find_layer(model, "top_activation")
    if layer is None:
        return None

    acts, grads = [], []
    h1 = layer.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))

    t = torch.tensor(arr, dtype=torch.float32).to(DEVICE)
    model.zero_grad()
    try:
        with torch.enable_grad():
            out = model(t, training=False)
            out[0, class_idx].backward()
    except Exception:
        h1.remove()
        h2.remove()
        return None

    h1.remove()
    h2.remove()

    if not acts or not grads:
        return None

    a = acts[0].detach()
    g = grads[0].detach()
    pooled = g.mean(dim=(0, 1, 2))
    hmap = torch.relu((a[0] * pooled).sum(dim=-1))
    mx = hmap.max()
    if mx > 0:
        hmap = hmap / mx
    return hmap.cpu().numpy().astype(np.float32)


def overlay_heatmap(img: Image.Image, hmap, alpha=0.45):
    orig = np.array(img.convert("RGB"))
    h, w = orig.shape[:2]
    resized = cv2.resize(np.asarray(hmap, dtype=np.float32), (w, h))
    colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET),
                           cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.uint8(colored * alpha + orig * (1 - alpha)))


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Model Details")
    if cfg:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Architecture:** EfficientNetB7 + ViT-B16")
        st.write(f"**Classes:** {len(cfg['classes'])}")
        st.write(f"**Test accuracy:** {cfg.get('test_accuracy', 0) * 100:.2f}%")
        st.markdown("**PSO ensemble weights**")
        st.progress(cfg["weights"][0], text=f"EfficientNet: {cfg['weights'][0]:.2f}")
        st.progress(cfg["weights"][1], text=f"ViT-B16: {cfg['weights'][1]:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### ℹ️ How it works")
    st.info(
        "Each image is passed through EfficientNetB7 (CNN) and ViT-B16 (Vision Transformer) "
        "independently. PSO was used to find the optimal blending weights on the validation set. "
        "The GradCAM heatmap shows which parts of the image EfficientNet focused on."
    )

# ── main ─────────────────────────────────────────────────────────────────────
st.title("🌊 DeepOcean: Sea Animal Classification")
st.markdown(
    "Upload a photo of a sea creature — the ensemble (EfficientNetB7 + ViT-B16) "
    "will classify it and show where it looked."
)

if err:
    st.error(err)
    st.stop()

col1, col2 = st.columns(2)

with col1:
    upload = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if upload:
        try:
            image = Image.open(upload)
            st.image(image, caption="Your image", use_container_width=True)
        except Exception:
            st.error("Could not open this file.")
            upload = None

with col2:
    if upload:
        with st.spinner("Running inference..."):
            try:
                top_classes, top_probs, arr = run_inference(image)
                pred, conf = top_classes[0], top_probs[0]

                if conf < 0.40:
                    st.warning(
                        f"Low confidence ({conf * 100:.1f}%) — this might not be one of the "
                        f"23 supported sea animal classes. The model always picks its closest match."
                    )

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<p style='text-align:center;color:#a0aec0;margin-bottom:0'>TOP PREDICTION</p>",
                            unsafe_allow_html=True)
                st.markdown(f"<h2 class='pred-label'>{pred}</h2>", unsafe_allow_html=True)
                st.markdown(
                    f"<p style='text-align:center;font-size:1.2rem'>Confidence: <b>{conf * 100:.1f}%</b></p>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("### Top-5 Probabilities")
                chart_df = pd.DataFrame({
                    "Class": top_classes,
                    "Confidence (%)": [p * 100 for p in top_probs],
                })
                st.bar_chart(chart_df.set_index("Class"), height=250, color="#3a86ff")

                st.markdown("### GradCAM — Model Attention")
                class_idx = cfg["classes"].index(pred)
                hmap = gradcam(eff_model, arr, class_idx)
                if hmap is not None:
                    st.image(
                        overlay_heatmap(image, hmap),
                        caption="Warmer colours = higher model attention",
                        use_container_width=True,
                    )
                else:
                    st.info("GradCAM not available for this input.")

            except Exception as e:
                st.error(f"Something went wrong: {e}")
    else:
        st.markdown(
            "<div class='card' style='text-align:center;color:#a0aec0;padding:50px 20px'>"
            "Upload an image to see the prediction."
            "</div>",
            unsafe_allow_html=True,
        )
