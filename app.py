import streamlit as st
import torch
import os
from PIL import Image
import yaml
from src.model import DynamicCNN
from src.data_setup import get_train_test_transforms, get_mean_and_std
from src. utils import get_list_labels

# --- CONFIGURATION ---
MODEL_PATH = "models/pokemon_cnn_best.pth"
SAMPLES_DIR = "samples/"

# --- CALLBACKS FOR MUTUAL EXCLUSION ---
def on_upload_change():
    # If a file is uploaded, clear the sample selector
    st.session_state.selector_key += 1

def on_sample_change():
    # If a sample is selected, clear the file uploader
    st.session_state.uploader_key += 1

# --- SESSION STATE INITIALIZATION ---
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "selector_key" not in st.session_state:
    st.session_state.selector_key = 0

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = DynamicCNN(
        n_layers=cfg['model']['n_layers'],
        n_filters=cfg['model']['n_filters'],
        kernel_sizes=cfg['model']['kernel_sizes'],
        dropout_rate=cfg['model']['dropout_rate'],
        fc_size=cfg['model']['fc_size'],
        num_classes=cfg["model"]["num_classes"]
    ).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, device

model, DEVICE = load_model()
CLASS_NAMES = get_list_labels()
mean, std = get_mean_and_std()
_, test_transform = get_train_test_transforms(mean=mean, std=std)

# --- UI LAYOUT ---
st.set_page_config(page_title="Pokémon Classifier", layout="wide")
st.title("⚡ Pokémon Species Classifier")
st.markdown("---")

# --- SIDEBAR & MAIN INPUT ---
col_sidebar, col_main = st.columns([1, 3])

with st.sidebar:
    st.header("🧪 Quick Test")
    sample_images = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', ".avif"))] if os.path.exists(SAMPLES_DIR) else []
    
    # Selecting a sample triggers on_sample_change callback
    selected_sample = st.selectbox(
        "Choose a sample:", 
        ["None"] + sample_images,
        key=f"sample_select_{st.session_state.selector_key}",
        on_change=on_sample_change
    )

with col_main:
    # Uploading a file triggers on_upload_change callback
    uploaded_file = st.file_uploader(
        "Or upload your own image...", 
        type=["jpg", "png", "jpeg", "webp", "avif"], 
        key=f"file_up_{st.session_state.uploader_key}",
        on_change=on_upload_change
    )

# --- FINAL IMAGE SELECTION ---
active_img = None
if uploaded_file:
    active_img = Image.open(uploaded_file).convert("RGB")
elif selected_sample != "None":
    active_img = Image.open(os.path.join(SAMPLES_DIR, selected_sample)).convert("RGB")

# --- UI DISPLAY & PREDICTION ---
if active_img:
    if st.button("🗑️ Remove Photo", use_container_width=True):
        st.session_state.uploader_key += 1
        st.session_state.selector_key += 1
        st.rerun()

    st.markdown("---")
    col_img, col_pred = st.columns(2)
    
    with col_img:
        st.image(active_img, caption="Active Image", use_container_width=True)
    
    with col_pred:
        img_tensor = test_transform(active_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            top5_p, top5_i = torch.topk(probs, 5)

        st.success(f"### Prediction: **{CLASS_NAMES[top5_i[0]]}**")
        st.metric("Confidence", f"{top5_p[0].item()*100:.1f}%")
        
        st.write("#### Top 5 Candidates")
        for i in range(5):
            st.write(f"**{CLASS_NAMES[top5_i[i]]}** ({top5_p[i].item()*100:.1f}%)")
            st.progress(top5_p[i].item())
else:
    st.info("👈 Select a sample from the sidebar or upload an image to begin.")