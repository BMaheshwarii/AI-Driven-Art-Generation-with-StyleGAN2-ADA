import os
import sys
import streamlit as st
import torch
from PIL import Image

# Ensure the StyleGAN2-ADA repository is available
if not os.path.isdir('stylegan2-ada-pytorch'):
    st.warning("Clone the 'stylegan2-ada-pytorch' repository into this directory before running.")
    st.stop()

# Add repo to Python path before importing dnnlib and legacy
sys.path.insert(0, 'stylegan2-ada-pytorch')

import dnnlib
import legacy

@st.cache_resource
def load_model(pkl_url: str = 'ffhq.pkl'):
    """
    Load the pretrained StyleGAN2-ADA generator.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with dnnlib.util.open_url(pkl_url) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].to(device)
    return G, device

@st.cache_data
def generate_art_image(_G, _device, seed: int, truncation_psi: float):
    """
    Generate an AI art image using the loaded generator.
    """
    torch.manual_seed(seed)
    z = torch.randn([1, _G.z_dim], device=_device)
    img_tensor = _G(z, None, truncation_psi=truncation_psi)[0]
    img = ((img_tensor.permute(1, 2, 0).cpu().numpy() * 127.5) + 127.5).astype('uint8')
    return Image.fromarray(img)


def main():
    st.title("AI-Driven Art Generation with StyleGAN2-ADA")

    # Model checkpoint input
    pkl_path = st.sidebar.text_input("Checkpoint Path/URL", 'ffhq.pkl')
    G, device = load_model(pkl_path)

    # Generation parameters
    seed = st.sidebar.number_input("Seed", min_value=0, max_value=10000, value=42)
    truncation_psi = st.sidebar.slider("Truncation Psi", 0.0, 1.0, 0.7)

    if st.sidebar.button("Generate Artwork"):
        img = generate_art_image(G, device, seed, truncation_psi)
        st.image(img, caption=f"Seed: {seed}, Psi: {truncation_psi}", use_column_width=True)

    st.markdown("---")
    st.write(
        "**Instructions:**\n"
        "1. Clone `stylegan2-ada-pytorch` into this directory.\n"
        "2. Place or provide a StyleGAN2-ADA checkpoint (e.g., `ffhq.pkl`).\n"
        "3. Run with `streamlit run app.py`."
    )

if __name__ == '__main__':
    main()
