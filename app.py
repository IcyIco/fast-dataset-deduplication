import streamlit as st
import numpy as np
from utils import load_image, calculate_phash, get_geometric_variations, calculate_ssim_score, get_hamming_distance

st.set_page_config(page_title="Memorization Analysis", layout="wide")

st.title("Diffusion Memorization Analysis")
st.markdown("""
This tool evaluates similarity between generated samples and training data using **Perceptual Hashing (pHash)** and **Structural Similarity (SSIM)**. 
It accounts for geometric transformations such as rotation and mirroring.
""")

# Sidebar
st.sidebar.header("Configuration")
h_thresh = st.sidebar.slider("Hamming Threshold (pHash)", 0, 20, 8)
s_thresh = st.sidebar.slider("SSIM Threshold", 0.0, 1.0, 0.45)

col1, col2 = st.columns(2)

img1, img2 = None, None

with col1:
    st.subheader("Reference Image")
    f1 = st.file_uploader("Upload Reference", key="f1")
    if f1:
        img1 = load_image(f1)
        st.image(img1, use_container_width=True)

with col2:
    st.subheader("Generated Image")
    f2 = st.file_uploader("Upload Generated", key="f2")
    if f2:
        img2 = load_image(f2)
        st.image(img2, use_container_width=True)

if img1 and img2:
    st.markdown("---")
    st.subheader("Results")
    
    # 1. Geometric pHash Calculation
    h1 = calculate_phash(img1)
    variations_h2 = get_geometric_variations(img2)
    
    # Find the BEST match among all rotations
    distances = [get_hamming_distance(h1, h_var) for h_var in variations_h2]
    best_dist = min(distances)
    
    # 2. SSIM Calculation
    ssim_val = calculate_ssim_score(img1, img2)
    
    # Display
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.metric("Min Hamming Distance", f"{best_dist}")
        
    with m2:
        st.metric("SSIM Score", f"{ssim_val:.3f}")
        
    with m3:
        is_mem = (best_dist <= h_thresh) and (ssim_val >= s_thresh)
        if is_mem:
            st.error("**Memorization Detected**")
        elif best_dist <= h_thresh:
            st.warning("**Low Confidence** (Hash match, SSIM mismatch)")
        else:
            st.success("**Distinct Images**")