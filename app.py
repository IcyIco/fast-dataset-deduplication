import streamlit as st
from PIL import Image
import imagehash
import requests
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(
    page_title="Diffusion Memorization Analysis Tool", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Header Section ---
st.title("Diffusion Model Memorization Analysis")
st.markdown("""
**System Overview:**
This utility evaluates the similarity between generated samples and training data using **Perceptual Hashing (pHash)** based on the Discrete Cosine Transform (DCT). 
It computes the Hamming Distance ($D_H$) to quantify structural similarity, providing a robust metric for detecting data memorization in diffusion models.
""")
st.markdown("---")

# --- Sidebar: Parameters ---
st.sidebar.markdown("## Parameter Configuration")
st.sidebar.info("Adjust the sensitivity threshold for duplicate detection.")

threshold = st.sidebar.slider(
    "Hamming Distance Threshold (τ)", 
    min_value=0, 
    max_value=20, 
    value=5, 
    help="Maximum Hamming distance to consider two images as perceptually identical."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Metric Definition:**")
st.sidebar.latex(r"D_H(h_1, h_2) = \sum_{i=1}^{64} |h_1[i] - h_2[i]|")

# --- Core Logic ---
def calculate_phash(image):
    """Calculates the perceptual hash of an image."""
    return imagehash.phash(image)

# --- Main Interface ---
col1, col2 = st.columns(2)

uploaded_file_1 = None
uploaded_file_2 = None
hash1 = None
hash2 = None

# Column 1: Reference
with col1:
    st.subheader("Reference Sample ($x_{ref}$)")
    uploaded_file_1 = st.file_uploader("Upload Training/Reference Image", type=['png', 'jpg', 'jpeg'], key="img1")
    
    if uploaded_file_1:
        image1 = Image.open(uploaded_file_1)
        st.image(image1, use_container_width=True, caption=f"Filename: {uploaded_file_1.name}")
        hash1 = calculate_phash(image1)
        st.text_input("pHash Value ($h_1$)", value=str(hash1), disabled=True)

# Column 2: Generated
with col2:
    st.subheader("Generated Sample ($x_{gen}$)")
    uploaded_file_2 = st.file_uploader("Upload Generated Image", type=['png', 'jpg', 'jpeg'], key="img2")
    
    if uploaded_file_2:
        image2 = Image.open(uploaded_file_2)
        st.image(image2, use_container_width=True, caption=f"Filename: {uploaded_file_2.name}")
        hash2 = calculate_phash(image2)
        st.text_input("pHash Value ($h_2$)", value=str(hash2), disabled=True)

# --- Analysis Section ---
if uploaded_file_1 and uploaded_file_2:
    st.markdown("---")
    st.subheader("Comparative Analysis Result")

    # Calculation
    distance = hash1 - hash2
    
    # Layout for metrics
    m_col1, m_col2, m_col3 = st.columns([1, 2, 2])
    
    with m_col1:
        st.metric(label="Hamming Distance", value=f"{distance}")

    with m_col2:
        st.metric(label="Threshold (τ)", value=threshold)

    with m_col3:
        # Logical Evaluation
        is_duplicate = distance <= threshold
        result_label = "Memorization Detected" if is_duplicate else "Distinct Samples"
        
        # Using a clean status indicator instead of colored text blocks
        if is_duplicate:
            st.error(f"**Result:** {result_label}")
        else:
            st.success(f"**Result:** {result_label}")

    # Detailed Explanation
    with st.expander("See Detailed Interpretation", expanded=True):
        if distance == 0:
            st.markdown("**Assessment:** The images are perceptually **identical**. This indicates a high probability of exact training data memorization.")
        elif distance <= threshold:
            st.markdown(f"**Assessment:** The images are **near-duplicates** ($D_H \le \\tau$). Minor variations (compression, resizing, or noise) are present, but the semantic structure remains memorized.")
        else:
            st.markdown(f"**Assessment:** The images are perceptually **distinct** ($D_H > \\tau$). No significant memorization detected under current threshold.")

else:
    st.info("Awaiting input: Please upload both Reference and Generated samples to initiate analysis.")