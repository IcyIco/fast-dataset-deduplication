# Diffusion Memorization Detector

## Overview

This repository contains an advanced auditing tool designed to detect data memorization in generative models (e.g., Diffusion Models). It calculates the similarity between generated images and a reference dataset to determine if a model has "memorized" training examples rather than generating novel content.

Unlike traditional hash-based tools, this system implements **Geometric Invariance** and **Deep Perceptual Verification** to detect memorization even when images are rotated, flipped, or structurally modified. It also utilizes **FAISS** for scalable, high-speed retrieval.

## Key Features

* **Geometric Invariance:** Automatically detects duplicates even if they are rotated ($90^\circ, 180^\circ, 270^\circ$) or mirrored/flipped.
* **Deep Perceptual Verification:** Uses **SSIM (Structural Similarity Index)** to cross-validate matches, filtering out false positives where pHash might collide but visual content differs.
* **Scalable Architecture:** Implements **FAISS** (Facebook AI Similarity Search) to convert $O(N)$ linear scanning into efficient vector search, suitable for large-scale datasets.
* **Dual Interface:** Provides both a high-performance CLI scanner and a Streamlit-based web dashboard for visual analysis.

## Installation

### Prerequisites

* Python 3.8 or higher
* pip package manager

### Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/fast-dataset-deduplication.git
    cd fast-dataset-deduplication
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Dependencies include `faiss-cpu`, `scikit-image`, `imagehash`, `numpy`, `pillow`, etc.)*

## Usage

### 1. Batch Scanning (CLI)

The `main.py` script is the core engine. It handles automatic index building and batch scanning against a target image.

1.  Open `main.py` and configure your paths:
    ```python
    # Path to the specific generated image to audit
    TARGET_IMAGE_PATH = "C:/path/to/generated_image.png"

    # Directory containing the original training dataset
    SEARCH_DIRECTORY = "C:/path/to/training_data"
    ```

2.  Run the scanner:
    ```bash
    python main.py
    ```
    * **First Run:** The system will scan the directory and build a FAISS index (`dataset.index`). This may take some time depending on dataset size.
    * **Subsequent Runs:** The system loads the pre-built index instantly for millisecond-level search.

### 2. Visual Analysis (GUI)

For a detailed side-by-side comparison of specific images with real-time metric visualization:

```bash
streamlit run app.py
```

## Interpretation of Results

The system uses a **Coarse-to-Fine** strategy. A "Memorization" verdict requires passing both the Hash filter and the SSIM verification.

| Metric | Threshold | Interpretation |
| :--- | :--- | :--- |
| **Hamming Distance** ($D_H$) | $\le 8$ | **Coarse Match:** The images share the same low-frequency "skeleton" or layout. Lower is better (0 is identical). |
| **SSIM Score** | $\ge 0.45$ | **Fine Confirmation:** The images share significant structural and luminance similarity. Higher is better (1.0 is identical). |

### Verdict Logic

* **MEMORIZED:** Distance $\le$ Threshold **AND** SSIM $\ge$ Threshold. (High confidence of data leakage).
* **PASS:** Distance is high **OR** SSIM is low (Likely a distinct image or a false positive hash collision).

## Methodology

1.  **Preprocessing & Vectorization:** All dataset images are converted to 64-bit Perceptual Hashes (pHash) and stored in a **FAISS Binary Index**.
2.  **Geometric Querying:** When checking a target image, the system generates 5 variants (Original, Flip, Rotations 90/180/270) and queries the index simultaneously.
3.  **Structural Verification:** Candidates returned by the vector search are re-evaluated using **SSIM** to ensure they are semantically and structurally similar to the target.

## License

This project is licensed under the MIT License.