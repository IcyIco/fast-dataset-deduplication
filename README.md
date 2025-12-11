# Image Memorization Detector

## Overview

This repository contains a tool designed to detect image memorization in generative models. It calculates the similarity between generated images and a reference dataset (e.g., training data) to determine if a model has "memorized" or reproduced training examples rather than generating novel content.

The core metric relies on computing the Hamming Distance between perceptual hashes of the images. This provides a robust measure of similarity that is resistant to minor transformations such as compression, resizing, or slight color shifts.

## Features

* **Perceptual Hashing:** Efficient comparison of high-dimensional image data.
* **Hamming Distance Calculation:** Quantifiable metric for image similarity.
* **Batch Processing:** Capable of scanning large datasets against target images.
* **Threshold-based Classification:** Automatically categorizes results into Identical, Near-Duplicate, or Distinct.

## Installation

### Prerequisites

* Python 3.8 or higher
* pip package manager

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the detection script by specifying the target image (the generated image) and the source directory (the reference/training dataset).

```bash
python detect.py --target ./generated_images/sample_01.png --source ./training_data/
```

### Arguments

* `--target`: Path to the image file you want to analyze.
* `--source`: Path to the directory containing reference/training images.
* `--recursive` (Optional): If set, searches through subdirectories in the source path.

## Interpretation of Results

The tool calculates a Hamming Distance score. Use the following table to interpret the findings:

| Distance | Verdict | Description |
| :--- | :--- | :--- |
| **0** | **Identical** | Statistically identical images. |
| **≤ 5** | **Near-Duplicate** | High probability of memorization. The image is a close variant of a training sample. |
| **> 10** | **Distinct** | The images are visually and statistically distinct. No evidence of memorization. |

## Example Output

```text
Scanning source directory...
[+] Found 5000 images in ./training_data/

Analyzing target: sample_01.png
----------------------------------------
Closest Match Found:
File: ./training_data/class_A/img_402.jpg
Distance: 3
Verdict: Near-Duplicate (High Probability of Memorization)
```

## Methodology

1.  **Preprocessing:** Images are resized and converted to grayscale to normalize input.
2.  **Hashing:** A perceptual hash is generated for the target image and every image in the source dataset.
3.  **Comparison:** The Hamming Distance is calculated between the target hash and all source hashes.
4.  **Minimization:** The algorithm identifies the source image with the minimum distance to the target.

## License

This project is licensed under the MIT License.