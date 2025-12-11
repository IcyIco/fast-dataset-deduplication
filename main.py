import os
import glob
import time
from utils import load_image, calculate_phash, get_hamming_distance

# --- Configuration Settings ---
# Path to the reference image (The "Training Data" to check against)
TARGET_IMAGE_PATH = "C:/Users/ICYICO/Desktop/fast-dataset-deduplication/image_6453b0.png"

# Directory containing the dataset to scan
SEARCH_DIRECTORY = "C:/Users/ICYICO/Desktop/fast-dataset-deduplication"

# Hamming Distance Threshold (Distance <= 5 implies potential memorization)
SIMILARITY_THRESHOLD = 5

def run_batch_scan():
    """
    Executes the batch scanning process.
    """
    start_time = time.time()
    
    print("==========================================")
    print("   DIFFUSION MEMORIZATION DETECTOR CLI    ")
    print("==========================================")
    print(f"[INFO] Target Image:   {os.path.basename(TARGET_IMAGE_PATH)}")
    print(f"[INFO] Search Dir:     {SEARCH_DIRECTORY}")
    print(f"[INFO] Threshold:      {SIMILARITY_THRESHOLD}")
    print("-" * 42)

    # 1. Load and Hash the Reference Image
    print("[INIT] Loading reference target...")
    target_img = load_image(TARGET_IMAGE_PATH)
    
    if target_img is None:
        print(f"[FATAL] Could not load target image at: {TARGET_IMAGE_PATH}")
        print("[FATAL] Aborting execution.")
        return

    target_hash = calculate_phash(target_img)
    print(f"[SUCCESS] Target Hash Computed: {target_hash}")
    print("-" * 42)

    # 2. Gather all image files in the directory
    # Supports .png, .jpg, and .jpeg extensions
    image_extensions = ["*.png", "*.jpg", "*.jpeg"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(SEARCH_DIRECTORY, ext)))

    total_files = len(image_files)
    print(f"[INFO] Found {total_files} images in directory. Starting scan...")
    print("-" * 42)

    # 3. Iterate and Compare
    duplicates_count = 0
    processed_count = 0

    for file_path in image_files:
        # Skip the target image itself if it exists in the same folder
        if os.path.abspath(file_path) == os.path.abspath(TARGET_IMAGE_PATH):
            continue

        filename = os.path.basename(file_path)
        processed_count += 1

        # Load candidate image
        current_img = load_image(file_path)
        if current_img is None:
            print(f"[WARN] Skipping unreadable file: {filename}")
            continue

        # Compute Hash and Distance
        current_hash = calculate_phash(current_img)
        distance = get_hamming_distance(target_hash, current_hash)

        # Evaluation
        if distance <= SIMILARITY_THRESHOLD:
            print(f"[ALERT] DUPLICATE DETECTED | File: {filename:<20} | Distance: {distance}")
            duplicates_count += 1
        else:
            # Optional: Comment this out if scanning large datasets to reduce noise
            # print(f"[PASS]  Distinct Image     | File: {filename:<20} | Distance: {distance}")
            pass

    # 4. Final Report
    elapsed_time = time.time() - start_time
    print("-" * 42)
    print("SCAN COMPLETION REPORT")
    print("-" * 42)
    print(f"Total Processed:    {processed_count}")
    print(f"Duplicates Found:   {duplicates_count}")
    print(f"Time Elapsed:       {elapsed_time:.2f} seconds")
    print("==========================================")

if __name__ == "__main__":
    run_batch_scan()