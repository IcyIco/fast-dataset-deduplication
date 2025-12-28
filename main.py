import os
import glob
import pickle
import faiss
import numpy as np
from utils import (
    load_image, calculate_phash, get_geometric_variations, 
    calculate_ssim_score, hashes_to_vectors
)

# --- Configuration ---
TARGET_IMAGE_PATH = "C:/Users/ICYICO/Desktop/fast-dataset-deduplication/image_6453b0.png"
SEARCH_DIRECTORY = "C:/Users/ICYICO/Desktop/fast-dataset-deduplication"

INDEX_FILE = "dataset.index"
MAPPING_FILE = "filenames.pkl"

# Thresholds
HAMMING_THRESHOLD = 8
SSIM_THRESHOLD = 0.45

def build_index_if_needed():
    """
    Builds the FAISS index if it does not already exist.
    """
    if os.path.exists(INDEX_FILE) and os.path.exists(MAPPING_FILE):
        return

    print("[INFO] Building Search Index...")
    image_files = glob.glob(os.path.join(SEARCH_DIRECTORY, "*.png")) + \
                  glob.glob(os.path.join(SEARCH_DIRECTORY, "*.jpg"))
    
    hash_list = []
    valid_files = []
    
    for i, fpath in enumerate(image_files):
        img = load_image(fpath)
        if img:
            hash_list.append(calculate_phash(img))
            valid_files.append(fpath)
        if i % 100 == 0: 
            print(f"      Processed {i} images...", end="\r")
            
    vectors = hashes_to_vectors(hash_list)
    
    # Initialize Binary Flat Index (64-bit)
    index = faiss.IndexBinaryFlat(64)
    index.add(vectors)
    
    faiss.write_index_binary(index, INDEX_FILE)
    with open(MAPPING_FILE, 'wb') as f:
        pickle.dump(valid_files, f)
    print(f"\n[INFO] Index built with {index.ntotal} items.")

def run_batch_scan():
    print("==========================================")
    print("   DIFFUSION MEMORIZATION DETECTOR        ")
    print("==========================================")
    
    # 1. Check Index
    build_index_if_needed()
    
    # 2. Load Index
    try:
        index = faiss.read_index_binary(INDEX_FILE)
        with open(MAPPING_FILE, 'rb') as f:
            filenames = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load index: {e}")
        return

    # 3. Process Target
    target_img = load_image(TARGET_IMAGE_PATH)
    if not target_img: 
        print(f"[ERROR] Could not load target: {TARGET_IMAGE_PATH}")
        return
    
    # Generate geometric variants
    target_hashes = get_geometric_variations(target_img)
    query_vectors = hashes_to_vectors(target_hashes)
    
    print(f"[INFO] Scanning against target: {os.path.basename(TARGET_IMAGE_PATH)}")
    
    # 4. Search Index
    # Query all variations; find top 10 matches for each
    D, I = index.search(query_vectors, k=10)
    
    found_candidates = set()
    
    print("-" * 65)
    print(f"{'FILE':<35} | {'DIST':<5} | {'SSIM':<6} | {'RESULT'}")
    print("-" * 65)

    # 5. Process Results
    for variant_idx, (distances, indices) in enumerate(zip(D, I)):
        for dist, idx in zip(distances, indices):
            if idx == -1: continue
            
            candidate_file = filenames[idx]
            
            if os.path.abspath(candidate_file) == os.path.abspath(TARGET_IMAGE_PATH):
                continue
            
            if candidate_file in found_candidates:
                continue

            if dist <= HAMMING_THRESHOLD:
                # SSIM Check
                candidate_img = load_image(candidate_file)
                ssim_val = calculate_ssim_score(target_img, candidate_img)
                
                status = "PASS"
                if ssim_val >= SSIM_THRESHOLD:
                    status = "MEMORIZED"
                    
                print(f"{os.path.basename(candidate_file):<35} | {dist:<5} | {ssim_val:.2f}   | {status}")
                
                if status == "MEMORIZED":
                    found_candidates.add(candidate_file)

    print("-" * 65)
    print(f"[REPORT] Total Verified Duplicates: {len(found_candidates)}")
    print("==========================================")

if __name__ == "__main__":
    run_batch_scan()