import requests
import imagehash
from PIL import Image
from io import BytesIO

def load_image_from_url(url):
    """
    Downloads an image from a URL and converts it to a PIL Image object.
    Returns None if the download fails.
    """
    try:
        # Use a timeout to prevent hanging on bad URLs
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"[ERROR] Failed to load image: {url}")
        return None

def get_hamming_distance(img1, img2):
    """
    Computes the Hamming Distance between two images using Perceptual Hashing (pHash).
    """
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)
    return hash1 - hash2

if __name__ == "__main__":
    print("--- Starting Diffusion Memorization Detection ---\n")

    # 1. Define the Target Image (The 'Original' from training data)
    target_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    
    # 2. Define Test Images (Simulating generated output)
    test_urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # Exact match
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg", # Different image
        "http://images.cocodataset.org/val2017/000000039769.jpg"   # Exact match
    ]

    # Load target
    target_img = load_image_from_url(target_url)

    if target_img:
        print(f"Target image loaded. Processing {len(test_urls)} test samples...\n")

        for index, url in enumerate(test_urls):
            test_img = load_image_from_url(url)
            
            if test_img:
                distance = get_hamming_distance(target_img, test_img)
                
                # output format: [ID] Distance | Status
                status = "SAFE"
                if distance == 0:
                    status = "DUPLICATE DETECTED"
                elif distance < 5:
                    status = "POTENTIAL MEMORIZATION"
                
                print(f"[Sample {index+1}] Distance: {distance} | {status}")
            
    else:
        print("[FATAL] Could not load target image.")