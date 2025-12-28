import imagehash
import requests
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO
from typing import Union, Optional, List
from skimage.metrics import structural_similarity as ssim

def load_image(source: Union[str, BytesIO]) -> Optional[Image.Image]:
    """Loads an image and converts to RGB."""
    try:
        if hasattr(source, "read"):
            img = Image.open(source)
        elif isinstance(source, str) and source.startswith(("http://", "https://")):
            response = requests.get(source, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(source)
        return img.convert('RGB')
    except Exception as e:
        print(f"[Error] Failed to load image source: {e}")
        return None

def calculate_phash(image: Image.Image, hash_size: int = 8) -> imagehash.ImageHash:
    """Computes standard pHash."""
    return imagehash.phash(image, hash_size=hash_size)

def get_geometric_variations(image: Image.Image) -> List[imagehash.ImageHash]:
    """
    Generates hashes for 0, 90, 180, 270 degree rotations and horizontal flip
    to handle geometric transformations.
    """
    hashes = []
    # Original & Flip
    hashes.append(calculate_phash(image))
    hashes.append(calculate_phash(ImageOps.mirror(image)))
    
    # Rotations
    for angle in [90, 180, 270]:
        hashes.append(calculate_phash(image.rotate(angle, expand=True)))
        
    return hashes

def calculate_ssim_score(img1: Image.Image, img2: Image.Image) -> float:
    """
    Calculates Structural Similarity Index (SSIM).
    Ranges from -1 to 1 (1.0 = Identical).
    """
    # Resize img2 to match img1
    img2_resized = img2.resize(img1.size, Image.Resampling.LANCZOS)
    
    # Convert to Grayscale for structural comparison
    im1_gray = np.array(img1.convert('L'))
    im2_gray = np.array(img2_resized.convert('L'))
    
    # Compute SSIM
    score, _ = ssim(im1_gray, im2_gray, full=True)
    return score

def get_hamming_distance(hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> int:
    return hash1 - hash2

def hashes_to_vectors(hash_list: List[imagehash.ImageHash]) -> np.ndarray:
    """
    Converts a list of ImageHash objects to a binary numpy matrix for vector search.
    """
    if not hash_list:
        return np.empty((0, 8), dtype='uint8')
    bool_arrays = np.vstack([h.hash.flatten() for h in hash_list])
    return np.packbits(bool_arrays, axis=1)