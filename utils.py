import imagehash
import requests
from PIL import Image
from io import BytesIO
from typing import Union, Optional

def load_image(source: Union[str, BytesIO]) -> Optional[Image.Image]:
    """
    Loads an image from a local path, a URL, or a file-like object.

    Args:
        source (str | BytesIO): File path, URL, or stream object.

    Returns:
        PIL.Image.Image: The loaded image object, or None if loading fails.
    """
    try:
        # Case 1: Streamlit Upload / File Object
        if hasattr(source, "read"):
            return Image.open(source)
        
        # Case 2: URL
        elif isinstance(source, str) and source.startswith(("http://", "https://")):
            response = requests.get(source, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        
        # Case 3: Local File Path
        else:
            return Image.open(source)
            
    except Exception as e:
        print(f"[Error] Failed to load image source: {e}")
        return None

def calculate_phash(image: Image.Image, hash_size: int = 8) -> imagehash.ImageHash:
    """
    Computes the Perceptual Hash (pHash) of an image using DCT.
    
    Args:
        image (PIL.Image): The input image.
        hash_size (int): The size of the hash (default 8x8).
        
    Returns:
        imagehash.ImageHash: The computed hash object.
    """
    return imagehash.phash(image, hash_size=hash_size)

def get_hamming_distance(hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> int:
    """
    Calculates the Hamming Distance between two perceptual hashes.
    
    Args:
        hash1, hash2: The hash objects to compare.
        
    Returns:
        int: The Hamming distance (0 means identical).
    """
    return hash1 - hash2