import hashlib
import io
from PIL import Image
import imagehash

class ContentHasher:
    @staticmethod
    def compute_image_phash(image_bytes: bytes) -> str:
        """Computes perceptual hash (pHash) for image duplicate detection."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return str(imagehash.phash(image))

    @staticmethod
    def compute_text_hash(text_content: str) -> str:
        """Computes SHA-256 hash for exact text payload matching."""
        return hashlib.sha256(text_content.encode("utf-8")).hexdigest()