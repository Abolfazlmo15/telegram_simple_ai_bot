"""Image processing utilities for vision engine."""
import logging
import io
import base64
from typing import Union, Optional
from PIL import Image
import asyncio

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Handles image preprocessing for vision models.
    Resizes, optimizes, and encodes images.
    """

    def __init__(self, max_size: int = 512, quality: int = 60):
        self.max_size = max_size
        self.quality = quality
        logger.info(f"Image processor initialized (max_size: {max_size}px, quality: {quality})")

    async def process_image(self, image_data: Union[bytes, Image.Image]) -> Image.Image:
        """Process image for optimal vision model input."""
        try:
            if isinstance(image_data, bytes):
                img = Image.open(io.BytesIO(image_data))
            else:
                img = image_data

            if img.mode in ('RGBA', 'P', 'LA'):
                img = img.convert('RGB')

            img = self._resize_if_needed(img)
            img = self._optimize_image(img)

            logger.info(f"Image processed: {img.size[0]}x{img.size[1]}")
            return img
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            raise

    def _resize_if_needed(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        if width > self.max_size or height > self.max_size:
            if width > height:
                new_width = self.max_size
                new_height = int(height * (self.max_size / width))
            else:
                new_height = self.max_size
                new_width = int(width * (self.max_size / height))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Image resized from {width}x{height} to {new_width}x{new_height}")
        return img

    def _optimize_image(self, img: Image.Image) -> Image.Image:
        return img

    def encode_to_base64(self, img: Image.Image, format: str = "JPEG") -> str:
        buffer = io.BytesIO()
        img.save(buffer, format=format, quality=self.quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return encoded

    def get_image_info(self, img: Image.Image) -> dict:
        return {
            "size": img.size,
            "mode": img.mode,
            "format": img.format,
            "width": img.width,
            "height": img.height
        }