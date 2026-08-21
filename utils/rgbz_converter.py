"""Convert .rgbz matrix files back to images."""
import struct
import zlib
from PIL import Image
from pathlib import Path
from typing import Optional


def rgbz_to_image(file_path: str, output_path: Optional[str] = None) -> Image.Image:
    """
    Read a .rgbz file and return a PIL Image.
    If output_path is provided, save the image there.
    """
    with open(file_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'RGBZ':
            raise ValueError("Invalid .rgbz file: magic header mismatch")
        width = struct.unpack('<H', f.read(2))[0]
        height = struct.unpack('<H', f.read(2))[0]
        compressed_len = struct.unpack('<I', f.read(4))[0]
        compressed_data = f.read(compressed_len)

    rle_data = zlib.decompress(compressed_data)
    # Decode RLE
    pixels = []
    idx = 0
    while idx < len(rle_data):
        run_length = rle_data[idx]
        idx += 1
        if idx + 2 >= len(rle_data):
            break
        r, g, b = rle_data[idx], rle_data[idx+1], rle_data[idx+2]
        idx += 3
        pixels.extend([(r, g, b)] * run_length)

    # Reshape to image
    img = Image.new('RGB', (width, height))
    img.putdata(pixels[:width*height])  # trim if needed

    if output_path:
        img.save(output_path)

    return img