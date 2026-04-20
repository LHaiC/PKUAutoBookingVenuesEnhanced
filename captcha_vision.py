import io

from PIL import Image


def whiten_watermark(image: Image.Image, x_ratio: float = 0.78, y_ratio: float = 0.85) -> Image.Image:
    """Whiten bottom-right watermark region (场馆预约) to prevent it being detected as text."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    wx = int(width * x_ratio)
    wy = int(height * y_ratio)
    if wx < width and wy < height:
        clean = rgb.copy()
        for y in range(wy, height):
            for x in range(wx, width):
                clean.putpixel((x, y), (255, 255, 255))
        return clean
    return rgb


def decode_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
