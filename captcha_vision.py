import io

from PIL import Image


def whiten_watermark(image: Image.Image, x_ratio: float = 0.78, y_ratio: float = 0.85) -> Image.Image:
    """Whiten bottom-right watermark region (场馆预约) to prevent it being detected as text."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    wx = int(width * x_ratio)
    wy = int(height * y_ratio)
    if wx < width and wy < height:
        white = Image.new("RGB", (width - wx, height - wy), (255, 255, 255))
        rgb.paste(white, (wx, wy))
    return rgb


def decode_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
