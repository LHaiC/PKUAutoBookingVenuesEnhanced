import io
from collections import deque

from PIL import Image


def decode_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def image_size(image: Image.Image) -> tuple[int, int]:
    return image.size


def validate_bbox(bbox: list[int], size: tuple[int, int]) -> bool:
    if not isinstance(bbox, (list, tuple)):
        return False
    if len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = [int(value) for value in bbox]
    except (TypeError, ValueError):
        return False
    width, height = size
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def bbox_center(bbox: list[int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _dark_mask(image: Image.Image, threshold: int = 90) -> list[list[bool]]:
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()
    return [[pixels[x, y] <= threshold for x in range(width)] for y in range(height)]


def detect_dark_regions(
    image: Image.Image,
    threshold: int = 90,
    min_area: int = 20,
) -> list[list[int]]:
    mask = _dark_mask(image, threshold=threshold)
    width, height = image.size
    visited = [[False for _ in range(width)] for _ in range(height)]
    regions = []

    for y in range(height):
        for x in range(width):
            if visited[y][x] or not mask[y][x]:
                continue

            queue = deque([(x, y)])
            visited[y][x] = True
            xs = []
            ys = []

            while queue:
                cx, cy = queue.popleft()
                xs.append(cx)
                ys.append(cy)
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if visited[ny][nx] or not mask[ny][nx]:
                        continue
                    visited[ny][nx] = True
                    queue.append((nx, ny))

            if len(xs) >= min_area:
                regions.append([min(xs), min(ys), max(xs) + 1, max(ys) + 1])

    regions.sort(key=lambda box: (box[0], box[1]))
    return regions


def refine_bbox_to_dark_pixels(
    image: Image.Image,
    bbox: list[int],
    threshold: int = 90,
) -> list[int]:
    if not validate_bbox(bbox, image.size):
        return bbox

    x1, y1, x2, y2 = bbox
    crop = image.crop((x1, y1, x2, y2))
    regions = detect_dark_regions(crop, threshold=threshold)
    if not regions:
        return bbox

    left = min(region[0] for region in regions) + x1
    top = min(region[1] for region in regions) + y1
    right = max(region[2] for region in regions) + x1
    bottom = max(region[3] for region in regions) + y1
    return [left, top, right, bottom]
