import io
import colorsys
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
    if any(not isinstance(value, int) or isinstance(value, bool) for value in bbox):
        return False
    x1, y1, x2, y2 = bbox
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


def _colored_text_pixel(r: int, g: int, b: int) -> bool:
    hue, saturation, value = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    hue *= 360
    if saturation < 0.28 or value > 0.97:
        return False
    # Some captchas use a saturated blue sky background. It has high value and
    # connects the whole image into one component unless we exclude it.
    if 180 <= hue <= 220 and 0.45 <= saturation < 0.75 and value > 0.85:
        return False
    # The captcha background commonly contains green plants. Excluding green
    # prevents those large background components from being treated as text.
    return not (45 <= hue <= 170)


def _merge_nearby_boxes(boxes: list[list[int]], margin: int = 4) -> list[list[int]]:
    merged = []
    for box in boxes:
        x1, y1, x2, y2 = box
        matched = False
        for existing in merged:
            ex1, ey1, ex2, ey2 = existing
            if x1 <= ex2 + margin and x2 + margin >= ex1 and y1 <= ey2 + margin and y2 + margin >= ey1:
                existing[:] = [min(ex1, x1), min(ey1, y1), max(ex2, x2), max(ey2, y2)]
                matched = True
                break
        if not matched:
            merged.append(list(box))
    return merged


def detect_colored_text_bboxes(
    image: Image.Image,
    min_area: int = 60,
) -> list[list[int]]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    mask = [
        [_colored_text_pixel(*pixels[x, y]) for x in range(width)]
        for y in range(height)
    ]
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
                for nx in (cx - 1, cx, cx + 1):
                    for ny in (cy - 1, cy, cy + 1):
                        if nx == cx and ny == cy:
                            continue
                        if nx < 0 or ny < 0 or nx >= width or ny >= height:
                            continue
                        if visited[ny][nx] or not mask[ny][nx]:
                            continue
                        visited[ny][nx] = True
                        queue.append((nx, ny))

            if len(xs) < min_area:
                continue
            box = [min(xs), min(ys), max(xs) + 1, max(ys) + 1]
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            density = len(xs) / (box_width * box_height)
            if density < 0.15:
                continue
            if not (8 <= box_width <= min(100, width * 0.35)):
                continue
            if not (8 <= box_height <= min(100, height * 0.65)):
                continue
            regions.append(box)

    regions.sort(key=lambda box: (box[0], box[1]))
    merged = _merge_nearby_boxes(regions)
    return [box for box in merged if validate_bbox(box, image.size)]


def filter_captcha_text_bboxes(
    boxes: list[list[int]],
    image_size: tuple[int, int] | None = None,
    min_width: int = 24,
    min_height: int = 24,
) -> list[list[int]]:
    max_bottom = None
    if image_size is not None:
        max_bottom = int(image_size[1] * 0.85)
    return [
        box
        for box in boxes
        if (
            len(box) == 4
            and box[2] - box[0] >= min_width
            and box[3] - box[1] >= min_height
            and (max_bottom is None or box[3] <= max_bottom)
        )
    ]


def isolate_colored_text(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    clean = Image.new("RGB", rgb.size, "white")
    source = rgb.load()
    target = clean.load()
    width, height = rgb.size

    for y in range(height):
        for x in range(width):
            pixel = source[x, y]
            if _colored_text_pixel(*pixel):
                target[x, y] = pixel

    return clean


def build_colored_text_strip(
    image: Image.Image,
    boxes: list[list[int]] | None = None,
    padding: int = 5,
    gap: int = 20,
    margin: int = 20,
) -> Image.Image | None:
    boxes = detect_colored_text_bboxes(image) if boxes is None else boxes
    if not boxes:
        return None

    clean = isolate_colored_text(image)
    width, height = clean.size
    crops = []
    max_height = 0

    for x1, y1, x2, y2 in boxes:
        left = max(0, x1 - padding)
        top = max(0, y1 - padding)
        right = min(width, x2 + padding)
        bottom = min(height, y2 + padding)
        crop = clean.crop((left, top, right, bottom))
        crops.append(crop)
        max_height = max(max_height, crop.height)

    strip_width = sum(crop.width for crop in crops) + gap * (len(crops) - 1) + margin * 2
    strip_height = max_height + margin * 2
    strip = Image.new("RGB", (strip_width, strip_height), "white")

    x = margin
    for crop in crops:
        y = margin + (max_height - crop.height) // 2
        strip.paste(crop, (x, y))
        x += crop.width + gap

    return strip


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
