import colorsys
import io
from collections import Counter, deque
from dataclasses import dataclass

from PIL import Image, ImageOps


@dataclass
class ProposalSet:
    boxes: list[list[int]]
    source: str
    preprocess_variant: str


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
    threshold: int = 130,
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
    if 180 <= hue <= 220 and 0.30 <= saturation < 0.65 and value > 0.65:
        return False
    return not (45 <= hue <= 170)


def _dominant_background_colors(image: Image.Image, limit: int = 5) -> list[tuple[int, int, int]]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    step = 2 if width * height > 20000 else 1
    buckets = Counter()

    sample_points = []
    for x in range(0, width, step):
        sample_points.append((x, 0))
        sample_points.append((x, height - 1))
    for y in range(0, height, step):
        sample_points.append((0, y))
        sample_points.append((width - 1, y))

    for x, y in sample_points:
        if 0 <= x < width and 0 <= y < height:
            r, g, b = pixels[x, y]
            buckets[(r // 16, g // 16, b // 16)] += 1

    total = sum(buckets.values())
    dominant = [
        (bucket[0] * 16 + 8, bucket[1] * 16 + 8, bucket[2] * 16 + 8)
        for bucket, count in buckets.most_common(limit)
        if count / total >= 0.04
    ]
    if dominant:
        return dominant
    bucket, _count = buckets.most_common(1)[0]
    return [(bucket[0] * 16 + 8, bucket[1] * 16 + 8, bucket[2] * 16 + 8)]


def _near_background_color(
    r: int,
    g: int,
    b: int,
    background_colors: list[tuple[int, int, int]],
    max_distance: int = 42,
) -> bool:
    max_distance_sq = max_distance * max_distance
    for br, bg, bb in background_colors:
        distance_sq = (r - br) ** 2 + (g - bg) ** 2 + (b - bb) ** 2
        if distance_sq <= max_distance_sq:
            return True
    return False


def _color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    dr, dg, db = c1[0] - c2[0], c1[1] - c2[1], c1[2] - c2[2]
    return (dr * dr + dg * dg + db * db) ** 0.5


def _region_color_stats(pixels: list[tuple[int, int, int]]) -> tuple[tuple[int, int, int], float]:
    n = len(pixels)
    if n == 0:
        return (0, 0, 0), 0.0
    mean_r = sum(pixel[0] for pixel in pixels) // n
    mean_g = sum(pixel[1] for pixel in pixels) // n
    mean_b = sum(pixel[2] for pixel in pixels) // n
    variance = sum(
        (pixel[0] - mean_r) ** 2 + (pixel[1] - mean_g) ** 2 + (pixel[2] - mean_b) ** 2
        for pixel in pixels
    ) / n
    return (mean_r, mean_g, mean_b), variance


def _split_box_by_colors(image: Image.Image, box: list[int]) -> list[list[int]]:
    x1, y1, x2, y2 = box
    pixels = image.load()
    width, height = image.size
    background_colors = _dominant_background_colors(image)
    region_pixels = []

    for y in range(max(0, y1), min(height, y2)):
        for x in range(max(0, x1), min(width, x2)):
            r, g, b = pixels[x, y]
            _hue, saturation, _value = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            if saturation < 0.12:
                continue
            if _near_background_color(r, g, b, background_colors, max_distance=42):
                continue
            region_pixels.append((x, y, r, g, b))

    if len(region_pixels) < 30:
        return [box]

    region_pixels.sort(key=lambda pixel: pixel[2] + pixel[3] + pixel[4])
    c1_idx = len(region_pixels) // 3
    c2_idx = 2 * len(region_pixels) // 3
    centroid1 = (region_pixels[c1_idx][2], region_pixels[c1_idx][3], region_pixels[c1_idx][4])
    centroid2 = (region_pixels[c2_idx][2], region_pixels[c2_idx][3], region_pixels[c2_idx][4])

    for _ in range(5):
        cluster1 = []
        cluster2 = []
        for pixel in region_pixels:
            _x, _y, r, g, b = pixel
            d1_sq = (r - centroid1[0]) ** 2 + (g - centroid1[1]) ** 2 + (b - centroid1[2]) ** 2
            d2_sq = (r - centroid2[0]) ** 2 + (g - centroid2[1]) ** 2 + (b - centroid2[2]) ** 2
            if d1_sq < d2_sq:
                cluster1.append(pixel)
            else:
                cluster2.append(pixel)
        if not cluster1 or not cluster2:
            return [box]
        centroid1 = (
            sum(pixel[2] for pixel in cluster1) // len(cluster1),
            sum(pixel[3] for pixel in cluster1) // len(cluster1),
            sum(pixel[4] for pixel in cluster1) // len(cluster1),
        )
        centroid2 = (
            sum(pixel[2] for pixel in cluster2) // len(cluster2),
            sum(pixel[3] for pixel in cluster2) // len(cluster2),
            sum(pixel[4] for pixel in cluster2) // len(cluster2),
        )

    color_dist_sq = (
        (centroid1[0] - centroid2[0]) ** 2
        + (centroid1[1] - centroid2[1]) ** 2
        + (centroid1[2] - centroid2[2]) ** 2
    )
    if color_dist_sq < 400:
        return [box]

    result = []
    for cluster in (cluster1, cluster2):
        if len(cluster) < 30:
            continue
        xs = [pixel[0] for pixel in cluster]
        ys = [pixel[1] for pixel in cluster]
        result.append([min(xs), min(ys), max(xs) + 1, max(ys) + 1])

    return result if result else [box]


def detect_uniform_color_regions(
    image: Image.Image,
    min_area: int = 30,
    color_variance_threshold: float = 300.0,
    saturation_threshold: float = 0.12,
    grow_color_distance: float = 35.0,
) -> list[list[int]]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    background_colors = _dominant_background_colors(rgb)
    mask = [[False for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            _hue, saturation, _value = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            if saturation < saturation_threshold:
                continue
            if _near_background_color(r, g, b, background_colors, max_distance=42):
                continue
            mask[y][x] = True

    visited = [[False for _ in range(width)] for _ in range(height)]
    regions = []

    for y in range(height):
        for x in range(width):
            if visited[y][x] or not mask[y][x]:
                continue

            seed_color = pixels[x, y]
            queue = deque([(x, y)])
            visited[y][x] = True
            region_pixels = []
            region_coords = []

            while queue:
                cx, cy = queue.popleft()
                region_pixels.append(pixels[cx, cy])
                region_coords.append((cx, cy))
                for nx in (cx - 1, cx, cx + 1):
                    for ny in (cy - 1, cy, cy + 1):
                        if nx < 0 or ny < 0 or nx >= width or ny >= height:
                            continue
                        if visited[ny][nx] or not mask[ny][nx]:
                            continue
                        if _color_distance(pixels[nx, ny], seed_color) > grow_color_distance:
                            continue
                        visited[ny][nx] = True
                        queue.append((nx, ny))

            if len(region_pixels) < min_area:
                continue

            _mean, variance = _region_color_stats(region_pixels)
            if variance > color_variance_threshold:
                continue

            xs = [coord[0] for coord in region_coords]
            ys = [coord[1] for coord in region_coords]
            regions.append([min(xs), min(ys), max(xs) + 1, max(ys) + 1])

    regions.sort(key=lambda box: (box[0], box[1]))
    return regions


def _merge_nearby_boxes(boxes: list[list[int]], margin: int = 1) -> list[list[int]]:
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


def detect_colored_text_bboxes(image: Image.Image, min_area: int = 30) -> list[list[int]]:
    uniform = detect_uniform_color_regions(image, min_area=min_area)
    dark = detect_dark_regions(image, threshold=90, min_area=15)

    merged = list(uniform)
    for dark_box in dark:
        overlapping = False
        for uniform_box in uniform:
            dx1, dy1, dx2, dy2 = dark_box
            ux1, uy1, ux2, uy2 = uniform_box
            overlap_x = max(0, min(dx2, ux2) - max(dx1, ux1))
            overlap_y = max(0, min(dy2, uy2) - max(dy1, uy1))
            overlap_area = overlap_x * overlap_y
            dark_area = max(1, (dx2 - dx1) * (dy2 - dy1))
            if overlap_area > dark_area * 0.5:
                overlapping = True
                break
        if not overlapping:
            merged.append(dark_box)

    result = _merge_nearby_boxes(merged, margin=4)
    result.sort(key=lambda box: (box[0], box[1]))
    result = [box for box in result if validate_bbox(box, image.size)]

    final_boxes = []
    for box in result:
        final_boxes.extend(_split_box_by_colors(image, box))

    final_boxes.sort(key=lambda box: (box[0], box[1]))
    deduped = []
    seen = set()
    for box in final_boxes:
        key = tuple(box)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(box)

    filtered = []
    for box in deduped:
        width = box[2] - box[0]
        height = box[3] - box[1]
        area = width * height
        if width < 10 or height < 15 or area < 200 or width > 100 or height > 100:
            continue
        filtered.append(box)
    return filtered


def filter_captcha_text_bboxes(
    boxes: list[list[int]],
    image_size: tuple[int, int] | None = None,
    min_width: int = 16,
    min_height: int = 16,
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
    background_colors = _dominant_background_colors(rgb)

    for y in range(height):
        for x in range(width):
            pixel = source[x, y]
            if _colored_text_pixel(*pixel) and not _near_background_color(*pixel, background_colors):
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


def prepare_captcha_boxes(image: Image.Image, refine: bool = True) -> list[list[int]]:
    proposals = generate_box_proposals(image)
    if not proposals:
        return []
    boxes = proposals[0].boxes
    if not refine:
        return boxes
    return [refine_bbox_to_dark_pixels(image, box) for box in boxes]


def _translate_boxes(boxes: list[list[int]], dx: int = 0, dy: int = 0) -> list[list[int]]:
    return [[x1 + dx, y1 + dy, x2 + dx, y2 + dy] for x1, y1, x2, y2 in boxes]


def generate_box_proposals(image: Image.Image) -> list[ProposalSet]:
    whitened = whiten_watermark(image)
    uniform = detect_colored_text_bboxes(whitened)
    dark = filter_captcha_text_bboxes(detect_dark_regions(whitened), whitened.size)
    padded_border = 2
    padded = filter_captcha_text_bboxes(
        detect_colored_text_bboxes(ImageOps.expand(whitened, border=padded_border, fill="white")),
        None,
    )
    padded = _translate_boxes(padded, dx=-padded_border, dy=-padded_border)
    return [
        ProposalSet(boxes=uniform, source="uniform_color_regions", preprocess_variant="whitened"),
        ProposalSet(boxes=dark, source="dark_regions", preprocess_variant="whitened"),
        ProposalSet(boxes=padded, source="uniform_color_regions", preprocess_variant="whitened_padded"),
    ]


def measure_box_size_consistency(boxes: list[list[int]]) -> dict[str, float | int | None]:
    widths = [box[2] - box[0] for box in boxes]
    heights = [box[3] - box[1] for box in boxes]
    median_w = sorted(widths)[len(widths) // 2] if widths else 0
    median_h = sorted(heights)[len(heights) // 2] if heights else 0
    penalties = [
        abs(width - median_w) + abs(height - median_h)
        for width, height in zip(widths, heights)
    ]
    worst = max(range(len(penalties)), key=penalties.__getitem__) if penalties else None
    score = 1.0 / (1.0 + (max(penalties) if penalties else 0))
    return {"score": score, "outlier_index": worst}
