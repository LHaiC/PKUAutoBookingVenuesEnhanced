"""
Test script for GLM-OCR captcha solver.

Reads tests/test.png, sends to local GLM-OCR service, and draws circles
around detected characters. Requires GLM-OCR service running at configured endpoint.

Usage:
    # 1. Start GLM-OCR service (one of):
    vllm serve zai-org/GLM-OCR --host 0.0.0.0 --port 8000
    # OR
    ollama run glm-ocr  # if endpoint configured for ollama

    # 2. Run this test:
    python tests/test_glm_ocr.py
"""

import base64
import json
import os
import re
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw
import requests
from captcha_vision import detect_colored_text_bboxes

# GLM-OCR endpoint configuration
GLM_ENDPOINT = os.environ.get("GLM_ENDPOINT", "http://localhost:8000/glmocr/parse")
GLM_TIMEOUT = 20
TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "test.png")
OUTPUT_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "test_output.png")
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "captcha_samples")
FAILURE_SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "captcha_failures",
)


def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def find_latest_captured_sample() -> tuple[str, list[str]] | None:
    for sample_dir in (FAILURE_SAMPLE_DIR, SAMPLE_DIR):
        if not os.path.isdir(sample_dir):
            continue

        metadata_files = sorted(
            [
                os.path.join(sample_dir, name)
                for name in os.listdir(sample_dir)
                if name.endswith(".json")
            ],
            reverse=True,
        )
        for metadata_path in metadata_files:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            image_name = metadata.get("image")
            targets = metadata.get("targets")
            if image_name and targets:
                image_path = os.path.join(sample_dir, image_name)
                if os.path.exists(image_path) and is_valid_image(image_path):
                    return image_path, targets
    return None


def load_image_as_base64(image_path: str) -> tuple[bytes, str]:
    """Load image and return (bytes content, base64 string)."""
    with open(image_path, "rb") as f:
        data = f.read()
    return data, base64.b64encode(data).decode("utf-8")


def call_glm_ocr(image_bytes: bytes, order_words: list[str] = None) -> dict:
    """
    Call GLM-OCR service to get character positions.

    Args:
        image_bytes: Raw image bytes
        order_words: Expected words to send as target click order

    Returns:
        GLM-OCR response JSON
    """
    data_uri = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    payload = {"images": [data_uri], "targets": order_words or []}

    print(f"Calling GLM-OCR at {GLM_ENDPOINT}...")
    response = requests.post(GLM_ENDPOINT, json=payload, timeout=GLM_TIMEOUT)
    response.raise_for_status()
    result = response.json()
    print(f"GLM-OCR response: {json.dumps(result, ensure_ascii=False)[:500]}")
    return result


def parse_glm_result(result: dict, order_words: list[str] = None) -> list:
    """
    Parse GLM-OCR response to extract character positions.
    Returns list of [char, x, y] for each detected character.
    """
    try:
        # GLM-OCR returns structured results
        # Response format may vary; extract text/bbox info
        results = result.get("results", result.get("response", []))

        if isinstance(results, list) and len(results) > 0:
            # Try to find text and position info
            chars_with_pos = []
            for item in results:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    if text and item.get("x") is not None and item.get("y") is not None:
                        chars_with_pos.append([text, int(item["x"]), int(item["y"])])
                        continue

                    bbox = item.get("bbox", [])
                    if text and len(bbox) == 4:
                        x = (int(bbox[0]) + int(bbox[2])) // 2
                        y = (int(bbox[1]) + int(bbox[3])) // 2
                        chars_with_pos.append([text, int(x), int(y)])
                elif isinstance(item, str):
                    # Plain text response
                    chars_with_pos.append([item, 0, 0])
            return chars_with_pos

        # Fallback: try to extract from raw response
        raw = str(result)
        # Look for patterns like ["字", x, y] or {"text":"字","bbox":[...]}
        matches = re.findall(r'["\"]([\u4e00-\u9fff])["\"][^]]*\[[\d,\s]+', raw)
        if matches:
            return [[m, 0, 0] for m in matches]

    except Exception as e:
        print(f"Failed to parse GLM result: {e}")

    return None


def draw_circles_on_image(
    image_path: str,
    output_path: str,
    chars_with_pos: list,
    order_words: list[str] = None,
) -> None:
    """
    Draw circles around detected characters on the image.

    Args:
        image_path: Input image path
        output_path: Output image path
        chars_with_pos: List of [char, x, y] for each character
        order_words: Words to highlight in order (from captcha prompt)
    """
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    order_words = order_words or []

    print(f"\nDetected {len(chars_with_pos)} characters:")
    for char, x, y in chars_with_pos:
        print(f"  '{char}' at ({x}, {y})")

    # Color mapping for order words
    colors = ["red", "green", "blue", "orange", "purple"]

    # Draw circle around each character
    radius = 25
    for idx, (char, x, y) in enumerate(chars_with_pos):
        # Determine color
        if char in order_words:
            color = colors[order_words.index(char) % len(colors)]
            label = f"[{order_words.index(char) + 1}] {char}"
        else:
            color = "gray"
            label = char

        # Draw circle outline
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline=color,
            width=3,
        )

        # Draw label
        draw.text((x + radius + 5, y - 10), label, fill=color)

    img.save(output_path)
    print(f"\nOutput saved to: {output_path}")


def draw_debug_boxes_on_image(image_path: str, output_path: str, order_words: list[str] = None) -> None:
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    boxes = detect_colored_text_bboxes(img)

    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 12)), f"box{idx}", fill="red")

    if order_words:
        draw.text((10, 10), "targets: " + ",".join(order_words), fill="blue")

    img.save(output_path)
    print(f"\nDebug overlay saved to: {output_path}")
    print(f"Detected {len(boxes)} colored candidate boxes")


def main():
    print("=" * 60)
    print("GLM-OCR Captcha Solver Test")
    print("=" * 60)

    captured_sample = find_latest_captured_sample()
    if captured_sample:
        test_image_path, order_words = captured_sample
        print(f"Using captured captcha sample: {test_image_path}")
    else:
        test_image_path = TEST_IMAGE_PATH
        order_words = ["件", "叶", "结"]
        print("Using fallback manual test image")

    # Check if test image exists
    if not os.path.exists(test_image_path):
        print(f"ERROR: Test image not found at {test_image_path}")
        sys.exit(1)

    print(f"Test image: {test_image_path}")

    # Load image
    image_bytes, _ = load_image_as_base64(test_image_path)
    print(f"Image size: {len(image_bytes)} bytes")

    print(f"Expected captcha words (in order): {order_words}")

    try:
        # Call GLM-OCR
        result = call_glm_ocr(image_bytes, order_words)

        # Parse results
        chars_with_pos = parse_glm_result(result, order_words)

        if chars_with_pos:
            print(f"\nParsed {len(chars_with_pos)} characters with positions")
            returned = [item[0] for item in chars_with_pos]
            if returned != order_words:
                raise AssertionError(f"Expected {order_words}, got {returned}")

            width, height = Image.open(test_image_path).size
            for char, x, y in chars_with_pos:
                if not (0 <= x <= width and 0 <= y <= height):
                    raise AssertionError(f"Coordinate out of bounds for {char}: {(x, y)}")
        else:
            print("\nCould not parse character positions from GLM-OCR response")
            chars_with_pos = []

    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Could not connect to GLM-OCR at {GLM_ENDPOINT}")
        print("Please ensure GLM-OCR service is running:")
        print("  python ocr_server_transformers.py --model models/GLM-OCR --port 8000")
        chars_with_pos = []

    except requests.exceptions.Timeout:
        print(f"\nERROR: GLM-OCR timeout after {GLM_TIMEOUT}s")
        chars_with_pos = []

    except AssertionError as e:
        print(f"\nERROR: {e}")
        raise

    except Exception as e:
        print(f"\nERROR: {e}")
        chars_with_pos = []

    # Draw results on image
    if chars_with_pos:
        draw_circles_on_image(test_image_path, OUTPUT_IMAGE_PATH, chars_with_pos, order_words)
    else:
        draw_debug_boxes_on_image(test_image_path, OUTPUT_IMAGE_PATH, order_words)
        print("(No OCR results; drew local colored candidate boxes)")
        sys.exit(1)


if __name__ == "__main__":
    main()
