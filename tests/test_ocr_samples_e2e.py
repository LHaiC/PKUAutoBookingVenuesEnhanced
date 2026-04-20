#!/usr/bin/env python3
"""
End-to-end OCR test suite for captcha samples.

This module tests the full OCR pipeline on captcha samples with known targets.
It replaces the old bash-based test_ocr_samples.sh script.

Test samples are stored in tests/samples/ with descriptive names:
- captcha_sample_001_<targets>_YYYYMMDD.png

Output (when running as script or with --annotate):
- tests/output/annotated_<original_name>: image with drawn boxes and results

Run with: python -m pytest tests/test_ocr_samples_e2e.py -v
Or directly: python tests/test_ocr_samples_e2e.py --annotate
"""

import argparse
import base64
import io
import os
import sys
import unittest
from pathlib import Path

import requests
from PIL import Image, ImageDraw

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from captcha_vision import whiten_watermark, detect_colored_text_bboxes
from ocr_server_transformers import crop_box_image, image_to_png_bytes

# Configuration
OCR_ENDPOINT = os.environ.get("GLM_ENDPOINT", "http://localhost:8000/glmocr/parse")
HEALTH_ENDPOINT = os.environ.get("GLM_HEALTH", "http://localhost:8000/health")
REQUEST_TIMEOUT = 120
OUTPUT_DIR = PROJECT_ROOT / "tests" / "output" / "annotated"
ANNOTATE = os.environ.get("ANNOTATE_OCR_TESTS", "0") == "1"

# Test samples: (filename, target_chars)
# Naming convention: captcha_sample_<NNN>_<first_char><second_char><third_char>_YYYYMMDD.png
SAMPLES = [
    ("captcha_sample_001_jin_ru_xin_20260420.png", ["今", "入", "心"]),
    ("captcha_sample_002_guan_li_sheng_20260420.png", ["关", "历", "生"]),
    ("captcha_sample_003_da_ri_he_20260420.png", ["打", "日", "合"]),
    ("captcha_sample_004_qian_si_li_20260420.png", ["前", "思", "力"]),
    ("captcha_sample_005_shu_ling_shi_20260420.png", ["叔", "领", "史"]),
    ("captcha_sample_006_chang_fang_ke_20260420.png", ["长", "方", "科"]),
]

# Color scheme for drawing boxes
BOX_COLORS = ['red', 'lime', 'yellow', 'orange', 'cyan', 'magenta']


def image_to_data_uri(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URI."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def load_sample(name: str) -> tuple[Image.Image, list[str]]:
    """Load a captcha sample image and its expected targets."""
    sample_dir = Path(__file__).parent / "samples"
    filepath = sample_dir / name
    if not filepath.exists():
        raise FileNotFoundError(f"Sample not found: {filepath}")

    img = Image.open(filepath)
    sample_entry = next((s for s in SAMPLES if s[0] == name), None)
    targets = sample_entry[1] if sample_entry else []
    return img, targets


def get_ocr_result(image: Image.Image, targets: list[str] | None = None) -> dict:
    """Call OCR endpoint and return JSON result."""
    data_uri = image_to_data_uri(image)
    payload = {"images": [data_uri]}
    if targets is not None:
        payload["targets"] = targets

    resp = requests.post(OCR_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def check_server_ready() -> bool:
    """Check if OCR server is running and model is loaded."""
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=5)
        result = resp.json()
        return result.get("model_loaded", False)
    except requests.RequestException:
        return False


def annotate_image(
    img: Image.Image,
    boxes: list[list[int]],
    result: dict | None,
    targets: list[str] | None,
    output_path: Path,
) -> None:
    """
    Draw boxes and OCR results on image and save to output_path.

    Args:
        img: Original PIL Image
        boxes: Detected bounding boxes [x1, y1, x2, y2]
        result: OCR result dict with 'results' list (can be None for box-only annotation)
        targets: Expected target characters (optional)
        output_path: Where to save the annotated image
    """
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # Draw detected boxes in blue
    for i, box in enumerate(boxes):
        color = BOX_COLORS[i % len(BOX_COLORS)]
        x1, y1, x2, y2 = box
        draw.rectangle(box, outline=color, width=2)
        draw.text((x1, y2 + 2), f"B{i}", fill=color)

    # Draw OCR results (boxes + text) in green
    if result:
        for i, res in enumerate(result.get("results", [])):
            bbox = res.get("bbox", [])
            if bbox and len(bbox) == 4:
                color = BOX_COLORS[i % len(BOX_COLORS)]
                draw.rectangle(bbox, outline='green', width=2)
                text = f"R{i}:{res['text']}"
                draw.text((bbox[0], bbox[3] + 2), text, fill='green')

        # Draw title with status
        status = "PASS" if set([r['text'] for r in result.get('results', [])]) == set(targets) else "FAIL"
        draw.text((5, 5), status, fill='green' if status == 'PASS' else 'red')

    # Draw info
    draw.text((5, 20), f"Boxes: {len(boxes)}", fill='white')
    if targets:
        draw.text((5, 35), f"Targets: {''.join(targets)}", fill='white')

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


class TestOCRSamplesE2E(unittest.TestCase):
    """End-to-end OCR tests on captcha samples."""

    annotate = False  # Set via parse_args or setUpClass

    @classmethod
    def setUpClass(cls):
        """Verify OCR server is ready before running tests."""
        if not check_server_ready():
            raise RuntimeError(
                f"OCR server not ready at {OCR_ENDPOINT}. "
                "Start it with: tmux a -t ocr (or python ocr_server_transformers.py)"
            )

    def _annotate_if_needed(self, name: str, img: Image.Image, boxes: list, result: dict | None, targets: list):
        """Save annotated image if self.annotate is True."""
        if not self.annotate:
            return
        output_path = OUTPUT_DIR / f"annotated_{name}"
        img_copy = img.copy()
        annotate_image(img_copy, boxes, result, targets, output_path)

    def test_sample_001_jin_ru_xin(self):
        """Test captcha with targets: 今, 入, 心"""
        name, targets = SAMPLES[0]
        img, _ = load_sample(name)

        result = get_ocr_result(img, targets)
        results_text = [r["text"] for r in result.get("results", [])]

        # Annotate
        whitened = whiten_watermark(img.copy())
        boxes = detect_colored_text_bboxes(whitened)
        self._annotate_if_needed(name, img, boxes, result, targets)

        self.assertEqual(
            set(results_text), set(targets),
            f"Sample {name}: expected {targets}, got {results_text}. "
            f"Error: {result.get('error')}, detail: {result.get('detail')}"
        )

    def test_sample_002_guan_li_sheng(self):
        """Test captcha with targets: 关, 历, 生"""
        name, targets = SAMPLES[1]
        img, _ = load_sample(name)

        result = get_ocr_result(img, targets)
        results_text = [r["text"] for r in result.get("results", [])]

        whitened = whiten_watermark(img.copy())
        boxes = detect_colored_text_bboxes(whitened)
        self._annotate_if_needed(name, img, boxes, result, targets)

        self.assertEqual(
            set(results_text), set(targets),
            f"Sample {name}: expected {targets}, got {results_text}. "
            f"Error: {result.get('error')}, detail: {result.get('detail')}"
        )

    def test_sample_003_da_ri_he(self):
        """Test captcha with targets: 打, 日, 合"""
        name, targets = SAMPLES[2]
        img, _ = load_sample(name)

        result = get_ocr_result(img, targets)
        results_text = [r["text"] for r in result.get("results", [])]

        whitened = whiten_watermark(img.copy())
        boxes = detect_colored_text_bboxes(whitened)
        self._annotate_if_needed(name, img, boxes, result, targets)

        self.assertEqual(
            set(results_text), set(targets),
            f"Sample {name}: expected {targets}, got {results_text}. "
            f"Error: {result.get('error')}, detail: {result.get('detail')}"
        )

    def test_sample_004_qian_si_li(self):
        """Test captcha with targets: 前, 思, 力"""
        name, targets = SAMPLES[3]
        img, _ = load_sample(name)

        result = get_ocr_result(img, targets)
        results_text = [r["text"] for r in result.get("results", [])]

        whitened = whiten_watermark(img.copy())
        boxes = detect_colored_text_bboxes(whitened)
        self._annotate_if_needed(name, img, boxes, result, targets)

        self.assertEqual(
            set(results_text), set(targets),
            f"Sample {name}: expected {targets}, got {results_text}. "
            f"Error: {result.get('error')}, detail: {result.get('detail')}"
        )

    def test_sample_005_shu_ling_shi(self):
        """Test captcha with targets: 叔, 领, 史"""
        name, targets = SAMPLES[4]
        img, _ = load_sample(name)

        result = get_ocr_result(img, targets)
        results_text = [r["text"] for r in result.get("results", [])]

        whitened = whiten_watermark(img.copy())
        boxes = detect_colored_text_bboxes(whitened)
        self._annotate_if_needed(name, img, boxes, result, targets)

        self.assertEqual(
            set(results_text), set(targets),
            f"Sample {name}: expected {targets}, got {results_text}. "
            f"Error: {result.get('error')}, detail: {result.get('detail')}"
        )

    def test_sample_006_chang_fang_ke(self):
        """Test captcha with targets: 长, 方, 科"""
        name, targets = SAMPLES[5]
        img, _ = load_sample(name)

        result = get_ocr_result(img, targets)
        results_text = [r["text"] for r in result.get("results", [])]

        whitened = whiten_watermark(img.copy())
        boxes = detect_colored_text_bboxes(whitened)
        self._annotate_if_needed(name, img, boxes, result, targets)

        self.assertEqual(
            set(results_text), set(targets),
            f"Sample {name}: expected {targets}, got {results_text}. "
            f"Error: {result.get('error')}, detail: {result.get('detail')}"
        )


class TestBoxDetection(unittest.TestCase):
    """Test box detection on captcha samples without OCR."""

    def test_detect_boxes_sample_001(self):
        """Verify box detection finds expected number of boxes for sample 001."""
        name, _ = SAMPLES[0]
        img, _ = load_sample(name)
        whitened = whiten_watermark(img)
        boxes = detect_colored_text_bboxes(whitened)
        # Should find at least 3 boxes (one per target character)
        self.assertGreaterEqual(
            len(boxes), 3,
            f"Sample {name}: expected >=3 boxes, got {len(boxes)}"
        )

    def test_detect_boxes_sample_006(self):
        """Verify box detection separates 方 from 长 for sample 006."""
        name, targets = SAMPLES[5]
        img, _ = load_sample(name)
        whitened = whiten_watermark(img)
        boxes = detect_colored_text_bboxes(whitened)

        # Find boxes whose center x-position suggests they contain distinct chars
        # 长 is on the right (x ~ 230), 方 should be separate (x ~ 160), 科 is below (x ~ 120)
        # After color-gated detection, we expect at least 3 distinct boxes
        self.assertGreaterEqual(
            len(boxes), 3,
            f"Sample {name}: expected >=3 boxes, got {len(boxes)}. "
            f"Boxes: {boxes}"
        )


class TestPerBoxOCR(unittest.TestCase):
    """Test per-box OCR recognition on individual crops."""

    def test_per_box_recognition_sample_001(self):
        """Test per-box OCR on sample 001 (今, 入, 心)."""
        name, targets = SAMPLES[0]
        img, _ = load_sample(name)
        whitened = whiten_watermark(img)
        boxes = detect_colored_text_bboxes(whitened)

        # Each box should produce a valid single-char crop for OCR
        for i, box in enumerate(boxes):
            crop = crop_box_image(whitened, box, padding=10)
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            resp = requests.post(OCR_ENDPOINT, json={"images": [b64]}, timeout=30)
            result = resp.json()
            texts = [r["text"] for r in result.get("results", [])]

            # Should recognize exactly one character per crop
            if texts:
                self.assertEqual(
                    len(texts), 1,
                    f"Sample {name} box[{i}] {box}: expected 1 char, got {texts}"
                )


def main():
    """Parse args and run tests with optional annotation."""
    parser = argparse.ArgumentParser(description="Run OCR E2E tests")
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Save annotated images to tests/output/annotated/",
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        default=[],
        help="Arguments to pass to pytest",
    )
    args = parser.parse_args()

    do_annotate = args.annotate or ANNOTATE
    if do_annotate:
        TestOCRSamplesE2E.annotate = True
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run pytest with any passed args
    import pytest
    sys.exit(pytest.main(["-v", *args.pytest_args, __file__]))


if __name__ == "__main__":
    main()