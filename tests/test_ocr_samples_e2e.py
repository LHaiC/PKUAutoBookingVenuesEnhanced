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

# Configuration
OCR_ENDPOINT = os.environ.get("OCR_ENDPOINT", "http://localhost:8000/glmocr/parse")
HEALTH_ENDPOINT = os.environ.get("OCR_HEALTH", "http://localhost:8000/health")
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


def _local_session() -> requests.Session:
    """Use direct localhost connections even if proxy env vars are set."""
    session = requests.Session()
    session.trust_env = False
    return session


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

    resp = _local_session().post(OCR_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def check_server_ready() -> bool:
    """Check if OCR server is running and model is loaded."""
    try:
        resp = _local_session().get(HEALTH_ENDPOINT, timeout=5)
        result = resp.json()
        # PaddleOCR server: {"status":"ok","model":"paddleocr"}
        # GLM server: {"status":"ok","model":"glm-ocr","model_loaded":true}
        if "model_loaded" in result:
            return result.get("model_loaded", False)
        # PaddleOCR health format - status ok means ready
        return result.get("status") == "ok"
    except requests.RequestException:
        return False


def test_health_endpoint_is_glm_server():
    response = _local_session().get(HEALTH_ENDPOINT, timeout=5)
    response.raise_for_status()
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model"] in {"glm-ocr", "glm_ocr_transformers"}


# PaddleOCR handles detection+recognition natively; no rule-based box detection needed.


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

    def _annotate_if_needed(self, name: str, img: Image.Image, result: dict | None, targets: list):
        """Save annotated image if self.annotate is True."""
        if not self.annotate:
            return
        output_path = OUTPUT_DIR / f"annotated_{name}"
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        if result:
            for i, res in enumerate(result.get("results", [])):
                bbox = res.get("bbox", [])
                if bbox and len(bbox) == 4:
                    color = BOX_COLORS[i % len(BOX_COLORS)]
                    draw.rectangle(bbox, outline='green', width=2)
                    text = f"R{i}:{res['text']}"
                    draw.text((bbox[0], bbox[3] + 2), text, fill='green')
            status = "PASS" if set([r['text'] for r in result.get('results', [])]) == set(targets) else "FAIL"
            draw.text((5, 5), status, fill='green' if status == 'PASS' else 'red')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img_copy.save(output_path)

    def test_sample_001_jin_ru_xin(self):
        """Test captcha with targets: 今, 入, 心"""
        name, targets = SAMPLES[0]
        img, _ = load_sample(name)

        result = get_ocr_result(img, targets)
        results_text = [r["text"] for r in result.get("results", [])]

        self._annotate_if_needed(name, img, result, targets)

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

        self._annotate_if_needed(name, img, result, targets)

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

        self._annotate_if_needed(name, img, result, targets)

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

        self._annotate_if_needed(name, img, result, targets)

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

        self._annotate_if_needed(name, img, result, targets)

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

        self._annotate_if_needed(name, img, result, targets)

        self.assertEqual(
            set(results_text), set(targets),
            f"Sample {name}: expected {targets}, got {results_text}. "
            f"Error: {result.get('error')}, detail: {result.get('detail')}"
        )


# PaddleOCR handles detection+recognition natively; no rule-based box detection or per-box OCR needed.


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
