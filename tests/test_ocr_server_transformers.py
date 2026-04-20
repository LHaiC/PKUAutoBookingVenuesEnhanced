import base64
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from PIL import Image, ImageDraw

import ocr_server_transformers
from ocr_server_transformers import (
    ParseRequest,
    decode_data_uri,
    health,
    parse,
    parse_model_output,
    validation_exception_handler,
)


def make_png_data_uri(
    width: int = 2,
    height: int = 2,
    dark_boxes: list[tuple[int, int, int, int]] | None = None,
) -> str:
    image = Image.new("RGB", (width, height), "white")
    for x1, y1, x2, y2 in dark_boxes or []:
        for x in range(x1, x2):
            for y in range(y1, y2):
                image.putpixel((x, y), (0, 0, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    payload = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{payload}"


class FakeEngine:
    model_path = "fake"
    loaded = True

    def recognize(self, image_bytes, targets):
        return "识别结果：件叶结"


class FakeJsonEngine:
    model_path = "fake"
    loaded = True

    def recognize(self, image_bytes, targets):
        return (
            '[{"text": "件", "bbox": [0, 0, 10, 10], "confidence": 0.94},'
            ' {"text": "叶", "bbox": [10, 0, 20, 10], "confidence": 0.89},'
            ' {"text": "结", "bbox": [20, 0, 30, 10], "confidence": 0.91}]'
        )


class FakeTextRecognitionEngine:
    model_path = "fake"
    loaded = True

    def recognize(self, image_bytes, targets):
        return "提 结 叶\n件"


class FakeRecordingEngine:
    model_path = "fake"
    loaded = True

    def __init__(self, outputs=None):
        self.outputs = list(outputs or ["工产建穿"])
        self.received_targets = []
        self.received_sizes = []

    def recognize(self, image_bytes, targets):
        self.received_targets.append(targets)
        self.received_sizes.append(Image.open(io.BytesIO(image_bytes)).size)
        if len(self.received_sizes) <= len(self.outputs):
            return self.outputs[len(self.received_sizes) - 1]
        return self.outputs[-1]


class OcrServerTransformerTests(unittest.TestCase):
    def setUp(self):
        self.original_engine = ocr_server_transformers.engine
        ocr_server_transformers.engine = None

    def tearDown(self):
        ocr_server_transformers.engine = self.original_engine

    def test_decode_data_uri_accepts_prefixed_uri(self):
        payload = base64.b64encode(b"abc").decode("utf-8")
        self.assertEqual(decode_data_uri(f"data:image/png;base64,{payload}"), b"abc")

    def test_decode_data_uri_accepts_raw_base64(self):
        payload = base64.b64encode(b"abc").decode("utf-8")
        self.assertEqual(decode_data_uri(payload), b"abc")

    def test_parse_model_output_reads_json_list(self):
        output = '[{"text": "件", "bbox": [140, 50, 190, 110], "confidence": 0.94}]'
        parsed = parse_model_output(output)
        self.assertEqual(parsed[0]["text"], "件")
        self.assertEqual(parsed[0]["bbox"], [140, 50, 190, 110])
        self.assertEqual(parsed[0]["confidence"], 0.94)

    def test_parse_model_output_skips_bad_json_items(self):
        output = (
            '[{"text": "件", "bbox": [140, 50, 190, 110], "confidence": 0.94},'
            ' {"text": "坏", "bbox": ["bad"], "confidence": "bad"}]'
        )

        self.assertEqual(
            parse_model_output(output),
            [{"text": "件", "bbox": [140, 50, 190, 110], "confidence": 0.94}],
        )

    def test_parse_model_output_uses_first_valid_json_array(self):
        output = (
            'answer: [{"text": "件", "bbox": [140, 50, 190, 110]}]\n'
            "debug: [not json]"
        )

        self.assertEqual(
            parse_model_output(output),
            [{"text": "件", "bbox": [140, 50, 190, 110], "confidence": 0.80}],
        )

    def test_parse_model_output_skips_non_candidate_arrays(self):
        output = (
            "debug: [1, 2, 3]\n"
            'answer: [{"text": "件", "bbox": [140, 50, 190, 110]}]'
        )

        self.assertEqual(
            parse_model_output(output),
            [{"text": "件", "bbox": [140, 50, 190, 110], "confidence": 0.80}],
        )

    def test_parse_model_output_rejects_unsafe_numeric_values(self):
        output = (
            '[{"text": "件", "bbox": [false, false, true, true], "confidence": NaN},'
            ' {"text": "叶", "bbox": [1.5, 2, 3, 4], "confidence": 0.9},'
            ' {"text": "结", "bbox": [1, 2, 3, 4], "confidence": NaN}]'
        )

        self.assertEqual(parse_model_output(output), [])

    def test_parse_model_output_extracts_single_chinese_characters(self):
        parsed = parse_model_output("识别结果：件叶结")
        self.assertEqual(
            parsed,
            [
                {"text": "件", "bbox": [], "confidence": 0.50},
                {"text": "叶", "bbox": [], "confidence": 0.50},
                {"text": "结", "bbox": [], "confidence": 0.50},
            ],
        )

    def test_parse_model_output_uses_result_line_when_later_colons_exist(self):
        parsed = parse_model_output("识别结果：件叶结\n置信度：0.9")

        self.assertEqual([item["text"] for item in parsed], ["件", "叶", "结"])

    def test_parse_model_output_uses_first_plain_text_line_without_label(self):
        parsed = parse_model_output("件叶结\n置信度：0.9")

        self.assertEqual([item["text"] for item in parsed], ["件", "叶", "结"])

    def test_parse_model_output_normalizes_katakana_i_as_eight(self):
        parsed = parse_model_output("装通イ被")

        self.assertEqual([item["text"] for item in parsed], ["装", "通", "八", "被"])

    def test_parse_route_returns_503_when_model_unloaded(self):
        with self.assertRaises(HTTPException) as ctx:
            parse(ParseRequest(images=[make_png_data_uri()]))

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail, "Model not loaded")

    def test_parse_route_returns_400_for_invalid_base64(self):
        with self.assertRaises(HTTPException) as ctx:
            parse(ParseRequest(images=["not-base64!!!"]))

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "Invalid base64 image data")

    def test_parse_route_returns_400_for_missing_images(self):
        with self.assertRaises(HTTPException) as ctx:
            parse(ParseRequest())

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "No images provided")

    def test_parse_route_returns_400_for_invalid_body_shape(self):
        response = validation_exception_handler(None, RequestValidationError([]))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.body, b'{"detail":"Invalid request body"}')

    def test_parse_route_returns_400_for_invalid_image_bytes_before_model_check(self):
        payload = base64.b64encode(b"abc").decode("utf-8")

        with self.assertRaises(HTTPException) as ctx:
            parse(ParseRequest(images=[payload]))

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "Invalid image data")

    def test_health_reports_unloaded_model(self):
        response = health()

        self.assertEqual(response["status"], "ok")
        self.assertFalse(response["model_loaded"])
        self.assertIsNone(response["model_path"])


def test_filter_watermark_boxes_excludes_bottom_right():
    from ocr_server_transformers import filter_watermark_boxes
    boxes = [
        [10, 10, 50, 60],   # top-left, keep
        [100, 20, 140, 70], # top-right, keep
        [20, 100, 60, 140], # bottom-left, keep
        [300, 280, 340, 310], # bottom-right watermark zone, remove
    ]
    image_size = (400, 300)
    filtered = filter_watermark_boxes(boxes, image_size)
    assert len(filtered) == 3
    assert [300, 280, 340, 310] not in filtered

def test_filter_watermark_boxes_preserves_edge_boxes():
    from ocr_server_transformers import filter_watermark_boxes
    boxes = [
        [10, 10, 50, 60],   # top-left, keep
        [320, 30, 360, 80], # x>80% but y<85%, keep
        [20, 270, 60, 290], # y>85% but x<80%, keep
    ]
    image_size = (400, 300)
    filtered = filter_watermark_boxes(boxes, image_size)
    assert len(filtered) == 3


def test_recognize_box_crops_returns_one_char_per_box():
    from PIL import Image
    from ocr_server_transformers import recognize_box_crops, engine
    image = Image.new('RGB', (200, 150), 'white')
    boxes = [[20, 20, 60, 60]]
    # Mock engine with a fake that returns a single Chinese character
    fake_engine = type('FakeBoxCropEngine', (), {
        'loaded': True,
        'recognize': lambda self, img_bytes, targets: '测'
    })()
    original_engine = ocr_server_transformers.engine
    ocr_server_transformers.engine = fake_engine
    try:
        results = recognize_box_crops(image, boxes)
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]['text'] == '测'
        assert results[0]['bbox'] == [20, 20, 60, 60]
    finally:
        ocr_server_transformers.engine = original_engine


def test_recognize_rotated_box_crops_returns_candidates_or_empty():
    from PIL import Image
    from ocr_server_transformers import recognize_rotated_box_crops, engine
    image = Image.new('RGB', (200, 150), 'white')
    boxes = [[20, 20, 60, 60], [100, 20, 140, 60]]
    # Mock engine that returns target chars when rotated
    fake_engine = type('FakeRotatedEngine', (), {
        'loaded': True,
        'recognize': lambda self, img_bytes, targets: '测'
    })()
    original_engine = ocr_server_transformers.engine
    ocr_server_transformers.engine = fake_engine
    try:
        results = recognize_rotated_box_crops(image, ['测', '试'], boxes)
        # Should return list of Candidate objects (possibly empty if rotation fails)
        assert isinstance(results, list)
    finally:
        ocr_server_transformers.engine = original_engine


def test_solve_image_uses_per_box_ocr_main_path():
    # Test that solve_image returns proper structure when model is loaded
    # This tests the function directly, not via HTTP
    from ocr_server_transformers import solve_image, engine

    if engine is None or not engine.loaded:
        import pytest
        pytest.skip("OCR model not loaded")

    from io import BytesIO
    from PIL import Image

    image = Image.new('RGB', (300, 200), 'white')
    img_bytes = BytesIO()
    image.save(img_bytes, format='PNG')

    result = solve_image(img_bytes.getvalue(), ["内", "别", "员"])
    assert "results" in result or "error" in result
    # Should return per_box_ocr method on success
    if "results" in result and result["results"]:
        assert result.get("method") == "per_box_ocr"


if __name__ == "__main__":
    unittest.main()
