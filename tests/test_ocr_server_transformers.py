import base64
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from PIL import Image

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
    """Returns single Chinese char output."""
    model_path = "fake"
    loaded = True

    def recognize(self, image_bytes):
        return "件"


class FakeMultiCharEngine:
    """Returns multiple Chinese chars (should fail per-box recognition)."""
    model_path = "fake"
    loaded = True

    def recognize(self, image_bytes):
        return "识别结果：件叶结"


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
        self.assertIn("件", parsed)

    def test_parse_model_output_skips_bad_json_items(self):
        output = (
            '[{"text": "件", "bbox": [140, 50, 190, 110], "confidence": 0.94},'
            ' {"text": "坏", "bbox": ["bad"], "confidence": "bad"}]'
        )
        parsed = parse_model_output(output)
        self.assertIn("件", parsed)

    def test_parse_model_output_uses_first_valid_json_array(self):
        output = (
            'answer: [{"text": "件", "bbox": [140, 50, 190, 110]}]\n'
            "debug: [not json]"
        )
        parsed = parse_model_output(output)
        self.assertIn("件", parsed)

    def test_parse_model_output_skips_non_candidate_arrays(self):
        output = (
            "debug: [1, 2, 3]\n"
            'answer: [{"text": "件", "bbox": [140, 50, 190, 110]}]'
        )
        parsed = parse_model_output(output)
        self.assertIn("件", parsed)

    def test_parse_model_output_extracts_single_chinese_characters(self):
        parsed = parse_model_output("识别结果：件叶结")
        self.assertEqual(parsed, ["件", "叶", "结"])

    def test_parse_model_output_uses_result_line_when_later_colons_exist(self):
        parsed = parse_model_output("识别结果：件叶结\n置信度：0.9")
        self.assertEqual(parsed, ["件", "叶", "结"])

    def test_parse_model_output_plain_text_extracts_chars(self):
        parsed = parse_model_output("件叶结\n置信度：0.9")
        self.assertIn("件", parsed)
        self.assertIn("叶", parsed)
        self.assertIn("结", parsed)

    def test_parse_model_output_normalizes_katakana_i_as_eight(self):
        parsed = parse_model_output("装通イ被")
        self.assertEqual(parsed, ["装", "通", "八", "被"])

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


class RecognizerUnitTests(unittest.TestCase):
    def setUp(self):
        self.original_engine = ocr_server_transformers.engine

    def tearDown(self):
        ocr_server_transformers.engine = self.original_engine

    def test_recognize_box_crops_returns_one_char_per_box(self):
        """Single-char engine output returns exactly one candidate per box."""
        ocr_server_transformers.engine = FakeEngine()
        image = Image.new("RGB", (200, 150), "white")
        boxes = [[20, 20, 60, 60]]
        results = ocr_server_transformers.recognize_box_crops(image, boxes, None)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "件")

    def test_recognize_box_crops_returns_empty_on_multi_char_output(self):
        """Multi-char engine output causes per-box failure → empty list."""
        ocr_server_transformers.engine = FakeMultiCharEngine()
        image = Image.new("RGB", (200, 150), "white")
        boxes = [[20, 20, 60, 60]]
        results = ocr_server_transformers.recognize_box_crops(image, boxes, ["件", "叶", "结"])
        self.assertEqual(results, [])

    def test_recognize_box_crop_skips_char_not_in_target_set(self):
        ocr_server_transformers.engine = FakeEngine()
        image = Image.new("RGB", (200, 150), "white")
        box = [20, 20, 60, 60]
        # "测" is not in engine output "件"
        result = ocr_server_transformers.recognize_box_crop(image, box, set(["测", "试"]))
        self.assertIsNone(result)

    def test_recognize_box_crop_accepts_matching_char(self):
        ocr_server_transformers.engine = FakeEngine()
        image = Image.new("RGB", (200, 150), "white")
        box = [20, 20, 60, 60]
        result = ocr_server_transformers.recognize_box_crop(image, box, set(["件", "叶"]))
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "件")


if __name__ == "__main__":
    unittest.main()
