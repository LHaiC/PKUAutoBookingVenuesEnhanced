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

    def test_parse_route_preserves_plain_text_results_without_targets(self):
        ocr_server_transformers.engine = FakeEngine()

        response = parse(ParseRequest(images=[make_png_data_uri()]))

        self.assertEqual(
            response["results"],
            [
                {"text": "件", "bbox": [], "confidence": 0.5},
                {"text": "叶", "bbox": [], "confidence": 0.5},
                {"text": "结", "bbox": [], "confidence": 0.5},
            ],
        )

    def test_parse_route_preserves_raw_json_results_without_targets(self):
        ocr_server_transformers.engine = FakeJsonEngine()

        response = parse(ParseRequest(images=[make_png_data_uri(30, 10, [(2, 2, 8, 8)])]))

        self.assertEqual(response["results"][0]["bbox"], [0, 0, 10, 10])
        self.assertNotIn("x", response["results"][0])
        self.assertNotIn("y", response["results"][0])

    def test_parse_route_returns_click_coordinates_for_targets(self):
        ocr_server_transformers.engine = FakeJsonEngine()
        image = make_png_data_uri(
            30,
            10,
            [(2, 2, 8, 8), (12, 2, 18, 8), (22, 2, 28, 8)],
        )

        response = parse(ParseRequest(images=[image], targets=["件", "叶", "结"]))

        self.assertEqual(
            response["results"],
            [
                {"text": "件", "bbox": [2, 2, 8, 8], "x": 5, "y": 5, "confidence": 0.94},
                {"text": "叶", "bbox": [12, 2, 18, 8], "x": 15, "y": 5, "confidence": 0.89},
                {"text": "结", "bbox": [22, 2, 28, 8], "x": 25, "y": 5, "confidence": 0.91},
            ],
        )

    def test_parse_route_fails_closed_for_plain_text_with_targets(self):
        ocr_server_transformers.engine = FakeEngine()

        response = parse(ParseRequest(images=[make_png_data_uri()], targets=["件", "叶", "结"]))

        self.assertEqual(response["results"], [])
        self.assertEqual(response["error"], "unsafe_ocr_output")

    def test_health_reports_unloaded_model(self):
        response = health()

        self.assertEqual(response["status"], "ok")
        self.assertFalse(response["model_loaded"])
        self.assertIsNone(response["model_path"])


if __name__ == "__main__":
    unittest.main()
