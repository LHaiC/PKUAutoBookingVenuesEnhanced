import base64
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from PIL import Image

import ocr_server_transformers
from ocr_server_transformers import app, decode_data_uri, parse_model_output


def make_png_data_uri() -> str:
    image = Image.new("RGB", (2, 2), "white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    payload = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{payload}"


class FakeEngine:
    model_path = "fake"
    loaded = True

    def recognize(self, image_bytes, targets):
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
        response = TestClient(app).post("/glmocr/parse", json={"images": [make_png_data_uri()]})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Model not loaded")

    def test_parse_route_returns_400_for_invalid_base64(self):
        response = TestClient(app).post("/glmocr/parse", json={"images": ["not-base64!!!"]})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Invalid base64 image data")

    def test_parse_route_returns_400_for_missing_images(self):
        response = TestClient(app).post("/glmocr/parse", json={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "No images provided")

    def test_parse_route_returns_400_for_invalid_image_bytes_before_model_check(self):
        payload = base64.b64encode(b"abc").decode("utf-8")

        response = TestClient(app).post("/glmocr/parse", json={"images": [payload]})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Invalid image data")

    def test_parse_route_preserves_plain_text_results_without_targets(self):
        ocr_server_transformers.engine = FakeEngine()

        response = TestClient(app).post("/glmocr/parse", json={"images": [make_png_data_uri()]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["results"],
            [
                {"text": "件", "bbox": [], "confidence": 0.5},
                {"text": "叶", "bbox": [], "confidence": 0.5},
                {"text": "结", "bbox": [], "confidence": 0.5},
            ],
        )

    def test_parse_route_fails_closed_for_plain_text_with_targets(self):
        ocr_server_transformers.engine = FakeEngine()

        response = TestClient(app).post(
            "/glmocr/parse",
            json={"images": [make_png_data_uri()], "targets": ["件", "叶", "结"]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"], [])
        self.assertEqual(response.json()["error"], "unsafe_ocr_output")


if __name__ == "__main__":
    unittest.main()
