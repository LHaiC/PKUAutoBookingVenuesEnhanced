import base64
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import ocr_server_transformers
from ocr_server_transformers import app, decode_data_uri, parse_model_output


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

    def test_parse_route_returns_503_when_model_unloaded(self):
        payload = base64.b64encode(b"abc").decode("utf-8")
        response = TestClient(app).post("/glmocr/parse", json={"images": [payload]})

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


if __name__ == "__main__":
    unittest.main()
