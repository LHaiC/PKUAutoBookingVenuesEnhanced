import base64
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException
from PIL import Image

import ocr_server_transformers
from ocr_server_transformers import (
    ParseRequest,
    health,
    parse,
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
    loaded = True
    model_path = "fake-model"

    def recognize(self, _image_bytes):
        return "识别结果：件"


class FakeMultiCharEngine:
    loaded = True
    model_path = "fake-model"

    def recognize(self, _image_bytes):
        return "识别结果：件叶结"


class OcrServerTransformerRouteTests(unittest.TestCase):
    def setUp(self):
        self.original_engine = ocr_server_transformers.engine
        ocr_server_transformers.engine = None

    def tearDown(self):
        ocr_server_transformers.engine = self.original_engine

    def test_parse_route_returns_503_when_model_unloaded(self):
        with self.assertRaises(HTTPException) as ctx:
            parse(ParseRequest(images=[make_png_data_uri()]))
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail, "Model not loaded")

    def test_health_reports_unloaded_model(self):
        response = health()
        self.assertEqual(response["status"], "ok")
        self.assertFalse(response["model_loaded"])
        self.assertIsNone(response["model_path"])


class RecognizerUnitTests(unittest.TestCase):
    def tearDown(self):
        ocr_server_transformers.engine = None

    def test_recognize_box_crops_returns_one_char_per_box(self):
        ocr_server_transformers.engine = FakeEngine()
        image = Image.new("RGB", (200, 150), "white")
        boxes = [[20, 20, 60, 60]]
        results = ocr_server_transformers.recognize_box_crops(image, boxes, None)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "件")

    def test_recognize_box_crops_returns_empty_on_multi_char_output(self):
        ocr_server_transformers.engine = FakeMultiCharEngine()
        image = Image.new("RGB", (200, 150), "white")
        boxes = [[20, 20, 60, 60]]
        results = ocr_server_transformers.recognize_box_crops(image, boxes, ["件", "叶", "结"])
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
