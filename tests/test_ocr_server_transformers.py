import base64
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException
from PIL import Image

import ocr_server_transformers
from captcha_matcher import Candidate
from captcha_vision import ProposalSet
from ocr_server_transformers import (
    ParseRequest,
    accept_solution,
    health,
    parse,
    score_proposal_set,
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


class OcrServerTransformerTests(unittest.TestCase):
    def test_score_proposal_set_prefers_full_target_coverage(self):
        image = Image.new("RGB", (160, 60), "white")
        proposal_good = ProposalSet(
            boxes=[[10, 10, 30, 30], [50, 10, 70, 30], [90, 10, 110, 30]],
            source="uniform_color_regions",
            preprocess_variant="whitened",
        )
        proposal_bad = ProposalSet(
            boxes=[[10, 10, 60, 35], [90, 10, 110, 30]],
            source="dark_regions",
            preprocess_variant="whitened",
        )
        good_candidates = [
            Candidate("今", proposal_good.boxes[0], 0.90),
            Candidate("入", proposal_good.boxes[1], 0.88),
            Candidate("心", proposal_good.boxes[2], 0.92),
        ]
        bad_candidates = [
            Candidate("今", proposal_bad.boxes[0], 0.55),
            Candidate("心", proposal_bad.boxes[1], 0.60),
        ]

        self.assertGreater(
            score_proposal_set(image, proposal_good, ["今", "入", "心"], good_candidates)["score"],
            score_proposal_set(image, proposal_bad, ["今", "入", "心"], bad_candidates)["score"],
        )

    def test_accept_solution_rejects_ambiguous_top_scores(self):
        accepted, reason = accept_solution(
            top_score={"score": 0.92, "matched": [{"text": "今"}, {"text": "入"}, {"text": "心"}]},
            runner_up_score={"score": 0.91, "matched": [{"text": "今"}, {"text": "入"}, {"text": "心"}]},
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "ambiguous_top_score")


if __name__ == "__main__":
    unittest.main()
