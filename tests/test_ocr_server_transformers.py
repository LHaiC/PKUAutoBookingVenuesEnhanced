import asyncio
import base64
import io
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException
import httpx
from PIL import Image

import ocr_server_transformers
from captcha_matcher import Candidate
from captcha_vision import ProposalSet
from ocr_server_transformers import (
    ParseRequest,
    accept_solution,
    app,
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


class FakeRecordingEngine:
    loaded = True
    model_path = "fake-model"

    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls = []

    def recognize(self, image_bytes):
        self.calls.append(image_bytes)
        if not self.outputs:
            raise AssertionError("No fake OCR outputs remaining")
        return self.outputs.pop(0)


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
    def tearDown(self):
        ocr_server_transformers.engine = None

    def post_json(self, path: str, payload: dict):
        async def send():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.post(path, json=payload)

        return asyncio.run(send())

    def test_score_proposal_set_prefers_full_target_coverage(self):
        image = Image.new("RGB", (160, 60), "white")
        proposal_good = ProposalSet(
            boxes=[[10, 10, 30, 30], [50, 10, 70, 30], [90, 10, 110, 30]],
            source="uniform_color_regions",
            preprocess_variant="whitened",
        )
        proposal_bad = ProposalSet(
            boxes=[[10, 10, 30, 30], [50, 10, 70, 30]],
            source="dark_regions",
            preprocess_variant="whitened",
        )
        good_candidates = [
            Candidate("今", proposal_good.boxes[0], 0.60),
            Candidate("入", proposal_good.boxes[1], 0.60),
            Candidate("心", proposal_good.boxes[2], 0.60),
        ]
        bad_candidates = [
            Candidate("今", proposal_bad.boxes[0], 0.60),
            Candidate("入", proposal_bad.boxes[1], 0.60),
        ]
        good_score = score_proposal_set(image, proposal_good, ["今", "入", "心"], good_candidates)
        bad_score = score_proposal_set(image, proposal_bad, ["今", "入", "心"], bad_candidates)

        self.assertEqual(len(good_score["matched"]), 3)
        self.assertEqual(len(bad_score["matched"]), 0)
        self.assertGreater(
            good_score["score"],
            bad_score["score"],
        )

    def test_accept_solution_rejects_ambiguous_top_scores(self):
        accepted, reason = accept_solution(
            top_score={
                "score": 0.92,
                "matched": [
                    {"text": "今", "bbox": [10, 10, 30, 30]},
                    {"text": "入", "bbox": [50, 10, 70, 30]},
                    {"text": "心", "bbox": [90, 10, 110, 30]},
                ],
            },
            runner_up_score={
                "score": 0.91,
                "matched": [
                    {"text": "今", "bbox": [12, 10, 32, 30]},
                    {"text": "入", "bbox": [52, 10, 72, 30]},
                    {"text": "心", "bbox": [92, 10, 112, 30]},
                ],
            },
            expected_match_count=3,
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "ambiguous_top_score")

    def test_parse_route_uses_best_proposal_set_for_targets(self):
        ocr_server_transformers.engine = FakeRecordingEngine(outputs=["今", "入", "心", "今入", "心"])
        payload = make_png_data_uri(width=160, height=60)
        proposals = [
            ProposalSet(
                boxes=[[10, 10, 30, 30], [50, 10, 70, 30], [90, 10, 110, 30]],
                source="uniform_color_regions",
                preprocess_variant="whitened",
            ),
            ProposalSet(
                boxes=[[12, 10, 32, 30], [52, 10, 72, 30]],
                source="dark_regions",
                preprocess_variant="whitened",
            ),
        ]

        with patch.object(ocr_server_transformers, "generate_box_proposals", return_value=proposals):
            response = self.post_json(
                "/glmocr/parse",
                {"images": [payload], "targets": ["今", "入", "心"]},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual([item["text"] for item in body["results"]], ["今", "入", "心"])
        self.assertEqual(body["method"], "glm_ocr_transformers_with_local_positioning")

    def test_parse_route_fails_closed_when_top_solution_is_ambiguous(self):
        ocr_server_transformers.engine = FakeRecordingEngine(outputs=["今", "入", "心"])
        payload = make_png_data_uri(width=160, height=60)
        proposals = [
            ProposalSet(
                boxes=[[10, 10, 30, 30], [50, 10, 70, 30], [150, 10, 170, 30]],
                source="uniform_color_regions",
                preprocess_variant="whitened",
            )
        ]

        with patch.object(ocr_server_transformers, "generate_box_proposals", return_value=proposals):
            response = self.post_json(
                "/glmocr/parse",
                {"images": [payload], "targets": ["今", "入", "心"]},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["results"], [])
        self.assertIn(body["error"], {"ambiguous_top_score", "incomplete_target_coverage"})

    def test_parse_route_recovers_failed_box_with_rotated_fallback(self):
        engine = FakeRecordingEngine(outputs=["今", "今入", "心", "今入", "入"])
        ocr_server_transformers.engine = engine
        payload = make_png_data_uri(width=160, height=60)
        proposals = [
            ProposalSet(
                boxes=[[10, 10, 30, 30], [50, 10, 70, 30], [90, 10, 110, 30]],
                source="uniform_color_regions",
                preprocess_variant="whitened",
            )
        ]

        with patch.object(ocr_server_transformers, "generate_box_proposals", return_value=proposals):
            response = self.post_json(
                "/glmocr/parse",
                {"images": [payload], "targets": ["今", "入", "心"]},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual([item["text"] for item in body["results"]], ["今", "入", "心"])
        self.assertEqual(len(engine.calls), 5)


if __name__ == "__main__":
    unittest.main()
