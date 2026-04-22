import base64
import io
import os
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException
from PIL import Image

import ocr_server_transformers
from captcha_matcher import Candidate
from captcha_vision import ProposalSet
from ocr_server_transformers import (
    GlmOcrEngine,
    OCR_MAX_NEW_TOKENS,
    ParseRequest,
    accept_solution,
    app,
    build_aligned_strip_candidates,
    dedupe_candidates,
    build_solver_response,
    choose_better_legacy_result,
    health,
    health_route,
    parse,
    parse_route,
    results_have_safe_click_geometry,
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


class FakeInputIds:
    shape = (1, 2)


class FakeBatchInputIds:
    def __init__(self, batch_size: int, prompt_tokens: int = 2):
        self.shape = (batch_size, prompt_tokens)


class FakeInputs(dict):
    def __init__(self):
        super().__init__(
            {
                "input_ids": FakeInputIds(),
                "attention_mask": "keep",
                "token_type_ids": "drop",
            }
        )
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeBatchInputs(dict):
    def __init__(self, batch_size: int):
        super().__init__(
            {
                "input_ids": FakeBatchInputIds(batch_size),
                "attention_mask": "keep",
                "token_type_ids": "drop",
            }
        )
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    def __init__(self):
        self.inputs = FakeInputs()
        self.messages = None
        self.kwargs = None
        self.image_path = None
        self.image_path_existed = False
        self.decode_args = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        self.image_path = messages[0]["content"][0]["url"]
        self.image_path_existed = os.path.exists(self.image_path)
        return self.inputs

    def decode(self, output_ids, skip_special_tokens=False):
        self.decode_args = (output_ids, skip_special_tokens)
        return "decoded-output"


class FakeBatchProcessor(FakeProcessor):
    def __init__(self):
        super().__init__()
        self.batch_inputs = None
        self.batch_messages = None
        self.batch_kwargs = None
        self.batch_decode_calls = []

    def apply_chat_template(self, messages, **kwargs):
        if messages and isinstance(messages[0], list):
            self.batch_messages = messages
            self.batch_kwargs = kwargs
            self.batch_inputs = FakeBatchInputs(len(messages))
            return self.batch_inputs
        return super().apply_chat_template(messages, **kwargs)

    def decode(self, output_ids, skip_special_tokens=False):
        self.batch_decode_calls.append((output_ids, skip_special_tokens))
        return f"decoded-{len(self.batch_decode_calls)}"


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[101, 102, 201, 202]]


class FakeBatchModel(FakeModel):
    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [
            [101, 102, 201, 202],
            [101, 102, 301, 302],
        ]


class RouteResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


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

    def test_health_route_reports_loaded_model(self):
        ocr_server_transformers.engine = FakeEngine()
        response = RouteResponse(200, health_route())
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["model_loaded"])
        self.assertEqual(body["model_path"], "fake-model")

    def test_app_registers_expected_routes(self):
        route_paths = {route.path for route in app.routes if hasattr(route, "path")}
        self.assertIn("/health", route_paths)
        self.assertIn("/glmocr/parse", route_paths)
        self.assertIn("/ocr/parse", route_paths)

    def test_results_have_safe_click_geometry_rejects_duplicate_centers(self):
        results = [
            {"text": "方", "bbox": [103, 69, 127, 91], "x": 115, "y": 80, "confidence": 0.5},
            {"text": "科", "bbox": [104, 69, 127, 91], "x": 115, "y": 80, "confidence": 0.5},
        ]
        self.assertFalse(results_have_safe_click_geometry(results))

    def test_build_solver_response_rejects_unsafe_click_geometry(self):
        image = Image.new("RGB", (160, 60), "white")
        response = build_solver_response(
            image,
            [
                {"text": "方", "bbox": [103, 69, 127, 91], "x": 115, "y": 80, "confidence": 0.5},
                {"text": "科", "bbox": [104, 69, 127, 91], "x": 115, "y": 80, "confidence": 0.5},
            ],
            "ok",
        )
        self.assertEqual(response["results"], [])
        self.assertEqual(response["error"], "unsafe_click_geometry")

    def test_build_solver_response_restores_original_box_when_refined_box_is_too_small(self):
        image = Image.new("RGB", (310, 155), "white")
        response = build_solver_response(
            image,
            [
                {
                    "text": "前",
                    "bbox": [208, 63, 225, 88],
                    "original_bbox": [202, 51, 238, 88],
                    "x": 216,
                    "y": 75,
                    "confidence": 0.6,
                },
                {
                    "text": "思",
                    "bbox": [160, 80, 193, 119],
                    "original_bbox": [160, 80, 193, 119],
                    "x": 176,
                    "y": 99,
                    "confidence": 0.6,
                },
                {
                    "text": "力",
                    "bbox": [91, 78, 127, 117],
                    "original_bbox": [91, 78, 127, 117],
                    "x": 109,
                    "y": 97,
                    "confidence": 0.6,
                },
            ],
            "ok",
        )

        self.assertEqual(response["results"][0]["bbox"], [202, 51, 238, 88])
        self.assertEqual((response["results"][0]["x"], response["results"][0]["y"]), (220, 69))

    def test_build_solver_response_shrinks_oversized_box_to_dark_region(self):
        image = Image.new("RGB", (160, 80), "white")
        for x in range(10, 40):
            for y in range(12, 47):
                image.putpixel((x, y), (0, 0, 0))
        for x in range(90, 120):
            for y in range(10, 45):
                image.putpixel((x, y), (0, 0, 0))
        for x in range(125, 155):
            for y in range(10, 45):
                image.putpixel((x, y), (0, 0, 0))

        response = build_solver_response(
            image,
            [
                {
                    "text": "科",
                    "bbox": [10, 10, 74, 47],
                    "original_bbox": [10, 10, 74, 47],
                    "x": 42,
                    "y": 28,
                    "confidence": 0.6,
                },
                {
                    "text": "方",
                    "bbox": [90, 10, 120, 45],
                    "original_bbox": [90, 10, 120, 45],
                    "x": 105,
                    "y": 27,
                    "confidence": 0.6,
                },
                {
                    "text": "长",
                    "bbox": [125, 10, 155, 45],
                    "original_bbox": [125, 10, 155, 45],
                    "x": 140,
                    "y": 27,
                    "confidence": 0.6,
                },
            ],
            "ok",
        )

        self.assertEqual(response["results"][0]["bbox"], [10, 12, 40, 47])
        self.assertEqual((response["results"][0]["x"], response["results"][0]["y"]), (25, 29))


class RecognizerUnitTests(unittest.TestCase):
    def tearDown(self):
        ocr_server_transformers.engine = None

    def test_recognize_with_cache_reuses_identical_payloads(self):
        recognize = Mock(return_value="识别结果：今")
        cache = {}
        payload = b"same-image"

        first = ocr_server_transformers._recognize_with_cache(recognize, cache, payload)
        second = ocr_server_transformers._recognize_with_cache(recognize, cache, payload)

        self.assertEqual(first, "识别结果：今")
        self.assertEqual(second, "识别结果：今")
        recognize.assert_called_once_with(payload)

    def test_recognize_box_crops_legacy_preserves_original_bbox_for_later_normalization(self):
        ocr_server_transformers.engine = FakeEngine()
        image = Image.new("RGB", (200, 150), "white")
        boxes = [[20, 20, 60, 60]]

        with patch.object(ocr_server_transformers, "refine_bbox_to_dark_pixels", return_value=[25, 25, 45, 45]):
            results = ocr_server_transformers.recognize_box_crops_legacy(image, boxes)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].bbox, [20, 20, 60, 60])
        self.assertEqual(results[0].original_bbox, [20, 20, 60, 60])

    def test_candidates_from_legacy_output_preserves_original_bbox_for_later_normalization(self):
        image = Image.new("RGB", (160, 60), "white")
        boxes = [[10, 10, 30, 30]]

        with patch.object(ocr_server_transformers, "refine_bbox_to_dark_pixels", return_value=[12, 12, 24, 24]):
            candidates = ocr_server_transformers.candidates_from_legacy_output(
                image,
                "识别结果：今",
                boxes,
                allow_index_attachment=True,
            )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].bbox, [10, 10, 30, 30])
        self.assertEqual(candidates[0].original_bbox, [10, 10, 30, 30])

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


class GlmOcrEngineRequestTests(unittest.TestCase):
    def test_recognize_builds_transformers_request_without_real_model(self):
        processor = FakeProcessor()
        model = FakeModel()
        engine = GlmOcrEngine("fake-model")
        engine.processor = processor
        engine.model = model

        output = engine.recognize(b"fake-image", ["件", "叶"])

        self.assertEqual(output, "decoded-output")
        self.assertTrue(processor.image_path_existed)
        self.assertFalse(os.path.exists(processor.image_path))
        self.assertEqual(processor.inputs.device, "cpu")
        self.assertNotIn("token_type_ids", model.generate_kwargs)
        self.assertIs(model.generate_kwargs["input_ids"], processor.inputs["input_ids"])
        self.assertEqual(model.generate_kwargs["max_new_tokens"], OCR_MAX_NEW_TOKENS)

    def test_recognize_batch_builds_transformers_request_without_real_model(self):
        engine = GlmOcrEngine("fake-model")
        processor = FakeBatchProcessor()
        model = FakeBatchModel()
        engine.processor = processor
        engine.model = model

        outputs = engine.recognize_batch([b"fake-image-1", b"fake-image-2"], ["件", "叶"])

        self.assertEqual(outputs, ["decoded-1", "decoded-2"])
        self.assertEqual(len(processor.batch_messages), 2)
        self.assertNotIn("token_type_ids", model.generate_kwargs)
        self.assertIs(model.generate_kwargs["input_ids"], processor.batch_inputs["input_ids"])
        self.assertEqual(model.generate_kwargs["max_new_tokens"], OCR_MAX_NEW_TOKENS)
        self.assertEqual(processor.batch_messages[0][0]["content"][1]["text"], "Text Recognition:")
        self.assertEqual(processor.batch_decode_calls, [([201, 202], False), ([301, 302], False)])


class OcrServerTransformerTests(unittest.TestCase):
    def tearDown(self):
        ocr_server_transformers.engine = None

    def post_json(self, path: str, payload: dict):
        if path not in {"/glmocr/parse", "/ocr/parse"}:
            raise AssertionError(f"Unexpected test path: {path}")
        try:
            result = parse_route(ParseRequest(**payload))
            if hasattr(result, "__await__"):
                import asyncio

                result = asyncio.run(result)
        except HTTPException as exc:
            return RouteResponse(exc.status_code, {"detail": exc.detail})
        return RouteResponse(200, result)

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

    def test_accept_solution_rejects_inconsistent_box_sizes(self):
        accepted, reason = accept_solution(
            top_score={
                "score": 0.92,
                "matched": [
                    {"text": "前", "bbox": [208, 63, 225, 88]},
                    {"text": "思", "bbox": [160, 80, 193, 119]},
                    {"text": "力", "bbox": [91, 78, 127, 117]},
                ],
            },
            runner_up_score=None,
            expected_match_count=3,
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "inconsistent_box_size")

    def test_parse_route_uses_best_proposal_set_for_targets(self):
        ocr_server_transformers.engine = FakeEngine()
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
        direct_candidates = [
            Candidate("今", proposals[0].boxes[0], 0.80),
            Candidate("入", proposals[0].boxes[1], 0.80),
            Candidate("心", proposals[0].boxes[2], 0.80),
        ]

        with (
            patch.object(ocr_server_transformers, "generate_box_proposals", return_value=proposals),
            patch.object(
                ocr_server_transformers,
                "recognize_box_candidates_with_recovery",
                side_effect=[
                    (direct_candidates, []),
                    AssertionError("second proposal should not run after early success"),
                ],
            ),
            patch.object(
                ocr_server_transformers,
                "solve_image_with_legacy_fallback",
                return_value={
                    "results": [],
                    "error": "ambiguous_top_score",
                    "method": "glm_ocr_transformers_with_local_positioning",
                    "image_size": [160, 60],
                },
            ) as fallback_mock,
        ):
            response = self.post_json(
                "/glmocr/parse",
                {"images": [payload], "targets": ["今", "入", "心"]},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual([item["text"] for item in body["results"]], ["今", "入", "心"])
        self.assertEqual(body["method"], "glm_ocr_transformers_with_local_positioning")
        fallback_mock.assert_not_called()

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

        with (
            patch.object(ocr_server_transformers, "generate_box_proposals", return_value=proposals),
            patch.object(
                ocr_server_transformers,
                "solve_image_with_legacy_fallback",
                return_value={
                    "results": [],
                    "error": "ambiguous_top_score",
                    "method": "glm_ocr_transformers_with_local_positioning",
                    "image_size": [160, 60],
                },
            ),
        ):
            response = self.post_json(
                "/glmocr/parse",
                {"images": [payload], "targets": ["今", "入", "心"]},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["results"], [])
        self.assertIn(body["error"], {"ambiguous_top_score", "incomplete_target_coverage"})

    def test_parse_route_recovers_failed_box_with_rotated_fallback(self):
        ocr_server_transformers.engine = FakeEngine()
        payload = make_png_data_uri(width=160, height=60)
        proposals = [
            ProposalSet(
                boxes=[[10, 10, 30, 30], [50, 10, 70, 30], [90, 10, 110, 30]],
                source="uniform_color_regions",
                preprocess_variant="whitened",
            )
        ]
        direct_candidates = [
            Candidate("今", proposals[0].boxes[0], 0.80),
            Candidate("心", proposals[0].boxes[2], 0.80),
        ]
        rotated_candidates = [Candidate("入", proposals[0].boxes[1], 0.50)]

        with (
            patch.object(ocr_server_transformers, "generate_box_proposals", return_value=proposals),
            patch.object(
                ocr_server_transformers,
                "recognize_box_candidates_with_recovery",
                return_value=(direct_candidates, [proposals[0].boxes[1]]),
            ),
            patch.object(
                ocr_server_transformers,
                "recognize_rotated_box_candidates",
                return_value=rotated_candidates,
            ),
            patch.object(ocr_server_transformers, "solve_image_with_legacy_fallback") as fallback_mock,
        ):
            response = self.post_json(
                "/glmocr/parse",
                {"images": [payload], "targets": ["今", "入", "心"]},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual([item["text"] for item in body["results"]], ["今", "入", "心"])
        fallback_mock.assert_not_called()

    def test_solve_image_invokes_legacy_fallback_when_current_path_rejects(self):
        image = Image.new("RGB", (160, 60), "white")
        proposal = ProposalSet(
            boxes=[[10, 10, 30, 30], [50, 10, 70, 30], [90, 10, 110, 30]],
            source="uniform_color_regions",
            preprocess_variant="whitened",
        )
        legacy_result = {
            "results": [
                {"text": "今", "bbox": [10, 10, 30, 30], "x": 20, "y": 20, "confidence": 0.5},
                {"text": "入", "bbox": [50, 10, 70, 30], "x": 60, "y": 20, "confidence": 0.5},
                {"text": "心", "bbox": [90, 10, 110, 30], "x": 100, "y": 20, "confidence": 0.5},
            ],
            "error": "ok",
            "method": "glm_ocr_transformers_with_local_positioning",
            "image_size": [160, 60],
        }

        with (
            patch.object(ocr_server_transformers, "generate_box_proposals", return_value=[proposal]),
            patch.object(
                ocr_server_transformers,
                "recognize_box_candidates_with_recovery",
                return_value=([], []),
            ),
            patch.object(ocr_server_transformers, "recognize_rotated_box_candidates", return_value=[]),
            patch.object(
                ocr_server_transformers,
                "solve_image_with_legacy_fallback",
                return_value=legacy_result,
            ) as fallback_mock,
        ):
            result = ocr_server_transformers.solve_image(image, ["今", "入", "心"])

        self.assertEqual(result, legacy_result)
        fallback_mock.assert_called_once()
        self.assertEqual(fallback_mock.call_args.args[1], ["今", "入", "心"])
        self.assertEqual(fallback_mock.call_args.args[2], "incomplete_target_coverage")

    def test_legacy_fallback_can_rescue_from_full_image_and_strip_views(self):
        ocr_server_transformers.engine = FakeRecordingEngine(
            outputs=[
                "识别结果：心入今",
                "识别结果：今入心",
            ]
        )
        image = Image.new("RGB", (160, 60), "white")
        strip_image = Image.new("RGB", (200, 60), "white")
        boxes = [[10, 10, 30, 30], [50, 10, 70, 30], [90, 10, 110, 30]]

        with (
            patch.object(ocr_server_transformers, "detect_colored_text_bboxes", return_value=boxes),
            patch.object(
                ocr_server_transformers,
                "generate_box_proposals",
                return_value=[
                    ProposalSet(boxes=boxes, source="uniform_color_regions", preprocess_variant="whitened")
                ],
            ),
            patch.object(ocr_server_transformers, "recognize_box_crops_legacy", return_value=[]),
            patch.object(ocr_server_transformers, "refine_bbox_to_dark_pixels", side_effect=lambda _image, box: box),
            patch.object(ocr_server_transformers, "build_colored_text_strip", return_value=strip_image),
            patch.object(ocr_server_transformers, "recognize_rotated_box_candidates_legacy", return_value=[]),
        ):
            result = ocr_server_transformers.solve_image_with_legacy_fallback(
                image,
                ["今", "入", "心"],
                "incomplete_target_coverage",
            )

        self.assertEqual(result["error"], "ok")
        self.assertEqual([item["text"] for item in result["results"]], ["今", "入", "心"])
        self.assertEqual(result["method"], "glm_ocr_transformers_with_local_positioning")

    def test_legacy_full_image_output_without_explicit_boxes_is_not_attached_by_order(self):
        image = Image.new("RGB", (160, 60), "white")
        boxes = [[10, 10, 30, 30], [50, 10, 70, 30], [90, 10, 110, 30]]

        candidates = ocr_server_transformers.candidates_from_legacy_output(
            image,
            "识别结果：心入今",
            boxes,
            allow_index_attachment=False,
        )

        self.assertEqual(candidates, [])

    def test_legacy_fallback_can_use_order_attached_original_output_as_last_resort(self):
        ocr_server_transformers.engine = FakeRecordingEngine(
            outputs=[
                "识别结果：长方科",
                "noise",
                "noise",
                "noise",
                "noise",
            ]
        )
        image = Image.new("RGB", (160, 60), "white")
        boxes = [[10, 10, 30, 30], [50, 10, 70, 30], [90, 10, 110, 30]]

        with (
            patch.object(ocr_server_transformers, "detect_colored_text_bboxes", return_value=boxes),
            patch.object(
                ocr_server_transformers,
                "generate_box_proposals",
                return_value=[
                    ProposalSet(boxes=boxes, source="uniform_color_regions", preprocess_variant="whitened")
                ],
            ),
            patch.object(ocr_server_transformers, "recognize_box_crops_legacy", return_value=[]),
            patch.object(ocr_server_transformers, "refine_bbox_to_dark_pixels", side_effect=lambda _image, box: box),
            patch.object(ocr_server_transformers, "build_colored_text_strip", return_value=image),
            patch.object(ocr_server_transformers, "recognize_rotated_box_candidates_legacy", return_value=[]),
        ):
            result = ocr_server_transformers.solve_image_with_legacy_fallback(
                image,
                ["长", "方", "科"],
                "incomplete_target_coverage",
            )

        self.assertEqual(result["error"], "ok")
        self.assertEqual([item["text"] for item in result["results"]], ["长", "方", "科"])

    def test_build_aligned_strip_candidates_prefers_direct_box_labels_for_subset_alignment(self):
        image = Image.new("RGB", (300, 120), "white")
        boxes = [
            [12, 24, 45, 64],
            [77, 128, 102, 155],
            [95, 47, 135, 85],
            [116, 98, 152, 147],
            [120, 97, 169, 132],
            [164, 26, 198, 61],
            [235, 57, 268, 91],
        ]
        direct_candidates = [
            Candidate("打", boxes[0], 0.60),
            Candidate("阵", boxes[2], 0.60),
            Candidate("目", boxes[5], 0.60),
            Candidate("合", boxes[6], 0.60),
        ]

        candidates = build_aligned_strip_candidates(
            image,
            "识别结果：打阵日合",
            boxes,
            direct_candidates,
            ["打", "日", "合"],
        )

        self.assertEqual([candidate.text for candidate in candidates], ["打", "阵", "日", "合"])
        self.assertEqual([candidate.bbox for candidate in candidates], [boxes[0], boxes[2], boxes[5], boxes[6]])

    def test_build_aligned_strip_candidates_can_fill_single_unused_box_for_missing_target(self):
        image = Image.new("RGB", (320, 160), "white")
        boxes = [
            [37, 104, 66, 136],
            [108, 77, 140, 112],
            [169, 34, 209, 75],
            [228, 20, 261, 54],
        ]
        direct_candidates = [Candidate("入", boxes[1], 0.60)]

        candidates = build_aligned_strip_candidates(
            image,
            "识别结果：入率今",
            boxes,
            direct_candidates,
            ["今", "入", "心"],
        )

        self.assertEqual([candidate.text for candidate in candidates], ["心", "入", "率", "今"])
        self.assertEqual([candidate.bbox for candidate in candidates], boxes)

    def test_dedupe_candidates_merges_identical_text_and_bbox(self):
        candidates = [
            Candidate("长", [221, 10, 253, 46], 0.50),
            Candidate("方", [143, 71, 183, 105], 0.60),
            Candidate("长", [221, 10, 253, 46], 0.60),
            Candidate("科", [108, 98, 142, 133], 0.60),
        ]

        deduped = dedupe_candidates(candidates)

        self.assertEqual(
            deduped,
            [
                Candidate("长", [221, 10, 253, 46], 0.60),
                Candidate("方", [143, 71, 183, 105], 0.60),
                Candidate("科", [108, 98, 142, 133], 0.60),
            ],
        )

    def test_dedupe_candidates_prefers_larger_overlapping_box_for_same_text(self):
        candidates = [
            Candidate("前", [208, 63, 225, 88], 0.60),
            Candidate("前", [202, 51, 238, 88], 0.60),
            Candidate("思", [160, 80, 193, 119], 0.60),
        ]

        deduped = dedupe_candidates(candidates)

        self.assertEqual(
            deduped,
            [
                Candidate("前", [202, 51, 238, 88], 0.60),
                Candidate("思", [160, 80, 193, 119], 0.60),
            ],
        )

    def test_dedupe_candidates_merges_contained_same_text_candidates(self):
        candidates = [
            Candidate("前", [202, 51, 238, 88], 0.60),
            Candidate("力", [91, 78, 127, 117], 0.60),
            Candidate("思", [91, 78, 190, 136], 0.50),
            Candidate("思", [160, 80, 193, 119], 0.60),
        ]

        deduped = dedupe_candidates(candidates)

        self.assertEqual(
            deduped,
            [
                Candidate("前", [202, 51, 238, 88], 0.60),
                Candidate("力", [91, 78, 127, 117], 0.60),
                Candidate("思", [160, 80, 193, 119], 0.60),
            ],
        )

    def test_choose_better_legacy_result_prefers_higher_total_confidence(self):
        lower = {
            "results": [
                {"text": "长", "bbox": [221, 10, 253, 46], "confidence": 0.5},
                {"text": "方", "bbox": [143, 71, 183, 105], "confidence": 0.5},
                {"text": "科", "bbox": [108, 98, 142, 133], "confidence": 0.5},
            ],
            "error": "ok",
        }
        higher = {
            "results": [
                {"text": "长", "bbox": [221, 10, 253, 46], "confidence": 0.6},
                {"text": "方", "bbox": [143, 71, 183, 105], "confidence": 0.6},
                {"text": "科", "bbox": [108, 98, 142, 133], "confidence": 0.6},
            ],
            "error": "ok",
        }

        self.assertIs(choose_better_legacy_result(None, lower), lower)
        self.assertIs(choose_better_legacy_result(lower, higher), higher)
        self.assertIs(choose_better_legacy_result(higher, lower), higher)

    def test_choose_better_legacy_result_rejects_inconsistent_box_sizes_even_with_higher_confidence(self):
        consistent = {
            "results": [
                {"text": "长", "bbox": [221, 10, 253, 46], "confidence": 0.55},
                {"text": "方", "bbox": [143, 71, 183, 105], "confidence": 0.55},
                {"text": "科", "bbox": [108, 98, 142, 133], "confidence": 0.55},
            ],
            "error": "ok",
        }
        inconsistent = {
            "results": [
                {"text": "长", "bbox": [221, 10, 253, 46], "confidence": 0.70},
                {"text": "方", "bbox": [143, 71, 183, 105], "confidence": 0.70},
                {"text": "科", "bbox": [108, 96, 182, 133], "confidence": 0.70},
            ],
            "error": "ok",
        }

        self.assertIs(choose_better_legacy_result(consistent, inconsistent), consistent)
        self.assertIs(choose_better_legacy_result(inconsistent, consistent), consistent)

    def test_solve_image_skips_rotated_recovery_when_direct_proposals_are_already_accepted(self):
        image = Image.new("RGB", (160, 60), "white")
        proposals = [
            ProposalSet(
                boxes=[[10, 10, 30, 30], [50, 10, 70, 30], [90, 10, 110, 30]],
                source="uniform_color_regions",
                preprocess_variant="whitened",
            ),
            ProposalSet(
                boxes=[[12, 10, 32, 30], [52, 10, 72, 30], [92, 10, 112, 30]],
                source="dark_regions",
                preprocess_variant="whitened",
            ),
        ]

        direct_candidates = [
            Candidate("今", proposals[0].boxes[0], 0.80),
            Candidate("入", proposals[0].boxes[1], 0.80),
            Candidate("心", proposals[0].boxes[2], 0.80),
        ]

        with (
            patch.object(ocr_server_transformers, "generate_box_proposals", return_value=proposals),
            patch.object(
                ocr_server_transformers,
                "recognize_box_candidates_with_recovery",
                side_effect=[
                    (direct_candidates, []),
                    ([], proposals[1].boxes),
                ],
            ),
            patch.object(ocr_server_transformers, "recognize_rotated_box_candidates") as rotated_mock,
            patch.object(ocr_server_transformers, "solve_image_with_legacy_fallback") as fallback_mock,
        ):
            result = ocr_server_transformers.solve_image(image, ["今", "入", "心"])

        self.assertEqual(result["error"], "ok")
        self.assertEqual([item["text"] for item in result["results"]], ["今", "入", "心"])
        rotated_mock.assert_not_called()
        fallback_mock.assert_not_called()

    def test_solve_image_returns_early_after_safe_strong_direct_hit(self):
        image = Image.new("RGB", (160, 60), "white")
        proposals = [
            ProposalSet(
                boxes=[[10, 10, 30, 30], [50, 10, 70, 30], [90, 10, 110, 30]],
                source="uniform_color_regions",
                preprocess_variant="whitened",
            ),
            ProposalSet(
                boxes=[[12, 10, 32, 30], [52, 10, 72, 30], [92, 10, 112, 30]],
                source="isolated_uniform",
                preprocess_variant="isolated",
            ),
        ]
        direct_candidates = [
            Candidate("今", proposals[0].boxes[0], 0.80),
            Candidate("入", proposals[0].boxes[1], 0.80),
            Candidate("心", proposals[0].boxes[2], 0.80),
        ]

        def unexpected_second_proposal(*_args, **_kwargs):
            raise AssertionError("second proposal should not be evaluated after strong direct hit")

        with (
            patch.object(ocr_server_transformers, "generate_box_proposals", return_value=proposals),
            patch.object(
                ocr_server_transformers,
                "recognize_box_candidates_with_recovery",
                side_effect=[
                    (direct_candidates, []),
                    unexpected_second_proposal,
                ],
            ),
            patch.object(ocr_server_transformers, "solve_image_with_legacy_fallback") as fallback_mock,
        ):
            result = ocr_server_transformers.solve_image(image, ["今", "入", "心"])

        self.assertEqual(result["error"], "ok")
        self.assertEqual([item["text"] for item in result["results"]], ["今", "入", "心"])
        fallback_mock.assert_not_called()

    def test_solve_image_reuses_cached_ocr_outputs_across_proposals(self):
        image = Image.new("RGB", (160, 60), "white")
        proposals = [
            ProposalSet(
                boxes=[[10, 10, 30, 30], [50, 10, 70, 30], [90, 10, 110, 30], [120, 10, 140, 30]],
                source="uniform_color_regions",
                preprocess_variant="whitened",
            ),
            ProposalSet(
                boxes=[[10, 10, 30, 30], [50, 10, 70, 30], [90, 10, 110, 30], [140, 10, 160, 30]],
                source="isolated_uniform",
                preprocess_variant="isolated",
            ),
        ]

        ocr_server_transformers.engine = Mock()
        ocr_server_transformers.engine.recognize = Mock(
            side_effect=lambda payload: {
                b"box:10,10,30,30": "今",
                b"box:50,10,70,30": "入",
                b"box:90,10,110,30": "心",
                b"box:120,10,140,30": "noise",
                b"box:140,10,160,30": "noise",
            }[payload]
        )

        def recognize_side_effect(_image, boxes, targets, recognizer=None, batch_recognizer=None, recognize_cache=None, stop_on_full_target_coverage=False):
            candidates = []
            failed = []
            for box in boxes:
                payload = f"box:{','.join(str(value) for value in box)}".encode("ascii")
                output = recognizer(payload)
                if output in set(targets):
                    candidates.append(Candidate(output, box, 0.80))
                else:
                    failed.append(box)
            return candidates, failed

        with (
            patch.object(ocr_server_transformers, "generate_box_proposals", return_value=proposals),
            patch.object(
                ocr_server_transformers,
                "recognize_box_candidates_with_recovery",
                side_effect=recognize_side_effect,
            ),
            patch.object(ocr_server_transformers, "recognize_rotated_box_candidates", return_value=[]),
            patch.object(
                ocr_server_transformers,
                "solve_image_with_legacy_fallback",
                return_value={
                    "error": "ok",
                    "results": [
                        {"text": "今", "bbox": [10, 10, 30, 30], "x": 20, "y": 20, "confidence": 0.8},
                        {"text": "入", "bbox": [50, 10, 70, 30], "x": 60, "y": 20, "confidence": 0.8},
                        {"text": "心", "bbox": [90, 10, 110, 30], "x": 100, "y": 20, "confidence": 0.8},
                    ],
                    "method": "glm_ocr_transformers_with_local_positioning",
                    "image_size": [160, 60],
                },
            ) as fallback_mock,
        ):
            result = ocr_server_transformers.solve_image(image, ["今", "入", "心"])

        self.assertEqual(result["error"], "ok")
        self.assertEqual(ocr_server_transformers.engine.recognize.call_count, 5)
        fallback_mock.assert_called_once()

    def test_solve_image_skips_fully_redundant_proposals(self):
        image = Image.new("RGB", (160, 60), "white")
        boxes = [[10, 10, 30, 30], [50, 10, 70, 30], [90, 10, 110, 30], [120, 10, 140, 30]]
        proposals = [
            ProposalSet(boxes=boxes, source="uniform_color_regions", preprocess_variant="whitened"),
            ProposalSet(boxes=list(boxes), source="isolated_uniform", preprocess_variant="isolated"),
        ]

        ocr_server_transformers.engine = Mock()
        ocr_server_transformers.engine.recognize = Mock(
            side_effect=lambda payload: {
                b"box:10,10,30,30": "今",
                b"box:50,10,70,30": "入",
                b"box:90,10,110,30": "心",
                b"box:120,10,140,30": "noise",
            }[payload]
        )

        def recognize_side_effect(_image, boxes, targets, recognizer=None, batch_recognizer=None, recognize_cache=None, stop_on_full_target_coverage=False):
            candidates = []
            failed = []
            for box in boxes:
                payload = f"box:{','.join(str(value) for value in box)}".encode("ascii")
                output = recognizer(payload)
                if output in set(targets):
                    candidates.append(Candidate(output, box, 0.80))
                else:
                    failed.append(box)
            return candidates, failed

        with (
            patch.object(ocr_server_transformers, "generate_box_proposals", return_value=proposals),
            patch.object(
                ocr_server_transformers,
                "recognize_box_candidates_with_recovery",
                side_effect=recognize_side_effect,
            ) as recognize_mock,
            patch.object(ocr_server_transformers, "recognize_rotated_box_candidates", return_value=[]),
            patch.object(
                ocr_server_transformers,
                "solve_image_with_legacy_fallback",
                return_value={"error": "incomplete_target_coverage", "results": [], "image_size": [160, 60]},
            ),
        ):
            ocr_server_transformers.solve_image(image, ["今", "入", "心"])

        self.assertEqual(ocr_server_transformers.engine.recognize.call_count, 4)
        self.assertEqual(recognize_mock.call_count, 1)

    def test_solve_image_only_recognizes_novel_boxes_in_later_proposals(self):
        image = Image.new("RGB", (160, 60), "white")
        proposals = [
            ProposalSet(
                boxes=[[10, 10, 30, 30], [50, 10, 70, 30], [90, 10, 110, 30], [120, 10, 140, 30]],
                source="uniform_color_regions",
                preprocess_variant="whitened",
            ),
            ProposalSet(
                boxes=[[10, 10, 30, 30], [50, 10, 70, 30], [90, 10, 110, 30], [140, 10, 160, 30]],
                source="isolated_uniform",
                preprocess_variant="isolated",
            ),
        ]

        ocr_server_transformers.engine = Mock()
        ocr_server_transformers.engine.recognize = Mock(
            side_effect=lambda payload: {
                b"box:10,10,30,30": "今",
                b"box:50,10,70,30": "入",
                b"box:90,10,110,30": "心",
                b"box:120,10,140,30": "noise",
                b"box:140,10,160,30": "noise",
            }[payload]
        )

        def recognize_side_effect(_image, boxes, targets, recognizer=None, batch_recognizer=None, recognize_cache=None, stop_on_full_target_coverage=False):
            candidates = []
            failed = []
            for box in boxes:
                payload = f"box:{','.join(str(value) for value in box)}".encode("ascii")
                output = recognizer(payload)
                if output in set(targets):
                    candidates.append(Candidate(output, box, 0.80))
                else:
                    failed.append(box)
            return candidates, failed

        with (
            patch.object(ocr_server_transformers, "generate_box_proposals", return_value=proposals),
            patch.object(
                ocr_server_transformers,
                "recognize_box_candidates_with_recovery",
                side_effect=recognize_side_effect,
            ) as recognize_mock,
            patch.object(ocr_server_transformers, "recognize_rotated_box_candidates", return_value=[]),
            patch.object(
                ocr_server_transformers,
                "solve_image_with_legacy_fallback",
                return_value={"error": "incomplete_target_coverage", "results": [], "image_size": [160, 60]},
            ),
        ):
            ocr_server_transformers.solve_image(image, ["今", "入", "心"])

        self.assertEqual(ocr_server_transformers.engine.recognize.call_count, 5)
        self.assertEqual(recognize_mock.call_count, 2)

    def test_solve_image_prioritizes_higher_quality_proposals_before_noisy_ones(self):
        image = Image.new("RGB", (160, 60), "white")
        noisy = ProposalSet(
            boxes=[
                [10, 10, 30, 30],
                [50, 10, 70, 30],
                [90, 10, 110, 30],
                [130, 10, 150, 30],
                [10, 35, 20, 44],
                [140, 35, 155, 55],
            ],
            source="uniform_color_regions",
            preprocess_variant="whitened",
        )
        clean = ProposalSet(
            boxes=[[12, 12, 34, 34], [52, 12, 74, 34], [92, 12, 114, 34]],
            source="isolated_uniform",
            preprocess_variant="isolated",
        )
        call_order = []

        def recognize_side_effect(_image, boxes, targets, recognizer=None, batch_recognizer=None, recognize_cache=None, stop_on_full_target_coverage=False):
            call_order.append(boxes)
            if boxes == clean.boxes:
                return (
                    [
                        Candidate("今", clean.boxes[0], 0.80),
                        Candidate("入", clean.boxes[1], 0.80),
                        Candidate("心", clean.boxes[2], 0.80),
                    ],
                    [],
                )
            return ([], boxes)

        with (
            patch.object(ocr_server_transformers, "generate_box_proposals", return_value=[noisy, clean]),
            patch.object(
                ocr_server_transformers,
                "recognize_box_candidates_with_recovery",
                side_effect=recognize_side_effect,
            ),
            patch.object(ocr_server_transformers, "solve_image_with_legacy_fallback") as fallback_mock,
        ):
            result = ocr_server_transformers.solve_image(image, ["今", "入", "心"])

        self.assertEqual(result["error"], "ok")
        self.assertEqual(call_order[0], clean.boxes)
        fallback_mock.assert_not_called()

    def test_recognize_box_candidates_stops_after_covering_all_targets(self):
        image = Image.new("RGB", (160, 60), "white")
        boxes = [
            [10, 10, 30, 30],
            [50, 10, 70, 30],
            [90, 10, 110, 30],
            [130, 10, 150, 30],
            [10, 35, 30, 55],
        ]
        calls = []

        def recognizer(payload):
            calls.append(payload)
            outputs = {
                b"box:10,10,30,30": "今",
                b"box:50,10,70,30": "入",
                b"box:90,10,110,30": "心",
                b"box:130,10,150,30": "噪",
                b"box:10,35,30,55": "声",
            }
            return outputs[payload]

        payloads = iter(
            [
                b"box:10,10,30,30",
                b"box:50,10,70,30",
                b"box:90,10,110,30",
            ]
        )
        with patch.object(
            ocr_server_transformers,
            "image_to_png_bytes",
            side_effect=lambda _crop: next(payloads),
        ):
            candidates, failed = ocr_server_transformers.recognize_box_candidates_with_recovery(
                image,
                boxes,
                ["今", "入", "心"],
                recognizer=recognizer,
                stop_on_full_target_coverage=True,
            )

        self.assertEqual([candidate.text for candidate in candidates], ["今", "入", "心"])
        self.assertEqual(failed, [])
        self.assertEqual(len(calls), 3)

    def test_recognize_box_candidates_uses_batch_recognizer_for_crop_chunks(self):
        image = Image.new("RGB", (160, 60), "white")
        boxes = [
            [10, 10, 30, 30],
            [50, 10, 70, 30],
            [90, 10, 110, 30],
            [130, 10, 150, 30],
        ]
        single_recognizer = Mock(side_effect=AssertionError("single recognizer should not be used"))
        batch_calls = []

        def batch_recognizer(payloads, _targets=None):
            batch_calls.append(list(payloads))
            outputs = {
                b"box:10,10,30,30": "今",
                b"box:50,10,70,30": "入",
                b"box:90,10,110,30": "心",
                b"box:130,10,150,30": "噪",
            }
            return [outputs[payload] for payload in payloads]

        payloads = iter(
            [
                b"box:10,10,30,30",
                b"box:50,10,70,30",
                b"box:90,10,110,30",
            ]
        )
        with patch.object(
            ocr_server_transformers,
            "image_to_png_bytes",
            side_effect=lambda _crop: next(payloads),
        ):
            candidates, failed = ocr_server_transformers.recognize_box_candidates_with_recovery(
                image,
                boxes,
                ["今", "入", "心"],
                recognizer=single_recognizer,
                batch_recognizer=batch_recognizer,
                stop_on_full_target_coverage=True,
            )

        self.assertEqual([candidate.text for candidate in candidates], ["今", "入", "心"])
        self.assertEqual(failed, [])
        self.assertEqual(len(batch_calls), 1)
        single_recognizer.assert_not_called()

    def test_solve_image_prefilters_clear_geometry_outlier_boxes_before_ocr(self):
        image = Image.new("RGB", (160, 60), "white")
        boxes = [
            [10, 10, 30, 30],
            [50, 10, 70, 30],
            [90, 10, 110, 30],
            [140, 10, 146, 17],
        ]
        proposals = [
            ProposalSet(boxes=boxes, source="uniform_color_regions", preprocess_variant="whitened"),
        ]

        ocr_server_transformers.engine = Mock()
        ocr_server_transformers.engine.recognize = Mock(
            side_effect=lambda payload: {
                b"box:10,10,30,30": "今",
                b"box:50,10,70,30": "入",
                b"box:90,10,110,30": "心",
            }[payload]
        )
        observed_boxes = []

        def recognize_side_effect(_image, candidate_boxes, targets, recognizer=None, batch_recognizer=None, recognize_cache=None, stop_on_full_target_coverage=False):
            observed_boxes.extend(candidate_boxes)
            candidates = []
            for box in candidate_boxes:
                payload = f"box:{','.join(str(value) for value in box)}".encode("ascii")
                output = recognizer(payload)
                if output in set(targets):
                    candidates.append(Candidate(output, box, 0.80))
            return candidates, []

        with (
            patch.object(ocr_server_transformers, "generate_box_proposals", return_value=proposals),
            patch.object(
                ocr_server_transformers,
                "recognize_box_candidates_with_recovery",
                side_effect=recognize_side_effect,
            ),
            patch.object(ocr_server_transformers, "solve_image_with_legacy_fallback") as fallback_mock,
        ):
            result = ocr_server_transformers.solve_image(image, ["今", "入", "心"])

        self.assertEqual(result["error"], "ok")
        self.assertEqual(observed_boxes, boxes[:3])
        self.assertEqual(ocr_server_transformers.engine.recognize.call_count, 3)
        fallback_mock.assert_not_called()

    def test_legacy_fallback_does_not_attach_strip_output_when_box_count_exceeds_targets(self):
        ocr_server_transformers.engine = FakeRecordingEngine(
            outputs=[
                "noise",
                "noise",
                "识别结果：包力思前",
                "noise",
                "noise",
                "noise",
                "noise",
            ]
        )
        image = Image.new("RGB", (160, 60), "white")
        boxes = [[10, 10, 30, 30], [40, 10, 60, 30], [70, 10, 90, 30], [100, 10, 120, 30]]

        with (
            patch.object(ocr_server_transformers, "detect_colored_text_bboxes", return_value=boxes),
            patch.object(
                ocr_server_transformers,
                "generate_box_proposals",
                return_value=[
                    ProposalSet(boxes=boxes, source="uniform_color_regions", preprocess_variant="whitened")
                ],
            ),
            patch.object(ocr_server_transformers, "refine_bbox_to_dark_pixels", side_effect=lambda _image, box: box),
            patch.object(ocr_server_transformers, "build_colored_text_strip", return_value=image),
            patch.object(ocr_server_transformers, "recognize_rotated_box_candidates_legacy", return_value=[]),
        ):
            result = ocr_server_transformers.solve_image_with_legacy_fallback(
                image,
                ["前", "思", "力"],
                "incomplete_target_coverage",
            )

        self.assertEqual(result["error"], "incomplete_target_coverage")
        self.assertEqual(result["results"], [])

    def test_legacy_fallback_can_retry_with_proposal_boxes_when_legacy_boxes_are_incomplete(self):
        ocr_server_transformers.engine = FakeRecordingEngine(outputs=["noise"] * 20)
        image = Image.new("RGB", (160, 60), "white")
        legacy_boxes = [[10, 10, 30, 30], [50, 10, 70, 30]]
        proposal_boxes = [[10, 10, 30, 30], [50, 10, 70, 30], [90, 10, 110, 30]]
        proposals = [
            ProposalSet(boxes=legacy_boxes, source="uniform_color_regions", preprocess_variant="whitened"),
            ProposalSet(boxes=proposal_boxes, source="dark_regions", preprocess_variant="whitened"),
        ]

        def recognize_side_effect(_image, boxes, padding=10, recognizer=None):
            if boxes == proposal_boxes:
                return [
                    Candidate("今", proposal_boxes[0], 0.60),
                    Candidate("入", proposal_boxes[1], 0.60),
                    Candidate("心", proposal_boxes[2], 0.60),
                ]
            return []

        with (
            patch.object(ocr_server_transformers, "detect_colored_text_bboxes", return_value=legacy_boxes),
            patch.object(ocr_server_transformers, "generate_box_proposals", return_value=proposals),
            patch.object(ocr_server_transformers, "recognize_box_crops_legacy", side_effect=recognize_side_effect),
            patch.object(ocr_server_transformers, "refine_bbox_to_dark_pixels", side_effect=lambda _image, box: box),
            patch.object(ocr_server_transformers, "recognize_rotated_box_candidates_legacy", return_value=[]),
        ):
            result = ocr_server_transformers.solve_image_with_legacy_fallback(
                image,
                ["今", "入", "心"],
                "incomplete_target_coverage",
            )

        self.assertEqual(result["error"], "ok")
        self.assertEqual([item["text"] for item in result["results"]], ["今", "入", "心"])

    def test_legacy_fallback_can_merge_candidates_across_box_sets(self):
        image = Image.new("RGB", (160, 60), "white")
        ocr_server_transformers.engine = FakeRecordingEngine(outputs=["noise"] * 10)
        legacy_boxes = [[10, 10, 30, 30], [50, 10, 70, 30]]
        proposal_boxes = [[50, 10, 70, 30], [90, 10, 110, 30]]
        proposals = [
            ProposalSet(boxes=legacy_boxes, source="uniform_color_regions", preprocess_variant="whitened"),
            ProposalSet(boxes=proposal_boxes, source="dark_regions", preprocess_variant="whitened"),
        ]

        def recognize_side_effect(_image, boxes, padding=10, recognizer=None):
            if boxes == legacy_boxes:
                return []
            if boxes == proposal_boxes:
                return [
                    Candidate("入", proposal_boxes[0], 0.60),
                    Candidate("今", proposal_boxes[1], 0.60),
                ]
            return []

        def rotated_side_effect(_image, targets, boxes, angles=(0, 90, 180, 270), recognizer=None):
            if boxes == legacy_boxes:
                return [Candidate("心", legacy_boxes[0], 0.50)]
            return []

        with (
            patch.object(ocr_server_transformers, "detect_colored_text_bboxes", return_value=legacy_boxes),
            patch.object(ocr_server_transformers, "generate_box_proposals", return_value=proposals),
            patch.object(ocr_server_transformers, "recognize_box_crops_legacy", side_effect=recognize_side_effect),
            patch.object(ocr_server_transformers, "recognize_rotated_box_candidates_legacy", side_effect=rotated_side_effect),
            patch.object(ocr_server_transformers, "refine_bbox_to_dark_pixels", side_effect=lambda _image, box: box),
            patch.object(ocr_server_transformers, "prepare_recognition_image_bytes", return_value=b"strip"),
        ):
            result = ocr_server_transformers.solve_image_with_legacy_fallback(
                image,
                ["今", "入", "心"],
                "incomplete_target_coverage",
            )

        self.assertEqual(result["error"], "ok")
        self.assertEqual([item["text"] for item in result["results"]], ["今", "入", "心"])

    def test_legacy_fallback_can_merge_aligned_strip_candidates_across_box_sets(self):
        image = Image.new("RGB", (160, 60), "white")
        ocr_server_transformers.engine = FakeRecordingEngine(outputs=["noise"] * 10)
        legacy_boxes = [[10, 10, 30, 30]]
        proposal_boxes = [[50, 10, 70, 30], [90, 10, 110, 30]]
        proposals = [
            ProposalSet(boxes=legacy_boxes, source="uniform_color_regions", preprocess_variant="whitened"),
            ProposalSet(boxes=proposal_boxes, source="dark_regions", preprocess_variant="whitened"),
        ]

        def aligned_side_effect(_image, _strip_output, boxes, _direct_candidates, _targets):
            if boxes == legacy_boxes:
                return [Candidate("心", legacy_boxes[0], 0.60)]
            if boxes == proposal_boxes:
                return [
                    Candidate("入", proposal_boxes[0], 0.60),
                    Candidate("今", proposal_boxes[1], 0.60),
                ]
            return []

        with (
            patch.object(ocr_server_transformers, "detect_colored_text_bboxes", return_value=legacy_boxes),
            patch.object(ocr_server_transformers, "generate_box_proposals", return_value=proposals),
            patch.object(ocr_server_transformers, "recognize_box_crops_legacy", return_value=[]),
            patch.object(ocr_server_transformers, "candidates_from_legacy_output", return_value=[]),
            patch.object(ocr_server_transformers, "build_aligned_strip_candidates", side_effect=aligned_side_effect),
            patch.object(ocr_server_transformers, "recognize_rotated_box_candidates_legacy", return_value=[]),
            patch.object(ocr_server_transformers, "prepare_recognition_image_bytes", return_value=b"strip"),
        ):
            result = ocr_server_transformers.solve_image_with_legacy_fallback(
                image,
                ["今", "入", "心"],
                "incomplete_target_coverage",
            )

        self.assertEqual(result["error"], "ok")
        self.assertEqual([item["text"] for item in result["results"]], ["今", "入", "心"])

    def test_legacy_fallback_can_merge_safe_order_attached_candidates_across_box_sets(self):
        image = Image.new("RGB", (160, 60), "white")
        ocr_server_transformers.engine = FakeRecordingEngine(outputs=["noise"] * 10)
        legacy_boxes = [[10, 10, 30, 30], [40, 10, 60, 30], [70, 10, 90, 30]]
        proposal_boxes = [[100, 10, 120, 30], [130, 10, 150, 30], [160, 10, 180, 30]]
        proposals = [
            ProposalSet(boxes=legacy_boxes, source="uniform_color_regions", preprocess_variant="whitened"),
            ProposalSet(boxes=proposal_boxes, source="dark_regions", preprocess_variant="whitened"),
        ]

        def legacy_output_side_effect(_image, output, boxes, allow_index_attachment=False):
            if not allow_index_attachment:
                return []
            if boxes == legacy_boxes:
                return [Candidate("心", legacy_boxes[0], 0.50)]
            if boxes == proposal_boxes:
                return [
                    Candidate("入", proposal_boxes[0], 0.50),
                    Candidate("今", proposal_boxes[1], 0.50),
                ]
            return []

        with (
            patch.object(ocr_server_transformers, "detect_colored_text_bboxes", return_value=legacy_boxes),
            patch.object(ocr_server_transformers, "generate_box_proposals", return_value=proposals),
            patch.object(ocr_server_transformers, "recognize_box_crops_legacy", return_value=[]),
            patch.object(ocr_server_transformers, "candidates_from_legacy_output", side_effect=legacy_output_side_effect),
            patch.object(ocr_server_transformers, "build_aligned_strip_candidates", return_value=[]),
            patch.object(ocr_server_transformers, "recognize_rotated_box_candidates_legacy", return_value=[]),
            patch.object(ocr_server_transformers, "prepare_recognition_image_bytes", return_value=b"strip"),
        ):
            result = ocr_server_transformers.solve_image_with_legacy_fallback(
                image,
                ["今", "入", "心"],
                "incomplete_target_coverage",
            )

        self.assertEqual(result["error"], "ok")
        self.assertEqual([item["text"] for item in result["results"]], ["今", "入", "心"])

    def test_legacy_fallback_skips_expensive_views_when_direct_box_candidates_are_already_safe(self):
        image = Image.new("RGB", (160, 60), "white")
        boxes = [[10, 10, 30, 30], [50, 10, 70, 30], [90, 10, 110, 30]]
        ocr_server_transformers.engine = Mock()
        ocr_server_transformers.engine.recognize = Mock()

        direct_candidates = [
            Candidate("今", boxes[0], 0.60),
            Candidate("入", boxes[1], 0.60),
            Candidate("心", boxes[2], 0.60),
        ]

        with (
            patch.object(ocr_server_transformers, "detect_colored_text_bboxes", return_value=boxes),
            patch.object(
                ocr_server_transformers,
                "generate_box_proposals",
                return_value=[ProposalSet(boxes=boxes, source="uniform_color_regions", preprocess_variant="whitened")],
            ),
            patch.object(ocr_server_transformers, "recognize_box_crops_legacy", return_value=direct_candidates),
            patch.object(ocr_server_transformers, "candidates_from_legacy_output") as original_candidates_mock,
            patch.object(ocr_server_transformers, "build_aligned_strip_candidates") as aligned_mock,
            patch.object(ocr_server_transformers, "prepare_recognition_image_bytes") as strip_mock,
            patch.object(ocr_server_transformers, "recognize_rotated_box_candidates_legacy") as rotated_mock,
        ):
            result = ocr_server_transformers.solve_image_with_legacy_fallback(
                image,
                ["今", "入", "心"],
                "incomplete_target_coverage",
            )

        self.assertEqual(result["error"], "ok")
        self.assertEqual([item["text"] for item in result["results"]], ["今", "入", "心"])
        ocr_server_transformers.engine.recognize.assert_not_called()
        original_candidates_mock.assert_not_called()
        aligned_mock.assert_not_called()
        strip_mock.assert_not_called()
        rotated_mock.assert_not_called()

    def test_parse_route_rejects_invalid_image_data(self):
        ocr_server_transformers.engine = FakeEngine()

        response = self.post_json("/glmocr/parse", {"images": ["not-base64"], "targets": ["件"]})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Invalid image data")


if __name__ == "__main__":
    unittest.main()
