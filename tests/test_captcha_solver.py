import os
import unittest
from unittest.mock import patch

from captcha_solver import CaptchaSolveError, CaptchaSolver
from ocr_server_transformers import GlmOcrEngine


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeInputIds:
    shape = (1, 2)


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


class FakeProcessor:
    def __init__(self):
        self.inputs = FakeInputs()
        self.messages = None
        self.kwargs = None
        self.image_path_existed = False
        self.decode_args = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        image_path = messages[0]["content"][0]["url"]
        self.image_path_existed = os.path.exists(image_path)
        return self.inputs

    def decode(self, output_ids, skip_special_tokens=False):
        self.decode_args = (output_ids, skip_special_tokens)
        return "decoded-output"


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[101, 102, 201, 202]]


class FakeElement:
    size = {"width": 224, "height": 179}
    rect = {"width": 224, "height": 179}


class FakeActions:
    instances = []

    def __init__(self, driver):
        self.driver = driver
        self.calls = []
        FakeActions.instances.append(self)

    def move_to_element_with_offset(self, element, x, y):
        self.calls.append((element, x, y))
        return self

    def click(self):
        return self

    def perform(self):
        return self


class CaptchaSolverTests(unittest.TestCase):
    def make_solver(self):
        return CaptchaSolver(
            glm_enabled=True,
            glm_endpoint="http://localhost:8000/",
            glm_timeout=3,
            allow_chaojiying_fallback=False,
            cy_username="",
            cy_password="",
            cy_soft_id="",
        )

    def test_solve_with_glm_posts_targets_and_uses_returned_xy(self):
        solver = self.make_solver()
        response = FakeResponse(
            {
                "image_size": [30, 10],
                "results": [
                    {"text": "叶", "x": 15, "y": 5, "bbox": [12, 2, 18, 8]},
                    {"text": "件", "x": 5, "y": 5, "bbox": [2, 2, 8, 8]},
                ]
            }
        )

        with patch("captcha_solver.requests.post", return_value=response) as post:
            result = solver._solve_with_glm(b"image-bytes", ["件", "叶"])

        self.assertEqual(result, [["件", 5, 5], ["叶", 15, 5]])
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["targets"], ["件", "叶"])
        self.assertEqual(list(payload), ["images", "targets"])

    def test_parse_glm_result_falls_back_to_bbox_center(self):
        solver = self.make_solver()

        result = solver._parse_glm_result(
            {"results": [{"text": "件", "bbox": [2, 2, 8, 8]}]},
            ["件"],
        )

        self.assertEqual(result, [["件", 5, 5]])

    def test_parse_glm_result_rejects_out_of_bounds_xy(self):
        solver = self.make_solver()

        result = solver._parse_glm_result(
            {"image_size": [30, 10], "results": [{"text": "件", "x": 30, "y": 5}]},
            ["件"],
        )

        self.assertIsNone(result)

    def test_parse_glm_result_does_not_reuse_candidate_for_duplicate_targets(self):
        solver = self.make_solver()

        result = solver._parse_glm_result(
            {"image_size": [30, 10], "results": [{"text": "件", "x": 5, "y": 5}]},
            ["件", "件"],
        )

        self.assertIsNone(result)

    def test_parse_glm_result_uses_distinct_candidates_for_duplicate_targets(self):
        solver = self.make_solver()

        result = solver._parse_glm_result(
            {
                "image_size": [30, 10],
                "results": [
                    {"text": "件", "x": 5, "y": 5},
                    {"text": "件", "x": 15, "y": 5},
                ],
            },
            ["件", "件"],
        )

        self.assertEqual(result, [["件", 5, 5], ["件", 15, 5]])

    def test_parse_glm_result_rejects_unsafe_numeric_values(self):
        solver = self.make_solver()

        result = solver._parse_glm_result(
            {
                "results": [
                    {"text": "件", "x": True, "y": 5},
                    {"text": "件", "bbox": [False, 2, 8, 8]},
                    {"text": "件", "bbox": [2.5, 2, 8, 8]},
                ]
            },
            ["件"],
        )

        self.assertIsNone(result)

    def test_click_captcha_scales_image_coordinates_to_element_offsets(self):
        solver = self.make_solver()
        FakeActions.instances = []

        solver._click_captcha(
            driver=object(),
            target_element=FakeElement(),
            words_loc=[["件", 224, 179]],
            order_words=["件"],
            image_size=(448, 358),
            actions_class=FakeActions,
        )

        self.assertEqual(len(FakeActions.instances), 1)
        self.assertEqual(FakeActions.instances[0].calls[0][1:], (0, 0))

    def test_click_captcha_rejects_out_of_bounds_image_coordinates(self):
        solver = self.make_solver()

        with self.assertRaisesRegex(ValueError, "out of image bounds"):
            solver._click_captcha(
                driver=object(),
                target_element=FakeElement(),
                words_loc=[["件", 448, 179]],
                order_words=["件"],
                image_size=(448, 358),
                actions_class=FakeActions,
            )

    def test_click_captcha_uses_distinct_candidates_for_duplicate_targets(self):
        solver = self.make_solver()
        FakeActions.instances = []

        solver._click_captcha(
            driver=object(),
            target_element=FakeElement(),
            words_loc=[["件", 224, 179], ["件", 100, 100]],
            order_words=["件", "件"],
            image_size=(448, 358),
            actions_class=FakeActions,
        )

        calls = FakeActions.instances[0].calls
        self.assertEqual(calls[0][1:], (0, 0))
        self.assertNotEqual(calls[0][1:], calls[1][1:])

    def test_click_captcha_rejects_missing_duplicate_candidate(self):
        solver = self.make_solver()

        with self.assertRaisesRegex(ValueError, "missing click target: 件"):
            solver._click_captcha(
                driver=object(),
                target_element=FakeElement(),
                words_loc=[["件", 224, 179]],
                order_words=["件", "件"],
                image_size=(448, 358),
                actions_class=FakeActions,
            )

    def test_glm_failure_does_not_fallback_by_default(self):
        class NoFallbackSolver(CaptchaSolver):
            def __init__(self):
                super().__init__(
                    True,
                    "http://localhost:8000",
                    3,
                    False,
                    "",
                    "",
                    "",
                )
                self.chaojiying_calls = 0
                self.clicks = 0

            def _get_captcha_info(self, driver):
                return object(), ["件"], b"image"

            def _solve_with_glm(self, image_content, order_words):
                return None

            def _solve_with_chaojiying(self, image_content, order_words):
                self.chaojiying_calls += 1
                return [["件", 5, 5]]

            def _click_captcha(self, driver, target_element, words_loc, order_words, image_size=None):
                self.clicks += 1

        solver = NoFallbackSolver()

        with self.assertRaisesRegex(CaptchaSolveError, "无法识别验证码"):
            solver.solve(driver=None)
        self.assertEqual(solver.chaojiying_calls, 0)
        self.assertEqual(solver.clicks, 0)

    def test_disabled_glm_does_not_use_chaojiying_without_fallback(self):
        class DisabledLocalSolver(CaptchaSolver):
            def __init__(self):
                super().__init__(
                    False,
                    "http://localhost:8000",
                    3,
                    False,
                    "",
                    "",
                    "",
                )
                self.chaojiying_calls = 0

            def _get_captcha_info(self, driver):
                return object(), ["件"], b"image"

            def _solve_with_chaojiying(self, image_content, order_words):
                self.chaojiying_calls += 1
                return [["件", 5, 5]]

        solver = DisabledLocalSolver()

        with self.assertRaisesRegex(CaptchaSolveError, "无法识别验证码"):
            solver.solve(driver=None)
        self.assertEqual(solver.chaojiying_calls, 0)

    def test_glm_failure_can_fallback_when_explicitly_allowed(self):
        class FallbackSolver(CaptchaSolver):
            def __init__(self):
                super().__init__(
                    True,
                    "http://localhost:8000",
                    3,
                    True,
                    "",
                    "",
                    "",
                )
                self.chaojiying_calls = 0
                self.clicks = 0

            def _get_captcha_info(self, driver):
                return object(), ["件"], b"image"

            def _solve_with_glm(self, image_content, order_words):
                return None

            def _solve_with_chaojiying(self, image_content, order_words):
                self.chaojiying_calls += 1
                return [["件", 5, 5]]

            def _click_captcha(self, driver, target_element, words_loc, order_words, image_size=None):
                self.clicks += 1

        solver = FallbackSolver()

        log = solver.solve(driver=None)

        self.assertIn("安全验证成功", log)
        self.assertEqual(solver.chaojiying_calls, 1)
        self.assertEqual(solver.clicks, 1)

    def test_solve_rejects_glm_result_when_click_order_does_not_match_targets(self):
        class WrongOrderSolver(CaptchaSolver):
            def __init__(self):
                super().__init__(
                    True,
                    "http://localhost:8000",
                    3,
                    False,
                    "",
                    "",
                    "",
                )
                self.clicks = 0

            def _get_captcha_info(self, driver):
                return object(), ["件", "叶"], b"image"

            def _solve_with_glm(self, image_content, order_words):
                return [["叶", 15, 5], ["件", 5, 5]]

            def _click_captcha(self, driver, target_element, words_loc, order_words, image_size=None):
                self.clicks += 1

        solver = WrongOrderSolver()

        with self.assertRaisesRegex(CaptchaSolveError, "顺序不匹配"):
            solver.solve(driver=None)
        self.assertEqual(solver.clicks, 0)

    def test_chaojiying_fallback_orders_distinct_candidates_before_clicking(self):
        class UnorderedFallbackSolver(CaptchaSolver):
            def __init__(self):
                super().__init__(
                    True,
                    "http://localhost:8000",
                    3,
                    True,
                    "",
                    "",
                    "",
                )
                self.clicked_words_loc = None

            def _get_captcha_info(self, driver):
                return object(), ["件", "叶"], b"image"

            def _solve_with_glm(self, image_content, order_words):
                return None

            def _solve_with_chaojiying(self, image_content, order_words):
                return [["叶", 15, 5], ["件", 5, 5]]

            def _click_captcha(self, driver, target_element, words_loc, order_words, image_size=None):
                self.clicked_words_loc = words_loc

        solver = UnorderedFallbackSolver()

        solver.solve(driver=None)

        self.assertEqual(solver.clicked_words_loc, [["件", 5, 5], ["叶", 15, 5]])


class GlmOcrEngineTests(unittest.TestCase):
    def test_recognize_builds_transformers_request_without_real_model(self):
        processor = FakeProcessor()
        model = FakeModel()
        engine = GlmOcrEngine("fake-model")
        engine.processor = processor
        engine.model = model

        output = engine.recognize(b"fake-image", ["件", "叶"])

        self.assertEqual(output, "decoded-output")
        self.assertTrue(processor.image_path_existed)
        self.assertEqual(processor.inputs.device, "cpu")
        self.assertNotIn("token_type_ids", model.generate_kwargs)
        self.assertIs(model.generate_kwargs["input_ids"], processor.inputs["input_ids"])
        self.assertIn("件、叶", processor.messages[0]["content"][1]["text"])
        self.assertEqual(processor.decode_args, ([201, 202], False))


if __name__ == "__main__":
    unittest.main()
