import os
import unittest
from unittest.mock import patch

from captcha_solver import CaptchaSolver
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


class CaptchaSolverTests(unittest.TestCase):
    def make_solver(self):
        return CaptchaSolver(
            glm_enabled=True,
            glm_endpoint="http://localhost:8000/",
            glm_timeout=3,
            cy_username="",
            cy_password="",
            cy_soft_id="",
        )

    def test_solve_with_glm_posts_targets_and_uses_returned_xy(self):
        solver = self.make_solver()
        response = FakeResponse(
            {
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
