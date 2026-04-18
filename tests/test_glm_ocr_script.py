import json
import os
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import tests.test_glm_ocr as glm_ocr_script


class GlmOcrScriptTests(unittest.TestCase):
    def test_parse_glm_result_prefers_returned_xy(self):
        result = {
            "results": [
                {"text": "件", "x": 5, "y": 6, "bbox": [1, 2, 9, 10]},
                {"text": "叶", "bbox": [10, 20, 30, 40]},
            ]
        }

        self.assertEqual(
            glm_ocr_script.parse_glm_result(result, ["件", "叶"]),
            [["件", 5, 6], ["叶", 20, 30]],
        )

    def test_call_glm_ocr_sends_targets(self):
        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"results": []}

        captured = {}

        def fake_post(url, json, timeout):
            captured["json"] = json
            return FakeResponse()

        original_post = glm_ocr_script.requests.post
        try:
            glm_ocr_script.requests.post = fake_post
            glm_ocr_script.call_glm_ocr(b"image", ["件", "叶"])
        finally:
            glm_ocr_script.requests.post = original_post

        self.assertEqual(captured["json"]["targets"], ["件", "叶"])

    def test_find_latest_captured_sample(self):
        original_sample_dir = glm_ocr_script.SAMPLE_DIR
        original_failure_sample_dir = glm_ocr_script.FAILURE_SAMPLE_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "captcha.png")
            metadata_path = os.path.join(tmpdir, "captcha.json")
            Image.new("RGB", (20, 20), "white").save(image_path)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump({"image": "captcha.png", "targets": ["件"]}, f)

            glm_ocr_script.SAMPLE_DIR = tmpdir
            glm_ocr_script.FAILURE_SAMPLE_DIR = os.path.join(tmpdir, "missing")
            try:
                self.assertEqual(
                    glm_ocr_script.find_latest_captured_sample(),
                    (image_path, ["件"]),
                )
            finally:
                glm_ocr_script.SAMPLE_DIR = original_sample_dir
                glm_ocr_script.FAILURE_SAMPLE_DIR = original_failure_sample_dir

    def test_find_latest_captured_sample_prefers_failure_samples(self):
        original_sample_dir = glm_ocr_script.SAMPLE_DIR
        original_failure_sample_dir = glm_ocr_script.FAILURE_SAMPLE_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            captured_dir = os.path.join(tmpdir, "captured")
            failure_dir = os.path.join(tmpdir, "failures")
            os.makedirs(captured_dir)
            os.makedirs(failure_dir)
            Image.new("RGB", (20, 20), "white").save(os.path.join(captured_dir, "old.png"))
            with open(os.path.join(captured_dir, "old.json"), "w", encoding="utf-8") as f:
                json.dump({"image": "old.png", "targets": ["旧"]}, f)
            Image.new("RGB", (20, 20), "white").save(os.path.join(failure_dir, "new.png"))
            with open(os.path.join(failure_dir, "new.json"), "w", encoding="utf-8") as f:
                json.dump({"image": "new.png", "targets": ["新"]}, f)

            glm_ocr_script.SAMPLE_DIR = captured_dir
            glm_ocr_script.FAILURE_SAMPLE_DIR = failure_dir
            try:
                self.assertEqual(
                    glm_ocr_script.find_latest_captured_sample(),
                    (os.path.join(failure_dir, "new.png"), ["新"]),
                )
            finally:
                glm_ocr_script.SAMPLE_DIR = original_sample_dir
                glm_ocr_script.FAILURE_SAMPLE_DIR = original_failure_sample_dir

    def test_find_latest_captured_sample_skips_invalid_images(self):
        original_sample_dir = glm_ocr_script.SAMPLE_DIR
        original_failure_sample_dir = glm_ocr_script.FAILURE_SAMPLE_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bad.png"), "wb") as f:
                f.write(b"not-an-image")
            with open(os.path.join(tmpdir, "bad.json"), "w", encoding="utf-8") as f:
                json.dump({"image": "bad.png", "targets": ["坏"]}, f)
            Image.new("RGB", (20, 20), "white").save(os.path.join(tmpdir, "good.png"))
            with open(os.path.join(tmpdir, "good.json"), "w", encoding="utf-8") as f:
                json.dump({"image": "good.png", "targets": ["好"]}, f)

            glm_ocr_script.SAMPLE_DIR = tmpdir
            glm_ocr_script.FAILURE_SAMPLE_DIR = os.path.join(tmpdir, "missing")
            try:
                self.assertEqual(
                    glm_ocr_script.find_latest_captured_sample(),
                    (os.path.join(tmpdir, "good.png"), ["好"]),
                )
            finally:
                glm_ocr_script.SAMPLE_DIR = original_sample_dir
                glm_ocr_script.FAILURE_SAMPLE_DIR = original_failure_sample_dir

    def test_main_exits_nonzero_when_no_positions_are_returned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "captcha.png")
            output_path = os.path.join(tmpdir, "output.png")
            image = Image.new("RGB", (80, 40), "white")
            for x in range(10, 30):
                for y in range(8, 28):
                    image.putpixel((x, y), (200, 20, 20))
            image.save(image_path)

            with patch.object(
                glm_ocr_script,
                "find_latest_captured_sample",
                return_value=(image_path, ["件"]),
            ), patch.object(
                glm_ocr_script,
                "call_glm_ocr",
                return_value={"results": [], "error": "unsafe_ocr_output"},
            ), patch.object(glm_ocr_script, "OUTPUT_IMAGE_PATH", output_path):
                with self.assertRaises(SystemExit) as exc:
                    glm_ocr_script.main()

            self.assertEqual(exc.exception.code, 1)
            self.assertTrue(os.path.exists(output_path))
            annotated = Image.open(output_path).convert("RGB")
            self.assertNotEqual(annotated.getpixel((10, 8)), (200, 20, 20))


if __name__ == "__main__":
    unittest.main()
