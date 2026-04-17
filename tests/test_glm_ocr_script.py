import json
import os
import tempfile
import unittest

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
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "captcha.png")
            metadata_path = os.path.join(tmpdir, "captcha.json")
            with open(image_path, "wb") as f:
                f.write(b"image")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump({"image": "captcha.png", "targets": ["件"]}, f)

            glm_ocr_script.SAMPLE_DIR = tmpdir
            try:
                self.assertEqual(
                    glm_ocr_script.find_latest_captured_sample(),
                    (image_path, ["件"]),
                )
            finally:
                glm_ocr_script.SAMPLE_DIR = original_sample_dir


if __name__ == "__main__":
    unittest.main()
