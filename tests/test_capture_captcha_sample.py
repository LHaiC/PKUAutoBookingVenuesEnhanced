import json
import os
import tempfile
import unittest

from tests.capture_captcha_sample import save_sample


class CaptureCaptchaSampleTests(unittest.TestCase):
    def test_save_sample_writes_image_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path, meta_path = save_sample(b"image-bytes", ["件", "叶"], tmpdir)

            with open(image_path, "rb") as f:
                self.assertEqual(f.read(), b"image-bytes")

            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)

        self.assertEqual(metadata["targets"], ["件", "叶"])
        self.assertEqual(metadata["image"], os.path.basename(image_path))
        self.assertEqual(metadata["source"], "live_webpage_capture")


if __name__ == "__main__":
    unittest.main()
