import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw

from captcha_vision import (
    bbox_center,
    decode_image,
    detect_dark_regions,
    image_size,
    refine_bbox_to_dark_pixels,
    validate_bbox,
)


def make_png_bytes() -> bytes:
    img = Image.new("RGB", (120, 80), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 15, 45, 55], fill="black")
    draw.rectangle([70, 20, 95, 60], fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class CaptchaVisionTests(unittest.TestCase):
    def test_decode_image_reads_bytes(self):
        img = decode_image(make_png_bytes())
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(img.size, (120, 80))

    def test_image_size_returns_width_height(self):
        self.assertEqual(image_size(decode_image(make_png_bytes())), (120, 80))

    def test_validate_bbox_accepts_inside_box(self):
        self.assertTrue(validate_bbox([20, 15, 45, 55], (120, 80)))

    def test_validate_bbox_rejects_outside_box(self):
        self.assertFalse(validate_bbox([20, 15, 130, 55], (120, 80)))

    def test_validate_bbox_rejects_malformed_inputs(self):
        self.assertFalse(validate_bbox(None, (120, 80)))
        self.assertFalse(validate_bbox("20,15,45,55", (120, 80)))
        self.assertFalse(validate_bbox([20, "bad", 45, 55], (120, 80)))
        self.assertFalse(validate_bbox(["20", "15", "45", "55"], (120, 80)))
        self.assertFalse(validate_bbox([20.9, 15.1, 45.2, 55.7], (120, 80)))

    def test_bbox_center(self):
        self.assertEqual(bbox_center([20, 15, 46, 55]), (33, 35))

    def test_detect_dark_regions_finds_two_regions(self):
        regions = detect_dark_regions(decode_image(make_png_bytes()))
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0], [20, 15, 46, 56])
        self.assertEqual(regions[1], [70, 20, 96, 61])

    def test_refine_bbox_to_dark_pixels_shrinks_padding_area(self):
        refined = refine_bbox_to_dark_pixels(
            decode_image(make_png_bytes()),
            [10, 5, 55, 65],
        )
        self.assertEqual(refined, [20, 15, 46, 56])

    def test_refine_bbox_to_dark_pixels_returns_malformed_bbox_unchanged(self):
        self.assertIsNone(refine_bbox_to_dark_pixels(decode_image(make_png_bytes()), None))


if __name__ == "__main__":
    unittest.main()
