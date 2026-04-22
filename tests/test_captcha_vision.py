import io
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw

from captcha_vision import (
    DARK_PROPOSAL_SOURCE,
    ISOLATED_DARK_PROPOSAL_VARIANT,
    ISOLATED_PROPOSAL_VARIANT,
    PADDED_PROPOSAL_VARIANT,
    PRIMARY_PROPOSAL_SOURCE,
    PRIMARY_PROPOSAL_VARIANT,
    bbox_center,
    build_colored_text_strip,
    decode_image,
    detect_colored_text_bboxes,
    detect_dark_regions,
    filter_captcha_text_bboxes,
    generate_box_proposals,
    image_size,
    measure_box_size_consistency,
    prepare_captcha_boxes,
    ProposalSet,
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

    def test_detect_colored_text_bboxes_orders_separate_regions_left_to_right(self):
        img = Image.new("RGB", (160, 60), (220, 240, 250))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 25, 35], fill=(200, 20, 20))
        draw.rectangle([50, 12, 67, 38], fill=(20, 120, 220))
        draw.rectangle([95, 9, 114, 36], fill=(230, 40, 180))

        self.assertEqual(
            detect_colored_text_bboxes(img),
            [[10, 10, 26, 36], [50, 12, 68, 39], [95, 9, 115, 37]],
        )

    def test_generate_box_proposals_returns_multiple_variants(self):
        img = Image.new("RGB", (160, 60), (220, 240, 250))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 25, 35], fill=(200, 20, 20))
        draw.rectangle([50, 12, 67, 38], fill=(20, 120, 220))
        draw.rectangle([95, 9, 114, 36], fill=(230, 40, 180))

        proposals = generate_box_proposals(img)
        proposal_map = {
            (proposal.source, proposal.preprocess_variant): proposal.boxes
            for proposal in proposals
        }

        self.assertEqual(
            [(proposal.source, proposal.preprocess_variant) for proposal in proposals],
            [
                (PRIMARY_PROPOSAL_SOURCE, PRIMARY_PROPOSAL_VARIANT),
                (DARK_PROPOSAL_SOURCE, PRIMARY_PROPOSAL_VARIANT),
                (PRIMARY_PROPOSAL_SOURCE, PADDED_PROPOSAL_VARIANT),
                (PRIMARY_PROPOSAL_SOURCE, ISOLATED_PROPOSAL_VARIANT),
                (DARK_PROPOSAL_SOURCE, ISOLATED_DARK_PROPOSAL_VARIANT),
            ],
        )
        self.assertTrue(all(len(item.boxes) >= 3 for item in proposals))
        self.assertEqual(
            proposal_map[(PRIMARY_PROPOSAL_SOURCE, PRIMARY_PROPOSAL_VARIANT)],
            [[10, 10, 26, 36], [50, 12, 68, 39], [95, 9, 115, 37]],
        )
        self.assertEqual(
            proposal_map[(DARK_PROPOSAL_SOURCE, PRIMARY_PROPOSAL_VARIANT)],
            [[10, 10, 26, 36], [50, 12, 68, 39], [95, 9, 115, 37]],
        )
        self.assertEqual(
            proposal_map[(PRIMARY_PROPOSAL_SOURCE, PADDED_PROPOSAL_VARIANT)],
            [[10, 10, 26, 36], [50, 12, 68, 39], [95, 9, 115, 37]],
        )
        self.assertEqual(
            proposal_map[(PRIMARY_PROPOSAL_SOURCE, ISOLATED_PROPOSAL_VARIANT)],
            [[10, 10, 26, 36], [50, 12, 68, 39], [95, 9, 115, 37]],
        )
        self.assertEqual(
            proposal_map[(DARK_PROPOSAL_SOURCE, ISOLATED_DARK_PROPOSAL_VARIANT)],
            [[10, 10, 26, 36], [50, 12, 68, 39], [95, 9, 115, 37]],
        )

    def test_box_size_consistency_flags_outlier_box(self):
        boxes = [[10, 10, 40, 40], [60, 10, 90, 40], [95, 8, 150, 55]]

        stats = measure_box_size_consistency(boxes)

        self.assertLess(stats["score"], 0.5)
        self.assertEqual(stats["outlier_index"], 2)

    def test_generate_box_proposals_normalizes_padded_edge_boxes(self):
        img = Image.new("RGB", (120, 50), (220, 240, 250))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 8, 13, 33], fill=(200, 20, 20))
        draw.rectangle([48, 10, 63, 35], fill=(20, 120, 220))
        draw.rectangle([101, 7, 119, 32], fill=(230, 40, 180))

        proposals = generate_box_proposals(img)
        proposal_map = {
            (proposal.source, proposal.preprocess_variant): proposal.boxes
            for proposal in proposals
        }

        expected = [[0, 8, 14, 34], [48, 10, 64, 36], [101, 7, 120, 33]]
        self.assertEqual(
            proposal_map[(PRIMARY_PROPOSAL_SOURCE, PADDED_PROPOSAL_VARIANT)],
            expected,
        )

    def test_generate_box_proposals_normalizes_padded_bottom_edge_boxes(self):
        img = Image.new("RGB", (120, 50), (220, 240, 250))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 35, 25, 49], fill=(200, 20, 20))
        draw.rectangle([48, 10, 63, 35], fill=(20, 120, 220))
        draw.rectangle([101, 7, 119, 32], fill=(230, 40, 180))

        proposals = generate_box_proposals(img)
        proposal_map = {
            (proposal.source, proposal.preprocess_variant): proposal.boxes
            for proposal in proposals
        }

        self.assertEqual(
            proposal_map[(PRIMARY_PROPOSAL_SOURCE, PADDED_PROPOSAL_VARIANT)],
            [[10, 35, 26, 50], [48, 10, 64, 36], [101, 7, 120, 33]],
        )

    def test_prepare_captcha_boxes_uses_primary_proposal_metadata(self):
        img = decode_image(make_png_bytes())
        primary = [[20, 15, 46, 56], [70, 20, 96, 61]]
        secondary = [[1, 1, 2, 2]]

        with patch(
            "captcha_vision.generate_box_proposals",
            return_value=[
                ProposalSet(boxes=secondary, source=DARK_PROPOSAL_SOURCE, preprocess_variant=PRIMARY_PROPOSAL_VARIANT),
                ProposalSet(
                    boxes=primary,
                    source=PRIMARY_PROPOSAL_SOURCE,
                    preprocess_variant=PRIMARY_PROPOSAL_VARIANT,
                ),
            ],
        ):
            self.assertEqual(prepare_captcha_boxes(img, refine=False), primary)

    def test_build_colored_text_strip_normalizes_regions_left_to_right(self):
        img = Image.new("RGB", (160, 80), (220, 240, 250))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 40, 20, 60], fill=(200, 20, 20))
        draw.rectangle([60, 5, 75, 25], fill=(20, 120, 220))
        draw.rectangle([100, 30, 120, 55], fill=(230, 40, 180))

        strip = build_colored_text_strip(img, padding=2, gap=4, margin=3)

        self.assertIsNotNone(strip)
        self.assertEqual(strip.size, (74, 36))
        pixels = strip.load()
        non_white_columns = [
            x
            for x in range(strip.width)
            if any(pixels[x, y] != (255, 255, 255) for y in range(strip.height))
        ]
        column_groups = []
        start = previous = non_white_columns[0]
        for x in non_white_columns[1:]:
            if x == previous + 1:
                previous = x
                continue
            column_groups.append((start, previous + 1))
            start = previous = x
        column_groups.append((start, previous + 1))

        self.assertEqual(column_groups, [(5, 16), (24, 40), (48, 69)])

    def test_build_colored_text_strip_returns_none_without_regions(self):
        img = Image.new("RGB", (160, 80), (220, 240, 250))

        self.assertIsNone(build_colored_text_strip(img))

    def test_filter_captcha_text_bboxes_removes_small_colored_noise(self):
        self.assertEqual(
            filter_captcha_text_bboxes(
                [
                    [97, 23, 136, 61],
                    [136, 73, 174, 106],
                    [158, 118, 184, 132],
                    [210, 46, 252, 85],
                ]
            ),
            [
                [97, 23, 136, 61],
                [136, 73, 174, 106],
                [210, 46, 252, 85],
            ],
        )

    def test_detect_colored_text_bboxes_ignores_saturated_photo_background(self):
        img = Image.new("RGB", (310, 155), (77, 193, 228))
        draw = ImageDraw.Draw(img)
        draw.rectangle([41, 34, 78, 69], fill=(85, 30, 165))
        draw.rectangle([93, 16, 126, 49], fill=(85, 30, 165))
        draw.rectangle([146, 74, 172, 92], fill=(160, 50, 160))
        draw.rectangle([169, 75, 180, 107], fill=(160, 50, 160))
        draw.rectangle([207, 6, 242, 39], fill=(210, 20, 25))
        draw.rectangle([77, 125, 109, 154], fill=(215, 90, 20))
        draw.rectangle([115, 125, 145, 154], fill=(215, 90, 20))

        boxes = filter_captcha_text_bboxes(detect_colored_text_bboxes(img), img.size)

        self.assertEqual(
            boxes,
            [
                [41, 34, 79, 70],
                [93, 16, 127, 50],
                [146, 74, 181, 108],
                [207, 6, 243, 40],
            ],
        )

    def test_detect_colored_text_bboxes_ignores_muted_photo_background(self):
        img = Image.new("RGB", (310, 155), (117, 164, 206))
        draw = ImageDraw.Draw(img)
        draw.rectangle([8, 71, 44, 107], fill=(85, 30, 165))
        draw.rectangle([102, 51, 136, 89], fill=(240, 90, 20))
        draw.rectangle([155, 51, 189, 85], fill=(25, 60, 140))
        draw.rectangle([207, 84, 240, 116], fill=(170, 45, 160))

        boxes = filter_captcha_text_bboxes(detect_colored_text_bboxes(img), img.size)

        self.assertEqual(
            boxes,
            [
                [8, 71, 45, 108],
                [102, 51, 137, 90],
                [155, 51, 190, 86],
                [207, 84, 241, 117],
            ],
        )


if __name__ == "__main__":
    unittest.main()
