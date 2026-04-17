import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captcha_matcher import (
    Candidate,
    MatchError,
    bbox_center,
    match_targets,
    normalize_candidates,
)


class CaptchaMatcherTests(unittest.TestCase):
    def test_bbox_center_uses_integer_midpoint(self):
        self.assertEqual(bbox_center([10, 20, 30, 50]), (20, 35))

    def test_match_targets_returns_requested_order(self):
        candidates = [
            Candidate(text="结", bbox=[70, 120, 130, 170], confidence=0.91),
            Candidate(text="件", bbox=[140, 50, 190, 110], confidence=0.94),
            Candidate(text="叶", bbox=[230, 90, 285, 140], confidence=0.89),
        ]

        matched = match_targets(["件", "叶", "结"], candidates, image_size=(448, 358))

        self.assertEqual(
            matched,
            [
                {"text": "件", "bbox": [140, 50, 190, 110], "x": 165, "y": 80, "confidence": 0.94},
                {"text": "叶", "bbox": [230, 90, 285, 140], "x": 257, "y": 115, "confidence": 0.89},
                {"text": "结", "bbox": [70, 120, 130, 170], "x": 100, "y": 145, "confidence": 0.91},
            ],
        )

    def test_missing_target_raises(self):
        candidates = [Candidate(text="件", bbox=[140, 50, 190, 110], confidence=0.94)]

        with self.assertRaisesRegex(MatchError, "missing target: 叶"):
            match_targets(["件", "叶"], candidates, image_size=(448, 358))

    def test_duplicate_target_raises(self):
        candidates = [
            Candidate(text="件", bbox=[140, 50, 190, 110], confidence=0.94),
            Candidate(text="件", bbox=[200, 50, 250, 110], confidence=0.93),
        ]

        with self.assertRaisesRegex(MatchError, "ambiguous target: 件"):
            match_targets(["件"], candidates, image_size=(448, 358))

    def test_low_confidence_raises(self):
        candidates = [Candidate(text="件", bbox=[140, 50, 190, 110], confidence=0.20)]

        with self.assertRaisesRegex(MatchError, "low confidence: 件"):
            match_targets(["件"], candidates, image_size=(448, 358), min_confidence=0.50)

    def test_out_of_bounds_bbox_raises(self):
        candidates = [Candidate(text="件", bbox=[140, 50, 999, 110], confidence=0.94)]

        with self.assertRaisesRegex(MatchError, "bbox out of bounds: 件"):
            match_targets(["件"], candidates, image_size=(448, 358))

    def test_normalize_candidates_skips_non_numeric_bbox_values(self):
        items = [{"text": "件", "bbox": [140, "bad", 190, 110], "confidence": 0.94}]

        self.assertEqual(normalize_candidates(items), [])

    def test_normalize_candidates_skips_non_sequence_bbox(self):
        items = [{"text": "件", "bbox": None, "confidence": 0.94}]

        self.assertEqual(normalize_candidates(items), [])

    def test_normalize_candidates_skips_non_numeric_confidence(self):
        items = [{"text": "件", "bbox": [140, 50, 190, 110], "confidence": "bad"}]

        self.assertEqual(normalize_candidates(items), [])

    def test_match_targets_returns_copied_bbox(self):
        candidate = Candidate(text="件", bbox=[140, 50, 190, 110], confidence=0.94)

        matched = match_targets(["件"], [candidate], image_size=(448, 358))
        matched[0]["bbox"][0] = 999

        self.assertEqual(candidate.bbox, [140, 50, 190, 110])


if __name__ == "__main__":
    unittest.main()
