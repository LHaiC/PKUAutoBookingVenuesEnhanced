import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from page_func import booking_venue_kind, venue_card_xpath


class PageFuncTests(unittest.TestCase):
    def test_venue_card_xpath_supports_wusi_badmiton_hall_alias(self):
        xpath = venue_card_xpath("五四羽毛球馆")

        self.assertIn("五四体育中心", xpath)
        self.assertIn("羽毛球馆", xpath)
        self.assertIn("//dl", xpath)

    def test_booking_venue_kind_normalizes_full_venue_names(self):
        self.assertEqual(booking_venue_kind("五四羽毛球馆"), "羽毛球馆")
        self.assertEqual(booking_venue_kind("邱德拔羽毛球场"), "羽毛球场")


if __name__ == "__main__":
    unittest.main()
