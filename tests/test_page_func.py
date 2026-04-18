import os
import sys
import unittest
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from page_func import booking_slot_page_and_column, booking_venue_kind, venue_card_xpath


class PageFuncTests(unittest.TestCase):
    def test_venue_card_xpath_supports_wusi_badmiton_hall_alias(self):
        xpath = venue_card_xpath("五四羽毛球馆")

        self.assertIn("五四体育中心", xpath)
        self.assertIn("羽毛球馆", xpath)
        self.assertIn("//dl", xpath)

    def test_booking_venue_kind_normalizes_full_venue_names(self):
        self.assertEqual(booking_venue_kind("五四羽毛球馆"), "羽毛球馆")
        self.assertEqual(booking_venue_kind("邱德拔羽毛球场"), "羽毛球场")

    def test_wusi_badminton_hall_supports_0650_first_slot(self):
        start_time = datetime.datetime.strptime("0650", "%H%M")

        self.assertEqual(booking_slot_page_and_column("五四羽毛球馆", start_time), (0, 1))
        self.assertEqual(booking_slot_page_and_column("五四体育中心-羽毛球馆", start_time), (0, 1))

    def test_badminton_hall_columns_advance_from_0650(self):
        start_time = datetime.datetime.strptime("1150", "%H%M")

        self.assertEqual(booking_slot_page_and_column("五四羽毛球馆", start_time), (1, 1))

    def test_qiudeba_badminton_court_keeps_0800_first_slot(self):
        start_time = datetime.datetime.strptime("0800", "%H%M")

        self.assertEqual(booking_slot_page_and_column("邱德拔羽毛球场", start_time), (0, 1))


if __name__ == "__main__":
    unittest.main()
