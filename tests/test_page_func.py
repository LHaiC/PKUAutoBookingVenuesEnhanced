import os
import sys
import unittest
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from page_func import (
    booking_venue_kind,
    time_column_from_rows,
    venue_card_xpath,
)


class PageFuncTests(unittest.TestCase):
    def test_venue_card_xpath_supports_wusi_badmiton_hall_alias(self):
        xpath = venue_card_xpath("五四羽毛球馆")

        self.assertIn("五四体育中心", xpath)
        self.assertIn("羽毛球馆", xpath)
        self.assertIn("//dl", xpath)

    def test_booking_venue_kind_normalizes_full_venue_names(self):
        self.assertEqual(booking_venue_kind("五四羽毛球馆"), "羽毛球馆")
        self.assertEqual(booking_venue_kind("邱德拔羽毛球场"), "羽毛球场")

    def test_time_column_from_rows_reads_0650_from_visible_table_headers(self):
        start_time = datetime.datetime.strptime("0650", "%H%M")
        rows = [
            ["场地", "06:50-07:50", "07:50-08:50", "08:50-09:50"],
            ["1号", "可预约", "已售", "已售"],
        ]

        self.assertEqual(time_column_from_rows(rows, start_time), 1)

    def test_time_column_from_rows_reads_0800_from_later_column(self):
        start_time = datetime.datetime.strptime("0800", "%H%M")
        rows = [
            ["场地", "06:00-07:00", "07:00-08:00", "08:00-09:00"],
        ]

        self.assertEqual(time_column_from_rows(rows, start_time), 3)

    def test_time_column_from_rows_returns_none_when_slot_not_visible(self):
        start_time = datetime.datetime.strptime("1150", "%H%M")
        rows = [
            ["场地", "06:50-07:50", "07:50-08:50", "08:50-09:50"],
        ]

        self.assertIsNone(time_column_from_rows(rows, start_time))


if __name__ == "__main__":
    unittest.main()
