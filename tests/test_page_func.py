import os
import sys
import unittest
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from page_func import (
    booking_venue_kind,
    click_pay,
    click_submit_order,
    time_column_from_rows,
    venue_card_xpath,
)


class FakePaymentElement:
    def __init__(self, text, displayed=True, width=120, height=40):
        self.text = text
        self.displayed = displayed
        self.size = {"width": width, "height": height}
        self.rect = {"width": width, "height": height}
        self.clicked = False

    def is_displayed(self):
        return self.displayed

    def get_attribute(self, name):
        return ""

    def click(self):
        self.clicked = True


class FakeSwitchTo:
    def window(self, _handle):
        pass


class FakePaymentDriver:
    window_handles = ["main"]

    def __init__(self, elements):
        self.elements = elements
        self.switch_to = FakeSwitchTo()
        self.page_source = "<html>payment</html>"
        self.scripts = []
        self.screenshot_path = None

    def find_elements(self, _by, _value):
        return self.elements

    def execute_script(self, script, element=None):
        self.scripts.append(script)
        if "click" in script and element is not None:
            element.clicked = True

    def save_screenshot(self, path):
        self.screenshot_path = path
        return True


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

    def test_click_pay_waits_for_manual_payment_by_default(self):
        campus_card = FakePaymentElement("校园卡付款")
        driver = FakePaymentDriver([campus_card])

        log = click_pay(driver, manual_wait_seconds=0)

        self.assertFalse(campus_card.clicked)
        self.assertIn("订单已提交，需要用户自行付款", log)

    def test_click_submit_order_clicks_visible_submit_button(self):
        submit = FakePaymentElement("提交")
        driver = FakePaymentDriver([submit])

        log = click_submit_order(driver, timeout=0)

        self.assertTrue(submit.clicked)
        self.assertIn("提交订单成功", log)


if __name__ == "__main__":
    unittest.main()
