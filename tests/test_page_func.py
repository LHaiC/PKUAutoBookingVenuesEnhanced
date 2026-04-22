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
    venue_scan_order,
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

    def is_enabled(self):
        return True

    def click(self):
        self.clicked = True


class FakeSwitchTo:
    def window(self, _handle):
        pass


class FakePaymentDriver:
    def __init__(self, elements):
        self.elements = elements
        self.window_handles = ["main"]
        self.switch_to = FakeSwitchTo()
        self.page_source = "<html>payment</html>"
        self.scripts = []
        self.screenshot_path = None

    def find_element(self, by, value):
        if value == "select-word":
            raise Exception("no captcha")
        return self.elements[0]

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

    def test_venue_scan_order_returns_full_permutation(self):
        order = venue_scan_order(6, "user|venue|1300")

        self.assertEqual(sorted(order), [1, 2, 3, 4, 5, 6])
        self.assertEqual(len(order), 6)

    def test_venue_scan_order_avoids_low_number_first_when_choices_exist(self):
        order = venue_scan_order(6, "user|venue|1300")

        self.assertGreaterEqual(order[0], 4)
        self.assertNotEqual(order, [1, 2, 3, 4, 5, 6])

    def test_click_pay_clicks_primary_pay_button_when_present(self):
        campus_card = FakePaymentElement("支付")
        driver = FakePaymentDriver([campus_card])

        class FakeWait:
            def __init__(self, _driver, _timeout):
                pass

            def until(self, _condition):
                return True

            def until_not(self, _condition):
                return True

        import page_func

        original_wait = page_func.WebDriverWait
        try:
            page_func.WebDriverWait = FakeWait
            log = click_pay(driver)
        finally:
            page_func.WebDriverWait = original_wait

        self.assertTrue(campus_card.clicked)
        self.assertIn("已点击支付按钮", log)

    def test_click_submit_order_clicks_visible_submit_button(self):
        submit = FakePaymentElement("提交")
        driver = FakePaymentDriver([submit])

        log = click_submit_order(driver, timeout=0)

        self.assertTrue(submit.clicked)
        self.assertIn("提交订单成功", log)


if __name__ == "__main__":
    unittest.main()
