import os
import sys
import unittest
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from page_func import (
    booking_venue_kind,
    click_pay,
    click_submit_order,
    free_venue_indices_from_statuses,
    header_row_signature,
    time_column_from_rows,
    time_column_index,
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
    def __init__(self, driver):
        self.driver = driver

    def window(self, _handle):
        self.driver.current_handle = _handle


class FakePaymentDriver:
    def __init__(self, elements):
        self.window_handles = ["main"]
        self.current_handle = "main"
        if isinstance(elements, dict):
            self.elements_by_handle = elements
            self.window_handles = list(elements)
            self.current_handle = self.window_handles[0]
        else:
            self.elements_by_handle = {"main": elements}
        self.switch_to = FakeSwitchTo(self)
        self.scripts = []
        self.screenshot_path = None

    @property
    def page_source(self):
        texts = "".join(element.text for element in self.elements_by_handle.get(self.current_handle, []))
        return f"<html><body>{texts}</body></html>"

    def find_element(self, by, value):
        if value == "select-word":
            raise Exception("no captcha")
        return self.elements_by_handle[self.current_handle][0]

    def find_elements(self, _by, _value):
        return self.elements_by_handle.get(self.current_handle, [])

    def execute_script(self, script, element=None):
        self.scripts.append(script)
        if "document.body.innerText" in script:
            return "\n".join(element.text for element in self.elements_by_handle.get(self.current_handle, []))
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

    def test_time_column_index_maps_header_slots_once(self):
        header = ["场地", "06:50-07:50", "07:50-08:50", "08:50-09:50"]

        self.assertEqual(
            time_column_index(header),
            {"06:50": 1, "07:50": 2, "08:50": 3},
        )

    def test_header_row_signature_is_stable_for_same_header(self):
        header = ["场地", "06:50-07:50", "07:50-08:50"]

        self.assertEqual(header_row_signature(header), header_row_signature(list(header)))

    def test_free_venue_indices_from_statuses_extracts_free_rows(self):
        statuses = [
            "reserveBlock position free",
            "reserveBlock position used",
            "reserveBlock position free",
            "",
        ]

        self.assertEqual(free_venue_indices_from_statuses(statuses), [1, 3])

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

    def test_click_pay_prefers_campus_card_and_real_pay_button_over_summary_text(self):
        summary = FakePaymentElement("请您支付：", width=400, height=120)
        campus_card = FakePaymentElement("电子校园卡", width=160, height=40)
        pay_button = FakePaymentElement("支付 （590s）", width=120, height=36)
        driver = FakePaymentDriver([summary, campus_card, pay_button])
        driver.window_handles = ["main", "order", "payment"]

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

        self.assertFalse(summary.clicked)
        self.assertTrue(campus_card.clicked)
        self.assertTrue(pay_button.clicked)
        self.assertIn("已选择电子校园卡", log)
        self.assertIn("已点击支付按钮", log)

    def test_click_pay_falls_back_to_non_blank_payment_window(self):
        payment_summary = FakePaymentElement("请您支付：", width=400, height=120)
        campus_card = FakePaymentElement("电子校园卡", width=160, height=40)
        pay_button = FakePaymentElement("支付 （590s）", width=120, height=36)
        driver = FakePaymentDriver(
            {
                "main": [FakePaymentElement("首页")],
                "payment": [payment_summary, campus_card, pay_button],
                "blank": [],
            }
        )
        driver.current_handle = "main"

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

        self.assertEqual(driver.current_handle, "payment")
        self.assertTrue(campus_card.clicked)
        self.assertTrue(pay_button.clicked)
        self.assertIn("已选择电子校园卡", log)

    def test_click_pay_reports_incomplete_when_pay_button_missing(self):
        payment_summary = FakePaymentElement("请您支付：", width=400, height=120)
        campus_card = FakePaymentElement("电子校园卡", width=160, height=40)
        driver = FakePaymentDriver([payment_summary, campus_card])

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

        self.assertIn("支付按钮点击失败: 找不到支付按钮", log)
        self.assertIn("付款未完成", log)
        self.assertNotIn("付款完成", log)

    def test_click_submit_order_clicks_visible_submit_button(self):
        submit = FakePaymentElement("提交")
        driver = FakePaymentDriver([submit])

        log = click_submit_order(driver, timeout=0)

        self.assertTrue(submit.clicked)
        self.assertIn("提交订单成功", log)


if __name__ == "__main__":
    unittest.main()
