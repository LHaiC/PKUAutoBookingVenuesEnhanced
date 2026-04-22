import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from main import load_config
from page_func import verify
from captcha_solver import solve_captcha


class ConfigLoadingTests(unittest.TestCase):
    def test_example_config_loads_without_duration(self):
        values = load_config("config.example.ini")

        self.assertEqual(len(values), 15)
        self.assertEqual(values[13], 20)
        self.assertFalse(values[-1])

    def test_glm_timeout_defaults_to_20_seconds_when_missing(self):
        config_text = """[login]
user_name = demo
password = secret

[wechat]
wechat_notice = false
SCKEY = test

[chaojiying]
username = test
password = test
soft_id = test

[glm_ocr]
enabled = true
endpoint = http://localhost:8000
allow_chaojiying_fallback = false
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.ini")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(config_text)

            values = load_config(config_path)

        self.assertEqual(values[13], 20)

    def test_booking_entry_points_are_importable_without_selenium_installed(self):
        self.assertTrue(callable(verify))
        self.assertTrue(callable(solve_captcha))

    def test_missing_selenium_fails_before_browser_options_are_called(self):
        original_webdriver = main.webdriver
        original_chrome_options = main.Chrome_Options
        original_firefox_options = main.Firefox_Options
        try:
            main.webdriver = None
            main.Chrome_Options = None
            main.Firefox_Options = None

            with self.assertRaisesRegex(RuntimeError, "pip install selenium"):
                main.ensure_selenium_available()
        finally:
            main.webdriver = original_webdriver
            main.Chrome_Options = original_chrome_options
            main.Firefox_Options = original_firefox_options


if __name__ == "__main__":
    unittest.main()
