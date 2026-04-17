import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import load_config
from page_func import verify
from captcha_solver import solve_captcha


class ConfigLoadingTests(unittest.TestCase):
    def test_example_config_loads_without_duration(self):
        values = load_config("config.example.ini")

        self.assertEqual(len(values), 15)
        self.assertFalse(values[-1])

    def test_booking_entry_points_are_importable_without_selenium_installed(self):
        self.assertTrue(callable(verify))
        self.assertTrue(callable(solve_captcha))


if __name__ == "__main__":
    unittest.main()
