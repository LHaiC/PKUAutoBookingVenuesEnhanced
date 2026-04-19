import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main


class MainFlowTests(unittest.TestCase):
    def test_page_submits_order_before_payment(self):
        calls = []

        def fake_load_config(_config):
            return (
                "user",
                "password",
                "venue",
                -1,
                "7-0650",
                "7-0750",
                False,
                "",
                "",
                "",
                "",
                True,
                "http://localhost:8000",
                10,
                False,
            )

        class FakeDriver:
            def quit(self):
                calls.append("quit")

        patches = {
            "load_config": fake_load_config,
            "judge_exceeds_days_limit": lambda _start, _end: (["7-0650"], ["7-0750"], [1], ""),
            "build_driver": lambda _browser, headless=True: FakeDriver(),
            "login": lambda *_args, **_kwargs: calls.append("login") or "login\n",
            "go_to_venue": lambda *_args, **_kwargs: calls.append("venue") or (True, "venue\n"),
            "book": lambda *_args, **_kwargs: calls.append("book") or (True, "book\n", "start", "end", 1),
            "click_agree": lambda *_args, **_kwargs: calls.append("agree") or "agree\n",
            "click_book": lambda *_args, **_kwargs: calls.append("book_confirm") or "book_confirm\n",
            "verify": lambda *_args, **_kwargs: calls.append("verify") or "verify\n",
            "click_submit_order": lambda *_args, **_kwargs: calls.append("submit") or "submit\n",
            "click_pay": lambda *_args, **_kwargs: calls.append("pay") or "pay\n",
            "log_status": lambda *_args, **_kwargs: calls.append("log"),
        }
        originals = {name: getattr(main, name) for name in patches}
        original_sleep = main.time.sleep
        original_imported_sleep = main.sleep
        try:
            for name, value in patches.items():
                setattr(main, name, value)
            main.time.sleep = lambda _seconds: None
            main.sleep = lambda _seconds: None

            self.assertTrue(main.page("config.ini", "firefox"))

            self.assertLess(calls.index("verify"), calls.index("submit"))
            self.assertLess(calls.index("submit"), calls.index("pay"))
        finally:
            for name, value in originals.items():
                setattr(main, name, value)
            main.time.sleep = original_sleep
            main.sleep = original_imported_sleep

    def test_page_waits_until_release_after_login_before_entering_venue(self):
        calls = []

        def fake_load_config(_config):
            return (
                "user",
                "password",
                "venue",
                -1,
                "7-0650",
                "7-0750",
                False,
                "",
                "",
                "",
                "",
                True,
                "http://localhost:8000",
                10,
                False,
            )

        class FakeDriver:
            def quit(self):
                calls.append("quit")

        patches = {
            "load_config": fake_load_config,
            "judge_exceeds_days_limit": lambda _start, _end: (["7-0650"], ["7-0750"], [1], ""),
            "build_driver": lambda _browser, headless=True: FakeDriver(),
            "login": lambda *_args, **_kwargs: calls.append("login") or "login\n",
            "wait_until_datetime": lambda *_args, **_kwargs: calls.append("wait"),
            "go_to_venue": lambda *_args, **_kwargs: calls.append("venue") or (False, "venue\n"),
            "log_status": lambda *_args, **_kwargs: calls.append("log"),
        }
        originals = {name: getattr(main, name) for name in patches}
        original_sleep = main.time.sleep
        original_imported_sleep = main.sleep
        try:
            for name, value in patches.items():
                setattr(main, name, value)
            main.time.sleep = lambda _seconds: None
            main.sleep = lambda _seconds: None

            self.assertFalse(main.page("config.ini", "firefox", wait_until="2026-04-19 12:00:00"))

            self.assertLess(calls.index("login"), calls.index("wait"))
            self.assertLess(calls.index("wait"), calls.index("venue"))
        finally:
            for name, value in originals.items():
                setattr(main, name, value)
            main.time.sleep = original_sleep
            main.sleep = original_imported_sleep


if __name__ == "__main__":
    unittest.main()
