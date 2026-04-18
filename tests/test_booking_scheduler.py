import datetime
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from booking_scheduler import (
    build_main_command,
    release_datetime_for_token,
    task_due,
    target_date_from_token,
)


class BookingSchedulerTests(unittest.TestCase):
    def test_absolute_token_release_time_is_three_days_before_noon(self):
        release = release_datetime_for_token("20260422-0650")

        self.assertEqual(release, datetime.datetime(2026, 4, 19, 12, 0))

    def test_weekday_token_maps_to_next_matching_day(self):
        today = datetime.date(2026, 4, 18)

        self.assertEqual(target_date_from_token("3-0650", today), datetime.date(2026, 4, 22))

    def test_task_is_due_one_minute_before_release_time(self):
        task = {"start_time": "20260422-0650"}

        self.assertFalse(task_due(task, datetime.datetime(2026, 4, 19, 11, 58, 59)))
        self.assertTrue(task_due(task, datetime.datetime(2026, 4, 19, 11, 59, 0)))

    def test_main_command_uses_config_and_once_flag(self):
        command = build_main_command({"config": "configs/wusi.ini", "browser": "firefox"})

        self.assertEqual(command[-6:], ["main.py", "--config", "configs/wusi.ini", "--browser", "firefox", "--once"])
        self.assertIn("--once", command)


if __name__ == "__main__":
    unittest.main()
