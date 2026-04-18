import datetime
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from booking_scheduler import (
    build_main_command,
    describe_task,
    normalize_task,
    release_datetime_for_token,
    start_tokens_for_task,
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
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "wusi.ini")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write("[type]\nvenue=五四体育中心-羽毛球馆\nvenue_num=-1\n")
                f.write("[time]\nstart_time=20260422-0650\nend_time=20260422-0750\n")
            task = {"config": config_path}

            self.assertFalse(task_due(task, datetime.datetime(2026, 4, 19, 11, 58, 59)))
            self.assertTrue(task_due(task, datetime.datetime(2026, 4, 19, 11, 59, 0)))

    def test_task_start_tokens_are_read_from_config_not_task_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "qdb.ini")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write("[type]\nvenue=邱德拔体育馆-羽毛球场\nvenue_num=-1\n")
                f.write("[time]\nstart_time=20260425-1300\nend_time=20260425-1500\n")

            self.assertEqual(start_tokens_for_task({"config": config_path, "start_time": "20260422-0650"}), ["20260425-1300"])

    def test_normalize_task_removes_legacy_start_time(self):
        task = normalize_task({
            "config": "configs/wusi.ini",
            "start_time": "20260422-0650",
            "venue": "五四体育中心-羽毛球馆",
            "booking_start_time": "20260425-1300",
            "release_at": "2026-04-22 12:00:00",
        })

        self.assertNotIn("start_time", task)
        self.assertNotIn("venue", task)
        self.assertNotIn("booking_start_time", task)
        self.assertNotIn("release_at", task)

    def test_describe_task_includes_booking_details_from_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "wusi.ini")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write("[type]\nvenue=五四体育中心-羽毛球馆\nvenue_num=-1\n")
                f.write("[time]\nstart_time=20260425-1300\nend_time=20260425-1500\n")

            described = describe_task({"config": config_path, "lead_seconds": 60}, today=datetime.date(2026, 4, 18))

            self.assertEqual(described["venue"], "五四体育中心-羽毛球馆")
            self.assertEqual(described["booking_start_time"], "20260425-1300")
            self.assertEqual(described["booking_end_time"], "20260425-1500")
            self.assertEqual(described["release_at"], "2026-04-22 12:00:00")
            self.assertEqual(described["run_after"], "2026-04-22 11:59:00")

    def test_main_command_uses_config_and_once_flag(self):
        command = build_main_command({"config": "configs/wusi.ini", "browser": "firefox"})

        self.assertEqual(command[-6:], ["main.py", "--config", "configs/wusi.ini", "--browser", "firefox", "--once"])
        self.assertIn("--once", command)


if __name__ == "__main__":
    unittest.main()
