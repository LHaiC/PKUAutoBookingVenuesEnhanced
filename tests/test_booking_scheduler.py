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
    run_task_until_success,
    start_tokens_for_task,
    task_due,
    target_date_from_token,
)
import booking_scheduler


class BookingSchedulerTests(unittest.TestCase):
    def test_absolute_token_release_time_is_three_days_before_noon(self):
        release = release_datetime_for_token("20260422-0650")

        self.assertEqual(release, datetime.datetime(2026, 4, 19, 12, 0))

    def test_weekday_token_maps_to_next_matching_day(self):
        today = datetime.date(2026, 4, 18)

        self.assertEqual(target_date_from_token("3-0650", today), datetime.date(2026, 4, 22))

    def test_same_weekday_token_maps_to_next_week(self):
        today = datetime.date(2026, 4, 18)

        self.assertEqual(target_date_from_token("6-1300", today), datetime.date(2026, 4, 25))
        self.assertEqual(release_datetime_for_token("6-1300", today), datetime.datetime(2026, 4, 22, 12, 0))

    def test_task_is_due_one_minute_before_release_time(self):
        task = {"config": "config.ini", "start_time": "20260422-0650", "end_time": "20260422-0750"}

        self.assertFalse(task_due(task, datetime.datetime(2026, 4, 19, 11, 58, 59)))
        self.assertTrue(task_due(task, datetime.datetime(2026, 4, 19, 11, 59, 0)))

    def test_task_start_tokens_are_read_from_task_payload(self):
        self.assertEqual(
            start_tokens_for_task({"config": "config.ini", "start_time": "20260422-0650"}),
            ["20260422-0650"],
        )

    def test_normalize_task_keeps_booking_fields(self):
        task = normalize_task({
            "config": "configs/wusi.ini",
            "start_time": "20260422-0650",
            "end_time": "20260422-0750",
            "venue": "五四体育中心-羽毛球馆",
            "booking_start_time": "20260425-1300",
            "release_at": "2026-04-22 12:00:00",
        })

        self.assertEqual(task["start_time"], "20260422-0650")
        self.assertEqual(task["end_time"], "20260422-0750")
        self.assertEqual(task["venue"], "五四体育中心-羽毛球馆")
        self.assertNotIn("booking_start_time", task)
        self.assertNotIn("release_at", task)

    def test_describe_task_includes_booking_details_from_task(self):
        described = describe_task(
            {
                "config": "config.ini",
                "venue": "五四体育中心-羽毛球馆",
                "venue_num": "-1",
                "start_time": "20260425-1300",
                "end_time": "20260425-1500",
                "lead_seconds": 60,
            },
            today=datetime.date(2026, 4, 18),
        )

        self.assertEqual(described["venue"], "五四体育中心-羽毛球馆")
        self.assertEqual(described["booking_start_time"], "20260425-1300")
        self.assertEqual(described["booking_end_time"], "20260425-1500")
        self.assertEqual(described["release_at"], "2026-04-22 12:00:00")
        self.assertEqual(described["booking_action_at"], "2026-04-22 12:00:00")
        self.assertEqual(described["run_after"], "2026-04-22 11:59:00")

    def test_main_command_uses_config_and_once_flag(self):
        command = build_main_command(
            {
                "config": "config.ini",
                "browser": "firefox",
                "venue": "五四体育中心-羽毛球馆",
                "venue_num": "-1",
                "start_time": "20260425-1300",
                "end_time": "20260425-1500",
            },
            today=datetime.date(2026, 4, 18),
        )

        self.assertIn("--once", command)
        self.assertIn("--wait-until", command)
        self.assertIn("--venue", command)
        self.assertIn("五四体育中心-羽毛球馆", command)
        self.assertIn("--start-time", command)
        self.assertIn("20260425-1300", command)

    def test_run_task_stops_immediately_when_config_has_no_login(self):
        statuses = []
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "bad.ini")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write("[glm_ocr]\nenabled = true\n")

            original_write = booking_scheduler.write_scheduler_status
            try:
                booking_scheduler.write_scheduler_status = lambda payload: statuses.append(payload)
                result = run_task_until_success(
                    {
                        "id": "bad",
                        "name": "bad task",
                        "config": config_path,
                        "venue": "五四体育中心-羽毛球馆",
                        "start_time": "20260425-1300",
                        "end_time": "20260425-1500",
                    }
                )
            finally:
                booking_scheduler.write_scheduler_status = original_write

        self.assertFalse(result)
        self.assertEqual(statuses[0]["status"], "invalid_config")
        self.assertIn("missing [login]", statuses[0]["error"])


if __name__ == "__main__":
    unittest.main()
