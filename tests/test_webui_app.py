import tempfile
import unittest
import json
from pathlib import Path

import web_dashboard.app as webapp


def write_config(path, venue, start_time, end_time):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "[type]",
            f"venue = {venue}",
            "venue_num = -1",
            "",
            "[time]",
            f"start_time = {start_time}",
            f"end_time = {end_time}",
            "",
        ]),
        encoding="utf-8",
    )


class WebUiAppTests(unittest.TestCase):
    def test_configs_endpoint_lists_real_configs_without_booking_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_config(root / "config.ini", "五四体育中心-羽毛球馆", "6-1300", "6-1500")
            write_config(root / "config_qdb.ini", "邱德拔体育馆-羽毛球场", "6-1300", "6-1500")
            write_config(root / "configs" / "wusi_morning.ini", "五四体育中心-羽毛球馆", "3-0650", "3-0850")
            write_config(root / "config.example.ini", "示例", "", "")

            original_root = webapp.ROOT_DIR
            try:
                webapp.ROOT_DIR = root
                response = webapp.app.test_client().get("/api/configs")
            finally:
                webapp.ROOT_DIR = original_root

        self.assertEqual(response.status_code, 200)
        configs = response.get_json()["configs"]
        by_path = {item["path"]: item for item in configs}

        self.assertEqual(set(by_path), {"config.ini", "config_qdb.ini", "configs/wusi_morning.ini"})
        self.assertTrue(by_path["config.ini"]["legacy_booking"])
        self.assertTrue(by_path["config_qdb.ini"]["legacy_booking"])
        self.assertTrue(by_path["configs/wusi_morning.ini"]["legacy_booking"])
        self.assertNotIn("venue", by_path["config.ini"])
        self.assertNotIn("start_time", by_path["config.ini"])

    def test_config_save_removes_legacy_booking_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_config(root / "config.ini", "五四体育中心-羽毛球馆", "6-1300", "6-1500")

            original_root = webapp.ROOT_DIR
            try:
                webapp.ROOT_DIR = root
                response = webapp.app.test_client().post(
                    "/api/config",
                    json={
                        "path": "config.ini",
                        "values": {
                            "login": {"user_name": "u", "password": "p"},
                            "type": {"venue": "should-not-save"},
                            "time": {"start_time": "should-not-save"},
                        },
                    },
                )
            finally:
                webapp.ROOT_DIR = original_root

            content = (root / "config.ini").read_text(encoding="utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("[type]", content)
        self.assertNotIn("[time]", content)
        self.assertIn("user_name = u", content)

    def test_status_endpoint_marks_stale_retrying_scheduler_as_stopped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scheduler_status.json").write_text(
                json.dumps({"status": "retrying", "task_name": "old"}),
                encoding="utf-8",
            )

            original_root = webapp.ROOT_DIR
            original_status = webapp.SCHEDULER_STATUS_FILE
            original_pid = webapp.SCHEDULER_PID_FILE
            try:
                webapp.ROOT_DIR = root
                webapp.SCHEDULER_PID_FILE = root / ".scheduler.pid"
                webapp.SCHEDULER_STATUS_FILE = root / "scheduler_status.json"
                response = webapp.app.test_client().get("/api/status")
            finally:
                webapp.ROOT_DIR = original_root
                webapp.SCHEDULER_STATUS_FILE = original_status
                webapp.SCHEDULER_PID_FILE = original_pid

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["scheduler_running"])
        self.assertEqual(payload["scheduler"]["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
