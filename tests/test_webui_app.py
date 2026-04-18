import tempfile
import unittest
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
    def test_configs_endpoint_lists_real_configs_with_booking_summary(self):
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
        self.assertEqual(by_path["config.ini"]["venue"], "五四体育中心-羽毛球馆")
        self.assertEqual(by_path["config_qdb.ini"]["start_time"], "6-1300")
        self.assertEqual(by_path["configs/wusi_morning.ini"]["end_time"], "3-0850")


if __name__ == "__main__":
    unittest.main()
