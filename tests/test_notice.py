import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import notice


class NoticeTests(unittest.TestCase):
    def test_pushplus_notification_skips_when_no_config(self):
        if not os.path.exists("config.ini"):
            self.skipTest("config.ini not found")
            return

        from configparser import ConfigParser
        conf = ConfigParser()
        conf.read("config.ini", encoding="utf8")
        if not conf.has_section("wechat"):
            self.skipTest("config.ini has no [wechat] section")
            return
        if not conf.getboolean("wechat", "wechat_notice", fallback=False):
            self.skipTest("config.ini wechat_notice is False")
            return
        sckey = conf.get("wechat", "SCKEY", fallback="").strip()
        if not sckey or sckey == "XXXX":
            self.skipTest("config.ini SCKEY is empty or placeholder")
            return

        result = notice.wechat_notification(
            user_name="测试用户",
            venue="测试场馆",
            venue_num="1",
            start_time="2026-04-20 10:00",
            end_time="2026-04-20 12:00",
            sckey=sckey,
        )
        self.assertIn("成功", result)


if __name__ == "__main__":
    unittest.main()
