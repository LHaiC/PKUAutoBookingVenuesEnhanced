import os
import unittest


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(ROOT_DIR, "web_dashboard", "templates", "index.html")


class WebUiTemplateTests(unittest.TestCase):
    def test_time_fields_are_not_duplicated_in_generic_config_grid(self):
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            template = f.read()

        fields_array = template.split("const fields = [", 1)[1].split("];", 1)[0]

        self.assertNotIn('["time", "start_time"]', fields_array)
        self.assertNotIn('["time", "end_time"]', fields_array)
        self.assertIn('id="advanced-start-time"', template)
        self.assertIn('id="advanced-end-time"', template)


if __name__ == "__main__":
    unittest.main()
