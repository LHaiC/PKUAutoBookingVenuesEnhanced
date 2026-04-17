import base64
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_server_transformers import decode_data_uri, parse_model_output


class OcrServerTransformerTests(unittest.TestCase):
    def test_decode_data_uri_accepts_prefixed_uri(self):
        payload = base64.b64encode(b"abc").decode("utf-8")
        self.assertEqual(decode_data_uri(f"data:image/png;base64,{payload}"), b"abc")

    def test_decode_data_uri_accepts_raw_base64(self):
        payload = base64.b64encode(b"abc").decode("utf-8")
        self.assertEqual(decode_data_uri(payload), b"abc")

    def test_parse_model_output_reads_json_list(self):
        output = '[{"text": "件", "bbox": [140, 50, 190, 110], "confidence": 0.94}]'
        parsed = parse_model_output(output)
        self.assertEqual(parsed[0]["text"], "件")
        self.assertEqual(parsed[0]["bbox"], [140, 50, 190, 110])
        self.assertEqual(parsed[0]["confidence"], 0.94)

    def test_parse_model_output_extracts_single_chinese_characters(self):
        parsed = parse_model_output("识别结果：件叶结")
        self.assertEqual(
            parsed,
            [
                {"text": "件", "bbox": [], "confidence": 0.50},
                {"text": "叶", "bbox": [], "confidence": 0.50},
                {"text": "结", "bbox": [], "confidence": 0.50},
            ],
        )


if __name__ == "__main__":
    unittest.main()
