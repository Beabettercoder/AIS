from __future__ import annotations

import unittest

from maritime_agent.datetime_parser import infer_date_order, parse_utc


class DatetimeParserTests(unittest.TestCase):
    def test_supported_time_families_normalize_to_utc(self):
        cases = {
            "2024-08-18T00:00:01Z": "2024-08-18T00:00:01Z",
            "2024-08-18 08:00:01+08:00": "2024-08-18T00:00:01Z",
            "1723939201": "2024-08-18T00:00:01Z",
            "1723939201000": "2024-08-18T00:00:01Z",
            "2024年8月18日 00时00分01秒": "2024-08-18T00:00:01Z",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(parse_utc(source, True), expected)

    def test_column_evidence_selects_day_first(self):
        profile = infer_date_order(["18/08/2024 00:00:01", "19/08/2024 00:00:02"])
        self.assertEqual(profile["date_order"], "DMY")
        self.assertEqual(parse_utc("18/08/2024 00:00:01", True, profile["date_order"]),
                         "2024-08-18T00:00:01Z")

    def test_naive_time_still_requires_explicit_timezone_policy(self):
        with self.assertRaisesRegex(ValueError, "必须明确选择按 UTC 解释"):
            parse_utc("18/08/2024 00:00:01", False, "DMY")


if __name__ == "__main__":
    unittest.main()
