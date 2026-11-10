from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from run_regression import answer_numbers, check_case


class RegressionEvaluatorTests(unittest.TestCase):
    def test_thousands_separator_and_tolerance(self):
        self.assertIn(18180.2, answer_numbers("18,180.2 条/小时"))
        case = {"expect": {"numbers": [{"value": 18180.1667, "tol": 0.1}]}}
        result = {"answer": "18,180.2 条/小时【aggregate:abc】",
                  "evidence": [{"evidence_id": "aggregate:abc"}], "warnings": [], "tool_trace": []}
        self.assertTrue(all(item["passed"] for item in check_case(case, result)))

    def test_message_mode_needs_matching_call_and_evidence(self):
        case = {"expect": {"call": {"tool": "get_vessel_track", "args": {
            "mmsi": "367457250", "granularity": "message"}}}}
        result = {"answer": "有记录【track:abc】", "evidence": [{"evidence_id": "track:abc"}],
                  "warnings": [], "tool_trace": [{"tool": "get_vessel_track",
                                                 "arguments": {"mmsi": "367457250"}}]}
        self.assertEqual([item["name"] for item in check_case(case, result) if not item["passed"]],
                         ["tool_arguments", "message_evidence"])

    def test_segment_count_is_not_gap_count(self):
        case = {"question": "有多少条 AIS 记录？"}
        result = {"answer": "有 1244 条，存在 1 段超过 15 分钟的观测空缺。",
                  "evidence": [{"evidence_id": "message_track:abc", "gap_segment_count": 1}],
                  "warnings": [], "tool_trace": []}
        failed = [item["name"] for item in check_case(case, result) if not item["passed"]]
        self.assertEqual(failed, ["gap_count_semantics"])
        result["answer"] = "有 1244 条、1 个连续片段，0 处超过 15 分钟的空缺。"
        self.assertTrue(all(item["passed"] for item in check_case(case, result)))

    def test_chinese_question_requires_chinese_answer(self):
        case = {"question": "今天可以通航吗？"}
        result = {"answer": "I cannot determine this today.",
                  "evidence": [{"evidence_id": "overview:abc"}], "warnings": [], "tool_trace": []}
        self.assertEqual([item["name"] for item in check_case(case, result) if not item["passed"]],
                         ["chinese_answer"])


if __name__ == "__main__":
    unittest.main()
