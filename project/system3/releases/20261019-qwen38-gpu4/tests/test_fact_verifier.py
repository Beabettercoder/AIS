from __future__ import annotations

import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from maritime_agent.agent import validate_answer
from maritime_agent.fact_verifier import (
    extract_evidence_numbers_and_claims,
    extract_metric_claims,
    is_claim_grounded,
    verify_numerical_grounding,
)


class FactVerifierTests(unittest.TestCase):
    def test_extract_metric_claims(self):
        sample_text = (
            "2024年7月6日第1轮观测发现，台风中心气压降至 978 hPa，最大风速达 48.2 节（相当于 24.8 m/s），"
            "阵风超过 65 knots。海浪达 4.5米浪高。在距离中心 3.2 公里（即 3200 米 或 1.73 海里）处，"
            "共有 12 艘船舶受影响，累计产生 12,109 条记录，模型置信度为 85%。"
            "前 5 条记录显示航向稳定。"
        )
        claims = extract_metric_claims(sample_text)
        categories = {c.category for c in claims}
        self.assertIn("pressure", categories)
        self.assertIn("wind_speed", categories)
        self.assertIn("wave_height", categories)
        self.assertIn("distance", categories)
        self.assertIn("vessel_count", categories)
        self.assertIn("record_count", categories)
        self.assertIn("percentage", categories)

        # Confirm non-metrics are filtered: 2024 (year), 1 (ordinal '第1轮'), 5 (pagination '前5条')
        values = {c.value for c in claims}
        self.assertNotIn(2024.0, values)
        self.assertNotIn(1.0, values)
        self.assertNotIn(5.0, values)

    def test_unit_conversions_and_grounding(self):
        # Evidence only has m/s and km and ratio
        evidence = {
            "evidence_id": "sensor_01",
            "wind_speed_mps": 24.8,
            "drift_km": 3.2,
            "confidence_ratio": 0.85,
            "pressure_hpa": 978.0,
            "vessels": [{"mmsi": "111"}, {"mmsi": "222"}],  # len = 2
        }
        all_nums, cat_nums = extract_evidence_numbers_and_claims(evidence)

        # 1. Knots converted to m/s: 48.2 knots * 0.5144 = 24.79 m/s ≈ 24.8
        claims_knots = extract_metric_claims("风速 48.2 节")
        self.assertTrue(is_claim_grounded(claims_knots[0], all_nums, cat_nums, set()))

        # 2. Meters converted to km: 3200 m / 1000 = 3.2 km
        claims_dist = extract_metric_claims("走锚 3200 米")
        self.assertTrue(is_claim_grounded(claims_dist[0], all_nums, cat_nums, set()))

        # 3. Percentage: 85% / 100 = 0.85
        claims_pct = extract_metric_claims("可信度为 85%")
        self.assertTrue(is_claim_grounded(claims_pct[0], all_nums, cat_nums, set()))

        # 4. Count from list length: 2 艘
        claims_vessel = extract_metric_claims("共 2 艘船")
        self.assertTrue(is_claim_grounded(claims_vessel[0], all_nums, cat_nums, set()))

    def test_hallucination_detection(self):
        evidence = {
            "evidence_id": "cube:beryl",
            "wind_speed": 48.0,
            "gust": 65.0,
            "pressure": 978.0,
            "vessel_count": 12,
        }
        # Hallucinated wind speed 99.5 knots and 88 vessels
        fake_text = "台风最大风速达到 99.5 节【cube:beryl】，受困船舶达到 88 艘【cube:beryl】。"
        warnings = verify_numerical_grounding(fake_text, evidence)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("99.5" in w and "风速/航速" in w for w in warnings))
        self.assertTrue(any("88" in w and "船舶/目标数量" in w for w in warnings))

        # Grounded text should pass with 0 warnings
        true_text = "台风最大风速为 48 节，阵风达 65 节，气压 978 hPa，涉及 12 艘船舶【cube:beryl】。"
        clean_warnings = verify_numerical_grounding(true_text, evidence)
        self.assertEqual(clean_warnings, [])

    def test_query_context_immunity(self):
        # When user asks about a hypothetical number, repeating it should not cause a false alarm
        evidence = {"evidence_id": "ev:1", "match_count": 0}
        question = "请检索风速在 55 节以上的船舶"
        answer = "在数据库中检索风速超过 55 节的船舶，未发现匹配结果【ev:1】。"
        warnings = verify_numerical_grounding(answer, evidence, context_text=question)
        self.assertEqual(warnings, [])

    def test_timestamp_is_not_treated_as_free_numbers(self):
        evidence = {
            "evidence_id": "ev:ts",
            "time": "2024-07-06T17:00:00Z",
            "value": 10.0,
        }
        # 17 from 17:00:00 should NOT allow 17 节 to be falsely verified
        fake_claim = "风速达到 17 节【ev:ts】。"
        warnings = verify_numerical_grounding(fake_claim, evidence)
        self.assertTrue(any("17" in w for w in warnings))

    def test_validate_answer_integration(self):
        messages = [
            HumanMessage(content="台风情况如何？"),
            ToolMessage(content=json.dumps({"evidence_id": "ev:ok", "wind_speed": 48.0}), tool_call_id="c1"),
            AIMessage(content="风速高达 120 节【ev:ok】。")
        ]
        answer, valid_ev, warnings = validate_answer(messages)
        self.assertEqual(len(valid_ev), 1)
        self.assertTrue(any("120" in w and "数值事实存疑" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
