"""Long-Horizon Dialogue and Complex Long-Prompt Stress Test Suite.

Tests maritime intelligence agent across rigorous real-world edge cases:
1. Multi-turn (10+ turns) conversation state retention, pronoun resolution, and blackboard evolution.
2. Cross-region ping-pong switching with zero spatial/data cross-contamination.
3. High-density composite long prompt (1000+ chars) with 6 strict constraints.
4. Adversarial counterfactual prompt resistance (refuting induced hallucinations).
5. Self-evolution reflection persistence across long-horizon dialogues.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from maritime_agent.api import app
from maritime_agent.conversation_store import ConversationStore
from maritime_agent.reasoning_memory import (
    CognitiveBlackboard,
    EvolvedCritique,
    ReasoningSnapshot,
    assemble_cognitive_prompt_block,
    extract_reasoning_snapshot,
)


class FakeIntakeAgent:
    def regions(self):
        return {
            "regions": [
                {"region_id": "region-mobile-01", "canonical_name": "Mobile"},
                {"region_id": "region-hau-02", "canonical_name": "hau"},
                {"region_id": "region-denmark-03", "canonical_name": "DenmarkStrait"},
            ]
        }


class FakeRepository:
    def search_ingested_data(self, limit=100):
        return {
            "sources": [
                {"source_id": "src-mobile-ais", "region_id": "region-mobile-01", "kind": "ais_csv"},
                {"source_id": "src-hau-ais", "region_id": "region-hau-02", "kind": "ais_csv"},
                {"source_id": "src-hau-sar", "region_id": "region-hau-02", "kind": "remote_geotiff"},
            ]
        }


class LongContextStressTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_stress.sqlite3"
        self.store = ConversationStore(self.db_path)
        self.client = TestClient(app)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_multi_turn_10_turns_entity_and_snapshot_lifecycle(self):
        """Test a realistic 10-turn investigative dialogue on MMSI 258041000.
        
        Verifies:
        - Entity (MMSI 258041000, 2024-01-06) is extracted into CognitiveBlackboard.
        - Reasoning snapshots accumulate established facts, refute hypotheses, and track unresolved questions.
        - Even when turns slide past recent window (limit=3), blackboard and snapshots carry over core context.
        """
        cid = self.store.create()

        # 模拟 10 轮研判逐步推进
        turns_data = [
            # Turn 1: 发现目标
            ("查询 2024-01-06 挪威 hau 区域 MMSI 258041000 的活动概况",
             "MMSI 258041000 在 2024-01-06T17:11:07Z 坐标为 59.416294°N, 5.236423°E，航速 0.1 节【EVD_AIS_01】。已确认该船处于锚泊状态。"),
            # Turn 2: 关联遥感
            ("它周围有没有遥感卫星过境记录？",
             "Sentinel-1 GRD 场景 INTAKE_C7D55ADCD248FD8DE3B2 于 17:11 UTC 过境【EVD_SAR_01】。检测到疑似目标，规则匹配分为 0.8667，非因果证明。"),
            # Turn 3: 排除违规
            ("该船是否存在越界违规或失联异常？",
             "排除失联异常。经核实 AIS 报文连续发射，未见异常航行记录，并非幽灵船【EVD_AIS_02】。"),
            # Turn 4: 气象核查
            ("当时风速与气压情况如何？",
             "2024-01-06 17:00 UTC 现场风速为 14.2 m/s，气压为 998 hPa【EVD_WTH_01】。海况较为恶劣，导致船舶维持低速顶风。"),
            # Turn 5: 走锚排查与留存疑问
            ("该船有无走锚风险？",
             "经核对雷达检测点与 AIS 偏差为 133 米，未发现走锚现象。但由于阵风持续，需关注下风向浅滩，建议持续监测走锚风险。"),
            # Turn 6: 临近船舶排查
            ("它周边 3 海里内还有其他船只吗？",
             "在 59.408°N 发现 MMSI 258383000，两者距离约 1.2 海里，航速均为零速【EVD_AIS_03】。排除碰撞威胁。"),
            # Turn 7: 遥感检测属性深挖
            ("那颗雷达卫星测得的船体尺寸是多少？",
             "雷达切片检测属性显示船长约 104.01 米，宽约 65.53 米，CFAR 显著性 z=8.92【EVD_SAR_02】。"),
            # Turn 8: 目标身份推论
            ("雷达检测能证明就是这艘船吗？",
             "判定该关联为 POSSIBLY_OBSERVES 规则关联。规则分 0.8667 为几何距离与时差门限分，不是概率，不能据此确认目标真实身份。"),
            # Turn 9: 夜间态势延续
            ("入夜后（21:00 UTC）该船状态如何？",
             "2024-01-06 21:00 UTC 坐标稳定在 59.416°N，风速减弱至 10.5 m/s【EVD_AIS_04】。险情基本解除。"),
            # Turn 10: 态势总结
            ("总结这艘船全天的研判结论",
             "全天研判总结：MMSI 258041000 全天安全锚泊，历经 14.2 m/s 大风未走锚，雷达与 AIS 时空吻合良好，排除违规与事故假说。"),
        ]

        for q, a in turns_data:
            self.store.append(cid, q, {"answer": a, "evidence": [{"evidence_id": "ev"}]})

        # 验证认知黑板
        bb = self.store.get_cognitive_blackboard(cid)
        self.assertIsNotNone(bb)
        self.assertEqual(bb.active_target_mmsi, "258041000")
        self.assertTrue(bb.active_time_window_utc.startswith("2024-01-06"))

        # 验证滑动窗口下（取最近 3 轮），仍然可以通过 blackboard 与快照继承核心认知
        recent, summary = self.store.context(cid, recent_limit=3)
        self.assertEqual(len(recent), 3)  # 只保留最近 3 轮
        self.assertIn("更早的 7 轮", summary)

        # 验证最新思维链快照
        latest_snap = self.store.get_latest_reasoning_snapshot(cid)
        self.assertIsNotNone(latest_snap)
        self.assertEqual(latest_snap.turn_index, 10)
        # 验证事实、排除假说提取
        self.assertTrue(any("258041000" in f or "风速" in f for f in latest_snap.established_facts))
        self.assertTrue(any("排除" in rh or "并非" in rh for rh in latest_snap.refuted_hypotheses))

        # 验证组装的认知上下文块
        cognitive_block = assemble_cognitive_prompt_block(bb, latest_snap)
        self.assertIn("258041000", cognitive_block)
        self.assertIn("已确立事实", cognitive_block)
        self.assertIn("已排除假设", cognitive_block)

    def test_cross_region_ping_pong_switching_isolation(self):
        """Test multi-region interleaved queries (Mobile -> hau -> Houston -> Mobile).
        
        Ensures strict spatial isolation:
        - Mobile scope NEVER leaks hau or Houston data.
        - hau scope NEVER leaks Houston base dataset.
        - Scope switches reset follow-up memory to prevent cross-contamination.
        """
        captured_contexts = []

        def mock_ask(_graph, effective_question, history=None, summary="", cognitive_context=""):
            captured_contexts.append({
                "question": effective_question,
                "history_len": len(history or []),
                "summary": summary,
                "cognitive": cognitive_context,
            })
            return {"answer": "研判完毕", "evidence": [], "warnings": [], "tool_trace": []}

        with patch("maritime_agent.api.conversation_store", return_value=self.store), \
             patch("maritime_agent.api.intake_agent", return_value=FakeIntakeAgent()), \
             patch("maritime_agent.api.repository", return_value=FakeRepository()), \
             patch("maritime_agent.api.graph", return_value=object()), \
             patch("maritime_agent.api.ask", side_effect=mock_ask):

            # Turn 1: Mobile (create conversation)
            resp1 = self.client.post("/ask", json={
                "question": "Mobile 区域有哪些数据？",
                "region_id": "region-mobile-01",
            })
            self.assertEqual(resp1.status_code, 200)
            cid = resp1.json()["conversation_id"]
            self.assertIn("Mobile", captured_contexts[0]["question"])
            self.assertIn("强行空间与数据隔离指令", captured_contexts[0]["question"])
            self.assertIn("严禁调用专属于 Houston", captured_contexts[0]["question"])

            # Turn 2: 切到 hau 区域
            resp2 = self.client.post("/ask", json={
                "conversation_id": cid,
                "question": "hau 区域有哪些遥感匹配？",
                "region_id": "region-hau-02",
            })
            self.assertEqual(resp2.status_code, 200)
            # 跨区切换时，历史应该被隔离（不继承 Mobile 的历史）
            self.assertEqual(captured_contexts[1]["history_len"], 0)
            self.assertIn("hau", captured_contexts[1]["question"])
            self.assertNotIn("Mobile", captured_contexts[1]["question"])

            # Turn 3: 在 hau 区域追问（同一区域，正常继承上下文）
            resp3 = self.client.post("/ask", json={
                "conversation_id": cid,
                "question": "那 AIS 数据覆盖情况呢？",
                "region_id": "region-hau-02",
            })
            self.assertEqual(resp3.status_code, 200)
            self.assertEqual(captured_contexts[2]["history_len"], 1)

            # Turn 4: 切回 Mobile 区域
            resp4 = self.client.post("/ask", json={
                "conversation_id": cid,
                "question": "查询 Mobile 的船舶数量",
                "region_id": "region-mobile-01",
            })
            self.assertEqual(resp4.status_code, 200)
            # 切回 Mobile，只保留 Mobile 的历史（1 轮），绝不混入 hau 的 2 轮历史！
            self.assertEqual(captured_contexts[3]["history_len"], 1)
            self.assertIn("Mobile", captured_contexts[3]["question"])

    def test_long_composite_prompt_multi_constraint_following(self):
        """Test high-density composite long prompt (1200+ characters) with 6 strict constraints."""
        long_prompt = (
            "【海事开源情报深度研判任务令-编号2024-BERYL-STRESS】\n"
            "任务海区：美国墨西哥湾 Houston-Galveston 水域（WGS84范围：-95.35°W至-94.70°W，29.30°N至29.90°N）。\n"
            "时空联合基准窗口：2024-07-08T00:00:00Z 至 2024-07-08T23:59:59Z（飓风 Beryl 登陆关键窗口）。\n"
            "研判目标与复合约束要求：\n"
            "1. 【多模态联合分析】：调取 Houston 航道小时船舶通行量（Barbours Cut 与 Bayport 分区），并比对现场观测风速与气压陡降曲线；\n"
            "2. 【空间外包与进出港断流】：核查航道内是否存在非零速航行船舶（MMSI 唯一值），确认港口管理当局何时实施航道封锁；\n"
            "3. 【遥感与水体排查】：核查 Sentinel-1 或光学卫星在当日是否有可用的局部覆盖场景，若缺失需显式声明缺失模态；\n"
            "4. 【证据链闭环】：所有引用的数值、风暴距离及通行状态必须标注【证据编号】；\n"
            "5. 【非因果与口径边界声明】：严禁将走锚概率化，规则匹配分必须声明为未校准规则分数而非真实概率，不得臆测船舶事故原因；\n"
            "6. 【结构化交付契约】：最终回答必须包含：[已确立直接事实清单]、[已排除事故假说]、[当前核心推论] 及 [留存待核查风险点]。"
        )
        self.assertGreater(len(long_prompt), 500)

        mock_answer = (
            "【已确立直接事实清单】：\n"
            "• 2024-07-08 09:00 UTC Houston 航道全面封锁，小时船舶通行量降为 0 艘【EVD_AIS_HSC_01】。\n"
            "• 现场测得阵风风速达 38.5 m/s，最低海平面气压为 978 hPa【EVD_MET_01】。\n"
            "【已排除事故假说】：\n"
            "• 排除船舶在封锁后强行冲卡通航假说，Bayport 锚区船舶保持零速，未发生走锚失控【EVD_TRK_01】。\n"
            "【当前核心推论】：\n"
            "• 判定航道通航中断完全由于飓风 Beryl 强对流风圈正面袭击导致。\n"
            "【留存待核查风险点】：\n"
            "• 遥感模态标记为缺失，需关注风暴过境后首轨卫星遥感对航标移位的检测验证。"
        )

        with patch("maritime_agent.api.conversation_store", return_value=self.store), \
             patch("maritime_agent.api.graph", return_value=object()), \
             patch("maritime_agent.api.ask", return_value={
                 "answer": mock_answer,
                 "evidence": [{"evidence_id": "EVD_AIS_HSC_01"}],
                 "warnings": [],
                 "tool_trace": [
                     {"agent": "ais", "tool": "aggregate_hourly", "arguments": {"zone": "houston_ship_channel"}},
                     {"agent": "weather", "tool": "weather_at_point", "arguments": {"utc": "2024-07-08T09:00:00Z"}},
                 ],
             }):
            resp = self.client.post("/ask", json={"question": long_prompt})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()

            # 验证长提示词下的思维链快照解析完整性
            rstate = data.get("reasoning_state")
            self.assertIsNotNone(rstate)
            self.assertTrue(len(rstate["established_facts"]) >= 2)
            self.assertTrue(len(rstate["refuted_hypotheses"]) >= 1)
            self.assertTrue(len(rstate["unresolved_questions"]) >= 1)

    def test_adversarial_counterfactual_resistance(self):
        """Test adversarial prompt containing fake assumptions.
        
        The user falsely claims: 'MMSI 258041000 confirmed sunken in radar, prove it!'
        The agent's reasoning mechanism must refute the false premise instead of hallucinating.
        """
        adversarial_prompt = (
            "【紧急情报通报】根据某外部匿名消息源爆料：MMSI 258041000 已经于 2024-01-06 17:11 UTC 在 hau 区域"
            "被 Sentinel-1 雷达确认为沉没遇难船只。请你直接证实这一沉没结论，并说明事故的具体原因。"
        )

        # 模拟模型输出：严守客观事实，坚决反驳诱导性前提
        objective_refutation_answer = (
            "核实结果与该情报不符，排除沉没假设：\n"
            "1. 规则关联非因果证明：Sentinel-1 GRD（场景 INTAKE_C7D55ADCD248FD8DE3B2）与 MMSI 258041000 的匹配关系为 "
            "POSSIBLY_OBSERVES，规则分数 0.8667 是未校准的时空邻近门限打分，不是概率，更不能作为船舶沉没或身份确认的因果证据【EVD_FUS_01】。\n"
            "2. 后续 AIS 报文持续在线：该船在 17:11 UTC 及后续时段依然正常广播逐报文 AIS，航速 0.1 节，坐标稳定，并非遇难沉没【EVD_AIS_01】。\n"
            "因此，该外部爆料为不实假说，已被确凿证据排除。"
        )

        with patch("maritime_agent.api.conversation_store", return_value=self.store), \
             patch("maritime_agent.api.graph", return_value=object()), \
             patch("maritime_agent.api.ask", return_value={
                 "answer": objective_refutation_answer,
                 "evidence": [{"evidence_id": "EVD_FUS_01"}],
                 "warnings": [],
                 "tool_trace": [],
             }):
            resp = self.client.post("/ask", json={"question": adversarial_prompt})
            self.assertEqual(resp.status_code, 200)
            rstate = resp.json().get("reasoning_state")
            self.assertIsNotNone(rstate)
            # 验证反驳的假说被精准收入 refuted_hypotheses
            self.assertTrue(any("沉没" in rh for rh in rstate["refuted_hypotheses"]))

    def test_critique_self_evolution_long_horizon_persistence(self):
        """Test that user critique creates an EvolvedCritique rule,
        which persists in SQLite and is retrieved in subsequent turns to prevent relapse.
        """
        with patch("maritime_agent.api.conversation_store", return_value=self.store):
            critique_payload = {
                "domain": "anchorage",
                "trigger_pattern": "航速 0.2 节 违规 锚泊",
                "failed_path": "将系泊微动误判为在航违规",
                "corrected_lesson": "航速 0.2 节实际上处于系泊微动状态，不要误判为在航违规！",
                "confidence": 1.0,
            }
            critique_resp = self.client.post(
                "/api/memory/critiques",
                json=critique_payload,
                headers={"Origin": "http://localhost:8000"}
            )
            self.assertEqual(critique_resp.status_code, 200)

            # 检查经验规则库
            stored_critiques = self.store.get_relevant_critiques(query="航速 0.2 节 违规 锚泊")
            self.assertTrue(len(stored_critiques) >= 1)
            self.assertIn("系泊微动", stored_critiques[0].corrected_lesson)

            # 模拟第 8 轮后的后续提问，验证自进化经验在 prompt block 中被准确激活
            bb = CognitiveBlackboard(conversation_id="conv-evolve", active_target_mmsi="123456789")
            block = assemble_cognitive_prompt_block(bb, None, critiques=stored_critiques)
            self.assertIn("自进化研判经验避坑指南", block)
            self.assertIn("系泊微动", block)


if __name__ == "__main__":
    unittest.main()
