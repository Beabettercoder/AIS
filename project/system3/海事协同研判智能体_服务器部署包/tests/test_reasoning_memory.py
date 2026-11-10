from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from fastapi.testclient import TestClient

from maritime_agent.api import app, repository
from maritime_agent.conversation_store import ConversationStore
from maritime_agent.reasoning_memory import (
    CognitiveBlackboard,
    EvolvedCritique,
    ReasoningSnapshot,
    assemble_cognitive_prompt_block,
    detect_and_evolve_critique,
    distill_tool_failure_critique,
    distill_verifier_warning_critique,
    extract_reasoning_snapshot,
    update_blackboard_state,
)


class ReasoningMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_conversations.sqlite3"
        self.store = ConversationStore(self.db_path)
        self.client = TestClient(app)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_extract_reasoning_snapshot(self):
        """测试从问答文本中精准提取思维链快照（事实、假说排除、当前推论、未决疑点）。"""
        question = "油轮 367425000 在 2024-07-08 12:00 是否发生走锚？"
        answer = (
            "经核查，休斯敦海区 2024-07-08 12:00 UTC 全域极大风速达到 21.2 m/s【EVD_WIND_01】。"
            "该油轮对地航速为 1.8 节，航向 312°，已排除正常巡航假设【EVD_TRACK_01】。"
            "分析判定当前处于严重顺风走锚漂移险情。"
            "后续需持续关注该轮与周边浅滩的防搁浅安全距离。"
        )
        evidence = [{"evidence_id": "EVD_WIND_01"}, {"evidence_id": "EVD_TRACK_01"}]
        active_scope = {"region_id": "region-houston"}

        snapshot = extract_reasoning_snapshot(
            turn_index=1,
            question=question,
            answer=answer,
            evidence=evidence,
            active_scope=active_scope,
        )

        self.assertEqual(snapshot.turn_index, 1)
        self.assertEqual(snapshot.target_entities.get("mmsi"), "367425000")
        self.assertEqual(snapshot.target_entities.get("region_id"), "region-houston")
        self.assertTrue(any("21.2 m/s" in f for f in snapshot.established_facts))
        self.assertTrue(any("排除正常巡航" in rh for rh in snapshot.refuted_hypotheses))
        self.assertTrue(any("走锚漂移" in inf for inf in snapshot.active_inferences))
        self.assertTrue(any("防搁浅" in uq or "浅滩" in uq for uq in snapshot.unresolved_questions))
        self.assertIn("EVD_WIND_01", snapshot.evidence_ids)

    def test_assemble_cognitive_prompt_block(self):
        """测试将黑板与上一轮思维断点转化为高质量的前置认知引导提示词。"""
        bb = CognitiveBlackboard(
            conversation_id="conv-123",
            active_region_id="region-houston",
            active_target_mmsi="367425000",
            active_time_window_utc="2024-07-08T12:00:00Z",
        )
        prev_snap = ReasoningSnapshot(
            turn_index=1,
            established_facts=["全域极大风速达到 21.2 m/s", "主航道实施封航"],
            refuted_hypotheses=["排除主动起航出港假设"],
            active_inferences=["判定为顺风走锚漂移"],
            unresolved_questions=["待核查该轮当前与浅水区的最小避碰余量"],
            evidence_ids=["EVD_WIND_01"],
        )
        critiques = [
            EvolvedCritique(
                critique_id="CRT-01",
                domain="anchorage",
                trigger_pattern="走锚",
                failed_path="此前未结合顺风流压夹角",
                corrected_lesson="锚地低速位移必须核验与风向顺风夹角偏差",
            )
        ]

        prompt_block = assemble_cognitive_prompt_block(bb, prev_snap, critiques)

        self.assertIn("态势认知黑板", prompt_block)
        self.assertIn("367425000", prompt_block)
        self.assertIn("上一轮思维链断点承接", prompt_block)
        self.assertIn("21.2 m/s", prompt_block)
        self.assertIn("排除主动起航出港假设", prompt_block)
        self.assertIn("待核查该轮当前与浅水区的最小避碰余量", prompt_block)
        self.assertIn("自进化研判经验避坑指南", prompt_block)
        self.assertIn("锚地低速位移必须核验与风向顺风夹角偏差", prompt_block)

    def test_detect_and_evolve_critique(self):
        """测试自动捕获用户业务纠错语义并提炼自进化经验。"""
        correction_q = "不对，这艘船当时不是主动出港，而是被大风吹着走锚了！"
        critique = detect_and_evolve_critique(correction_q, history=[])
        self.assertIsNotNone(critique)
        self.assertEqual(critique.domain, "anchorage")
        self.assertIn("用户明确指示业务规则", critique.corrected_lesson)

        normal_q = "请帮我查一下这艘船的吃水深度。"
        critique_none = detect_and_evolve_critique(normal_q, history=[])
        self.assertIsNone(critique_none)

    def test_conversation_store_reasoning_lifecycle(self):
        """测试会话存储中完整的思维链快照保存、黑板更新与自进化反思生命周期。"""
        cid = self.store.create()

        # 第 1 轮
        q1 = "休斯敦 7月8日 台风登陆时油轮 367425000 状态如何？"
        resp1 = {
            "answer": "经核查，休斯敦风速达到 21.2 m/s【EVD_WIND_01】，油轮 367425000 航速 1.8 节，判定为走锚漂移。",
            "evidence": [{"evidence_id": "EVD_WIND_01"}],
            "selected_region_id": "region-houston",
        }
        self.store.append(cid, q1, resp1)

        # 验证快照是否保存
        snap1 = self.store.get_latest_reasoning_snapshot(cid)
        self.assertIsNotNone(snap1)
        self.assertEqual(snap1.target_entities.get("mmsi"), "367425000")

        # 验证黑板是否更新
        bb = self.store.get_cognitive_blackboard(cid)
        self.assertEqual(bb.active_region_id, "region-houston")
        self.assertEqual(bb.active_target_mmsi, "367425000")

        # 第 2 轮带纠错
        q2 = "不对，它实际上是避风抛锚状态！"
        resp2 = {
            "answer": "收到更正指示，已修正研判为在锚地系泊避风。",
            "evidence": [],
        }
        self.store.append(cid, q2, resp2)

        # 验证反思库是否自动沉淀新规则
        critiques = self.store.get_relevant_critiques(query="抛锚")
        self.assertTrue(len(critiques) >= 1)
        self.assertIn("用户明确指示业务规则", critiques[0].corrected_lesson)

    def test_api_memory_endpoints(self):
        """测试 /api/memory/blackboard 和 /api/memory/critiques 系列管理接口工作正常。"""
        # 1. 查反思库
        res = self.client.get("/api/memory/critiques")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("critiques", data)

        # 2. 手动创建反思
        create_payload = {
            "domain": "ais",
            "trigger_pattern": "航速低于0.2节",
            "failed_path": "误判为失控",
            "corrected_lesson": "在锚地且对地航速低于0.2节应视为正常系泊",
            "confidence": 1.0,
        }
        create_res = self.client.post("/api/memory/critiques", json=create_payload)
        self.assertEqual(create_res.status_code, 200)
        critique_id = create_res.json()["critique"]["critique_id"]

        # 3. 反馈评分 (upvote)
        fb_res = self.client.post(f"/api/memory/critiques/{critique_id}/feedback", json={"action": "upvote"})
        self.assertEqual(fb_res.status_code, 200)
        self.assertEqual(fb_res.json()["confidence"], 1.2)

        # 4. 删除反思
        del_res = self.client.delete(f"/api/memory/critiques/{critique_id}")
        self.assertEqual(del_res.status_code, 200)
        self.assertEqual(del_res.json()["deleted_critique_id"], critique_id)

    def test_greeting_and_unrecognized_input_handling(self):
        """测试日常问候与无意义测试短串快速响应，不触发底层多源海事工具链。"""
        # 1. 问候语测试
        res_hi = self.client.post("/ask", json={"question": "你好"})
        self.assertEqual(res_hi.status_code, 200)
        data_hi = res_hi.json()
        self.assertIn("您好！我是海事态势调查智能体", data_hi["answer"])
        self.assertEqual(len(data_hi.get("tool_trace", [])), 0)
        self.assertEqual(len(data_hi.get("evidence", [])), 0)

        # 2. 问候语（带空格/标点）
        res_hi2 = self.client.post("/ask", json={"question": "你好呀！"})
        self.assertEqual(res_hi2.status_code, 200)
        self.assertIn("您好！我是海事态势调查智能体", res_hi2.json()["answer"])

        # 3. 随机无语义数字测试（如 1123）
        res_num = self.client.post("/ask", json={"question": "1123"})
        self.assertEqual(res_num.status_code, 200)
        data_num = res_num.json()
        self.assertIn("未能识别具体的研判指令", data_num["answer"])
        self.assertIn("1123", data_num["answer"])
        self.assertEqual(len(data_num.get("tool_trace", [])), 0)

    def test_distill_tool_failure_critique(self):
        """测试工具调用返回空匹配时自主提炼避坑规则。"""
        # 1. 遥感场景检索 0 匹配
        trace_remote = [{"tool": "search_remote_sensing", "arguments": {"date_utc": "2024-07-08"}}]
        evidence_remote = [{"evidence_id": "remote_search:houston", "match_count": 0, "returned": 0, "date_utc": "2024-07-08"}]
        critique_remote = distill_tool_failure_critique(trace_remote, evidence_remote)
        self.assertIsNotNone(critique_remote)
        self.assertEqual(critique_remote.domain, "remote_sensing")
        self.assertEqual(critique_remote.source, "tool_failure")
        self.assertIn("前后邻近日期", critique_remote.corrected_lesson)

        # 2. 轨迹查询 0 匹配
        trace_track = [{"tool": "search_vessel_tracks", "arguments": {"mmsi": "258041000"}}]
        evidence_track = [{"evidence_id": "track_search:258041000", "returned": 0, "tracks": []}]
        critique_track = distill_tool_failure_critique(trace_track, evidence_track)
        self.assertIsNotNone(critique_track)
        self.assertEqual(critique_track.domain, "ais")
        self.assertEqual(critique_track.source, "tool_failure")
        self.assertIn("确切报文起止时段", critique_track.corrected_lesson)

    def test_distill_verifier_warning_critique(self):
        """测试核验器发现虚假数值或未引证事实时沉淀反幻觉规则。"""
        warnings = ["数值事实存疑: 声称风速/航速 '99.5节' 未在引用证据中找到对应数值支持"]
        critique = distill_verifier_warning_critique(warnings, question="台风风速是多少？")
        self.assertIsNotNone(critique)
        self.assertEqual(critique.domain, "metocean")
        self.assertEqual(critique.source, "verifier_warning")
        self.assertIn("以工具实测数据为基准", critique.corrected_lesson)
        self.assertIn("99.5节", critique.failed_path)

    def test_blackboard_nav_state_machine_and_conflict_resolution(self):
        """测试黑板态势状态机状态迁移（锚泊 -> 走锚漂移）以及事实冲突消解。"""
        cid = self.store.create()

        # 第 1 轮：正常锚泊
        resp1 = {
            "answer": "油轮 367425000 在休斯敦主锚地正常锚泊系泊【ev1】。",
            "evidence": [{
                "evidence_id": "ev1",
                "vessel_name": "MT BERYL",
                "length": 180,
                "width": 32,
                "latitude": 29.35,
                "longitude": -94.75,
            }],
            "selected_region_id": "region-houston",
        }
        self.store.append(cid, "该轮目前什么状态？", resp1)
        bb1 = self.store.get_cognitive_blackboard(cid)
        self.assertEqual(bb1.nav_status, "锚泊系泊")
        self.assertEqual(bb1.vessel_static_profile.get("length"), 180)
        self.assertEqual(bb1.latest_coordinates, (29.35, -94.75))
        self.assertTrue(any("正常锚泊系泊" in f for f in bb1.active_facts))

        # 第 2 轮：发生台风走锚，状态机迁移
        resp2 = {
            "answer": "风速超过 25 m/s【ev2】，判定处于走锚漂移险情【ev3】。",
            "evidence": [{"evidence_id": "ev2"}, {"evidence_id": "ev3"}],
            "selected_region_id": "region-houston",
        }
        self.store.append(cid, "台风过境后呢？", resp2)
        bb2 = self.store.get_cognitive_blackboard(cid)
        self.assertEqual(bb2.nav_status, "走锚漂移")
        # 验证旧的正常锚泊被走锚漂移冲刷消解
        self.assertTrue(any("走锚漂移" in f for f in bb2.active_facts))
        self.assertFalse(any("正常锚泊" in f for f in bb2.active_facts))

    def test_critique_confidence_decay_and_filtering(self):
        """测试经验置信度衰减、打分奖惩与低质量规则动态休眠过滤。"""
        cid_critique = "CRT-TEST-DECAY"
        critique = EvolvedCritique(
            critique_id=cid_critique,
            domain="ais",
            trigger_pattern="测试模式",
            failed_path="误判",
            corrected_lesson="测试经验准则",
            confidence=1.0,
        )
        self.store.save_evolved_critique(critique)

        # 验证初始能被召回
        res_init = self.store.get_relevant_critiques(query="测试模式", min_confidence=0.5)
        self.assertTrue(any(c.critique_id == cid_critique for c in res_init))

        # 连续两次 downvote 扣分: 1.0 -> 0.7 -> 0.4
        self.store.adjust_critique_confidence(cid_critique, -0.3)
        self.store.adjust_critique_confidence(cid_critique, -0.3)

        # 验证衰减至 0.4 后，被 min_confidence=0.5 门槛过滤休眠
        res_decayed = self.store.get_relevant_critiques(query="测试模式", min_confidence=0.5)
        self.assertFalse(any(c.critique_id == cid_critique for c in res_decayed))

        # 验证 assemble_cognitive_prompt_block 同样排除了衰减规则
        decayed_critique = EvolvedCritique(
            critique_id="CRT-LOW",
            domain="ais",
            trigger_pattern="低置信度",
            failed_path="踩坑",
            corrected_lesson="不应展示的规则",
            confidence=0.4,
        )
        prompt_block = assemble_cognitive_prompt_block(None, None, [decayed_critique])
        self.assertNotIn("不应展示的规则", prompt_block)

    def test_bushi_ershi_pattern_distillation(self):
        """测试对 '不是 A 而是 B' 复杂纠错句式的结构化拆解与黄金规则提炼。"""
        question = "不对，这艘船当时不是主动巡航，而是被大风吹着走锚漂移！"
        critique = detect_and_evolve_critique(question)
        self.assertIsNotNone(critique)
        self.assertEqual(critique.domain, "anchorage")
        self.assertIn("主动巡航", critique.failed_path)
        self.assertIn("走锚漂移", critique.corrected_lesson)

