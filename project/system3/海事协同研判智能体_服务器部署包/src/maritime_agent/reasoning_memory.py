"""Cognitive Blackboard, Persistent Reasoning Trajectory, and Self-Evolution Engine.

This module provides human-level cognitive continuity across conversation turns:
1. Cognitive Blackboard: Tracks active entities, spatial-temporal scopes, navigational states,
   and dynamic high-SNR facts with conflict resolution.
2. Reasoning Snapshot: Records established facts, refuted hypotheses, active inferences,
   and unresolved questions at the end of each turn (reasoning checkpoints).
3. Self-Evolution Reflexion: Distills lessons from user corrections, tool failures,
   and fact-verifier warnings, injecting them into future reasoning loops to prevent recurring mistakes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import re
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ReasoningSnapshot:
    """Represents a frozen reasoning checkpoint at the conclusion of an investigation turn."""
    turn_index: int
    created_at_utc: str = field(default_factory=_utc_now)
    target_entities: dict[str, Any] = field(default_factory=dict)
    established_facts: list[str] = field(default_factory=list)
    refuted_hypotheses: list[str] = field(default_factory=list)
    active_inferences: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReasoningSnapshot:
        return cls(
            turn_index=data.get("turn_index", 0),
            created_at_utc=data.get("created_at_utc", _utc_now()),
            target_entities=data.get("target_entities") or {},
            established_facts=data.get("established_facts") or [],
            refuted_hypotheses=data.get("refuted_hypotheses") or [],
            active_inferences=data.get("active_inferences") or [],
            unresolved_questions=data.get("unresolved_questions") or [],
            evidence_ids=data.get("evidence_ids") or [],
        )


@dataclass
class CognitiveBlackboard:
    """Session-level workspace maintaining active entities, scope, navigational state, and high-SNR facts."""
    conversation_id: str
    active_region_id: str = ""
    active_target_mmsi: str = ""
    active_vessel_name: str = ""
    vessel_static_profile: dict[str, Any] = field(default_factory=dict)
    nav_status: str = ""  # e.g., '锚泊系泊', '走锚漂移', '机航在航', '靠泊作业'
    latest_coordinates: tuple[float, float] | None = None
    active_time_window_utc: str = ""
    cumulative_established_facts: list[str] = field(default_factory=list)
    active_facts: list[str] = field(default_factory=list)
    latest_snapshot: ReasoningSnapshot | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "active_region_id": self.active_region_id,
            "active_target_mmsi": self.active_target_mmsi,
            "active_vessel_name": self.active_vessel_name,
            "vessel_static_profile": self.vessel_static_profile,
            "nav_status": self.nav_status,
            "latest_coordinates": list(self.latest_coordinates) if self.latest_coordinates else None,
            "active_time_window_utc": self.active_time_window_utc,
            "cumulative_established_facts": self.cumulative_established_facts,
            "active_facts": self.active_facts,
            "latest_snapshot": self.latest_snapshot.to_dict() if self.latest_snapshot else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CognitiveBlackboard:
        snap_data = data.get("latest_snapshot")
        latest_snap = ReasoningSnapshot.from_dict(snap_data) if snap_data else None
        coords = data.get("latest_coordinates")
        coords_tuple = tuple(coords) if coords and isinstance(coords, (list, tuple)) and len(coords) == 2 else None
        return cls(
            conversation_id=data.get("conversation_id", ""),
            active_region_id=data.get("active_region_id", ""),
            active_target_mmsi=data.get("active_target_mmsi", ""),
            active_vessel_name=data.get("active_vessel_name", ""),
            vessel_static_profile=data.get("vessel_static_profile") or {},
            nav_status=data.get("nav_status", ""),
            latest_coordinates=coords_tuple,
            active_time_window_utc=data.get("active_time_window_utc", ""),
            cumulative_established_facts=data.get("cumulative_established_facts") or [],
            active_facts=data.get("active_facts") or [],
            latest_snapshot=latest_snap,
        )


@dataclass
class EvolvedCritique:
    """Self-evolution feedback rule distilled from user corrections, tool failures, or verifier warnings."""
    critique_id: str
    domain: str  # e.g., 'ais', 'metocean', 'fusion', 'general', 'anchorage', 'remote_sensing', 'gis'
    trigger_pattern: str  # keywords or concepts triggering this critique
    failed_path: str  # previous mistake or failure path
    corrected_lesson: str  # distilled golden rule
    confidence: float = 1.0
    hit_count: int = 0
    success_count: int = 0
    source: str = "user_correction"  # 'user_correction', 'tool_failure', 'verifier_warning', 'manual'
    created_at_utc: str = field(default_factory=_utc_now)
    updated_at_utc: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvolvedCritique:
        return cls(
            critique_id=data.get("critique_id", ""),
            domain=data.get("domain", "general"),
            trigger_pattern=data.get("trigger_pattern", ""),
            failed_path=data.get("failed_path", ""),
            corrected_lesson=data.get("corrected_lesson", ""),
            confidence=float(data.get("confidence", 1.0)),
            hit_count=int(data.get("hit_count", 0)),
            success_count=int(data.get("success_count", 0)),
            source=data.get("source", "user_correction"),
            created_at_utc=data.get("created_at_utc") or _utc_now(),
            updated_at_utc=data.get("updated_at_utc") or _utc_now(),
        )


def extract_reasoning_snapshot(
    turn_index: int,
    question: str,
    answer: str,
    evidence: list[Any] | None = None,
    active_scope: dict[str, Any] | None = None,
) -> ReasoningSnapshot:
    """Extracts a structured reasoning checkpoint from a turn's question, answer, and evidence."""
    evidence_list = evidence or []
    target_entities: dict[str, Any] = {}
    if active_scope:
        target_entities.update({k: v for k, v in active_scope.items() if v})

    # 1. 提取实体 (MMSI、船名、海区、时间)
    mmsi_match = re.search(r"\b(2\d{8}|3\d{8}|4\d{8}|5\d{8}|6\d{8}|7\d{8})\b", question + " " + answer)
    if mmsi_match and not target_entities.get("mmsi"):
        target_entities["mmsi"] = mmsi_match.group(1)

    # 提取时间戳
    time_match = re.search(r"\b(\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2})?(?:Z)?)?)\b", question + " " + answer)
    if time_match and not target_entities.get("time_window"):
        target_entities["time_window"] = time_match.group(1)

    # 提取证据编号
    ev_ids = []
    for ev in evidence_list:
        if isinstance(ev, dict) and ev.get("evidence_id"):
            ev_ids.append(str(ev["evidence_id"]))
        elif isinstance(ev, str) and ev.startswith("EVD_"):
            ev_ids.append(ev)

    text_ev_matches = re.findall(r"【([A-Za-z0-9:_-]+)】", answer)
    for m in text_ev_matches:
        if m not in ev_ids:
            ev_ids.append(m)

    # 2. 事实抽取 (Established Facts)
    established_facts = []
    sentences = [s.strip() for s in re.split(r"[。\n;；]", answer) if len(s.strip()) > 8]
    for s in sentences:
        if any(kw in s for kw in ("风速", "气压", "航速", "节", "艘", "台风", "坐标", "已确认", "核实", "m/s", "hPa", "UTC", "锚泊", "系泊", "走锚", "巡航", "在航", "靠泊", "停泊")):
            clean_s = re.sub(r"【.*?】", "", s).strip()
            if clean_s and clean_s not in established_facts:
                established_facts.append(clean_s[:160])
        if len(established_facts) >= 5:
            break

    # 3. 排除假设 (Refuted Hypotheses)
    refuted_hypotheses = []
    for s in sentences:
        if any(neg in s for neg in ("排除", "并非", "不是", "并未", "未发现", "无走锚", "不存在", "未见异常")):
            clean_s = re.sub(r"【.*?】", "", s).strip()
            if clean_s and clean_s not in refuted_hypotheses:
                refuted_hypotheses.append(clean_s[:160])
        if len(refuted_hypotheses) >= 3:
            break

    # 4. 当前核心推论 (Active Inferences)
    active_inferences = []
    for s in sentences:
        if any(inf in s for inf in ("判定", "表明", "推断", "显示为", "处于", "由于", "导致", "险情", "顺风走锚")):
            clean_s = re.sub(r"【.*?】", "", s).strip()
            if clean_s and clean_s not in active_inferences:
                active_inferences.append(clean_s[:160])
        if len(active_inferences) >= 3:
            break

    # 5. 未决疑问 / 留存疑点 (Unresolved Questions)
    unresolved_questions = []
    for s in sentences:
        if any(unres in s for unres in ("需进一步", "待核实", "建议持续", "有待", "后续需", "需关注", "持续关注", "防范", "防搁浅", "盲区", "风险")):
            clean_s = re.sub(r"【.*?】", "", s).strip()
            if clean_s and clean_s not in unresolved_questions:
                unresolved_questions.append(clean_s[:160])
        if len(unresolved_questions) >= 3:
            break

    return ReasoningSnapshot(
        turn_index=turn_index,
        created_at_utc=_utc_now(),
        target_entities=target_entities,
        established_facts=established_facts,
        refuted_hypotheses=refuted_hypotheses,
        active_inferences=active_inferences,
        unresolved_questions=unresolved_questions,
        evidence_ids=ev_ids,
    )


def update_blackboard_state(
    blackboard: CognitiveBlackboard,
    snapshot: ReasoningSnapshot,
    response: dict[str, Any],
) -> CognitiveBlackboard:
    """
    Updates the session's CognitiveBlackboard with new snapshot data, resolves fact conflicts,
    and advances the vessel navigational state machine.
    """
    active_scope = {
        "region_id": response.get("selected_region_id"),
        "source_id": response.get("selected_source_id"),
    }
    if active_scope.get("region_id"):
        blackboard.active_region_id = active_scope["region_id"]
    elif snapshot.target_entities.get("region_id"):
        blackboard.active_region_id = snapshot.target_entities["region_id"]

    if snapshot.target_entities.get("mmsi"):
        blackboard.active_target_mmsi = snapshot.target_entities["mmsi"]
    if snapshot.target_entities.get("time_window"):
        blackboard.active_time_window_utc = snapshot.target_entities["time_window"]

    # 1. Update static vessel profile and coordinates from evidence
    for ev in response.get("evidence", []):
        if not isinstance(ev, dict):
            continue
        if ev.get("vessel_name") and not blackboard.active_vessel_name:
            blackboard.active_vessel_name = str(ev["vessel_name"])
        for key in ("length", "width", "vessel_type", "draught", "callsign"):
            if ev.get(key) is not None:
                blackboard.vessel_static_profile[key] = ev[key]
        if ev.get("latitude") is not None and ev.get("longitude") is not None:
            try:
                blackboard.latest_coordinates = (float(ev["latitude"]), float(ev["longitude"]))
            except (ValueError, TypeError):
                pass

    # 2. Advance Navigational State Machine
    full_text = " ".join([
        str(response.get("answer") or ""),
        " ".join(snapshot.established_facts),
        " ".join(snapshot.active_inferences),
    ])
    if any(w in full_text for w in ("走锚", "漂移险情", "顺风漂移", "失控走锚")):
        blackboard.nav_status = "走锚漂移"
    elif any(w in full_text for w in ("锚泊", "抛锚", "系泊", "锚地避风")):
        blackboard.nav_status = "锚泊系泊"
    elif any(w in full_text for w in ("靠泊", "停靠", "码头作业")):
        blackboard.nav_status = "靠泊作业"
    elif any(w in full_text for w in ("巡航", "在航", "航行", "机航")):
        blackboard.nav_status = "机航在航"

    # 3. Dynamic Active Facts (Conflict Resolution & Sliding Window)
    for fact in snapshot.established_facts:
        if fact not in blackboard.cumulative_established_facts:
            blackboard.cumulative_established_facts.append(fact)

    new_active = list(blackboard.active_facts)
    for new_fact in snapshot.established_facts:
        clean_new = re.sub(r"【.*?】", "", new_fact).strip()
        if not clean_new:
            continue
        
        conflict_detected = False
        for idx, old_fact in enumerate(new_active):
            # Resolve conflict: anchor drag supersedes normal anchor/underway
            if ("走锚" in clean_new and "正常" in old_fact) or \
               ("排除" in clean_new and any(w in old_fact for w in ("在航", "巡航", "出港"))):
                new_active[idx] = clean_new
                conflict_detected = True
                break
        if not conflict_detected and clean_new not in new_active:
            new_active.insert(0, clean_new)

    blackboard.active_facts = new_active[:5]
    blackboard.latest_snapshot = snapshot
    return blackboard


def distill_tool_failure_critique(
    tool_trace: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> EvolvedCritique | None:
    """
    Inspects tool execution trace and evidence to autonomously distill an experience critique
    when a tool returns 0 matches or fails unexpectedly.
    """
    if not tool_trace:
        return None

    for call in tool_trace:
        tool_name = call.get("tool") or call.get("name") or ""
        args = call.get("arguments") or call.get("args") or {}

        for ev in evidence:
            if not isinstance(ev, dict):
                continue
            ev_id = str(ev.get("evidence_id", ""))
            
            # 1. Remote sensing empty search
            if "remote" in tool_name or ev_id.startswith("remote_search"):
                match_count = ev.get("match_count")
                returned = ev.get("returned")
                if match_count == 0 or returned == 0:
                    date_val = str(args.get("date_utc") or ev.get("date_utc") or "")
                    critique_id = f"CRT-TOOL-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
                    return EvolvedCritique(
                        critique_id=critique_id,
                        domain="remote_sensing",
                        trigger_pattern=f"遥感影像检索 {date_val}".strip(),
                        failed_path=f"在指定日期 {date_val} 直接调用 {tool_name} 检索遥感场景返回 0 条匹配",
                        corrected_lesson="遥感卫星过轨存在回归周期，若目标日无覆盖，应先查询前后邻近日期或结合 situation_cube_overview 确认覆盖编目。",
                        confidence=0.9,
                        source="tool_failure",
                        created_at_utc=_utc_now(),
                    )

            # 2. Track search empty
            elif "track" in tool_name or ev_id.startswith("track_search"):
                returned = ev.get("returned")
                tracks = ev.get("tracks")
                if returned == 0 or (isinstance(tracks, list) and len(tracks) == 0):
                    mmsi_val = str(args.get("mmsi") or "")
                    critique_id = f"CRT-TOOL-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
                    return EvolvedCritique(
                        critique_id=critique_id,
                        domain="ais",
                        trigger_pattern=f"轨迹查询 {mmsi_val}".strip(),
                        failed_path=f"调用 {tool_name} 针对 MMSI {mmsi_val} 未检索到轨迹记录",
                        corrected_lesson="检索单船轨迹无返回时，应先调用 search_ingested_data 或 data_overview 核实该船在数据库中的确切报文起止时段。",
                        confidence=0.9,
                        source="tool_failure",
                        created_at_utc=_utc_now(),
                    )

            # 3. GIS object empty search
            elif "gis" in tool_name or ev_id.startswith("gis_search"):
                returned = ev.get("returned")
                objs = ev.get("objects")
                if returned == 0 or (isinstance(objs, list) and len(objs) == 0):
                    q_val = str(args.get("query") or "")
                    critique_id = f"CRT-TOOL-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
                    return EvolvedCritique(
                        critique_id=critique_id,
                        domain="gis",
                        trigger_pattern=f"GIS要素检索 {q_val}".strip(),
                        failed_path=f"调用 {tool_name} 检索关键词 '{q_val}' 返回 0 条要素",
                        corrected_lesson="GIS要素检索未命中时，建议放宽匹配关键词或改用 category 范围检索（如 anchorage、fairway）。",
                        confidence=0.9,
                        source="tool_failure",
                        created_at_utc=_utc_now(),
                    )

    return None


def distill_verifier_warning_critique(
    warnings: list[str],
    question: str = "",
) -> EvolvedCritique | None:
    """
    Distills an anti-hallucination critique when Fact Verifier or citation validator
    raises a warning regarding ungrounded numerical claims or invalid citations.
    """
    if not warnings:
        return None

    for w in warnings:
        # 1. Numerical hallucination warning from fact_verifier
        num_match = re.search(r"数值事实存疑:\s*声称([^']+)\s*'([^']+)'\s*未在引用证据中找到对应数值支持", w)
        if num_match:
            metric_type = num_match.group(1).strip()
            claimed_val = num_match.group(2).strip()
            domain = "metocean" if any(k in metric_type for k in ("风速", "航速", "气压", "浪高")) else "ais"
            critique_id = f"CRT-WARN-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            return EvolvedCritique(
                critique_id=critique_id,
                domain=domain,
                trigger_pattern=f"{metric_type} {claimed_val}",
                failed_path=f"回答中给出了未经工具实测证据支持的虚假数值 '{claimed_val}'",
                corrected_lesson=f"陈述{metric_type}结论时必须严格以工具实测数据为基准，严禁脱离证据陈述未经证实的量化数值（曾捏造: '{claimed_val}'）。",
                confidence=1.0,
                source="verifier_warning",
                created_at_utc=_utc_now(),
            )

        # 2. Citation ledger mismatch
        if "回答中存在未由本轮工具结果提供的证据编号" in w or "没有有效证据引用" in w:
            critique_id = f"CRT-WARN-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            return EvolvedCritique(
                critique_id=critique_id,
                domain="general",
                trigger_pattern="证据引用缺失或捏造",
                failed_path="回答中编造了不存在的 evidence_id 或未能给出合法证据引用",
                corrected_lesson="回答中的每一项研判结论必须使用【真实evidence_id】标记，严禁沿用上一轮或伪造证据编号。",
                confidence=1.1,
                source="verifier_warning",
                created_at_utc=_utc_now(),
            )

    return None


def detect_and_evolve_critique(question: str, history: list[dict[str, Any]] | None = None) -> EvolvedCritique | None:
    """Detects user correction patterns and distills a structured self-evolution critique lesson."""
    correction_keywords = (
        "不是", "不对", "搞错了", "更正", "不要以为", "当时没有", "实际上是",
        "应该算作", "实际处于", "并不是", "并没有", "搞反了", "应该为", "实际上", "请注意不是"
    )
    if not any(kw in question for kw in correction_keywords):
        return None

    # Determine domain
    domain = "general"
    if any(w in question for w in ("走锚", "锚泊", "停泊", "抛锚", "偏位", "漂移", "锚链")):
        domain = "anchorage"
    elif any(w in question for w in ("风速", "风向", "气象", "台风", "海况", "气压", "阵风", "涌浪")):
        domain = "metocean"
    elif any(w in question for w in ("暗标", "暗船", "雷达", "遥感", "切片", "融合", "伴航")):
        domain = "fusion"
    elif any(w in question for w in ("航速", "航向", "航道", "ais", "mmsi", "船名", "吃水", "轨迹")):
        domain = "ais"

    critique_id = f"CRT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    # Extract clean correction lesson and failed path if pattern like "不是 A 而是 B"
    bushi_ershi = re.search(r"(?:不是|并非|并未)\s*(.+?)\s*[,，]?\s*(?:而是|实际上是|应该为|实际处于)\s*([^，,。！!]+)", question)
    if bushi_ershi:
        failed_part = bushi_ershi.group(1).strip()
        correct_part = bushi_ershi.group(2).strip()
        failed_path = f"此前误判为'{failed_part}'"
        corrected_lesson = f"用户明确指示业务规则：{question[:140]}（应判定为'{correct_part}'，不可误判为'{failed_part}'）"
    else:
        failed_path = "此前未结合用户纠正要点或误判了通航/气象工况"
        corrected_lesson = f"用户明确指示业务规则：{question[:140]}"

    return EvolvedCritique(
        critique_id=critique_id,
        domain=domain,
        trigger_pattern=question[:60],
        failed_path=failed_path,
        corrected_lesson=corrected_lesson,
        confidence=1.0,
        source="user_correction",
        created_at_utc=_utc_now(),
    )


def assemble_cognitive_prompt_block(
    blackboard: CognitiveBlackboard | None,
    previous_snapshot: ReasoningSnapshot | None,
    critiques: list[EvolvedCritique] | None = None,
) -> str:
    """Assembles a high signal-to-noise cognitive state block to inject into the LLM context.
    
    This ensures that subsequent questions do not restart from scratch, but seamlessly
    continue along the established reasoning chain.
    """
    blocks = []

    # 1. 活跃态势黑板
    if blackboard:
        bb_parts = []
        if blackboard.active_region_id:
            bb_parts.append(f"当前锁定海区: {blackboard.active_region_id}")
        if blackboard.active_target_mmsi:
            bb_parts.append(f"当前聚焦船舶 MMSI: {blackboard.active_target_mmsi}")
        if blackboard.active_vessel_name:
            bb_parts.append(f"当前船舶名: {blackboard.active_vessel_name}")
        if blackboard.nav_status:
            bb_parts.append(f"当前研判态势/状态: {blackboard.nav_status}")
        if blackboard.latest_coordinates:
            bb_parts.append(f"最新已知坐标: {blackboard.latest_coordinates[0]:.4f}, {blackboard.latest_coordinates[1]:.4f}")
        if blackboard.active_time_window_utc:
            bb_parts.append(f"基准时窗: {blackboard.active_time_window_utc}")
        if blackboard.active_facts:
            bb_parts.append("活跃已确立态势事实:")
            for af in blackboard.active_facts[:3]:
                bb_parts.append(f"  • {af}")
        if bb_parts:
            blocks.append("【🧠 态势认知黑板 (Cognitive Blackboard)】\n" + "\n".join(f"- {p}" if not p.startswith(" ") else p for p in bb_parts))

    # 2. 上一轮思维链断点承接 (Reasoning Checkpoint)
    if previous_snapshot and (
        previous_snapshot.established_facts
        or previous_snapshot.active_inferences
        or previous_snapshot.unresolved_questions
    ):
        snap_parts = []
        snap_parts.append(f"上一轮(Turn #{previous_snapshot.turn_index})已确立事实:")
        for f in previous_snapshot.established_facts[:4]:
            snap_parts.append(f"  • {f}")

        if previous_snapshot.refuted_hypotheses:
            snap_parts.append("上一轮已排除假设(无需重走死胡同):")
            for rh in previous_snapshot.refuted_hypotheses[:2]:
                snap_parts.append(f"  ✕ {rh}")

        if previous_snapshot.active_inferences:
            snap_parts.append("上一轮核心推论:")
            for inf in previous_snapshot.active_inferences[:2]:
                snap_parts.append(f"  ➔ {inf}")

        if previous_snapshot.unresolved_questions:
            snap_parts.append("留存待深入探查疑问(优先承接深挖):")
            for uq in previous_snapshot.unresolved_questions[:2]:
                snap_parts.append(f"  ❓ {uq}")

        snap_parts.append(
            "【认知指示】: 若用户当前问题为顺承追问或省略指代（如'那它周围呢'、'为什么'、'当时风速呢'），"
            "必须紧密承接上述已证事实与未决疑问深挖；若用户输入为新话题、日常问候闲聊（如'你好'）或明确更换分析目标，"
            "应聚焦当前问题本身，切勿生搬硬套或强行复述上一轮无关结论！"
        )
        blocks.append("【🔗 上一轮思维链断点承接 (Reasoning Trajectory Continuation)】\n" + "\n".join(snap_parts))

    # 3. 关联自进化经验避坑提示 (Filter confidence >= 0.6)
    if critiques:
        active_critiques = [c for c in critiques if getattr(c, "confidence", 1.0) >= 0.6]
        if active_critiques:
            critique_parts = ["【⚡ 自进化研判经验避坑指南 (Evolved Heuristics)】:"]
            for c in active_critiques[:3]:
                critique_parts.append(f"- [{c.domain}] {c.corrected_lesson} (曾踩坑: {c.failed_path})")
            blocks.append("\n".join(critique_parts))

    return "\n\n".join(blocks)
