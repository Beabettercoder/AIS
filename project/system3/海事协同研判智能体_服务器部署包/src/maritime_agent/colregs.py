"""COLREGs (1972) Navigational Conflict & Collision Risk Evaluation Engine.

Provides deterministic kinematic and maritime regulatory reasoning:
1. Local tangent-plane relative motion kinematics (DCPA, TCPA).
2. Multi-factor non-linear Collision Risk Index (CRI).
3. International Regulations for Preventing Collisions at Sea (COLREGs 1972) scenario classification:
   - Rule 13: Overtaking (追越局面)
   - Rule 14: Head-on (对遇局面)
   - Rule 15: Crossing (交叉相遇局面 - 让路船 / 直航船)
   - Rule 18: Responsibilities between Vessels / NUC / Anchor-dragging (失去控制/走锚失控船责任条款)
4. Multi-vessel traffic conflict matrix scanning and pairwise risk ranking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any


class EncounterType(str, Enum):
    HEAD_ON = "head_on"                          # Rule 14: 对遇局面 (双方均须向右转向)
    CROSSING_GIVE_WAY = "crossing_give_way"      # Rule 15: 交叉相遇-本船让路 (他船在本船右舷)
    CROSSING_STAND_ON = "crossing_stand_on"      # Rule 15: 交叉相遇-本船直航 (他船在本船左舷)
    OVERTAKING_GIVE_WAY = "overtaking_give_way"  # Rule 13: 追越局面-本船追越让路
    OVERTAKEN_STAND_ON = "overtaken_stand_on"    # Rule 13: 追越局面-本船被追越保向保速
    ANCHOR_DRIFT_NUC = "anchor_drift_nuc"        # Rule 18: 走锚失控与在航船险情 (在航船须全面避让)
    SAFE_PASSING = "safe_passing"                # 安全通过 / 无逼近风险


@dataclass
class CollisionRiskRecord:
    vessel1_mmsi: str
    vessel1_name: str
    vessel2_mmsi: str
    vessel2_name: str
    distance_nm: float
    distance_m: float
    dcpa_nm: float
    dcpa_m: float
    tcpa_minutes: float
    cri: float                                    # Collision Risk Index [0.0, 1.0]
    encounter_type: EncounterType
    encounter_desc: str
    obligation_v1: str                           # 船舶 1 避让义务
    obligation_v2: str                           # 船舶 2 避让义务
    is_urgent: bool                              # True if CRI >= 0.7 or (DCPA < 0.2 nm and 0 < TCPA < 10 min)
    evidence_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["encounter_type"] = self.encounter_type.value
        return data


def calculate_dcpa_tcpa(
    lat1: float, lon1: float, sog1_kn: float, cog1_deg: float,
    lat2: float, lon2: float, sog2_kn: float, cog2_deg: float,
) -> tuple[float, float, float, float]:
    """
    Computes distance, bearing, DCPA (nautical miles), and TCPA (minutes)
    between two vessels on a local tangent plane.
    
    Returns:
        (distance_nm, true_bearing_1_to_2_deg, dcpa_nm, tcpa_min)
    """
    # 1. Great circle approximation to local nautical mile tangent plane
    # 1 degree of latitude = 60 nautical miles
    dy = (lat2 - lat1) * 60.0
    mean_lat_rad = math.radians((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * 60.0 * math.cos(mean_lat_rad)
    
    distance_nm = math.hypot(dx, dy)
    bearing_1_to_2 = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

    # 2. Velocity vectors (knots, cartesian where +x = East, +y = North)
    c1_rad = math.radians(cog1_deg)
    v1x = sog1_kn * math.sin(c1_rad)
    v1y = sog1_kn * math.cos(c1_rad)

    c2_rad = math.radians(cog2_deg)
    v2x = sog2_kn * math.sin(c2_rad)
    v2y = sog2_kn * math.cos(c2_rad)

    # Relative velocity of Vessel 2 relative to Vessel 1: V_r = V2 - V1
    vrx = v2x - v1x
    vry = v2y - v1y
    vr_speed = math.hypot(vrx, vry)

    # If relative speed is negligible (co-moving or both stopped)
    if vr_speed < 0.05:
        return round(distance_nm, 3), round(bearing_1_to_2, 1), round(distance_nm, 3), 0.0

    # 3. TCPA = - (P · V_r) / |V_r|^2 (hours)
    # P = (dx, dy)
    p_dot_vr = dx * vrx + dy * vry
    tcpa_hours = -p_dot_vr / (vr_speed ** 2)
    tcpa_min = tcpa_hours * 60.0

    # 4. DCPA = |P + V_r * TCPA|
    cpa_x = dx + vrx * tcpa_hours
    cpa_y = dy + vry * tcpa_hours
    dcpa_nm = math.hypot(cpa_x, cpa_y)

    return (
        round(distance_nm, 3),
        round(bearing_1_to_2, 1),
        round(dcpa_nm, 3),
        round(tcpa_min, 1),
    )


def solve_continuous_segment_cpa(
    lat1: float, lon1: float, sog1_kn: float, cog1_deg: float,
    lat2: float, lon2: float, sog2_kn: float, cog2_deg: float,
    duration_sec: float = 3600.0,
) -> tuple[float, float, float, float]:
    """
    在时空航段持续区间 [0, duration_sec] 内，通过微分导数解析解（d(|D(t)|^2)/dt = 0），
    精确求解两船在局部切平面上的闭式极限连续最近逼近距离 DCPA（米与海里）与发生该极值的精确偏移秒数 delta_t_sec。

    Returns:
        (min_distance_m, min_distance_nm, delta_t_sec, initial_distance_m)
    """
    mean_lat_rad = math.radians((lat1 + lat2) / 2.0)
    dy = (lat2 - lat1) * 111139.0
    dx = (lon2 - lon1) * 111139.0 * math.cos(mean_lat_rad)
    initial_dist_m = math.hypot(dx, dy)

    m_per_sec = 0.5144444444444445
    c1_rad = math.radians(cog1_deg)
    v1x = sog1_kn * m_per_sec * math.sin(c1_rad)
    v1y = sog1_kn * m_per_sec * math.cos(c1_rad)

    c2_rad = math.radians(cog2_deg)
    v2x = sog2_kn * m_per_sec * math.sin(c2_rad)
    v2y = sog2_kn * m_per_sec * math.cos(c2_rad)

    vrx = v2x - v1x
    vry = v2y - v1y
    vr2 = vrx ** 2 + vry ** 2

    if vr2 < 1e-4:
        return round(initial_dist_m, 2), round(initial_dist_m / 1852.0, 4), 0.0, round(initial_dist_m, 2)

    p_dot_vr = dx * vrx + dy * vry
    tau_cpa = -p_dot_vr / vr2
    tau_best = max(0.0, min(float(duration_sec), tau_cpa))

    cpa_x = dx + vrx * tau_best
    cpa_y = dy + vry * tau_best
    min_dist_m = math.hypot(cpa_x, cpa_y)
    min_dist_nm = min_dist_m / 1852.0

    return round(min_dist_m, 2), round(min_dist_nm, 4), round(tau_best, 1), round(initial_dist_m, 2)


def calculate_collision_risk_index(
    distance_nm: float,
    dcpa_nm: float,
    tcpa_min: float,
    safe_distance_nm: float = 1.0,
    safe_dcpa_nm: float = 0.5,
    warning_tcpa_min: float = 18.0,
) -> float:
    """
    Computes Collision Risk Index (CRI) in range [0.0, 1.0].
    Takes into account geometric proximity, shortest approach distance,
    and temporal urgency.
    """
    # If TCPA is significantly negative, vessels are diverging and danger has passed
    if tcpa_min < -1.5:
        return 0.0

    # 1. Proximity factor U_D
    if distance_nm <= safe_distance_nm:
        u_d = 1.0
    elif distance_nm >= 5.0:
        u_d = 0.0
    else:
        u_d = (5.0 - distance_nm) / (5.0 - safe_distance_nm)

    # 2. DCPA factor U_dcpa
    if dcpa_nm <= 0.1:
        u_dcpa = 1.0
    elif dcpa_nm >= safe_dcpa_nm:
        u_dcpa = 0.0
    else:
        u_dcpa = (safe_dcpa_nm - dcpa_nm) / (safe_dcpa_nm - 0.1)

    # 3. TCPA factor U_tcpa
    if tcpa_min <= 0.0:
        u_tcpa = 0.5 if dcpa_nm <= 0.2 else 0.0
    elif tcpa_min <= 4.0:
        u_tcpa = 1.0
    elif tcpa_min >= warning_tcpa_min:
        u_tcpa = 0.0
    else:
        u_tcpa = (warning_tcpa_min - tcpa_min) / (warning_tcpa_min - 4.0)

    # Multi-factor weighted synthesis
    cri = 0.45 * u_dcpa + 0.35 * u_tcpa + 0.20 * u_d

    # Critical amplification: close quarters with imminent approach
    if dcpa_nm < 0.2 and 0.0 < tcpa_min < 10.0:
        cri = min(1.0, cri * 1.35)

    return round(max(0.0, min(1.0, cri)), 3)


def classify_colregs_encounter(
    cog1_deg: float,
    sog1_kn: float,
    cog2_deg: float,
    sog2_kn: float,
    bearing_1_to_2_deg: float,
    is_anchor_drag1: bool = False,
    is_anchor_drag2: bool = False,
    is_anchored1: bool = False,
    is_anchored2: bool = False,
) -> tuple[EncounterType, str, str, str]:
    """
    Classifies the encounter scenario according to COLREGs 1972:
    - Rule 18: Responsibilities (走锚/失去控制 NUC)
    - Rule 13: Overtaking (追越)
    - Rule 14: Head-on (对遇)
    - Rule 15: Crossing (交叉相遇)
    
    Returns:
        (encounter_type, encounter_desc, obligation_v1, obligation_v2)
    """
    # 1. Rule 18: Anchor-dragging or Not Under Command (NUC)
    if is_anchor_drag1 or (is_anchored1 and sog1_kn > 0.8):
        desc = "走锚失控船与在航船险情 (COLREGs Rule 18 责任条款)"
        v1_ob = "本船处于走锚失控/失去控制状态，无自主避让能力，应鸣放警示信号并通报交管"
        v2_ob = "他船为正常机动在航船，依法负有全部让路义务，必须及早采取大幅避让措施"
        return EncounterType.ANCHOR_DRIFT_NUC, desc, v1_ob, v2_ob

    if is_anchor_drag2 or (is_anchored2 and sog2_kn > 0.8):
        desc = "走锚失控船与在航船险情 (COLREGs Rule 18 责任条款)"
        v1_ob = "本船为正常机动在航船，依法负有全部让路义务，必须及早采取大幅避让措施"
        v2_ob = "他船处于走锚失控状态，无自主避让能力，应鸣放声光信号"
        return EncounterType.ANCHOR_DRIFT_NUC, desc, v1_ob, v2_ob

    # If both vessels are stopped or moored (no navigation maneuvering underway)
    if sog1_kn < 0.5 and sog2_kn < 0.5:
        return EncounterType.SAFE_PASSING, "两船均处于静止/系泊状态", "保持系泊观察", "保持系泊观察"

    # Relative bearing of Ship 2 as seen from Ship 1
    # 0 = directly ahead, 90 = starboard beam, 180 = directly astern, 270 = port beam
    rel_bearing_v1 = (bearing_1_to_2_deg - cog1_deg + 360.0) % 360.0

    # Relative bearing of Ship 1 as seen from Ship 2
    bearing_2_to_1_deg = (bearing_1_to_2_deg + 180.0) % 360.0
    rel_bearing_v2 = (bearing_2_to_1_deg - cog2_deg + 360.0) % 360.0

    # Course difference
    diff_cog = abs(cog1_deg - cog2_deg)
    if diff_cog > 180.0:
        diff_cog = 360.0 - diff_cog

    # 2. Rule 13: Overtaking (追越局面)
    # Ship 1 is overtaking Ship 2 if Ship 1 is coming up > 22.5° abaft the beam of Ship 2
    # rel_bearing_v2 in (112.5°, 247.5°) and SOG1 > SOG2
    if 112.5 < rel_bearing_v2 < 247.5 and sog1_kn > sog2_kn:
        desc = "追越局面 (COLREGs Rule 13 追越条款)"
        v1_ob = "本船为追越船（让路船），必须及早采取行动，从他船安全距离外绕过，并在彻底驶过让清前保持让清"
        v2_ob = "他船为被追越船（直航船），应保向保速"
        return EncounterType.OVERTAKING_GIVE_WAY, desc, v1_ob, v2_ob

    # Ship 2 is overtaking Ship 1
    if 112.5 < rel_bearing_v1 < 247.5 and sog2_kn > sog1_kn:
        desc = "追越局面 (COLREGs Rule 13 追越条款)"
        v1_ob = "本船为被追越船（直航船），应保向保速"
        v2_ob = "他船为追越船（让路船），负有全面让路义务，须主动绕行让清本船"
        return EncounterType.OVERTAKEN_STAND_ON, desc, v1_ob, v2_ob

    # 3. Rule 14: Head-on (对遇局面)
    # Nearly reciprocal courses (course difference >= 168° or near 180°)
    # and both are seeing each other nearly dead ahead (within 6 degrees)
    is_v1_ahead = (rel_bearing_v1 <= 6.0 or rel_bearing_v1 >= 354.0)
    is_v2_ahead = (rel_bearing_v2 <= 6.0 or rel_bearing_v2 >= 354.0)
    if diff_cog >= 168.0 and (is_v1_ahead or is_v2_ahead):
        desc = "对遇局面 (COLREGs Rule 14 对遇条款)"
        v1_ob = "双方均为让路船，本船必须及早向右转向，以便互从对方左舷驶过"
        v2_ob = "双方均为让路船，他船依法亦须及早向右转向，不得左转"
        return EncounterType.HEAD_ON, desc, v1_ob, v2_ob

    # 4. Rule 15: Crossing (交叉相遇局面)
    # If Ship 2 is on Ship 1's starboard (006° to 112.5°)
    if 6.0 < rel_bearing_v1 <= 112.5:
        desc = "交叉相遇局面 - 本船让路 (COLREGs Rule 15 交叉条款)"
        v1_ob = "他船在本船右舷，本船为让路船，必须及早采取大幅右转避让，如环境许可不得从他船船头穿越"
        v2_ob = "他船为直航船，应保持航向和航速，并密切戒备"
        return EncounterType.CROSSING_GIVE_WAY, desc, v1_ob, v2_ob

    # If Ship 2 is on Ship 1's port side (247.5° to 354°)
    if 247.5 <= rel_bearing_v1 < 354.0:
        desc = "交叉相遇局面 - 本船直航 (COLREGs Rule 15 交叉条款)"
        v1_ob = "他船在本船左舷，本船为直航船，应保向保速；如他船显然未按规定避让，本船应协同采取行动"
        v2_ob = "他船为让路船，依法必须主动采取大幅度避让措施，不得抢越本船船头"
        return EncounterType.CROSSING_STAND_ON, desc, v1_ob, v2_ob

    return EncounterType.SAFE_PASSING, "安全会遇航态", "正常了望，保持航向航速", "正常了望，保持航向航速"


def evaluate_traffic_conflict_matrix(
    vessels: list[dict[str, Any]],
    target_mmsi: str = "",
    min_cri: float = 0.4,
    max_distance_nm: float = 4.0,
    top_limit: int = 10,
) -> list[CollisionRiskRecord]:
    """
    Evaluates pairwise navigational conflicts across concurrent vessels in a snapshot.
    Filters and ranks by Collision Risk Index (CRI).
    """
    records: list[CollisionRiskRecord] = []
    n = len(vessels)
    if n < 2:
        return []

    # Fast indexing
    vessel_map = {str(v.get("mmsi")): v for v in vessels if v.get("mmsi")}
    target_mmsi_clean = str(target_mmsi).strip()

    # If target_mmsi specified, only evaluate target against all others
    if target_mmsi_clean and target_mmsi_clean in vessel_map:
        v1 = vessel_map[target_mmsi_clean]
        for mmsi2, v2 in vessel_map.items():
            if mmsi2 == target_mmsi_clean:
                continue
            rec = _evaluate_pair(v1, v2)
            if rec and rec.cri >= min_cri and rec.distance_nm <= max_distance_nm:
                records.append(rec)
    else:
        # Full pairwise upper-triangle evaluation (bounded to prevent n^2 blowup)
        # Sort candidates to prioritize moving/active vessels
        active_vessels = [v for v in vessels if float(v.get("sog_kn") or 0.0) > 0.3 or "走锚" in str(v.get("motion_state", ""))]
        eval_pool = active_vessels[:60] if len(active_vessels) >= 2 else vessels[:40]
        
        for i in range(len(eval_pool)):
            for j in range(i + 1, len(eval_pool)):
                v1 = eval_pool[i]
                v2 = eval_pool[j]
                rec = _evaluate_pair(v1, v2)
                if rec and rec.cri >= min_cri and rec.distance_nm <= max_distance_nm:
                    records.append(rec)

    # Sort descending by CRI (highest collision danger first)
    records.sort(key=lambda r: (r.cri, -r.dcpa_nm), reverse=True)
    return records[:top_limit]


def _evaluate_pair(v1: dict[str, Any], v2: dict[str, Any]) -> CollisionRiskRecord | None:
    try:
        lat1 = float(v1["lat"])
        lon1 = float(v1["lon"])
        sog1 = float(v1.get("sog_kn") or v1.get("sog") or 0.0)
        cog1 = float(v1.get("cog_deg") or v1.get("cog") or 0.0)

        lat2 = float(v2["lat"])
        lon2 = float(v2["lon"])
        sog2 = float(v2.get("sog_kn") or v2.get("sog") or 0.0)
        cog2 = float(v2.get("cog_deg") or v2.get("cog") or 0.0)
    except (KeyError, ValueError, TypeError):
        return None

    dist_nm, bearing12, dcpa_nm, tcpa_min = calculate_dcpa_tcpa(
        lat1, lon1, sog1, cog1,
        lat2, lon2, sog2, cog2,
    )

    cri = calculate_collision_risk_index(dist_nm, dcpa_nm, tcpa_min)

    # Check for anchor drag status
    m_state1 = str(v1.get("motion_state") or "").lower()
    m_state2 = str(v2.get("motion_state") or "").lower()
    is_drag1 = "走锚" in m_state1 or "drift" in m_state1
    is_drag2 = "走锚" in m_state2 or "drift" in m_state2
    is_anc1 = "锚" in m_state1 or "moored" in m_state1 or "anchor" in m_state1
    is_anc2 = "锚" in m_state2 or "moored" in m_state2 or "anchor" in m_state2

    enc_type, enc_desc, ob1, ob2 = classify_colregs_encounter(
        cog1, sog1, cog2, sog2, bearing12,
        is_anchor_drag1=is_drag1, is_anchor_drag2=is_drag2,
        is_anchored1=is_anc1, is_anchored2=is_anc2,
    )

    is_urgent = (cri >= 0.7) or (dcpa_nm < 0.25 and 0.0 < tcpa_min < 12.0)
    m1 = str(v1.get("mmsi", ""))
    m2 = str(v2.get("mmsi", ""))
    ev_id = f"colregs:{m1}_{m2}"

    return CollisionRiskRecord(
        vessel1_mmsi=m1,
        vessel1_name=str(v1.get("name") or f"MMSI {m1}"),
        vessel2_mmsi=m2,
        vessel2_name=str(v2.get("name") or f"MMSI {m2}"),
        distance_nm=dist_nm,
        distance_m=round(dist_nm * 1852.0, 1),
        dcpa_nm=dcpa_nm,
        dcpa_m=round(dcpa_nm * 1852.0, 1),
        tcpa_minutes=tcpa_min,
        cri=cri,
        encounter_type=enc_type,
        encounter_desc=enc_desc,
        obligation_v1=ob1,
        obligation_v2=ob2,
        is_urgent=is_urgent,
        evidence_id=ev_id,
    )
