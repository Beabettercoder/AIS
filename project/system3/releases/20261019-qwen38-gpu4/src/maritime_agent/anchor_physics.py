"""Hydrodynamic Anchor Mechanics & Dragging Kinematics Evaluation Engine.

Provides deterministic physical modeling and maritime mechanical calculations:
1. Environmental drag forces:
   - Wind aerodynamic drag force (OCIMF/IMO standard: F_wind = 0.5 * rho_air * C_w * A_w * V^2)
   - Current hydrodynamic drag force (F_curr = 0.5 * rho_water * C_c * A_c * V_c^2)
   - Environmental force vector combination (magnitude in kN, force bearing in degrees)
2. Seabed holding capacity and anchoring resistance:
   - Holding ground coefficients (lambda_bottom: mud, sand, clay, gravel, rock)
   - Anchor mass holding resistance (R_anchor = lambda_bottom * W_anchor * g)
   - Seabed chain friction resistance (R_chain = mu_chain * w_chain * L_bed * g)
   - Total holding capacity (R_holding)
3. Kinematic trajectory & swing circle boundary evaluation:
   - Safety swing circle radius (R_swing = L_chain + L_oa)
   - Distance from anchor drop coordinate (D_anchor)
   - Kinematic displacement vector alignment with environmental force bearing (Delta_psi)
4. Comprehensive dragging risk assessment:
   - Drag Force Ratio (DFR = F_env / R_holding)
   - Status classification: NORMAL_AT_ANCHOR, YAWING_SWINGING, CRITICAL_WARNING, CONFIRMED_DRAGGING, UNDERWAY_MANEUVER
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any


class AnchorStatus(str, Enum):
    NORMAL_AT_ANCHOR = "normal_at_anchor"          # 正常锚泊受控 (回旋圆内，抓力充足，航速接近 0)
    YAWING_SWINGING = "yawing_swinging"            # 锚泊正常回旋偏摆 (随风流在安全回旋圈内钟摆运动)
    CRITICAL_WARNING = "critical_warning"          # 走锚临界预警 (外力逼近抓力上限 DFR >= 0.8 或逼近圆周)
    CONFIRMED_DRAGGING = "confirmed_dragging"      # 确诊失控走锚 (外力突破极限持力 DFR > 1.0 且位移与合力方向吻合)
    UNDERWAY_MANEUVER = "underway_maneuver"        # 主动在航机动 (航向逆风流或航速较高，非受控漂移)
    UNKNOWN = "unknown"                            # 参数缺失或态势未明


class SeabedType(str, Enum):
    MUD = "mud"                     # 软泥/淤泥 (抓力比 ~3.5)
    SANDY_MUD = "sandy_mud"         # 砂质黏土/砂泥 (抓力比 ~4.0)
    SAND = "sand"                   # 细砂/中砂 (抓力比 ~4.5)
    CLAY = "clay"                   # 坚硬黏土 (抓力比 ~5.0)
    GRAVEL = "gravel"               # 粗砾石/卵石 (抓力比 ~1.8，抓底差)
    ROCK = "rock"                   # 礁石/基岩 (抓力比 ~1.0，易滑脱)


# 典型底质抓力比系数 (Holding Ground Ratio lambda)
HOLDING_GROUND_RATIO = {
    SeabedType.MUD: 3.5,
    SeabedType.SANDY_MUD: 4.0,
    SeabedType.SAND: 4.5,
    SeabedType.CLAY: 5.0,
    SeabedType.GRAVEL: 1.8,
    SeabedType.ROCK: 1.0,
}


@dataclass
class AnchorStabilityRecord:
    mmsi: str
    vessel_name: str
    status: AnchorStatus
    status_desc: str
    drag_force_ratio: float               # DFR = F_env / R_holding
    environmental_force_kn: float         # 环境总推力 (千牛 kN)
    holding_capacity_kn: float            # 锚系极限持力 (千牛 kN)
    wind_force_kn: float                  # 风压力分量 (kN)
    current_force_kn: float               # 水流力分量 (kN)
    env_force_bearing_deg: float          # 环境合力推向 (0-360 度真方位)
    drift_bearing_deg: float | None       # 实测漂移/移动真方位 (0-360 度)
    drift_bearing_delta_deg: float | None # 漂向与推向夹角偏差 (度)
    current_speed_knots: float            # 船舶实测当前航速 (节)
    distance_from_drop_m: float           # 距离抛锚点距离 (米)
    safety_swing_radius_m: float          # 理论安全回旋半径 (米)
    is_outside_swing_circle: bool         # 是否已突破安全回旋半径
    seabed_type: str                      # 底质类型
    is_urgent: bool                       # 是否属于险情 (走锚或临界)
    physics_evidence: str                 # 机理计算与分析凭据摘要
    evidence_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两经纬度球面距离 (米)。"""
    r_earth = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return r_earth * c


def true_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算从点 1 到点 2 的真方位角 (0-360 度)。"""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def calculate_wind_force(
    wind_speed_knots: float,
    wind_direction_from_deg: float,
    heading_deg: float,
    length_m: float,
    beam_m: float,
    draft_m: float,
    air_draft_m: float | None = None,
) -> tuple[float, float, float]:
    """计算船舶受风压力 (OCIMF 经验工程标准)。

    返回: (总风阻力 kN, 船体纵向风力 kN, 船体横向风力 kN)
    """
    rho_air = 1.225  # kg/m^3
    v_wind_ms = wind_speed_knots * 0.514444

    # 迎风面与侧风面有效受风面积估算
    freeboard = max(2.0, (air_draft_m - draft_m) if air_draft_m else (beam_m * 0.6))
    front_area = beam_m * freeboard
    lateral_area = length_m * freeboard * 0.85

    # 相对风向角 theta (0 度为正顶风，90 度为右舷受风，180 度为正顺风)
    # wind_direction_from_deg 是风的来向，风向推向为 (wind_direction_from_deg + 180) % 360
    rel_wind = (wind_direction_from_deg - heading_deg) % 360.0
    rel_rad = math.radians(rel_wind)

    # OCIMF 风压阻力系数经验近似
    # 纵向系数 Cx, 横向系数 Cy
    c_x = 0.85 * math.cos(rel_rad)
    c_y = 0.95 * math.sin(rel_rad)

    # 纵向力与横向力 (牛顿 N)
    f_x = 0.5 * rho_air * c_x * front_area * (v_wind_ms ** 2)
    f_y = 0.5 * rho_air * c_y * lateral_area * (v_wind_ms ** 2)

    f_total_n = math.sqrt(f_x ** 2 + f_y ** 2)
    return (f_total_n / 1000.0, abs(f_x) / 1000.0, abs(f_y) / 1000.0)


def calculate_current_force(
    current_speed_knots: float,
    current_direction_deg: float,
    heading_deg: float,
    length_m: float,
    beam_m: float,
    draft_m: float,
) -> tuple[float, float, float]:
    """计算水动力水流阻力 (kN)。

    返回: (总水流阻力 kN, 纵向流阻力 kN, 横向流阻力 kN)
    """
    rho_water = 1025.0  # kg/m^3 (海水密度)
    v_curr_ms = current_speed_knots * 0.514444

    front_underwater = beam_m * draft_m
    lateral_underwater = length_m * draft_m * 0.9

    rel_curr = (current_direction_deg - heading_deg) % 360.0
    rel_rad = math.radians(rel_curr)

    c_cx = 0.15 * math.cos(rel_rad)
    c_cy = 0.80 * math.sin(rel_rad)

    f_cx = 0.5 * rho_water * c_cx * front_underwater * (v_curr_ms ** 2)
    f_cy = 0.5 * rho_water * c_cy * lateral_underwater * (v_curr_ms ** 2)

    f_total_n = math.sqrt(f_cx ** 2 + f_cy ** 2)
    return (f_total_n / 1000.0, abs(f_cx) / 1000.0, abs(f_cy) / 1000.0)


def combine_environmental_forces(
    f_wind_kn: float,
    wind_direction_from_deg: float,
    f_curr_kn: float,
    current_direction_to_deg: float,
) -> tuple[float, float]:
    """计算环境合力大小 (kN) 及合力推向真方位 (0-360 度)。"""
    # 风推向为来向 + 180 度
    wind_push_bearing = (wind_direction_from_deg + 180.0) % 360.0
    wind_push_rad = math.radians(wind_push_bearing)
    curr_push_rad = math.radians(current_direction_to_deg)

    # 正北 (y) 与 正东 (x) 分量
    f_x = f_wind_kn * math.sin(wind_push_rad) + f_curr_kn * math.sin(curr_push_rad)
    f_y = f_wind_kn * math.cos(wind_push_rad) + f_curr_kn * math.cos(curr_push_rad)

    f_total = math.sqrt(f_x ** 2 + f_y ** 2)
    bearing_rad = math.atan2(f_x, f_y)
    bearing_deg = (math.degrees(bearing_rad) + 360.0) % 360.0
    return f_total, bearing_deg


def estimate_anchor_holding_capacity(
    seabed_type: SeabedType | str,
    vessel_length_m: float,
    vessel_beam_m: float,
    vessel_draft_m: float,
    anchor_weight_kg: float | None = None,
    chain_length_m: float | None = None,
    chain_diameter_mm: float | None = None,
    water_depth_m: float = 15.0,
) -> tuple[float, float, float]:
    """估算底质与锚链系统的综合极限抗抓力 (kN)。

    返回: (极限总持力 kN, 锚爪抓力 kN, 卧底链摩擦持力 kN)
    """
    g = 9.80665

    # 1. 估算或取标准霍尔/斯派克大抓力锚重
    if anchor_weight_kg is None or anchor_weight_kg <= 0:
        # 基于船舶主尺度近似经验公式: 舾装数 Equipment Number 关联
        displacement_est = vessel_length_m * vessel_beam_m * vessel_draft_m * 0.70  # 排水体积(吨)
        anchor_weight_kg = max(500.0, 0.035 * (displacement_est ** (2.0 / 3.0)) * 1000.0)

    # 底质抓力系数
    if isinstance(seabed_type, str):
        try:
            sb_enum = SeabedType(seabed_type.lower())
        except ValueError:
            sb_enum = SeabedType.SANDY_MUD
    else:
        sb_enum = seabed_type

    holding_ratio = HOLDING_GROUND_RATIO.get(sb_enum, 3.5)
    r_anchor_n = anchor_weight_kg * g * holding_ratio
    r_anchor_kn = r_anchor_n / 1000.0

    # 2. 卧底链摩擦持力估算
    if chain_diameter_mm is None or chain_diameter_mm <= 0:
        chain_diameter_mm = max(24.0, math.sqrt(anchor_weight_kg) * 0.7)
    # 锚链在空中的单位米重 w_air = 0.0219 * d^2 (kg/m)
    w_chain_m = 0.0219 * (chain_diameter_mm ** 2)
    # 海水中浮力折减系数约 0.87
    w_submerged_m = w_chain_m * 0.87

    if chain_length_m is None or chain_length_m <= 0:
        # 一般出链长度为水深的 4~6 倍 (如 5 倍水深)
        chain_length_m = max(100.0, water_depth_m * 5.0)

    # 悬链段长度近似为 sqrt(2 * depth * T / w)，简化为约 1.5 ~ 2.0 倍水深悬垂，其余为卧底链
    suspended_length = min(chain_length_m, water_depth_m * 1.8)
    bed_length = max(0.0, chain_length_m - suspended_length)

    # 卧底链与海床摩擦系数 mu 约为 0.75
    mu_chain = 0.75
    r_chain_n = bed_length * w_submerged_m * g * mu_chain
    r_chain_kn = r_chain_n / 1000.0

    total_holding_kn = r_anchor_kn + r_chain_kn
    return total_holding_kn, r_anchor_kn, r_chain_kn


def assess_anchor_stability(
    mmsi: str,
    vessel_name: str,
    anchor_drop_lat: float,
    anchor_drop_lon: float,
    current_lat: float,
    current_lon: float,
    sog_knots: float,
    cog_deg: float,
    heading_deg: float,
    wind_speed_knots: float,
    wind_direction_from_deg: float,
    vessel_length_m: float = 120.0,
    vessel_beam_m: float = 20.0,
    vessel_draft_m: float = 6.5,
    current_speed_knots: float = 1.0,
    current_direction_to_deg: float = 180.0,
    seabed_type: str = "sandy_mud",
    water_depth_m: float = 15.0,
    chain_length_m: float | None = None,
    anchor_weight_kg: float | None = None,
) -> AnchorStabilityRecord:
    """综合研判船舶锚泊稳定性、流体力学受力平衡与走锚机理态势。"""
    # 1. 距离抛锚点距离与理论安全回旋圈
    dist_m = haversine_distance_m(anchor_drop_lat, anchor_drop_lon, current_lat, current_lon)
    if chain_length_m is None or chain_length_m <= 0:
        chain_length_m = max(120.0, water_depth_m * 5.0)
    swing_radius_m = chain_length_m + vessel_length_m
    is_outside = dist_m > (swing_radius_m * 1.10)

    # 2. 计算受风压阻力与流水阻力
    f_wind_kn, _, _ = calculate_wind_force(
        wind_speed_knots=wind_speed_knots,
        wind_direction_from_deg=wind_direction_from_deg,
        heading_deg=heading_deg,
        length_m=vessel_length_m,
        beam_m=vessel_beam_m,
        draft_m=vessel_draft_m,
    )
    f_curr_kn, _, _ = calculate_current_force(
        current_speed_knots=current_speed_knots,
        current_direction_deg=current_direction_to_deg,
        heading_deg=heading_deg,
        length_m=vessel_length_m,
        beam_m=vessel_beam_m,
        draft_m=vessel_draft_m,
    )

    # 3. 合成外力推力矢量
    f_env_kn, env_push_deg = combine_environmental_forces(
        f_wind_kn=f_wind_kn,
        wind_direction_from_deg=wind_direction_from_deg,
        f_curr_kn=f_curr_kn,
        current_direction_to_deg=current_direction_to_deg,
    )

    # 4. 计算海底极限持力能力
    r_holding_kn, r_anchor_kn, r_chain_kn = estimate_anchor_holding_capacity(
        seabed_type=seabed_type,
        vessel_length_m=vessel_length_m,
        vessel_beam_m=vessel_beam_m,
        vessel_draft_m=vessel_draft_m,
        anchor_weight_kg=anchor_weight_kg,
        chain_length_m=chain_length_m,
        water_depth_m=water_depth_m,
    )

    # 5. 受力比 DFR (Drag Force Ratio)
    dfr = (f_env_kn / max(0.1, r_holding_kn))

    # 6. 位移矢量与环境推力方向一致性核查
    drift_bearing: float | None = None
    delta_deg: float | None = None
    if dist_m > 15.0:
        drift_bearing = true_bearing_deg(anchor_drop_lat, anchor_drop_lon, current_lat, current_lon)
        # 计算两角度最小差值
        diff = abs(drift_bearing - env_push_deg) % 360.0
        delta_deg = diff if diff <= 180.0 else (360.0 - diff)
    elif sog_knots >= 0.5:
        drift_bearing = cog_deg
        diff = abs(drift_bearing - env_push_deg) % 360.0
        delta_deg = diff if diff <= 180.0 else (360.0 - diff)

    # 7. 规则状态分类判断
    # 状态 A: 主动机航操纵 (Underway Maneuver) - 航速较高或严重逆风航行
    if sog_knots >= 3.0 or (sog_knots >= 1.5 and delta_deg is not None and delta_deg >= 110.0):
        status = AnchorStatus.UNDERWAY_MANEUVER
        status_desc = f"船舶处于在航机动操纵状态（实测航速 {sog_knots:.1f} 节），航向非被动风流漂移，非走锚失控。"
        is_urgent = False

    # 状态 B: 确诊失控走锚 (Confirmed Dragging)
    # 判据 1: DFR > 1.05 且航速持续 > 0.5 节且突破回旋圈
    # 判据 2: 明显突破回旋圈 (距离 > 1.2 * 半径)，且漂移矢量与环境推向高度一致 (delta <= 50 度)，且航速 > 0.4 节
    elif (dfr >= 1.0 and (is_outside or sog_knots >= 0.8)) or (
        is_outside and delta_deg is not None and delta_deg <= 50.0 and sog_knots >= 0.5
    ):
        status = AnchorStatus.CONFIRMED_DRAGGING
        status_desc = (
            f"确诊发生失控走锚！环境推力 ({f_env_kn:.1f} kN) 突破底质持力极限 ({r_holding_kn:.1f} kN, DFR={dfr:.2f})，"
            f"船体已偏离抛锚点 {dist_m:.0f} 米（突破回旋半径 {swing_radius_m:.0f} 米），"
            f"漂移航向与风流推力轴线一致（偏差 {delta_deg:.1f}°），呈被动滑移态势。"
        )
        is_urgent = True

    # 状态 C: 走锚临界预警 (Critical Warning)
    elif dfr >= 0.80 or (dist_m >= swing_radius_m * 0.90 and sog_knots >= 0.6):
        status = AnchorStatus.CRITICAL_WARNING
        status_desc = (
            f"走锚临界预警！当前环境负荷率达 {dfr*100:.1f}%（推力 {f_env_kn:.1f} kN / 持力 {r_holding_kn:.1f} kN），"
            f"船体距安全边缘仅余 {max(0.0, swing_radius_m - dist_m):.0f} 米，随时可能发生锚爪脱底滑移。"
        )
        is_urgent = True

    # 状态 D: 正常回旋偏摆 (Yawing / Swinging)
    elif dist_m <= swing_radius_m and (0.2 <= sog_knots < 1.5):
        status = AnchorStatus.YAWING_SWINGING
        status_desc = (
            f"船舶处于正常锚泊回旋偏摆中。偏离抛锚点 {dist_m:.0f} 米在安全回旋圈 ({swing_radius_m:.0f} 米) 范围内，"
            f"持力安全裕度充足 (DFR={dfr:.2f})，随风流钟摆摆动属于正常锚泊物理动态。"
        )
        is_urgent = False

    # 状态 E: 正常受控锚泊 (Normal at Anchor)
    else:
        status = AnchorStatus.NORMAL_AT_ANCHOR
        status_desc = (
            f"船舶受控平稳锚泊。偏离原点 {dist_m:.0f} 米，航速 {sog_knots:.1f} 节极低，"
            f"底质极限持力 ({r_holding_kn:.1f} kN) 远大于外力推力 ({f_env_kn:.1f} kN, DFR={dfr:.2f})，状态安全稳固。"
        )
        is_urgent = False

    evidence_summary = (
        f"MMSI={mmsi} | 状态={status.value} | DFR={dfr:.2f} | 推力={f_env_kn:.1f}kN "
        f"(风={f_wind_kn:.1f}kN, 流={f_curr_kn:.1f}kN, 推向={env_push_deg:.0f}°) | "
        f"持力={r_holding_kn:.1f}kN (锚={r_anchor_kn:.1f}kN, 链={r_chain_kn:.1f}kN, 底质={seabed_type}) | "
        f"位移={dist_m:.0f}m / 安全半径={swing_radius_m:.0f}m"
    )

    return AnchorStabilityRecord(
        mmsi=mmsi,
        vessel_name=vessel_name,
        status=status,
        status_desc=status_desc,
        drag_force_ratio=round(dfr, 3),
        environmental_force_kn=round(f_env_kn, 1),
        holding_capacity_kn=round(r_holding_kn, 1),
        wind_force_kn=round(f_wind_kn, 1),
        current_force_kn=round(f_curr_kn, 1),
        env_force_bearing_deg=round(env_push_deg, 1),
        drift_bearing_deg=round(drift_bearing, 1) if drift_bearing is not None else None,
        drift_bearing_delta_deg=round(delta_deg, 1) if delta_deg is not None else None,
        current_speed_knots=round(sog_knots, 2),
        distance_from_drop_m=round(dist_m, 1),
        safety_swing_radius_m=round(swing_radius_m, 1),
        is_outside_swing_circle=is_outside,
        seabed_type=str(seabed_type),
        is_urgent=is_urgent,
        physics_evidence=evidence_summary,
        evidence_id=f"anchor_physics:{mmsi}:{int(dist_m)}m",
    )
