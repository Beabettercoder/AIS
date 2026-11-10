"""Tests for Hydrodynamic Anchor Mechanics & Dragging Kinematics Evaluation Engine."""

import math
import pytest

from maritime_agent.anchor_physics import (
    AnchorStatus,
    SeabedType,
    assess_anchor_stability,
    calculate_current_force,
    calculate_wind_force,
    combine_environmental_forces,
    estimate_anchor_holding_capacity,
    haversine_distance_m,
    true_bearing_deg,
)


def test_distance_and_bearing_calculation():
    # 经度 0.001 度在赤道附近约 111 米
    lat1, lon1 = 29.5000, -94.8000
    lat2, lon2 = 29.5010, -94.8000
    dist = haversine_distance_m(lat1, lon1, lat2, lon2)
    assert 105.0 < dist < 115.0

    bearing = true_bearing_deg(lat1, lon1, lat2, lon2)
    # 正北方向应接近 0 度或 360 度
    assert abs(bearing - 0.0) < 1.0 or abs(bearing - 360.0) < 1.0

    # 正东方向
    lat3, lon3 = 29.5000, -94.7990
    bearing_east = true_bearing_deg(lat1, lon1, lat3, lon3)
    assert 88.0 < bearing_east < 92.0


def test_wind_and_current_force_calculation():
    # 30 节强风吹拂 150 米船舶
    f_wind, f_x, f_y = calculate_wind_force(
        wind_speed_knots=30.0,
        wind_direction_from_deg=0.0,
        heading_deg=0.0,  # 正顶风
        length_m=150.0,
        beam_m=25.0,
        draft_m=8.0,
    )
    assert f_wind > 5.0  # 应产生可观的风压 (kN)
    assert f_x > 0.0     # 主要是纵向力

    # 横风情况下侧向力明显增加
    f_cross, _, f_y_cross = calculate_wind_force(
        wind_speed_knots=30.0,
        wind_direction_from_deg=90.0,
        heading_deg=0.0,  # 正横风 (右舷)
        length_m=150.0,
        beam_m=25.0,
        draft_m=8.0,
    )
    assert f_cross > f_wind  # 侧投影面积远大于正投影面积，侧风合力应明显更大
    assert f_y_cross > 10.0

    # 水流力
    f_curr, _, _ = calculate_current_force(
        current_speed_knots=2.0,
        current_direction_deg=180.0,
        heading_deg=0.0,
        length_m=150.0,
        beam_m=25.0,
        draft_m=8.0,
    )
    assert f_curr > 0.0


def test_anchor_holding_capacity_different_seabeds():
    # 比较不同底质的持力
    mud_cap, _, _ = estimate_anchor_holding_capacity(
        seabed_type=SeabedType.MUD,
        vessel_length_m=160.0,
        vessel_beam_m=25.0,
        vessel_draft_m=9.0,
    )
    sand_cap, _, _ = estimate_anchor_holding_capacity(
        seabed_type=SeabedType.SAND,
        vessel_length_m=160.0,
        vessel_beam_m=25.0,
        vessel_draft_m=9.0,
    )
    rock_cap, _, _ = estimate_anchor_holding_capacity(
        seabed_type=SeabedType.ROCK,
        vessel_length_m=160.0,
        vessel_beam_m=25.0,
        vessel_draft_m=9.0,
    )

    # 细砂抓力 > 淤泥抓力 > 礁石岩石
    assert sand_cap > mud_cap > rock_cap
    assert rock_cap > 0.0


def test_normal_at_anchor_and_yawing():
    # 平稳受控锚泊：航速 0.1 节，位移 20 米，微风微流
    rec = assess_anchor_stability(
        mmsi="123456789",
        vessel_name="TEST_VESSEL_NORMAL",
        anchor_drop_lat=29.5000,
        anchor_drop_lon=-94.8000,
        current_lat=29.5001,
        current_lon=-94.8001,
        sog_knots=0.1,
        cog_deg=45.0,
        heading_deg=45.0,
        wind_speed_knots=8.0,
        wind_direction_from_deg=45.0,
        vessel_length_m=100.0,
        vessel_beam_m=16.0,
        vessel_draft_m=5.0,
    )
    assert rec.status == AnchorStatus.NORMAL_AT_ANCHOR
    assert not rec.is_urgent
    assert not rec.is_outside_swing_circle
    assert rec.drag_force_ratio < 0.5

    # 正常回旋偏摆：回旋圈内，航速在 0.5 节左右钟摆运动
    rec_swing = assess_anchor_stability(
        mmsi="123456789",
        vessel_name="TEST_VESSEL_SWING",
        anchor_drop_lat=29.5000,
        anchor_drop_lon=-94.8000,
        current_lat=29.5005,
        current_lon=-94.8005,  # 约 60-70 米，远小于安全回旋圈 (180米左右)
        sog_knots=0.6,
        cog_deg=120.0,
        heading_deg=110.0,
        wind_speed_knots=15.0,
        wind_direction_from_deg=300.0,
        vessel_length_m=100.0,
        vessel_beam_m=16.0,
        vessel_draft_m=5.0,
    )
    assert rec_swing.status == AnchorStatus.YAWING_SWINGING
    assert not rec_swing.is_urgent
    assert not rec_swing.is_outside_swing_circle


def test_confirmed_dragging_under_hurricane_beryl():
    # 模拟飓风 Beryl 场景：55 节狂风 + 强横向风压 + 泥质海床
    # 船舶突破安全回旋圈（偏离 350 米，回旋半径约 180 米），航速 1.6 节，漂移方向与风推向一致
    # 风来向 120 度，风推向 (120+180) = 300 度 (向西北推)
    # 船舶从 (29.5000, -94.8000) 向西北漂移到 (29.5020, -94.8030)
    drop_lat, drop_lon = 29.5000, -94.8000
    curr_lat, curr_lon = 29.5025, -94.8035  # 明显偏离原点

    rec = assess_anchor_stability(
        mmsi="987654321",
        vessel_name="BERYL_DRIFTING_SHIP",
        anchor_drop_lat=drop_lat,
        anchor_drop_lon=drop_lon,
        current_lat=curr_lat,
        current_lon=curr_lon,
        sog_knots=1.8,
        cog_deg=310.0,
        heading_deg=220.0,  # 侧风受压严重
        wind_speed_knots=58.0,  # 飓风级强风
        wind_direction_from_deg=130.0,
        vessel_length_m=180.0,
        vessel_beam_m=28.0,
        vessel_draft_m=10.0,
        seabed_type="mud",
    )
    assert rec.status == AnchorStatus.CONFIRMED_DRAGGING
    assert rec.is_urgent is True
    assert rec.is_outside_swing_circle is True
    assert rec.drag_force_ratio >= 0.80  # 高负荷滑脱阈值
    assert rec.drift_bearing_delta_deg is not None
    assert rec.drift_bearing_delta_deg < 45.0  # 位移与外力推向高度一致

    # 极端超载走锚：72 节强风导致 DFR >= 1.0 极限超载
    rec_overload = assess_anchor_stability(
        mmsi="987654321",
        vessel_name="OVERLOAD_DRIFTING_SHIP",
        anchor_drop_lat=drop_lat,
        anchor_drop_lon=drop_lon,
        current_lat=curr_lat,
        current_lon=curr_lon,
        sog_knots=2.2,
        cog_deg=310.0,
        heading_deg=220.0,
        wind_speed_knots=72.0,  # 严重超强台风/狂风
        wind_direction_from_deg=130.0,
        vessel_length_m=180.0,
        vessel_beam_m=28.0,
        vessel_draft_m=10.0,
        seabed_type="mud",
    )
    assert rec_overload.status == AnchorStatus.CONFIRMED_DRAGGING
    assert rec_overload.drag_force_ratio >= 1.0


def test_underway_maneuver_exclusion():
    # 船舶航速 8.5 节，朝 180 度主动南下航行，风来自 180 度（正迎风航行）
    rec = assess_anchor_stability(
        mmsi="555666777",
        vessel_name="UNDERWAY_SHIP",
        anchor_drop_lat=29.5000,
        anchor_drop_lon=-94.8000,
        current_lat=29.4950,
        current_lon=-94.8000,
        sog_knots=8.5,
        cog_deg=180.0,
        heading_deg=180.0,
        wind_speed_knots=30.0,
        wind_direction_from_deg=180.0,  # 顶风航行
        vessel_length_m=150.0,
        vessel_beam_m=22.0,
        vessel_draft_m=7.0,
    )
    assert rec.status == AnchorStatus.UNDERWAY_MANEUVER
    assert not rec.is_urgent
