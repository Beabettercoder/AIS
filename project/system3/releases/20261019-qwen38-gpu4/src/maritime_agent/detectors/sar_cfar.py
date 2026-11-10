"""Small, explainable SAR CFAR baseline for georeferenced GeoTIFF assets."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


from .sar_geometry import resolve_sar_geometry
from .doppler_shift import restore_ground_position_from_sar_echo

DETECTOR_VERSION = "sar_cfar_v7"


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = phi2 - phi1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 6371008.8 * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _box_sum(values, radius: int):
    import numpy as np

    padded = np.pad(values.astype(np.float64, copy=False), radius, mode="constant")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    size = radius * 2 + 1
    return (integral[size:, size:] - integral[:-size, size:]
            - integral[size:, :-size] + integral[:-size, :-size])


def _local_zscore(values, valid, background_radius: int, guard_radius: int):
    import numpy as np

    weighted = np.where(valid, values, 0.0).astype(np.float64, copy=False)
    outer_n = _box_sum(valid.astype(np.float64, copy=False), background_radius)
    outer_sum = _box_sum(weighted, background_radius)
    outer_sq = _box_sum(weighted * weighted, background_radius)
    if guard_radius:
        inner_n = _box_sum(valid.astype(np.float64, copy=False), guard_radius)
        inner_sum = _box_sum(weighted, guard_radius)
        inner_sq = _box_sum(weighted * weighted, guard_radius)
        outer_n, outer_sum, outer_sq = outer_n - inner_n, outer_sum - inner_sum, outer_sq - inner_sq
    count = np.maximum(outer_n, 1.0)
    mean = outer_sum / count
    variance = np.maximum(outer_sq / count - mean * mean, 1e-6)
    z = (values.astype(np.float64, copy=False) - mean) / np.sqrt(variance)
    return z.astype(np.float32), outer_n.astype(np.float32)


def _tile_peaks(score, candidate, cell_size: int, max_detections: int, nms_radius: int):
    import numpy as np

    height, width = score.shape
    padded_h = math.ceil(height / cell_size) * cell_size
    padded_w = math.ceil(width / cell_size) * cell_size
    work = np.full((padded_h, padded_w), -np.inf, dtype="float32")
    work[:height, :width] = np.where(candidate, score, -np.inf)
    tiles = work.reshape(padded_h // cell_size, cell_size,
                         padded_w // cell_size, cell_size).transpose(0, 2, 1, 3)
    flat = tiles.reshape(tiles.shape[0], tiles.shape[1], -1)
    indices = flat.argmax(axis=2)
    values = flat.max(axis=2)
    tile_rows, tile_cols = np.where(np.isfinite(values))
    peaks = []
    for tile_row, tile_col in zip(tile_rows.tolist(), tile_cols.tolist()):
        index = int(indices[tile_row, tile_col])
        row = tile_row * cell_size + index // cell_size
        col = tile_col * cell_size + index % cell_size
        if row < height and col < width:
            peaks.append((float(score[row, col]), row, col))
    peaks.sort(reverse=True)
    selected: list[tuple[float, int, int]] = []
    radius_sq = nms_radius * nms_radius
    for item in peaks:
        _, row, col = item
        if any((row - kept[1]) ** 2 + (col - kept[2]) ** 2 <= radius_sq for kept in selected):
            continue
        selected.append(item)
        if len(selected) >= max_detections:
            break
    return selected


def _component_features(score, valid, row: int, col: int, support_floor: float,
                        radius: int, pixel_size_m: tuple[float, float]):
    """Describe the connected bright support around one CFAR peak.

    This deliberately keeps detection and interpretation separate: a component
    is a radar-bright object, not automatically a vessel.
    """
    import numpy as np

    height, width = score.shape
    r0, r1 = max(0, row - radius), min(height, row + radius + 1)
    c0, c1 = max(0, col - radius), min(width, col + radius + 1)
    support = valid[r0:r1, c0:c1] & np.isfinite(score[r0:r1, c0:c1]) \
        & (score[r0:r1, c0:c1] >= support_floor)
    seed = (row - r0, col - c0)
    support[seed] = True
    pending, seen = [seed], {seed}
    while pending:
        rr, cc = pending.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if (dr or dc) and 0 <= nr < support.shape[0] and 0 <= nc < support.shape[1] \
                        and support[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    pending.append((nr, nc))
    coordinates = np.asarray(sorted(seen), dtype="float64")
    rows, cols = coordinates[:, 0], coordinates[:, 1]
    bbox_h = int(rows.max() - rows.min() + 1)
    bbox_w = int(cols.max() - cols.min() + 1)
    # PCA gives a stable orientation and elongation even when the component is
    # oblique. Add one pixel to convert centre spread into an object extent.
    if len(coordinates) >= 2:
        xy = np.column_stack((cols * pixel_size_m[0], -rows * pixel_size_m[1]))
        centered = xy - xy.mean(axis=0)
        covariance = centered.T @ centered / max(len(xy) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        axis = eigenvectors[:, order[0]]
        projections = centered @ eigenvectors[:, order]
        major_m = float(np.ptp(projections[:, 0]) + max(pixel_size_m))
        minor_m = float(np.ptp(projections[:, 1]) + min(pixel_size_m))
        orientation = float((math.degrees(math.atan2(axis[1], axis[0])) + 180.0) % 180.0)
    else:
        major_m, minor_m, orientation = max(pixel_size_m), min(pixel_size_m), 0.0
    major_m, minor_m = max(major_m, minor_m), min(major_m, minor_m)
    return {
        "component_pixel_count": int(len(coordinates)),
        "component_bbox_height_px": bbox_h,
        "component_bbox_width_px": bbox_w,
        "component_length_m": round(major_m, 2),
        "component_width_m": round(minor_m, 2),
        "component_elongation": round(major_m / max(minor_m, 1e-6), 3),
        "component_orientation_deg": round(orientation, 2),
        "component_truncated": bool(r0 == 0 or c0 == 0 or r1 == height or c1 == width
                                      or rows.min() == 0 or cols.min() == 0
                                      or rows.max() == support.shape[0] - 1
                                      or cols.max() == support.shape[1] - 1),
    }


def _pixel_size_metres(transform, crs) -> tuple[float, float]:
    """Estimate ground sampling distance at the raster centre."""
    from rasterio.transform import xy
    from rasterio.warp import transform as transform_coordinates

    x0, y0 = xy(transform, 0, 0, offset="center")
    x1, y1 = xy(transform, 0, 1, offset="center")
    x2, y2 = xy(transform, 1, 0, offset="center")
    lon, lat = transform_coordinates(crs, "EPSG:4326", [x0, x1, x2], [y0, y1, y2])

    def metres(a_lon, a_lat, b_lon, b_lat):
        phi1, phi2 = math.radians(a_lat), math.radians(b_lat)
        dphi, dlambda = phi2 - phi1, math.radians(b_lon - a_lon)
        value = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return 6371008.8 * 2 * math.atan2(math.sqrt(value), math.sqrt(max(0.0, 1 - value)))

    return max(metres(lon[0], lat[0], lon[1], lat[1]), 0.01), \
        max(metres(lon[0], lat[0], lon[2], lat[2]), 0.01)


def detect_sar_targets(assets: list[dict[str, Any]], parameters: dict[str, Any]) -> dict[str, Any]:
    """Return point detections from one or more co-registered SAR amplitude assets."""
    try:
        import numpy as np
        import rasterio
        from rasterio.transform import xy
        from rasterio.warp import transform as transform_coordinates
    except ImportError as exc:  # pragma: no cover - exercised when optional extra is absent
        raise RuntimeError("SAR 检测需要安装项目的 remote-sensing 可选依赖") from exc

    background_radius = int(parameters.get("background_radius_px", 24))
    guard_radius = int(parameters.get("guard_radius_px", 3))
    threshold = float(parameters.get("threshold_z", 5.0))
    min_percentile = float(parameters.get("min_percentile", 99.0))
    cell_size = int(parameters.get("cell_size_px", 8))
    nms_radius = int(parameters.get("nms_radius_px", 10))
    # CFAR background windows are incomplete at the image boundary.  Peaks in
    # that area are usually crop/nodata artefacts and must not be presented to
    # the association layer as vessels.
    edge_margin = int(parameters.get(
        "edge_margin_px", max(background_radius + guard_radius, 32)))
    max_detections = int(parameters.get("max_detections", 300))
    component_support_delta = float(parameters.get("component_support_delta_z", 1.5))
    component_radius = int(parameters.get("component_radius_px", 12))
    if not (guard_radius < background_radius and 0 < cell_size and 0 < max_detections
            and edge_margin >= 0 and component_support_delta >= 0 and component_radius > 0):
        raise ValueError("CFAR 参数无效：guard 必须小于 background，cell/max_detections 必须为正数")

    scores, amplitudes, valid_masks, metadata = [], {}, [], []
    reference = None
    for asset in assets:
        with rasterio.open(Path(asset["path"])) as source:
            band = source.read(1, masked=True).astype("float32")
            values = np.log1p(band.filled(0.0))
            valid = ~np.ma.getmaskarray(band) & np.isfinite(values) & (band.filled(0.0) > 0)
            current = (source.width, source.height, source.transform, source.crs)
            if reference is None:
                reference = current
            elif current != reference:
                raise ValueError("SAR 多极化资产必须具有相同网格、仿射变换和 CRS")
            zscore, count = _local_zscore(values, valid, background_radius, guard_radius)
            zscore[count < 32] = -np.inf
            scores.append(zscore)
            valid_masks.append(valid)
            amplitudes[str(asset.get("polarization") or asset["asset_id"])] = band.filled(0.0)
            metadata.append({"asset_id": asset["asset_id"], "polarization": asset.get("polarization"),
                             "crs": source.crs.to_string(), "bbox": list(source.bounds),
                             "width": source.width, "height": source.height,
                             "transform": tuple(source.transform)})
    if not scores or reference is None:
        raise ValueError("检测任务没有可读取的 SAR 资产")
    combined = np.max(np.stack(scores), axis=0)
    valid = np.logical_and.reduce(valid_masks)
    valid_mask_path = parameters.get("valid_mask_path")
    external_mask = None
    if valid_mask_path:
        with rasterio.open(Path(valid_mask_path)) as mask_source:
            mask_grid = (mask_source.width, mask_source.height,
                         mask_source.transform, mask_source.crs)
            if mask_grid != reference:
                raise ValueError("valid_mask_path 必须与SAR资产使用相同网格、变换和CRS")
            external_mask = mask_source.read(1) > 0
            valid &= external_mask
    finite_values = combined[np.isfinite(combined) & valid]
    if not finite_values.size:
        return {"detections": [], "assets": metadata, "summary": {"candidate_pixels": 0}}
    percentile_floor = float(np.percentile(finite_values, min_percentile))
    candidate = valid & np.isfinite(combined) & (combined >= max(threshold, percentile_floor))
    candidate_before_valid_mask = int((np.logical_and.reduce(valid_masks)
                                       & np.isfinite(combined)
                                       & (combined >= max(threshold, percentile_floor))).sum())
    candidate_before_edge_filter = int(candidate.sum())
    if edge_margin:
        height, width = candidate.shape
        if edge_margin * 2 >= min(height, width):
            raise ValueError("edge_margin_px 过大，未留下有效检测区域")
        candidate[:edge_margin, :] = False
        candidate[-edge_margin:, :] = False
        candidate[:, :edge_margin] = False
    crs = reference[3]
    pixel_size_m = _pixel_size_metres(reference[2], crs)
    transform = reference[2]

    # --- 多普勒频移补偿与平台几何解析 ---
    enable_doppler = bool(parameters.get("enable_doppler_compensation", True))
    geometry = resolve_sar_geometry(parameters)

    # --- 阶段 1: AIS 先验反向引导局部聚光灯检测 (Prior-Guided ROI Spotlighting with Doppler Alignment) ---
    prior_targets = parameters.get("prior_targets") or []
    prior_peaks = []
    if prior_targets:
        from rasterio.transform import rowcol
        spotlight_radius_m = float(parameters.get("spotlight_radius_m", 150.0 if enable_doppler else 350.0))
        max_dist_m = float(parameters.get("max_spotlight_dist_m", 250.0 if enable_doppler else 450.0))
        search_radius_px = max(int(round(spotlight_radius_m / max(min(pixel_size_m), 1.0))), 6)
        candidate_spotlight_floor = max(threshold - 1.8, 1.8)

        for prior in prior_targets:
            p_lon, p_lat = float(prior["lon"]), float(prior["lat"])
            g_lon = float(prior.get("ground_lon", p_lon))
            g_lat = float(prior.get("ground_lat", p_lat))
            try:
                if str(crs).upper() not in ("EPSG:4326", "WGS 84"):
                    xs, ys = transform_coordinates("EPSG:4326", crs, [p_lon], [p_lat])
                    x_crs, y_crs = xs[0], ys[0]
                else:
                    x_crs, y_crs = p_lon, p_lat
                center_row, center_col = rowcol(transform, x_crs, y_crs)
            except Exception:
                continue

            height, width = combined.shape
            if not (edge_margin <= center_row < height - edge_margin and
                    edge_margin <= center_col < width - edge_margin):
                continue

            r0, r1 = max(0, center_row - search_radius_px), min(height, center_row + search_radius_px + 1)
            c0, c1 = max(0, center_col - search_radius_px), min(width, center_col + search_radius_px + 1)

            sub_score = combined[r0:r1, c0:c1]
            sub_valid = valid[r0:r1, c0:c1] & np.isfinite(sub_score)
            if not np.any(sub_valid):
                continue

            masked_sub = np.where(sub_valid, sub_score, -np.inf)
            local_max_idx = np.argmax(masked_sub)
            local_max_r, local_max_c = np.unravel_index(local_max_idx, masked_sub.shape)
            best_z = float(sub_score[local_max_r, local_max_c])

            if best_z >= candidate_spotlight_floor:
                best_row = r0 + int(local_max_r)
                best_col = c0 + int(local_max_c)

                min_comp_pixels = int(parameters.get("min_spotlight_component_pixels", 1))
                if min_comp_pixels > 1:
                    comp_check = _component_features(
                        combined, valid, best_row, best_col,
                        max(threshold - component_support_delta, 1.2),
                        component_radius, pixel_size_m
                    )
                    if comp_check["component_pixel_count"] < min_comp_pixels and best_z < threshold:
                        continue

                x, y = xy(transform, best_row, best_col, offset="center")
                det_lons, det_lats = transform_coordinates(crs, "EPSG:4326", [x], [y])
                det_lon, det_lat = float(det_lons[0]), float(det_lats[0])
                dist_m = _haversine_m(p_lon, p_lat, det_lon, det_lat)

                if dist_m <= max_dist_m:
                    sog = prior.get("sog_kn")
                    cog = prior.get("cog_deg")
                    if enable_doppler and sog is not None and sog > 0.5 and cog is not None:
                        restored = restore_ground_position_from_sar_echo(
                            image_lon=det_lon,
                            image_lat=det_lat,
                            sog_kn=sog,
                            cog_deg=cog,
                            geometry=geometry,
                        )
                        ground_lon = restored["ground_lon"]
                        ground_lat = restored["ground_lat"]
                        doppler_shift_m = restored["azimuth_shift_m"]
                        true_ground_error_m = round(_haversine_m(g_lon, g_lat, ground_lon, ground_lat), 1)
                    else:
                        ground_lon = det_lon
                        ground_lat = det_lat
                        doppler_shift_m = 0.0
                        true_ground_error_m = round(dist_m, 1)

                    peak_entry = (
                        best_z, int(best_row), int(best_col),
                        {
                            "is_prior_match": True,
                            "detection_source": "prior_guided_spotlight",
                            "matched_mmsi": str(prior.get("mmsi")),
                            "prior_vessel_name": prior.get("vessel_name"),
                            "prior_sog_kn": sog,
                            "prior_cog_deg": cog,
                            "prior_distance_m": round(dist_m, 1),
                            "image_echo_lon": det_lon,
                            "image_echo_lat": det_lat,
                            "restored_ground_lon": ground_lon,
                            "restored_ground_lat": ground_lat,
                            "doppler_compensated": enable_doppler,
                            "doppler_azimuth_shift_m": doppler_shift_m,
                            "ground_alignment_error_m": true_ground_error_m,
                        }
                    )
                    # 避免不同先验锁定同一回波质心冲突，保留距离最近者
                    existing_idx = next((i for i, p in enumerate(prior_peaks)
                                         if p[1] == int(best_row) and p[2] == int(best_col)), None)
                    if existing_idx is not None:
                        if dist_m < prior_peaks[existing_idx][3]["prior_distance_m"]:
                            prior_peaks[existing_idx] = peak_entry
                    else:
                        prior_peaks.append(peak_entry)
                    # 在全图候选掩膜中抹去该合规船周边（残差掩膜），避免盲检重复触发
                    mask_r0 = max(0, best_row - nms_radius)
                    mask_r1 = min(height, best_row + nms_radius + 1)
                    mask_c0 = max(0, best_col - nms_radius)
                    mask_c1 = min(width, best_col + nms_radius + 1)
                    candidate[mask_r0:mask_r1, mask_c0:mask_c1] = False

    # --- 阶段 2: 残差开阔海域地毯式盲检 (Residual Blind Area Scanning) ---
    remaining_quota = max(10, max_detections - len(prior_peaks))
    blind_peaks_raw = _tile_peaks(combined, candidate, cell_size, remaining_quota, nms_radius)
    blind_peaks = [
        (item[0], item[1], item[2],
         {
             "is_prior_match": False,
             "detection_source": "residual_blind_scan",
             "is_dark_vessel_candidate": True,
         })
        for item in blind_peaks_raw
    ]

    all_peaks = prior_peaks + blind_peaks
    all_peaks.sort(key=lambda item: (-int(item[3].get("is_prior_match", False)), -item[0]))
    all_peaks = all_peaks[:max_detections]

    detections = []
    for rank, (zscore, row, col, extra_props) in enumerate(all_peaks, 1):
        x, y = xy(reference[2], row, col, offset="center")
        lon, lat = transform_coordinates(crs, "EPSG:4326", [x], [y])
        raw_det_lon, raw_det_lat = float(lon[0]), float(lat[0])
        final_lon = extra_props.get("restored_ground_lon", raw_det_lon)
        final_lat = extra_props.get("restored_ground_lat", raw_det_lat)
        z_excess = zscore - threshold
        if extra_props.get("is_prior_match"):
            score = float(1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, z_excess + 1.2)))))
        else:
            score = float(1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, z_excess)))))
        score = min(0.999, max(0.01, round(score, 4)))
        component = _component_features(
            combined, valid, row, col, max(threshold - component_support_delta, 0.0),
            component_radius, pixel_size_m)
        amplitude_features = {}
        for name, values in amplitudes.items():
            patch = values[max(0, row - 1):row + 2, max(0, col - 1):col + 2]
            amplitude_features[name] = {
                "peak": float(values[row, col]),
                "local_3x3_mean": float(np.mean(patch)),
            }
        polarizations = {str(name).upper(): values for name, values in amplitudes.items()}
        vv_vh_ratio_db = None
        if "VV" in polarizations and "VH" in polarizations:
            vv = max(float(polarizations["VV"][row, col]), 1e-12)
            vh = max(float(polarizations["VH"][row, col]), 1e-12)
            vv_vh_ratio_db = round(10.0 * math.log10(vv / vh), 3)
        detections.append({"rank": int(rank), "row": int(row), "col": int(col),
                           "lon": final_lon, "lat": final_lat,
                           "image_echo_lon": raw_det_lon, "image_echo_lat": raw_det_lat,
                           "detection_confidence": score, "cfar_z": zscore,
                           "amplitudes": {name: float(values[row, col])
                                          for name, values in amplitudes.items()},
                           "polarization_features": amplitude_features,
                           "vv_vh_ratio_db": vv_vh_ratio_db, **extra_props, **component})
    return {"detections": detections, "assets": metadata,
            "summary": {"candidate_pixels": int(candidate.sum()), "peak_count": len(detections),
                        "prior_guided_count": len(prior_peaks),
                        "residual_blind_count": len(blind_peaks),
                        "edge_margin_px": edge_margin,
                        "pixel_size_m": [round(value, 3) for value in pixel_size_m],
                        "component_support_delta_z": component_support_delta,
                        "valid_mask_fraction": (round(float(external_mask.mean()), 6)
                                                if external_mask is not None else None),
                        "masked_candidate_pixels_removed": candidate_before_valid_mask - candidate_before_edge_filter,
                        "edge_candidate_pixels_removed": candidate_before_edge_filter - int(candidate.sum()),
                        "threshold_z": threshold, "effective_floor_z": max(threshold, percentile_floor),
                        "doppler_compensation_enabled": enable_doppler,
                        "sar_geometry": geometry.to_dict()}}

