from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import zlib
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config

HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"
PPT_NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


@dataclass(frozen=True)
class Component:
    component_id: int
    area: int
    min_row: int
    max_row: int
    min_col: int
    max_col: int
    centroid_row: float
    centroid_col: float

    @property
    def width(self) -> int:
        return self.max_col - self.min_col + 1

    @property
    def height(self) -> int:
        return self.max_row - self.min_row + 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def well_id_from_path(path: Path, suffix: str) -> str:
    return path.name.split(suffix)[0]


def qbin(series: pd.Series, n_bins: int, prefix: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0 or values.nunique(dropna=True) <= 1:
        return pd.Series([f"{prefix}_all"] * len(series), index=series.index, dtype=object)
    try:
        labels = pd.qcut(
            values.rank(method="first"), q=min(n_bins, values.notna().sum()), labels=False
        )
    except ValueError:
        labels = pd.cut(values, bins=min(n_bins, values.nunique(dropna=True)), labels=False)
    out = labels.astype("Int64").astype(str).where(values.notna(), "missing")
    return prefix + "_" + out


def fixed_azimuth_bin(series: pd.Series, n_bins: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    edges = np.linspace(-180.0, 180.0, n_bins + 1)
    labels = pd.cut(values, bins=edges, include_lowest=True, labels=False)
    out = labels.astype("Int64").astype(str).where(values.notna(), "missing")
    return "az_" + out


def summarize_horizontal_well(path: Path) -> dict[str, Any]:
    well_id = well_id_from_path(path, HORIZONTAL_SUFFIX)
    df = pd.read_csv(path)
    eval_mask = df["TVT_input"].isna()
    known_mask = df["TVT_input"].notna()
    eval_indices = np.flatnonzero(eval_mask.to_numpy())
    first_eval = int(eval_indices[0]) if len(eval_indices) else -1
    last_eval = int(eval_indices[-1]) if len(eval_indices) else -1

    prefix = (
        df.loc[: max(first_eval - 1, 0), "TVT_input"]
        if first_eval > 0
        else df["TVT_input"].iloc[:0]
    )
    known_tvt = df.loc[known_mask, "TVT_input"]
    if prefix.notna().any():
        last_known_tvt = float(prefix.dropna().iloc[-1])
        median_known_tvt = float(prefix.dropna().median())
        prefix_tvt_range = float(prefix.max() - prefix.min())
    elif known_tvt.notna().any():
        last_known_tvt = float(known_tvt.iloc[-1])
        median_known_tvt = float(known_tvt.median())
        prefix_tvt_range = float(known_tvt.max() - known_tvt.min())
    else:
        last_known_tvt = float("nan")
        median_known_tvt = float("nan")
        prefix_tvt_range = float("nan")

    x = df["X"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    z = df["Z"].to_numpy(dtype=float)
    md = df["MD"].to_numpy(dtype=float)
    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])
    dz = float(z[-1] - z[0])
    md_span = float(np.nanmax(md) - np.nanmin(md))
    step = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    chord = math.sqrt(dx * dx + dy * dy + dz * dz)
    tortuosity = float(np.nansum(step) / max(chord, 1e-6))

    return {
        "well_id": well_id,
        "n_rows": int(len(df)),
        "eval_rows": int(eval_mask.sum()),
        "known_rows": int(known_mask.sum()),
        "first_eval_row": first_eval,
        "last_eval_row": last_eval,
        "eval_length": int(eval_mask.sum()),
        "prefix_length": int((np.arange(len(df)) < first_eval).sum())
        if first_eval >= 0
        else int(len(df)),
        "centroid_x": float(np.nanmean(x)),
        "centroid_y": float(np.nanmean(y)),
        "start_x": float(x[0]),
        "start_y": float(y[0]),
        "end_x": float(x[-1]),
        "end_y": float(y[-1]),
        "delta_x": dx,
        "delta_y": dy,
        "delta_z": dz,
        "signed_azimuth_deg": math.degrees(math.atan2(dy, dx)),
        "median_known_tvt": median_known_tvt,
        "last_known_tvt": last_known_tvt,
        "median_full_tvt": float(df["TVT"].median()) if "TVT" in df else float("nan"),
        "prefix_tvt_range": prefix_tvt_range,
        "gr_coverage": float(df["GR"].notna().mean()) if "GR" in df else float("nan"),
        "gr_missing_rate": float(df["GR"].isna().mean()) if "GR" in df else float("nan"),
        "md_min": float(np.nanmin(md)),
        "md_max": float(np.nanmax(md)),
        "md_span": md_span,
        "z_min": float(np.nanmin(z)),
        "z_max": float(np.nanmax(z)),
        "z_span": float(np.nanmax(z) - np.nanmin(z)),
        "dz_dmd": dz / md_span if md_span > 0 else float("nan"),
        "bbox_dx": float(np.nanmax(x) - np.nanmin(x)),
        "bbox_dy": float(np.nanmax(y) - np.nanmin(y)),
        "bbox_dz": float(np.nanmax(z) - np.nanmin(z)),
        "tortuosity": tortuosity,
    }


def typewell_exact_groups(train_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(train_dir.glob(f"*{TYPEWELL_SUFFIX}")):
        well_id = well_id_from_path(path, TYPEWELL_SUFFIX)
        rows.append(
            {
                "well_id": well_id,
                "typewell_exact_sha16": sha256_file(path)[:16],
                "typewell_path_name": path.name,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["well_id", "typewell_group_id", "typewell_group_size"])
    frame["typewell_group_id"] = "exact_" + frame["typewell_exact_sha16"]
    frame["typewell_group_size"] = frame.groupby("typewell_group_id")["well_id"].transform("size")
    return frame[["well_id", "typewell_group_id", "typewell_group_size", "typewell_exact_sha16"]]


def build_well_metadata(paths: ExperimentPaths, config: dict[str, Any]) -> pd.DataFrame:
    max_wells = get_nested(config, "audit.max_wells")
    train_files = sorted(paths.train_data_dir.glob(f"*{HORIZONTAL_SUFFIX}"))
    if max_wells is not None:
        train_files = train_files[: int(max_wells)]
    if not train_files:
        raise FileNotFoundError(f"No train horizontal well files found in {paths.train_data_dir}")

    meta = pd.DataFrame(summarize_horizontal_well(path) for path in train_files).sort_values(
        "well_id"
    )
    groups = typewell_exact_groups(paths.train_data_dir)
    meta = meta.merge(groups, on="well_id", how="left")
    meta["typewell_group_id"] = meta["typewell_group_id"].fillna("missing_" + meta["well_id"])
    meta["typewell_group_size"] = meta["typewell_group_size"].fillna(1).astype(int)

    bins = get_nested(config, "audit.binning") or {}
    meta["azimuth_bin"] = fixed_azimuth_bin(
        meta["signed_azimuth_deg"], int(bins.get("azimuth_bins", 4))
    )
    meta["tvt_bin"] = qbin(
        meta["median_known_tvt"].fillna(meta["median_full_tvt"]),
        int(bins.get("tvt_bins", 4)),
        "tvt",
    )
    meta["x_bin"] = qbin(meta["centroid_x"], int(bins.get("spatial_x_bins", 4)), "x")
    meta["y_bin"] = qbin(meta["centroid_y"], int(bins.get("spatial_y_bins", 4)), "y")
    meta["spatial_bin"] = meta["x_bin"].astype(str) + "__" + meta["y_bin"].astype(str)
    meta["eval_length_bin"] = qbin(
        meta["eval_length"], int(bins.get("eval_length_bins", 4)), "eval_len"
    )
    meta["prefix_length_bin"] = qbin(
        meta["prefix_length"], int(bins.get("prefix_length_bins", 4)), "prefix_len"
    )
    meta["gr_bin"] = qbin(meta["gr_coverage"], int(bins.get("gr_bins", 4)), "gr")

    x_min, x_max = float(meta["centroid_x"].min()), float(meta["centroid_x"].max())
    y_min, y_max = float(meta["centroid_y"].min()), float(meta["centroid_y"].max())
    meta["centroid_x_norm"] = (meta["centroid_x"] - x_min) / max(x_max - x_min, 1e-9)
    meta["centroid_y_norm"] = (meta["centroid_y"] - y_min) / max(y_max - y_min, 1e-9)
    return meta.reset_index(drop=True)


def paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_png_rgba(data: bytes) -> np.ndarray:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not a PNG file")
    offset = 8
    width = height = bit_depth = color_type = None
    compressed = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _compression, _filter, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            if bit_depth != 8 or interlace != 0:
                raise ValueError(f"unsupported PNG bit_depth={bit_depth} interlace={interlace}")
        elif chunk_type == b"IDAT":
            compressed.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None or bit_depth is None or color_type is None:
        raise ValueError("PNG missing IHDR")

    channels_by_type = {0: 1, 2: 3, 4: 2, 6: 4}
    if color_type not in channels_by_type:
        raise ValueError(f"unsupported PNG color_type={color_type}")
    channels = channels_by_type[color_type]
    bpp = channels
    stride = width * channels
    raw = zlib.decompress(bytes(compressed))
    rows = np.zeros((height, stride), dtype=np.uint8)
    pos = 0
    prev = np.zeros(stride, dtype=np.uint8)
    for row_idx in range(height):
        filter_type = raw[pos]
        pos += 1
        scan = np.frombuffer(raw[pos : pos + stride], dtype=np.uint8).copy()
        pos += stride
        recon = np.empty_like(scan)
        for i, value in enumerate(scan):
            left = int(recon[i - bpp]) if i >= bpp else 0
            up = int(prev[i])
            up_left = int(prev[i - bpp]) if i >= bpp else 0
            if filter_type == 0:
                recon[i] = value
            elif filter_type == 1:
                recon[i] = (int(value) + left) & 0xFF
            elif filter_type == 2:
                recon[i] = (int(value) + up) & 0xFF
            elif filter_type == 3:
                recon[i] = (int(value) + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                recon[i] = (int(value) + paeth_predictor(left, up, up_left)) & 0xFF
            else:
                raise ValueError(f"unsupported PNG filter type {filter_type}")
        rows[row_idx] = recon
        prev = recon

    arr = rows.reshape(height, width, channels)
    if color_type == 0:
        rgb = np.repeat(arr[:, :, :1], 3, axis=2)
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
    elif color_type == 2:
        rgb = arr
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
    elif color_type == 4:
        rgb = np.repeat(arr[:, :, :1], 3, axis=2)
        alpha = arr[:, :, 1:2]
    else:
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3:4]
    return np.concatenate([rgb, alpha], axis=2)


def slide_image_path(pptx_path: Path, slide_number: int) -> str:
    slide_rel = f"ppt/slides/_rels/slide{slide_number}.xml.rels"
    with ZipFile(pptx_path) as archive:
        rel_xml = archive.read(slide_rel)
    root = ET.fromstring(rel_xml)
    for rel in root.findall("rel:Relationship", PPT_NS):
        target = rel.attrib.get("Target", "")
        rel_type = rel.attrib.get("Type", "")
        if "image" in rel_type and target:
            if target.startswith("../"):
                return "ppt/" + target[3:]
            return str(Path(f"ppt/slides/{target}"))
    raise FileNotFoundError(f"No image relationship found for slide {slide_number}")


def connected_components(mask: np.ndarray) -> list[Component]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[Component] = []
    component_id = 0
    coords = np.argwhere(mask)
    for start_row, start_col in coords:
        if visited[start_row, start_col]:
            continue
        component_id += 1
        queue: deque[tuple[int, int]] = deque([(int(start_row), int(start_col))])
        visited[start_row, start_col] = True
        area = 0
        sum_row = 0
        sum_col = 0
        min_row = max_row = int(start_row)
        min_col = max_col = int(start_col)
        while queue:
            row, col = queue.popleft()
            area += 1
            sum_row += row
            sum_col += col
            min_row = min(min_row, row)
            max_row = max(max_row, row)
            min_col = min(min_col, col)
            max_col = max(max_col, col)
            for nr in range(max(0, row - 1), min(height, row + 2)):
                for nc in range(max(0, col - 1), min(width, col + 2)):
                    if not visited[nr, nc] and mask[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        components.append(
            Component(
                component_id=component_id,
                area=area,
                min_row=min_row,
                max_row=max_row,
                min_col=min_col,
                max_col=max_col,
                centroid_row=sum_row / area,
                centroid_col=sum_col / area,
            )
        )
    return components


def extract_red_points_from_ppt(
    config: dict[str, Any], paths: ExperimentPaths
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ppt_name = (
        get_nested(config, "data.official_pptx") or "AI_wellbore_geology_prediction_task_en.pptx"
    )
    pptx_path = paths.raw_data_dir / str(ppt_name)
    slide_number = int(get_nested(config, "ppt.slide_number") or 10)
    if not pptx_path.exists():
        raise FileNotFoundError(f"Official PPTX not found: {pptx_path}")

    image_path = slide_image_path(pptx_path, slide_number)
    with ZipFile(pptx_path) as archive:
        image_bytes = archive.read(image_path)
    rgba = decode_png_rgba(image_bytes)
    rgb = rgba[:, :, :3].astype(np.int16)
    alpha = rgba[:, :, 3].astype(np.int16)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    max_rgb = rgb.max(axis=2)
    min_rgb = rgb.min(axis=2)

    red_cfg = get_nested(config, "ppt.red_detection") or {}
    red_mask = (
        (alpha > int(red_cfg.get("alpha_min", 10)))
        & (red >= int(red_cfg.get("red_min", 130)))
        & ((red - green) >= int(red_cfg.get("red_green_margin", 45)))
        & ((red - blue) >= int(red_cfg.get("red_blue_margin", 45)))
        & (green <= int(red_cfg.get("green_max", 170)))
        & (blue <= int(red_cfg.get("blue_max", 170)))
    )
    color_mask = (alpha > 10) & ((max_rgb - min_rgb) >= 35) & (max_rgb <= 252)
    color_rows, color_cols = np.where(color_mask)
    if len(color_rows) == 0:
        color_rows, color_cols = np.where(red_mask)
    if len(color_rows) == 0:
        raise RuntimeError("No colored pixels found in PPT slide image")
    plot_min_row = int(color_rows.min())
    plot_max_row = int(color_rows.max())
    plot_min_col = int(color_cols.min())
    plot_max_col = int(color_cols.max())

    min_area = int(red_cfg.get("min_component_area", 4))
    max_area = int(red_cfg.get("max_component_area", 5000))
    max_width = int(red_cfg.get("max_component_width", 320))
    max_height = int(red_cfg.get("max_component_height", 320))
    components = [
        comp
        for comp in connected_components(red_mask)
        if min_area <= comp.area <= max_area
        and comp.width <= max_width
        and comp.height <= max_height
    ]
    rows = []
    for comp in sorted(components, key=lambda item: (item.centroid_col, item.centroid_row)):
        x_norm = (comp.centroid_col - plot_min_col) / max(plot_max_col - plot_min_col, 1)
        y_norm = 1.0 - (comp.centroid_row - plot_min_row) / max(plot_max_row - plot_min_row, 1)
        if -0.05 <= x_norm <= 1.05 and -0.05 <= y_norm <= 1.05:
            rows.append(
                {
                    "target_id": f"red_{len(rows):04d}",
                    "component_id": comp.component_id,
                    "pixel_centroid_col": comp.centroid_col,
                    "pixel_centroid_row": comp.centroid_row,
                    "pixel_area": comp.area,
                    "pixel_width": comp.width,
                    "pixel_height": comp.height,
                    "x_norm": float(np.clip(x_norm, 0.0, 1.0)),
                    "y_norm": float(np.clip(y_norm, 0.0, 1.0)),
                }
            )
    points = pd.DataFrame(rows)
    meta = {
        "pptx_path": str(pptx_path),
        "pptx_sha256": sha256_file(pptx_path),
        "slide_number": slide_number,
        "slide_image_path": image_path,
        "slide_image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "image_width": int(rgba.shape[1]),
        "image_height": int(rgba.shape[0]),
        "red_pixel_count": int(red_mask.sum()),
        "colored_pixel_count": int(color_mask.sum()),
        "red_component_count": len(components),
        "plot_bbox_pixel": {
            "min_row": plot_min_row,
            "max_row": plot_max_row,
            "min_col": plot_min_col,
            "max_col": plot_max_col,
        },
        "status": "ok" if len(points) else "no_red_components_after_filter",
    }
    return points, meta


def fallback_target_points(config: dict[str, Any]) -> pd.DataFrame:
    grid = int(get_nested(config, "audit.fallback_grid_size") or 5)
    rows = []
    for i in range(grid):
        for j in range(grid):
            rows.append(
                {
                    "target_id": f"fallback_{i:02d}_{j:02d}",
                    "component_id": -1,
                    "pixel_centroid_col": np.nan,
                    "pixel_centroid_row": np.nan,
                    "pixel_area": np.nan,
                    "pixel_width": np.nan,
                    "pixel_height": np.nan,
                    "x_norm": (j + 0.5) / grid,
                    "y_norm": (i + 0.5) / grid,
                }
            )
    return pd.DataFrame(rows)


def add_ppt_distances(meta: pd.DataFrame, target_points: pd.DataFrame) -> pd.DataFrame:
    point_xy = target_points[["x_norm", "y_norm"]].to_numpy(dtype=float)
    well_xy = meta[["centroid_x_norm", "centroid_y_norm"]].to_numpy(dtype=float)
    distances = np.sqrt(((well_xy[:, None, :] - point_xy[None, :, :]) ** 2).sum(axis=2))
    nearest_idx = distances.argmin(axis=1)
    out = meta.copy()
    out["ppt_red_distance"] = distances[np.arange(len(out)), nearest_idx]
    out["nearest_target_id"] = target_points.iloc[nearest_idx]["target_id"].to_numpy()
    out["nearest_target_x_norm"] = target_points.iloc[nearest_idx]["x_norm"].to_numpy(dtype=float)
    out["nearest_target_y_norm"] = target_points.iloc[nearest_idx]["y_norm"].to_numpy(dtype=float)
    return out


def greedy_target_balanced_selection(
    meta: pd.DataFrame,
    target_points: pd.DataFrame,
    target_count: int,
    *,
    distinct_typewell_group: bool,
) -> list[str]:
    selected: list[str] = []
    selected_set: set[str] = set()
    selected_groups: set[str] = set()
    point_xy = target_points[["x_norm", "y_norm"]].to_numpy(dtype=float)
    work = meta.reset_index(drop=True).copy()
    well_xy = work[["centroid_x_norm", "centroid_y_norm"]].to_numpy(dtype=float)
    distances = np.sqrt(((well_xy[:, None, :] - point_xy[None, :, :]) ** 2).sum(axis=2))
    order_by_target = [
        np.argsort(distances[:, idx], kind="mergesort") for idx in range(len(target_points))
    ]

    max_passes = max(target_count * 4, len(target_points) * 2)
    for step in range(max_passes):
        if len(selected) >= target_count:
            break
        target_idx = step % len(target_points)
        for row_idx in order_by_target[target_idx]:
            well_id = str(work.iloc[row_idx]["well_id"])
            group_id = str(work.iloc[row_idx]["typewell_group_id"])
            if well_id in selected_set:
                continue
            if distinct_typewell_group and group_id in selected_groups:
                continue
            selected.append(well_id)
            selected_set.add(well_id)
            selected_groups.add(group_id)
            break

    if len(selected) < target_count:
        for _, row in work.sort_values(
            ["ppt_red_distance", "well_id"], kind="mergesort"
        ).iterrows():
            well_id = str(row["well_id"])
            if well_id in selected_set:
                continue
            if distinct_typewell_group and str(row["typewell_group_id"]) in selected_groups:
                continue
            selected.append(well_id)
            selected_set.add(well_id)
            selected_groups.add(str(row["typewell_group_id"]))
            if len(selected) >= target_count:
                break
    return selected


def build_holdout_assignments(
    config: dict[str, Any], meta: pd.DataFrame, target_points: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_count = int(
        get_nested(config, "audit.target_holdout_wells")
        or min(200, max(1, round(len(meta) * 0.25)))
    )
    target_count = min(target_count, max(1, len(meta) - 1))
    spatial_selected = greedy_target_balanced_selection(
        meta, target_points, target_count, distinct_typewell_group=False
    )
    purged_selected = greedy_target_balanced_selection(
        meta, target_points, target_count, distinct_typewell_group=True
    )

    assignment = meta.copy()
    assignment["verification_like_spatial_role"] = np.where(
        assignment["well_id"].isin(spatial_selected), "valid", "train"
    )
    selected_groups = set(
        assignment.loc[assignment["well_id"].isin(purged_selected), "typewell_group_id"].astype(str)
    )
    assignment["verification_like_typewell_purged_role"] = "train"
    assignment.loc[
        assignment["typewell_group_id"].astype(str).isin(selected_groups),
        "verification_like_typewell_purged_role",
    ] = "purged_train_excluded"
    assignment.loc[
        assignment["well_id"].isin(purged_selected), "verification_like_typewell_purged_role"
    ] = "valid"

    holdout_rows = []
    for variant, selected in [
        ("verification_like_spatial", spatial_selected),
        ("verification_like_typewell_purged", purged_selected),
    ]:
        selected_frame = assignment[assignment["well_id"].isin(selected)].copy()
        rank_map = {well_id: rank + 1 for rank, well_id in enumerate(selected)}
        selected_frame["selection_rank"] = selected_frame["well_id"].map(rank_map)
        selected_frame["variant"] = variant
        holdout_rows.append(selected_frame)
    holdout = pd.concat(holdout_rows, ignore_index=True)
    keep = [
        "variant",
        "selection_rank",
        "well_id",
        "typewell_group_id",
        "typewell_group_size",
        "ppt_red_distance",
        "nearest_target_id",
        "centroid_x",
        "centroid_y",
        "centroid_x_norm",
        "centroid_y_norm",
        "signed_azimuth_deg",
        "eval_length",
        "prefix_length",
        "gr_coverage",
        "median_known_tvt",
        "spatial_bin",
        "azimuth_bin",
        "eval_length_bin",
        "prefix_length_bin",
        "gr_bin",
        "tvt_bin",
    ]
    return assignment, holdout[keep].sort_values(["variant", "selection_rank", "well_id"])


def distribution_report(assignment: pd.DataFrame, target_points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fields = [
        "x_bin",
        "y_bin",
        "spatial_bin",
        "azimuth_bin",
        "eval_length_bin",
        "prefix_length_bin",
        "gr_bin",
        "tvt_bin",
    ]
    subsets = {
        "all_train": assignment,
        "verification_like_spatial_valid": assignment[
            assignment["verification_like_spatial_role"] == "valid"
        ],
        "verification_like_typewell_purged_valid": assignment[
            assignment["verification_like_typewell_purged_role"] == "valid"
        ],
        "verification_like_typewell_purged_excluded": assignment[
            assignment["verification_like_typewell_purged_role"] == "purged_train_excluded"
        ],
    }
    for field in fields:
        all_bins = sorted(
            set().union(*(set(frame[field].astype(str)) for frame in subsets.values()))
        )
        for bin_value in all_bins:
            row = {"field": field, "bin": bin_value}
            for subset_name, frame in subsets.items():
                count = int((frame[field].astype(str) == bin_value).sum())
                row[f"{subset_name}_count"] = count
                row[f"{subset_name}_share"] = count / len(frame) if len(frame) else 0.0
            rows.append(row)

    point_frame = target_points.copy()
    point_frame["target_x_bin"] = pd.cut(
        point_frame["x_norm"], bins=np.linspace(0, 1, 5), include_lowest=True
    ).astype(str)
    point_frame["target_y_bin"] = pd.cut(
        point_frame["y_norm"], bins=np.linspace(0, 1, 5), include_lowest=True
    ).astype(str)
    for field in ["target_x_bin", "target_y_bin"]:
        for bin_value, count in point_frame[field].value_counts().sort_index().items():
            rows.append(
                {
                    "field": field,
                    "bin": str(bin_value),
                    "ppt_target_count": int(count),
                    "ppt_target_share": float(count / len(point_frame))
                    if len(point_frame)
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def write_outputs(
    paths: ExperimentPaths,
    config: dict[str, Any],
    target_points: pd.DataFrame,
    ppt_meta: dict[str, Any],
    assignment: pd.DataFrame,
    holdout: pd.DataFrame,
    report: pd.DataFrame,
) -> dict[str, Any]:
    paths.ensure_output_dirs()
    prefix = get_nested(config, "audit.output_prefix") or EXPERIMENT_NAME
    artifacts = {
        "ppt_red_points": paths.artifacts_dir / f"{prefix}_ppt_red_points.csv",
        "well_metadata": paths.artifacts_dir / f"{prefix}_well_metadata.csv",
        "holdout_wells": paths.artifacts_dir / f"{prefix}_holdout_wells.csv",
        "fold_assignments": paths.artifacts_dir / f"{prefix}_fold_assignments.csv",
        "distribution_report": paths.artifacts_dir / f"{prefix}_distribution_report.csv",
        "summary": paths.artifacts_dir / f"{prefix}_summary.json",
    }
    target_points.to_csv(artifacts["ppt_red_points"], index=False)
    assignment.to_csv(artifacts["well_metadata"], index=False)
    holdout.to_csv(artifacts["holdout_wells"], index=False)
    assignment[
        [
            "well_id",
            "verification_like_spatial_role",
            "verification_like_typewell_purged_role",
            "typewell_group_id",
            "ppt_red_distance",
            "nearest_target_id",
        ]
    ].to_csv(artifacts["fold_assignments"], index=False)
    report.to_csv(artifacts["distribution_report"], index=False)

    spatial_valid = assignment[assignment["verification_like_spatial_role"] == "valid"]
    purged_valid = assignment[assignment["verification_like_typewell_purged_role"] == "valid"]
    purged_excluded = assignment[
        assignment["verification_like_typewell_purged_role"] == "purged_train_excluded"
    ]
    summary = {
        "experiment": EXPERIMENT_NAME,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "holdout_created",
        "route": get_nested(config, "experiment.route"),
        "validation_strategy": get_nested(config, "validation.strategy"),
        "target_holdout_wells": int(
            get_nested(config, "audit.target_holdout_wells") or len(spatial_valid)
        ),
        "train_wells": int(len(assignment)),
        "ppt_extraction": ppt_meta,
        "target_points": int(len(target_points)),
        "spatial_holdout": {
            "valid_wells": int(len(spatial_valid)),
            "median_ppt_red_distance": float(spatial_valid["ppt_red_distance"].median()),
            "max_ppt_red_distance": float(spatial_valid["ppt_red_distance"].max()),
            "distinct_typewell_groups": int(spatial_valid["typewell_group_id"].nunique()),
        },
        "typewell_purged_holdout": {
            "valid_wells": int(len(purged_valid)),
            "purged_train_excluded_wells": int(len(purged_excluded)),
            "median_ppt_red_distance": float(purged_valid["ppt_red_distance"].median()),
            "max_ppt_red_distance": float(purged_valid["ppt_red_distance"].max()),
            "distinct_typewell_groups": int(purged_valid["typewell_group_id"].nunique()),
        },
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }
    artifacts["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "holdout_created",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "key_idea": (
            "Build a fixed hidden-like spatial holdout from official PPT "
            "verification map distribution."
        ),
        "summary": summary,
    }
    paths.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return summary


def run_audit(
    config: dict[str, Any] | None = None, paths: ExperimentPaths | None = None
) -> dict[str, Any]:
    config = config or load_config()
    paths = paths or ExperimentPaths()
    paths.ensure_output_dirs()
    meta = build_well_metadata(paths, config)
    try:
        target_points, ppt_meta = extract_red_points_from_ppt(config, paths)
    except Exception as exc:
        if not bool(get_nested(config, "ppt.allow_fallback")):
            raise
        target_points = fallback_target_points(config)
        ppt_meta = {
            "status": "fallback_grid",
            "error": f"{type(exc).__name__}: {exc}",
            "slide_number": int(get_nested(config, "ppt.slide_number") or 10),
        }
    meta = add_ppt_distances(meta, target_points)
    assignment, holdout = build_holdout_assignments(config, meta, target_points)
    report = distribution_report(assignment, target_points)
    return write_outputs(paths, config, target_points, ppt_meta, assignment, holdout, report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hidden-like spatial holdout from official PPT."
    )
    parser.add_argument(
        "--allow-local", action="store_true", help="Compatibility flag for local smoke runs."
    )
    return parser.parse_args()


def main() -> None:
    _args = parse_args()
    paths = ExperimentPaths()
    config = load_config()
    summary = run_audit(config=config, paths=paths)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
