#!/usr/bin/env python3
"""Build a dependency-free, interactive 3D Canvas EDA of all well paths.

The competition's X/Y/Z trajectory coordinates are available only in
``*__horizontal_well.csv``.  The paired typewell files do not contain spatial
coordinates, so this report plots every available horizontal-well trajectory
from both train and test.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


TRAJECTORY_COLUMNS = ["MD", "X", "Y", "Z"]
TVT_COLUMNS = ["TVT", "TVT_input"]
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
FORMATION_COLORS = {
    "ANCC": "#f9c74f",
    "ASTNU": "#b27cff",
    "ASTNL": "#4cc9f0",
    "EGFDU": "#43aa8b",
    "EGFDL": "#577590",
    "BUDA": "#f94144",
}
FORMATION_GRID_SHAPE = (52, 42)  # x, y: compact enough for responsive Canvas rendering


def well_id_from_path(path: Path) -> str:
    return path.name.split("__", 1)[0]


def evenly_sample_indices(count: int, max_points: int) -> np.ndarray:
    """Uniformly decimate a polyline while preserving its first and last point."""
    if count <= max_points:
        return np.arange(count, dtype=np.int64)
    positions = np.linspace(0, count - 1, max_points, dtype=np.int64)
    positions[0] = 0
    positions[-1] = count - 1
    return positions


def build_formation_surfaces(
    records: list[dict],
    coordinate_min: np.ndarray,
    coordinate_max: np.ndarray,
    centre: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    """Interpolate train-only formation contacts onto compact 3D height meshes."""
    nx, ny = FORMATION_GRID_SHAPE
    x_grid = np.linspace(coordinate_min[0], coordinate_max[0], nx)
    y_grid = np.linspace(coordinate_min[1], coordinate_max[1], ny)
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
    query = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    x_span = max(float(coordinate_max[0] - coordinate_min[0]), 1e-9)
    y_span = max(float(coordinate_max[1] - coordinate_min[1]), 1e-9)
    max_gap = 3.0 * float(np.hypot(x_span / (nx - 1), y_span / (ny - 1)))
    surfaces: list[dict] = []
    summaries: list[dict] = []

    for formation in FORMATIONS:
        xyz: list[np.ndarray] = []
        for record in records:
            if record["split"] != "train":
                continue
            values = record["formations"][formation]
            mask = np.isfinite(values)
            if mask.any():
                xyz.append(np.column_stack([record["points"][mask, :2], values[mask]]))
        if not xyz:
            continue

        samples = np.concatenate(xyz, axis=0)
        # Aggregate to the target grid first.  This keeps triangulation stable
        # and avoids feeding hundreds of thousands of nearly collinear samples
        # into Qhull.
        ix = np.clip(
            np.rint((samples[:, 0] - coordinate_min[0]) / x_span * (nx - 1)).astype(int),
            0,
            nx - 1,
        )
        iy = np.clip(
            np.rint((samples[:, 1] - coordinate_min[1]) / y_span * (ny - 1)).astype(int),
            0,
            ny - 1,
        )
        flat = iy * nx + ix
        counts = np.bincount(flat, minlength=nx * ny)
        sums = np.bincount(flat, weights=samples[:, 2], minlength=nx * ny)
        support_mask = counts > 0
        support = query[support_mask]
        support_z = sums[support_mask] / counts[support_mask]
        if len(support) < 3:
            continue

        contact_z = griddata(support, support_z, query, method="linear", fill_value=np.nan)
        nearest_distance, _ = cKDTree(support).query(query, k=1)
        contact_z[nearest_distance > max_gap] = np.nan
        contact_z = contact_z.reshape(ny, nx)
        relative_x = mesh_x.ravel() - centre[0]
        relative_y = mesh_y.ravel() - centre[1]
        relative_z = contact_z.ravel() - centre[2]
        mesh_points: list[float | None] = []
        for x_value, y_value, z_value in zip(relative_x, relative_y, relative_z):
            mesh_points.extend(
                [
                    round(float(x_value), 3),
                    round(float(y_value), 3),
                    round(float(z_value), 3) if np.isfinite(z_value) else None,
                ]
            )
        finite_vertices = int(np.isfinite(contact_z).sum())
        quad_mask = (
            np.isfinite(contact_z[:-1, :-1])
            & np.isfinite(contact_z[:-1, 1:])
            & np.isfinite(contact_z[1:, :-1])
            & np.isfinite(contact_z[1:, 1:])
        )
        surfaces.append(
            {
                "name": formation,
                "color": FORMATION_COLORS[formation],
                "nx": nx,
                "ny": ny,
                "p": mesh_points,
            }
        )
        summaries.append(
            {
                "name": formation,
                "color": FORMATION_COLORS[formation],
                "source_samples": int(len(samples)),
                "support_grid_cells": int(support_mask.sum()),
                "mesh_vertices": finite_vertices,
                "mesh_quads": int(quad_mask.sum()),
                "max_interpolation_gap_m": max_gap,
            }
        )
    return surfaces, summaries


def load_paths(data_root: Path, max_points_per_well: int) -> tuple[list[dict], dict]:
    records: list[dict] = []
    summaries: list[dict] = []

    for split in ("train", "test"):
        paths = sorted((data_root / split).glob("*__horizontal_well.csv"))
        for path in paths:
            # Test CSVs do not necessarily carry the hidden TVT target, whereas
            # train files do.  Read either TVT column if it exists.
            frame = pd.read_csv(
                path,
                usecols=lambda column: column
                in TRAJECTORY_COLUMNS + TVT_COLUMNS + FORMATIONS,
            )
            original_rows = len(frame)
            frame = frame.dropna(subset=TRAJECTORY_COLUMNS).sort_values("MD")
            points = frame[["X", "Y", "Z"]].to_numpy(dtype=np.float64)
            tvt = pd.to_numeric(
                frame.get("TVT", pd.Series(np.nan, index=frame.index)), errors="coerce"
            ).to_numpy(dtype=np.float64)
            tvt_input = pd.to_numeric(
                frame.get("TVT_input", pd.Series(np.nan, index=frame.index)), errors="coerce"
            ).to_numpy(dtype=np.float64)
            # TVT is preferred.  TVT_input fills the visible prefix in test data;
            # unavailable target intervals remain uncoloured in the HTML view.
            tvt = np.where(np.isfinite(tvt), tvt, tvt_input)
            # Consecutive duplicate positions add no visible information.
            if len(points) > 1:
                keep = np.r_[True, np.any(np.diff(points, axis=0) != 0.0, axis=1)]
                frame = frame.iloc[np.flatnonzero(keep)]
                points = points[keep]
                tvt = tvt[keep]
            if len(points) < 2:
                continue

            display_indices = evenly_sample_indices(len(points), max_points_per_well)
            display_points = points[display_indices]
            display_tvt = tvt[display_indices]
            display_formations = {
                formation: pd.to_numeric(
                    frame.get(formation, pd.Series(np.nan, index=frame.index)),
                    errors="coerce",
                ).to_numpy(dtype=np.float64)[display_indices]
                for formation in FORMATIONS
            }
            well_id = well_id_from_path(path)
            records.append(
                {
                    "id": well_id,
                    "split": split,
                    "points": display_points,
                    "tvt": display_tvt,
                    "formations": display_formations,
                }
            )
            summaries.append(
                {
                    "well_id": well_id,
                    "split": split,
                    "source_file": str(path.relative_to(data_root)),
                    "rows_in_file": int(original_rows),
                    "valid_unique_path_points": int(len(points)),
                    "plotted_points": int(len(display_points)),
                    "md_start": float(frame["MD"].iloc[0]),
                    "md_end": float(frame["MD"].iloc[-1]),
                    "x_start": float(points[0, 0]),
                    "y_start": float(points[0, 1]),
                    "z_start": float(points[0, 2]),
                    "x_end": float(points[-1, 0]),
                    "y_end": float(points[-1, 1]),
                    "z_end": float(points[-1, 2]),
                }
            )

    if not records:
        raise FileNotFoundError(f"No usable horizontal well CSV files below {data_root}")

    all_points = np.concatenate([record["points"] for record in records], axis=0)
    all_tvt = np.concatenate([record["tvt"] for record in records])
    finite_tvt = all_tvt[np.isfinite(all_tvt)]
    centre = (all_points.min(axis=0) + all_points.max(axis=0)) / 2.0
    span = all_points.max(axis=0) - all_points.min(axis=0)
    surfaces, formation_summaries = build_formation_surfaces(
        records, all_points.min(axis=0), all_points.max(axis=0), centre
    )
    global_summary = {
        "data_root": str(data_root),
        "well_count": len(records),
        "train_well_count": sum(record["split"] == "train" for record in records),
        "test_well_count": sum(record["split"] == "test" for record in records),
        "source_rows": int(sum(item["rows_in_file"] for item in summaries)),
        "plotted_vertices": int(sum(item["plotted_points"] for item in summaries)),
        "max_points_per_well": max_points_per_well,
        "coordinate_min": {axis: float(value) for axis, value in zip("xyz", all_points.min(axis=0))},
        "coordinate_max": {axis: float(value) for axis, value in zip("xyz", all_points.max(axis=0))},
        "coordinate_centre": {axis: float(value) for axis, value in zip("xyz", centre)},
        "coordinate_span": {axis: float(value) for axis, value in zip("xyz", span)},
        "tvt_color_value_count": int(len(finite_tvt)),
        "tvt_color_low": float(np.quantile(finite_tvt, 0.02)),
        "tvt_color_high": float(np.quantile(finite_tvt, 0.98)),
        "tvt_color_semantics": "TVT; TVT が無い行は TVT_input、両方無い行は未着色",
        "formation_surfaces": formation_summaries,
        "wells": summaries,
    }

    for record in records:
        relative = record.pop("points") - centre
        # Three decimal places retain millimetre precision in this metre-scale dataset
        # while keeping the self-contained HTML responsive to open.
        record["p"] = np.round(relative, 3).ravel().tolist()
        record["t"] = [
            round(float(value), 3) if np.isfinite(value) else None
            for value in record.pop("tvt")
        ]
        formation_values = record.pop("formations")
        record["f"] = {
            formation: [
                round(float(value - centre[2]), 3) if np.isfinite(value) else None
                for value in values
            ]
            for formation, values in formation_values.items()
        }
    return records, surfaces, global_summary


def render_html(records: list[dict], surfaces: list[dict], summary: dict) -> str:
    payload = json.dumps(records, separators=(",", ":"))
    formation_payload = json.dumps(surfaces, separators=(",", ":"))
    title = "ROGII: 全 horizontal well 軌跡 — インタラクティブ 3D EDA"
    summary_text = (
        f"{summary['well_count']} paths ({summary['train_well_count']} train / "
        f"{summary['test_well_count']} test), {summary['plotted_vertices']:,} plotted vertices"
    )
    options = []
    for record in sorted(records, key=lambda item: (item["split"], item["id"])):
        options.append(
            f'<option value="{html.escape(record["split"] + ":" + record["id"])}">'
            f'{html.escape(record["split"])} · {html.escape(record["id"])}</option>'
        )
    select_options = "\n".join(options)
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; overflow: hidden; background: #08111d; color: #e7edf5; }}
  canvas {{ display: block; width: 100vw; height: 100vh; cursor: grab; touch-action: none; }}
  canvas:active {{ cursor: grabbing; }}
  #panel {{ position: fixed; z-index: 2; left: 16px; top: 16px; max-width: min(440px, calc(100vw - 32px));
            padding: 14px 16px; border: 1px solid #3a4b60; border-radius: 10px;
            background: rgb(8 17 29 / 90%); backdrop-filter: blur(8px); box-shadow: 0 8px 24px #0008; }}
  h1 {{ margin: 0 0 5px; font-size: 16px; }}
  #summary, #help {{ margin: 0; color: #b8c5d5; font-size: 12px; line-height: 1.45; }}
  #help {{ margin-top: 9px; }}
  .controls {{ display: grid; grid-template-columns: auto auto 1fr auto auto; align-items: center; gap: 8px; margin-top: 11px; font-size: 12px; }}
  select, button {{ min-width: 0; color: #edf4ff; border: 1px solid #53677c; border-radius: 5px; background: #152536; padding: 5px 7px; }}
  .wide {{ grid-column: 1 / -1; width: 100%; }}
  button {{ cursor: pointer; }} button:hover {{ background: #20384f; }}
  label {{ white-space: nowrap; }}
  #legend {{ position: fixed; z-index: 2; right: 16px; bottom: 16px; padding: 10px 12px; border-radius: 8px;
             background: rgb(8 17 29 / 88%); border: 1px solid #3a4b60; font-size: 12px; line-height: 1.7; }}
  .swatch {{ display:inline-block; width: 12px; height: 3px; vertical-align: middle; margin: 0 6px 2px 0; }}
  .train {{ background:#53b5f5; }} .test {{ background:#ffad4f; }}
  .gradient {{ width: 72px; height: 8px; border-radius: 3px; background: linear-gradient(90deg, #440154, #3b528b, #21918c, #5ec962, #fde725); }}
  #tooltip {{ position: fixed; z-index: 3; pointer-events: none; display: none; padding: 5px 8px; border-radius: 5px;
              color: #fff; font-size: 12px; background: rgb(0 0 0 / 85%); border: 1px solid #71849a; }}
  @media (max-width: 620px) {{ #panel {{ left: 8px; top: 8px; padding: 10px 12px; }} #legend {{ right: 8px; bottom: 8px; }} }}
</style>
</head>
<body>
<canvas id="plot" aria-label="interactive 3D well-path plot"></canvas>
<section id="panel">
  <h1>全 horizontal well の 3D 軌跡</h1>
  <p id="summary">{html.escape(summary_text)}</p>
  <div class="controls">
    <label><input id="train" type="checkbox" checked> train</label>
    <label><input id="test" type="checkbox" checked> test</label>
    <span></span>
    <button id="reset" type="button">視点を戻す</button>
    <button id="top" type="button">上面図</button>
    <select id="color-mode" class="wide"><option value="split">色: train / test</option><option value="tvt">色: TVT（経路内で連続色分け）</option><option value="formations">表示: 地層境界面（train）</option></select>
    <button id="clear-wells" type="button">well 選択解除（全表示）</button>
    <label for="well" class="wide">表示する well（未選択なら全表示。Ctrl/Cmd または Shift で複数選択）</label>
    <select id="well" class="wide" multiple size="7">{select_options}</select>
  </div>
  <p id="help">左ドラッグ: 回転　・　右/中ドラッグ または Shift+左ドラッグ: 横・縦移動　・　ホイール/ピンチ: ズーム<br>地層モードは、未選択時は train の半透明境界面、well 選択時は選択 well 上の境界線だけを表示します。</p>
</section>
<aside id="legend"></aside><div id="tooltip"></div>
<script>
"use strict";
const WELLS = {payload};
const FORMATION_SURFACES = {formation_payload};
const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");
const trainBox = document.getElementById("train"), testBox = document.getElementById("test");
const wellSelect = document.getElementById("well"), colorMode = document.getElementById("color-mode");
const legend = document.getElementById("legend"), tooltip = document.getElementById("tooltip");
const tvtColorLow = {summary['tvt_color_low']:.6f}, tvtColorHigh = {summary['tvt_color_high']:.6f};
const VIRIDIS_STOPS = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
let yaw = -0.72, pitch = -0.48, zoom = 0.82, panX = 0, panY = 0, dirty = true;
let pointer = null, previousDistance = null, lastView = null, hoverPoint = null, hoverFrame = null;
const initial = {{yaw, pitch, zoom, panX, panY}};
const plotExtent = {max(summary['coordinate_span'].values()):.6f};
function makePalette(size) {{
  const out = [];
  for (let index = 0; index < size; index++) {{
    const position = index / Math.max(size - 1, 1) * (VIRIDIS_STOPS.length - 1);
    const left = Math.floor(position), right = Math.min(left + 1, VIRIDIS_STOPS.length - 1), fraction = position - left;
    const rgb = VIRIDIS_STOPS[left].map((value, channel) => Math.round(value + (VIRIDIS_STOPS[right][channel] - value) * fraction));
    out.push("rgb(" + rgb.join(",") + ")");
  }}
  return out;
}}
const TVT_PALETTE = makePalette(28), MISSING_TVT_BIN = TVT_PALETTE.length;
const MIN_ZOOM = 0.08, MAX_ZOOM = 100, WHEEL_ZOOM_SPEED = 0.004;
function resize() {{
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(innerWidth * ratio); canvas.height = Math.round(innerHeight * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0); dirty = true;
}}
function project(x, y, z, cy, sy, cp, sp, scale, cx, cyy) {{
  const rx = x * cy - y * sy, ry = x * sy + y * cy;
  const py = ry * cp - z * sp, depth = ry * sp + z * cp;
  return [cx + rx * scale, cyy - py * scale, depth];
}}
function segmentBin(left, right) {{
  const value = Number.isFinite(right) ? right : left;
  if (!Number.isFinite(value)) return MISSING_TVT_BIN;
  const fraction = Math.max(0, Math.min(1, (value - tvtColorLow) / Math.max(tvtColorHigh - tvtColorLow, 1e-9)));
  return Math.min(TVT_PALETTE.length - 1, Math.floor(fraction * TVT_PALETTE.length));
}}
function selectedWellKeys() {{ return new Set(Array.from(wellSelect.selectedOptions, option => option.value)); }}
function isVisible(well, selected) {{
  if ((well.split === "train" && !trainBox.checked) || (well.split === "test" && !testBox.checked)) return false;
  return selected.size === 0 || selected.has(well.split + ":" + well.id);
}}
function traceWell(well, view, target) {{
  const p = well.p;
  for (let index = 0; index < p.length; index += 3) {{
    const point = project(p[index], p[index + 1], p[index + 2], view.cy, view.sy, view.cp, view.sp, view.scale, view.cx, view.cyy);
    if (index === 0) target.moveTo(point[0], point[1]); else target.lineTo(point[0], point[1]);
  }}
}}
function drawSplitPaths(visible, view, focused) {{
  for (const [, well] of visible) {{
    const path = new Path2D(); traceWell(well, view, path);
    ctx.strokeStyle = well.split === "train" ? "#53b5f5" : "#ffad4f";
    ctx.globalAlpha = focused ? 0.85 : 0.48; ctx.lineWidth = focused ? 1.2 : 0.7; ctx.stroke(path);
  }}
}}
function appendTvtSegments(well, view, paths) {{
  const p = well.p, values = well.t;
  let previous = project(p[0], p[1], p[2], view.cy, view.sy, view.cp, view.sp, view.scale, view.cx, view.cyy);
  for (let pointIndex = 1, offset = 3; offset < p.length; pointIndex++, offset += 3) {{
    const current = project(p[offset], p[offset + 1], p[offset + 2], view.cy, view.sy, view.cp, view.sp, view.scale, view.cx, view.cyy);
    const bin = segmentBin(values[pointIndex - 1], values[pointIndex]);
    paths[bin].moveTo(previous[0], previous[1]); paths[bin].lineTo(current[0], current[1]); previous = current;
  }}
}}
function drawTvtPaths(visible, view) {{
  const paths = Array.from({{length: TVT_PALETTE.length + 1}}, () => new Path2D());
  for (const [, well] of visible) appendTvtSegments(well, view, paths);
  ctx.globalAlpha = 0.62; ctx.lineWidth = 0.78;
  for (let index = 0; index < paths.length; index++) {{ ctx.strokeStyle = index === MISSING_TVT_BIN ? "#7b8796" : TVT_PALETTE[index]; ctx.stroke(paths[index]); }}
}}
function drawFormationSurfaces(view) {{
  const ordered = [];
  for (const surface of FORMATION_SURFACES) {{
    const p = surface.p; let depth = 0, count = 0;
    for (let offset = 0; offset < p.length; offset += 3) {{
      if (!Number.isFinite(p[offset + 2])) continue;
      depth += (p[offset] * view.sy + p[offset + 1] * view.cy) * view.sp + p[offset + 2] * view.cp; count++;
    }}
    if (count) ordered.push([depth / count, surface]);
  }}
  ordered.sort((left, right) => left[0] - right[0]);
  for (const [, surface] of ordered) {{
    const p = surface.p, projected = new Array(surface.nx * surface.ny);
    for (let index = 0, offset = 0; index < projected.length; index++, offset += 3) {{
      projected[index] = Number.isFinite(p[offset + 2]) ? project(p[offset], p[offset + 1], p[offset + 2], view.cy, view.sy, view.cp, view.sp, view.scale, view.cx, view.cyy) : null;
    }}
    const mesh = new Path2D();
    for (let row = 0; row < surface.ny - 1; row++) {{
      for (let col = 0; col < surface.nx - 1; col++) {{
        const a = projected[row * surface.nx + col], b = projected[row * surface.nx + col + 1];
        const c = projected[(row + 1) * surface.nx + col + 1], d = projected[(row + 1) * surface.nx + col];
        if (!a || !b || !c || !d) continue;
        mesh.moveTo(a[0], a[1]); mesh.lineTo(b[0], b[1]); mesh.lineTo(c[0], c[1]); mesh.lineTo(d[0], d[1]); mesh.closePath();
      }}
    }}
    ctx.fillStyle = surface.color; ctx.globalAlpha = 0.13; ctx.fill(mesh);
    ctx.strokeStyle = surface.color; ctx.globalAlpha = 0.42; ctx.lineWidth = 0.38; ctx.stroke(mesh);
  }}
  ctx.globalAlpha = 1;
}}
function drawWellFormationTraces(visible, view) {{
  for (const [, well] of visible) {{
    for (const surface of FORMATION_SURFACES) {{
      const contactZ = well.f ? well.f[surface.name] : null;
      if (!contactZ) continue;
      const p = well.p, trace = new Path2D(); let connected = false;
      for (let pointIndex = 0, offset = 0; offset < p.length; pointIndex++, offset += 3) {{
        if (!Number.isFinite(contactZ[pointIndex])) {{ connected = false; continue; }}
        const point = project(p[offset], p[offset + 1], contactZ[pointIndex], view.cy, view.sy, view.cp, view.sp, view.scale, view.cx, view.cyy);
        if (connected) trace.lineTo(point[0], point[1]); else trace.moveTo(point[0], point[1]);
        connected = true;
      }}
      ctx.strokeStyle = surface.color; ctx.globalAlpha = 0.98; ctx.lineWidth = 2.1; ctx.stroke(trace);
    }}
  }}
  ctx.globalAlpha = 1;
}}
function draw() {{
  if (!dirty) {{ requestAnimationFrame(draw); return; }}
  dirty = false;
  const width = innerWidth, height = innerHeight;
  ctx.clearRect(0, 0, width, height);
  const gradient = ctx.createLinearGradient(0, 0, 0, height); gradient.addColorStop(0, "#0b1d31"); gradient.addColorStop(1, "#060b13");
  ctx.fillStyle = gradient; ctx.fillRect(0, 0, width, height);
  const view = {{cy: Math.cos(yaw), sy: Math.sin(yaw), cp: Math.cos(pitch), sp: Math.sin(pitch), scale: Math.min(width, height) * zoom / plotExtent, cx: width / 2 + panX, cyy: height / 2 + panY}};
  const selection = selectedWellKeys(), visible = [];
  for (const well of WELLS) {{
    if (!isVisible(well, selection)) continue;
    const p = well.p; let meanDepth = 0;
    for (let index = 0; index < p.length; index += 3) meanDepth += (p[index] * view.sy + p[index + 1] * view.cy) * view.sp + p[index + 2] * view.cp;
    visible.push([meanDepth / (p.length / 3), well]);
  }}
  visible.sort((left, right) => left[0] - right[0]); lastView = {{...view, wells: visible.map(entry => entry[1])}};
  if (colorMode.value === "formations") {{
    if (selection.size > 0) drawWellFormationTraces(visible, view); else drawFormationSurfaces(view);
    drawSplitPaths(visible, view, selection.size > 0);
  }}
  else if (colorMode.value === "tvt") drawTvtPaths(visible, view); else drawSplitPaths(visible, view, selection.size > 0);
  ctx.globalAlpha = 1;
  const ox = 58, oy = height - 58, axis = 30;
  ctx.strokeStyle = "#c6d4e2"; ctx.fillStyle = "#c6d4e2"; ctx.lineWidth = 1.2; ctx.font = "11px system-ui";
  for (const [name, x, y, z] of [["X", axis, 0, 0], ["Y", 0, axis, 0], ["Z", 0, 0, axis]]) {{
    const point = project(x, y, z, view.cy, view.sy, view.cp, view.sp, 1, ox, oy); ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(point[0], point[1]); ctx.stroke(); ctx.fillText(name, point[0] + 3, point[1] + 3);
  }}
  requestAnimationFrame(draw);
}}
function distanceToSegmentSquared(px, py, ax, ay, bx, by) {{
  const dx = bx - ax, dy = by - ay, lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) return (px - ax) ** 2 + (py - ay) ** 2;
  const fraction = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lengthSquared));
  return (px - (ax + fraction * dx)) ** 2 + (py - (ay + fraction * dy)) ** 2;
}}
function hideTooltip() {{ tooltip.style.display = "none"; }}
function resolveHover() {{
  hoverFrame = null; if (!hoverPoint || !lastView) return;
  const view = lastView, px = hoverPoint.x, py = hoverPoint.y; let nearest = null, nearestDistance = 100;
  for (const well of view.wells) {{
    const p = well.p; let previous = project(p[0], p[1], p[2], view.cy, view.sy, view.cp, view.sp, view.scale, view.cx, view.cyy);
    for (let offset = 3; offset < p.length; offset += 3) {{
      const current = project(p[offset], p[offset + 1], p[offset + 2], view.cy, view.sy, view.cp, view.sp, view.scale, view.cx, view.cyy);
      const distance = distanceToSegmentSquared(px, py, previous[0], previous[1], current[0], current[1]);
      if (distance < nearestDistance) {{ nearestDistance = distance; nearest = well; }}
      previous = current;
    }}
  }}
  if (!nearest) {{ hideTooltip(); return; }}
  tooltip.textContent = nearest.split + " · " + nearest.id;
  tooltip.style.left = Math.min(px + 13, innerWidth - 150) + "px"; tooltip.style.top = Math.min(py + 13, innerHeight - 32) + "px"; tooltip.style.display = "block";
}}
function queueHover(event) {{
  if (pointer) return; hoverPoint = {{x: event.clientX, y: event.clientY}};
  if (hoverFrame === null) hoverFrame = requestAnimationFrame(resolveHover);
}}
function updateLegend() {{
  if (colorMode.value === "formations") {{
    const detail = selectedWellKeys().size > 0 ? "選択 well の地層境界線" : "train 境界の補間面";
    legend.innerHTML = FORMATION_SURFACES.map(surface => "<span class='swatch' style='background:" + surface.color + "'></span>" + surface.name).join("<br>") + "<br><span style='color:#b8c5d5'>" + detail + "</span>";
  }}
  else if (colorMode.value === "tvt") legend.innerHTML = "<span class='swatch gradient'></span>TVT: " + tvtColorLow.toFixed(1) + " – " + tvtColorHigh.toFixed(1) + "<br><span class='swatch' style='background:#7b8796'></span>TVT / TVT_input 不明";
  else legend.innerHTML = "<span class='swatch train'></span>train horizontal well<br><span class='swatch test'></span>test horizontal well";
}}
canvas.addEventListener("pointerdown", event => {{ hideTooltip(); pointer = {{id: event.pointerId, x: event.clientX, y: event.clientY, mode: (event.button !== 0 || event.shiftKey) ? "pan" : "rotate"}}; canvas.setPointerCapture(event.pointerId); }});
canvas.addEventListener("pointermove", event => {{
  if (!pointer || pointer.id !== event.pointerId) {{ queueHover(event); return; }}
  const dx = event.clientX - pointer.x, dy = event.clientY - pointer.y;
  if (pointer.mode === "pan") {{ panX += dx; panY += dy; }} else {{ yaw += dx * 0.008; pitch = Math.max(-1.52, Math.min(1.52, pitch + dy * 0.008)); }}
  pointer = {{...pointer, x: event.clientX, y: event.clientY}}; dirty = true;
}});
canvas.addEventListener("pointerup", event => {{ if (pointer && pointer.id === event.pointerId) {{ pointer = null; queueHover(event); }} }});
canvas.addEventListener("pointercancel", () => {{ pointer = null; }}); canvas.addEventListener("pointerleave", hideTooltip); canvas.addEventListener("contextmenu", event => event.preventDefault());
canvas.addEventListener("wheel", event => {{ event.preventDefault(); zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * Math.exp(-event.deltaY * WHEEL_ZOOM_SPEED))); dirty = true; }}, {{passive: false}});
canvas.addEventListener("touchmove", event => {{ if (event.touches.length !== 2) {{ previousDistance = null; return; }} const [a, b] = event.touches; const distance = Math.hypot(a.clientX-b.clientX, a.clientY-b.clientY); if (previousDistance) {{ zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * distance / previousDistance)); dirty = true; }} previousDistance = distance; }}, {{passive: true}});
canvas.addEventListener("touchend", () => {{ previousDistance = null; }});
for (const element of [trainBox, testBox]) element.addEventListener("change", () => {{ dirty = true; }});
wellSelect.addEventListener("change", () => {{ updateLegend(); dirty = true; }});
colorMode.addEventListener("change", () => {{ updateLegend(); dirty = true; }});
document.getElementById("clear-wells").addEventListener("click", () => {{ for (const option of wellSelect.options) option.selected = false; updateLegend(); dirty = true; }});
document.getElementById("reset").addEventListener("click", () => {{ ({{yaw, pitch, zoom, panX, panY}} = initial); dirty = true; }});
document.getElementById("top").addEventListener("click", () => {{ yaw = 0; pitch = -1.56; zoom = 0.82; panX = 0; panY = 0; dirty = true; }});
window.addEventListener("resize", resize); updateLegend(); resize(); draw();
</script>
</body>
</html>"""


def write_readme(output_dir: Path, summary: dict) -> None:
    coordinate_span = summary["coordinate_span"]
    formation_details = ", ".join(
        f"{surface['name']}: {surface['mesh_quads']:,} quads"
        for surface in summary["formation_surfaces"]
    )
    text = f"""# 全 horizontal well 軌跡のインタラクティブ 3D EDA

`all_horizontal_well_paths_3d.html` をブラウザで開くと、全 {summary['well_count']} 本（train {summary['train_well_count']} 本、test {summary['test_well_count']} 本）の座標軌跡を回転・ズームできます。

- 通常モードでは train は青、test は橙です。
- TVT モードでは各経路を TVT（無い箇所は `TVT_input`）で連続的に色分けし、両方無い区間は灰色です。
- 地層境界面モードでは train horizontal well の `ANCC`、`ASTNU`、`ASTNL`、`EGFDU`、`EGFDL`、`BUDA` を色分けした半透明 3D メッシュとして重ねます。well を選択した場合は全体メッシュを隠し、選択 well 上の対応する地層境界線だけを表示します。test にはこれらの列がないため、test well を選択した場合は地層境界線を表示できません。
- 左ドラッグで回転、右/中ドラッグまたは Shift+左ドラッグで画面を横・縦に移動できます。ホイールまたはピンチで最大 100 倍までズームできます。
- 軌跡にマウスを重ねると、最も近い well の split と well ID を表示します。
- well リストは複数選択でき、選択した well だけを表示します。未選択なら全 well を表示し、「well 選択解除（全表示）」で戻せます。
- typewell CSV には X/Y/Z 座標がないため、描画対象は対応する horizontal well のみです。
- 各経路は端点を保った一様間引き（最大 {summary['max_points_per_well']} 点）で描画しています。元の全点数は `path_summary.json` にあります。

## 実行結果

- 入力行数: {summary['source_rows']:,}
- 描画頂点数: {summary['plotted_vertices']:,}
- X span: {coordinate_span['x']:.2f} m
- Y span: {coordinate_span['y']:.2f} m
- Z span: {coordinate_span['z']:.2f} m
- TVT 色範囲（全描画頂点の 2–98 percentile）: {summary['tvt_color_low']:.2f} – {summary['tvt_color_high']:.2f}
- TVT 色に使える頂点数: {summary['tvt_color_value_count']:,}
- 地層境界メッシュ: {formation_details}

生成: `studies/plot_all_well_paths_3d.py`
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("studies/well_path_3d_eda_20260711")
    )
    parser.add_argument("--max-points-per-well", type=int, default=250)
    args = parser.parse_args()
    if args.max_points_per_well < 2:
        raise ValueError("--max-points-per-well must be at least 2")

    records, surfaces, summary = load_paths(args.data_root, args.max_points_per_well)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.output_dir / "all_horizontal_well_paths_3d.html"
    html_path.write_text(render_html(records, surfaces, summary), encoding="utf-8")
    (args.output_dir / "path_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(args.output_dir, summary)
    print(f"Wrote interactive plot: {html_path}")
    print(json.dumps({key: value for key, value in summary.items() if key != "wells"}, indent=2))


if __name__ == "__main__":
    main()
