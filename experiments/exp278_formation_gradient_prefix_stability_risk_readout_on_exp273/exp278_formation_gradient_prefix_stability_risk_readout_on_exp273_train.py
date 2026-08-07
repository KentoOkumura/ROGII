# %% [markdown]
# # exp278 formation gradient prefix stability risk readout on exp273 — train
#
# exp273 の保存済み 2D-gradient candidate を再生成せず、known prefix の window 間で
# formation plane が不安定な well ほど candidate family が悪化するかを監査する。

# %% [markdown]
# ## Contents
# 1. Imports and immutable configuration
# 2. Compute, leakage, and decision contract
# 3. Input resolution, SHA, and fixed exp273 outcome helpers
# 4. Deterministic Huber plane and full-prefix parity helpers
# 5. Prefix-stability feature construction
# 6. Outcome attachment and five-fold readout helpers
# 7. Setup and fixed input checks
# 8. Recompute prefix diagnostics and freeze target-free risk
# 9. Attach outcomes, evaluate guards, and save generated artifacts

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from IPython import get_ipython
from IPython.display import display

EXPERIMENT_NAME = "exp278_formation_gradient_prefix_stability_risk_readout_on_exp273"
OUTPUT_PREFIX = "exp278"
PACKAGE_DIR = Path.cwd()
CONFIG_CANDIDATES = [
    PACKAGE_DIR / "config.yaml",
    PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
]
CONFIG_PATH = next((path for path in CONFIG_CANDIDATES if path.is_file()), None)
if CONFIG_PATH is None:
    raise FileNotFoundError(f"Could not locate config.yaml; checked={CONFIG_CANDIDATES}")
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
EXECUTE_NOTEBOOK = get_ipython() is not None


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_decompressed_gzip(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_frame_sha256(frame: pd.DataFrame, sort_columns: Sequence[str]) -> str:
    ordered = frame.sort_values(list(sort_columns), kind="stable").reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(ordered.columns), separators=(",", ":")).encode())
    digest.update(json.dumps([str(dtype) for dtype in ordered.dtypes]).encode())
    hashed = pd.util.hash_pandas_object(ordered, index=False, categorize=True).to_numpy(np.uint64)
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        to_jsonable(dict(payload)), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


# %% [markdown]
# ## 2. Compute, leakage, and decision contract
#
# 本 notebook は 0 variant / 0 config / 0 trained fold / 0 booster。fold は学習用ではなく、
# stable well hash で固定した readout stratum である。outcome は stability feature
# 凍結後にだけ接続する。

# %%
if EXECUTE_NOTEBOOK:
    execution = CONFIG["execution"]
    compute_contract = {
        "experiment": CONFIG["experiment"]["name"],
        "route": CONFIG["experiment"]["route"],
        "stage": execution["stage"],
        "variants": int(execution["variants"]),
        "lightgbm_configs": int(execution["lightgbm_configs"]),
        "folds_trained": int(execution["folds_trained"]),
        "total_boosters": int(execution["total_boosters"]),
        "parent_control_retraining": bool(execution["parent_control_retraining"]),
        "hmm_paths_generated": int(execution["hmm_paths_generated"]),
        "gpu": bool(CONFIG["runtime"]["kaggle"]["enable_gpu"]),
        "internet": bool(CONFIG["runtime"]["kaggle"]["enable_internet"]),
        "inference": bool(execution["inference_enabled"]),
        "submission": bool(execution["submission_enabled"]),
    }
    display(compute_contract)
    assert compute_contract == {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "stage": "formation_gradient_prefix_stability_risk_readout",
        "variants": 0,
        "lightgbm_configs": 0,
        "folds_trained": 0,
        "total_boosters": 0,
        "parent_control_retraining": False,
        "hmm_paths_generated": 0,
        "gpu": False,
        "internet": False,
        "inference": False,
        "submission": False,
    }
    assert CONFIG["risk"]["components"] == [
        "gradient_angle_disagreement",
        "gradient_magnitude_log_ratio",
        "plane_rmse_log_ratio",
        "rank_ratio_absolute_gap",
        "condition_number_log_ratio",
        "validity_flip",
    ]
    assert CONFIG["risk"]["fit_from_outcome"] is False
    assert CONFIG["guards"]["primary"]["secondary_can_rescue"] is False
    assert CONFIG["guards"]["primary"]["candidate_specific_can_rescue"] is False
    if not Path("/kaggle/input").exists() or not Path("/kaggle/working").exists():
        raise RuntimeError("The first full exp278 readout must run on Kaggle CPU.")
    if not bool(execution["run_approved"]):
        raise RuntimeError(
            "Kaggle CPU readout is not approved. Set execution.run_approved=true only "
            "for the approved canonical push."
        )
    print("Leakage contract")
    for rule in CONFIG["validation"]["leakage_policy"]:
        print("-", rule)


# %% [markdown]
# ## 3. Input resolution, SHA, and fixed exp273 outcome helpers
#
# aggregate CSV は byte SHA、shard gzip は raw/decompressed SHAで fail-closed にする。
# shardからは candidate prediction と true TVTを outcome parity にだけ使い、risk featureへ渡さない。


# %%
def resolve_fixed_file(spec: Mapping[str, Any], expected_key: str = "expected_sha256") -> Path:
    expected = str(spec[expected_key])
    candidates: list[Path] = []
    for value in spec["candidates"]:
        path = Path(str(value))
        if path.is_file():
            candidates.append(path)
    filename = str(spec["filename"])
    for root in (Path("/kaggle/input"), Path("/tmp"), PACKAGE_DIR):
        if root.exists():
            candidates.extend(root.rglob(filename))
    checked: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if key in seen or not path.is_file() or path.stat().st_size == 0:
            continue
        seen.add(key)
        actual = sha256_file(path)
        checked.append({"path": str(path), "sha256": actual})
        if actual == expected:
            return path
    raise FileNotFoundError(
        f"No {filename} matched {expected_key}={expected}; checked={checked[:30]}"
    )


def resolve_raw_train_dir(patterns: Sequence[str]) -> Path:
    for value in patterns:
        path = Path(str(value))
        if path.is_dir() and next(path.glob("*__horizontal_well.csv"), None) is not None:
            return path
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for path in sorted(kaggle_input.rglob("train")):
            if path.is_dir() and next(path.glob("*__horizontal_well.csv"), None) is not None:
                return path
    raise FileNotFoundError("Could not resolve raw competition train directory")


def stable_outer_fold(well: str, n_folds: int) -> int:
    payload = f"exp278::outer_fold::{well}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
    return int(value % int(n_folds))


def stream_shard_candidate_rmse(
    path: Path,
    candidates: Sequence[str],
    *,
    chunk_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    usecols = ["well", "true_tvt", *candidates]
    accumulators: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    total_rows = 0
    wells: set[str] = set()
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunk_rows)):
        chunk["well"] = chunk["well"].astype(str)
        truth = pd.to_numeric(chunk["true_tvt"], errors="raise").to_numpy(np.float64)
        if not np.isfinite(truth).all():
            raise ValueError(f"non-finite true_tvt in {path}")
        total_rows += len(chunk)
        wells.update(chunk["well"].unique())
        for candidate in candidates:
            prediction = pd.to_numeric(chunk[candidate], errors="raise").to_numpy(np.float64)
            if not np.isfinite(prediction).all():
                raise ValueError(f"non-finite {candidate} in {path}")
            partial = (
                pd.DataFrame(
                    {
                        "well": chunk["well"].to_numpy(),
                        "squared_error": np.square(truth - prediction),
                    }
                )
                .groupby("well", sort=False)["squared_error"]
                .agg(["count", "sum"])
            )
            for well, row in partial.iterrows():
                state = accumulators[(str(well), str(candidate))]
                state[0] += float(row["count"])
                state[1] += float(row["sum"])
    rows = [
        {
            "well": well,
            "candidate": candidate,
            "rows": int(count),
            "recomputed_rmse": float(math.sqrt(squared_error / count)),
        }
        for (well, candidate), (count, squared_error) in accumulators.items()
    ]
    result = pd.DataFrame(rows).sort_values(["candidate", "well"], kind="stable")
    summary = {
        "path": str(path),
        "rows": int(total_rows),
        "wells": int(len(wells)),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": sha256_decompressed_gzip(path),
    }
    return result.reset_index(drop=True), summary


def build_candidate_outcomes(
    by_well: pd.DataFrame,
    recomputed: pd.DataFrame,
    candidates: Sequence[str],
    *,
    parity_atol: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "candidate",
        "well",
        "rows",
        "rmse",
        "scalar_rmse",
        "delta_rmse_vs_scalar",
    }
    missing = sorted(required.difference(by_well.columns))
    if missing:
        raise ValueError(f"exp273 by-well metrics missing columns: {missing}")
    selected = by_well[by_well["candidate"].isin(candidates)].copy()
    selected["well"] = selected["well"].astype(str)
    if len(selected) != len(candidates) * selected["well"].nunique():
        raise ValueError("exp273 gradient by-well metrics are not a complete candidate grid")
    parity = selected.merge(
        recomputed,
        on=["well", "candidate"],
        how="outer",
        validate="one_to_one",
        suffixes=("_saved", "_shard"),
        indicator=True,
    )
    if not parity["_merge"].eq("both").all():
        raise ValueError("shard/by-well candidate grid mismatch")
    if not np.array_equal(
        pd.to_numeric(parity["rows_saved"], errors="raise").to_numpy(np.int64),
        pd.to_numeric(parity["rows_shard"], errors="raise").to_numpy(np.int64),
    ):
        raise ValueError("shard/by-well row-count mismatch")
    parity["rmse_abs_diff"] = np.abs(
        pd.to_numeric(parity["rmse"], errors="raise").to_numpy(np.float64)
        - pd.to_numeric(parity["recomputed_rmse"], errors="raise").to_numpy(np.float64)
    )
    parity["parity_pass"] = parity["rmse_abs_diff"] <= float(parity_atol)
    if not parity["parity_pass"].all():
        worst = parity.nlargest(10, "rmse_abs_diff")[
            ["well", "candidate", "rmse", "recomputed_rmse", "rmse_abs_diff"]
        ]
        raise ValueError(f"candidate RMSE parity failed:\n{worst}")
    selected = selected.sort_values(["well", "candidate"], kind="stable")
    grouped = selected.groupby("well", sort=True)
    outcomes = grouped.agg(
        eval_rows=("rows", "first"),
        scalar_rmse=("scalar_rmse", "first"),
        gradient_bank_mean_delta_rmse_vs_scalar=("delta_rmse_vs_scalar", "mean"),
        gradient_bank_max_delta_rmse_vs_scalar=("delta_rmse_vs_scalar", "max"),
        gradient_bank_min_delta_rmse_vs_scalar=("delta_rmse_vs_scalar", "min"),
        regressed_candidate_count=("delta_rmse_vs_scalar", lambda values: int((values > 0).sum())),
    ).reset_index()
    scalar_spread = grouped["scalar_rmse"].agg(lambda values: float(values.max() - values.min()))
    if float(scalar_spread.max()) > 1.0e-12:
        raise ValueError("scalar RMSE differs across candidate rows for a well")
    candidate_wide = selected.pivot(
        index="well", columns="candidate", values="delta_rmse_vs_scalar"
    ).rename(columns=lambda value: f"delta_rmse_{value}")
    outcomes = outcomes.merge(
        candidate_wide.reset_index(), on="well", how="left", validate="one_to_one"
    )
    parity_columns = [
        "well",
        "candidate",
        "rows_saved",
        "rows_shard",
        "rmse",
        "recomputed_rmse",
        "rmse_abs_diff",
        "parity_pass",
    ]
    return (
        outcomes,
        selected,
        parity[parity_columns].sort_values(["candidate", "well"], kind="stable"),
    )


# %% [markdown]
# ## 4. Deterministic Huber plane and full-prefix parity helpers
#
# exp273 と同じ centered XY、SVD geometry、Huber IRLSを使う。full-prefix 再計算は
# 保存済みdiagnosticsとの hard parity guardで、window readoutの実装 driftを検出する。


# %%
def axial_azimuth_coverage(x: np.ndarray, y: np.ndarray, min_step: float) -> tuple[float, int]:
    dx = np.diff(np.asarray(x, dtype=np.float64))
    dy = np.diff(np.asarray(y, dtype=np.float64))
    step = np.hypot(dx, dy)
    valid = np.isfinite(dx) & np.isfinite(dy) & (step > float(min_step))
    if not valid.any():
        return 0.0, 0
    theta = np.arctan2(dy[valid], dx[valid])
    resultant = np.hypot(np.mean(np.cos(2.0 * theta)), np.mean(np.sin(2.0 * theta)))
    return float(np.clip(1.0 - resultant, 0.0, 1.0)), int(valid.sum())


def fit_formation_plane(
    known_prefix: pd.DataFrame,
    plane_config: Mapping[str, Any],
) -> dict[str, Any]:
    required = ["X", "Y", "Z", "TVT_input"]
    numeric = {
        column: pd.to_numeric(known_prefix[column], errors="coerce").to_numpy(np.float64)
        for column in required
    }
    finite = np.ones(len(known_prefix), dtype=bool)
    for values in numeric.values():
        finite &= np.isfinite(values)
    x = numeric["X"][finite]
    y = numeric["Y"][finite]
    surface = (numeric["TVT_input"] + numeric["Z"])[finite]
    n_points = int(len(surface))
    min_points = int(plane_config["min_points"])
    min_xy_step = float(plane_config["min_xy_step"])
    min_rank_ratio = float(plane_config["min_rank_ratio"])
    max_condition = float(plane_config["max_condition_number"])
    min_azimuth = float(plane_config["min_azimuth_coverage"])
    reasons: list[str] = []

    x0 = float(np.median(x)) if n_points else 0.0
    y0 = float(np.median(y)) if n_points else 0.0
    s0 = float(np.median(surface)) if n_points else 0.0
    dx = x - x0
    dy = y - y0
    geometry = np.column_stack([dx, dy]) if n_points else np.empty((0, 2))
    singular = np.linalg.svd(geometry, compute_uv=False) if n_points else np.array([])
    singular_1 = float(singular[0]) if len(singular) >= 1 else 0.0
    singular_2 = float(singular[1]) if len(singular) >= 2 else 0.0
    rank_ratio = singular_2 / singular_1 if singular_1 > 0.0 else 0.0
    condition_number = singular_1 / singular_2 if singular_2 > 0.0 else float("inf")
    azimuth_coverage, azimuth_steps = axial_azimuth_coverage(x, y, min_xy_step)

    if n_points < min_points:
        reasons.append("too_few_points")
    if singular_1 <= min_xy_step or singular_2 <= 0.0:
        reasons.append("xy_rank_below_2")
    if rank_ratio < min_rank_ratio:
        reasons.append("rank_ratio_below_guard")
    if condition_number > max_condition:
        reasons.append("condition_number_above_guard")
    if azimuth_coverage < min_azimuth:
        reasons.append("azimuth_coverage_below_guard")

    diagnostic_gradient = np.zeros(2, dtype=np.float64)
    plane_rmse = float("nan")
    robust_scale = float("nan")
    iterations = 0
    diagnostic_fit_valid = bool(
        n_points >= min_points
        and singular_1 > min_xy_step
        and singular_2 > 0.0
        and np.isfinite(surface).all()
    )
    if diagnostic_fit_valid:
        design = np.column_stack([dx, dy, np.ones(n_points, dtype=np.float64)])
        target = surface - s0
        beta = np.linalg.lstsq(design, target, rcond=None)[0]
        weights = np.ones(n_points, dtype=np.float64)
        huber_delta = float(plane_config["huber_delta"])
        max_iterations = int(plane_config["max_iterations"])
        tolerance = float(plane_config["convergence_tolerance"])
        for iteration in range(1, max_iterations + 1):
            residual = target - design @ beta
            residual_center = float(np.median(residual))
            robust_scale = max(
                1.4826 * float(np.median(np.abs(residual - residual_center))), 1.0e-9
            )
            cutoff = huber_delta * robust_scale
            absolute = np.abs(residual - residual_center)
            weights = np.ones(n_points, dtype=np.float64)
            outside = absolute > cutoff
            weights[outside] = cutoff / absolute[outside]
            root_weight = np.sqrt(weights)
            beta_next = np.linalg.lstsq(
                design * root_weight[:, None], target * root_weight, rcond=None
            )[0]
            iterations = iteration
            delta = float(np.max(np.abs(beta_next - beta)))
            beta = beta_next
            if delta <= tolerance * (1.0 + float(np.max(np.abs(beta)))):
                break
        residual = target - design @ beta
        plane_rmse = float(np.sqrt(np.mean(residual**2)))
        diagnostic_gradient = np.asarray(beta[:2], dtype=np.float64)
        if not np.isfinite(diagnostic_gradient).all() or not np.isfinite(plane_rmse):
            diagnostic_fit_valid = False
            reasons.append("non_finite_plane_fit")

    valid = len(reasons) == 0 and diagnostic_fit_valid
    generation_gradient = diagnostic_gradient.copy() if valid else np.zeros(2, dtype=np.float64)
    return {
        "valid": bool(valid),
        "diagnostic_fit_valid": bool(diagnostic_fit_valid),
        "fallback_reason": "ok" if valid else ";".join(dict.fromkeys(reasons)),
        "n_points": n_points,
        "x_center": x0,
        "y_center": y0,
        "surface_center": s0,
        "singular_value_1": singular_1,
        "singular_value_2": singular_2,
        "rank_ratio": float(rank_ratio),
        "condition_number": float(condition_number),
        "azimuth_coverage": float(azimuth_coverage),
        "azimuth_valid_steps": int(azimuth_steps),
        "gradient_x": float(diagnostic_gradient[0]),
        "gradient_y": float(diagnostic_gradient[1]),
        "gradient_magnitude": float(np.linalg.norm(diagnostic_gradient)),
        "generation_gradient_x": float(generation_gradient[0]),
        "generation_gradient_y": float(generation_gradient[1]),
        "generation_plane_rmse": float(plane_rmse) if valid else float("nan"),
        "plane_rmse": float(plane_rmse),
        "robust_scale": float(robust_scale),
        "iterations": int(iterations),
    }


def recompute_prefix_plane_diagnostics(
    raw_train_dir: Path,
    wells: Sequence[str],
    plane_config: Mapping[str, Any],
    windows: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostic_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for index, well in enumerate(sorted(str(value) for value in wells)):
        path = raw_train_dir / f"{well}__horizontal_well.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        horizontal = pd.read_csv(path, usecols=["MD", "X", "Y", "Z", "TVT_input"])
        known = horizontal[horizontal["TVT_input"].notna()].copy()
        if known.empty:
            raise ValueError(f"well={well} has no known prefix")
        manifest_rows.append(
            {
                "source": "raw_horizontal",
                "well": well,
                "path": str(path),
                "file_sha256": sha256_file(path),
                "rows": int(len(horizontal)),
                "known_prefix_rows": int(len(known)),
            }
        )
        for window in windows:
            name = str(window["name"])
            rows = window.get("rows")
            window_frame = known if rows is None else known.tail(int(rows))
            plane = fit_formation_plane(window_frame, plane_config)
            condition = float(plane["condition_number"])
            if not np.isfinite(condition):
                condition = float(np.finfo(np.float32).max)
            diagnostic_rows.append(
                {
                    "well": well,
                    "window": name,
                    "prefix_rows_total": int(len(known)),
                    "window_rows": int(len(window_frame)),
                    "gradient_valid": int(plane["valid"]),
                    "diagnostic_fit_valid": int(plane["diagnostic_fit_valid"]),
                    "gradient_fallback_reason": str(plane["fallback_reason"]),
                    "xy_rank_ratio": float(plane["rank_ratio"]),
                    "xy_condition_number": condition,
                    "azimuth_coverage": float(plane["azimuth_coverage"]),
                    "plane_rmse": (
                        float(plane["plane_rmse"]) if np.isfinite(plane["plane_rmse"]) else -1.0
                    ),
                    "generation_plane_rmse": (
                        float(plane["generation_plane_rmse"])
                        if np.isfinite(plane["generation_plane_rmse"])
                        else -1.0
                    ),
                    "gradient_x": float(plane["gradient_x"]),
                    "gradient_y": float(plane["gradient_y"]),
                    "gradient_magnitude": float(plane["gradient_magnitude"]),
                    "generation_gradient_x": float(plane["generation_gradient_x"]),
                    "generation_gradient_y": float(plane["generation_gradient_y"]),
                    "robust_scale": (
                        float(plane["robust_scale"]) if np.isfinite(plane["robust_scale"]) else -1.0
                    ),
                    "iterations": int(plane["iterations"]),
                }
            )
        if (index + 1) % 50 == 0 or index + 1 == len(wells):
            print(f"prefix plane diagnostics: {index + 1}/{len(wells)} wells")
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(["well", "window"], kind="stable")
    manifest = pd.DataFrame(manifest_rows).sort_values(["well"], kind="stable")
    return diagnostics.reset_index(drop=True), manifest.reset_index(drop=True)


def verify_full_plane_parity(
    diagnostics: pd.DataFrame,
    saved: pd.DataFrame,
    *,
    atol: float,
    rtol: float,
) -> pd.DataFrame:
    full = diagnostics[diagnostics["window"].eq("full")].copy()
    saved = saved.copy()
    saved["well"] = saved["well"].astype(str)
    required = {
        "well",
        "prefix_rows",
        "gradient_valid",
        "gradient_fallback_reason",
        "xy_rank_ratio",
        "xy_condition_number",
        "azimuth_coverage",
        "plane_rmse",
        "gradient_x_center",
        "gradient_y_center",
    }
    missing = sorted(required.difference(saved.columns))
    if missing:
        raise ValueError(f"saved plane diagnostics missing columns: {missing}")
    merged = full.merge(
        saved[list(required)],
        on="well",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(merged) != len(full) or not merged["_merge"].eq("both").all():
        raise ValueError("full plane/saved diagnostics well mismatch")
    exact_pairs = [
        ("prefix_rows_total", "prefix_rows"),
        ("gradient_valid_x", "gradient_valid_y"),
        ("gradient_fallback_reason_x", "gradient_fallback_reason_y"),
    ]
    for actual_column, saved_column in exact_pairs:
        if not merged[actual_column].astype(str).equals(merged[saved_column].astype(str)):
            mismatch = merged[
                merged[actual_column].astype(str) != merged[saved_column].astype(str)
            ][["well", actual_column, saved_column]].head(10)
            raise ValueError(f"full plane exact parity failed:\n{mismatch}")
    numeric_pairs = [
        ("xy_rank_ratio_x", "xy_rank_ratio_y"),
        ("xy_condition_number_x", "xy_condition_number_y"),
        ("azimuth_coverage_x", "azimuth_coverage_y"),
        ("generation_plane_rmse", "plane_rmse_y"),
        ("generation_gradient_x", "gradient_x_center"),
        ("generation_gradient_y", "gradient_y_center"),
    ]
    parity_rows: list[dict[str, Any]] = []
    for actual_column, saved_column in numeric_pairs:
        actual = pd.to_numeric(merged[actual_column], errors="raise").to_numpy(np.float64)
        expected = pd.to_numeric(merged[saved_column], errors="raise").to_numpy(np.float64)
        difference = np.abs(actual - expected)
        passed = np.isclose(actual, expected, atol=float(atol), rtol=float(rtol))
        parity_rows.append(
            {
                "field": saved_column,
                "rows": int(len(merged)),
                "max_abs_diff": float(np.max(difference)),
                "mean_abs_diff": float(np.mean(difference)),
                "parity_pass": bool(passed.all()),
            }
        )
        if not passed.all():
            bad = merged.loc[~passed, ["well", actual_column, saved_column]].head(10)
            raise ValueError(f"full plane numeric parity failed for {saved_column}:\n{bad}")
    return pd.DataFrame(parity_rows)


# %% [markdown]
# ## 5. Prefix-stability feature construction
#
# 3 window の全 pairを計算し、各 component の最大 disagreement を取る。component はすべて
# `[0,1]`、primary risk は6成分の単純平均で、outcome や outer-train fitを使わない。


# %%
def normalized_log_ratio(left: float, right: float, *, epsilon: float, clip: float) -> float:
    value = abs(math.log((max(float(left), 0.0) + epsilon) / (max(float(right), 0.0) + epsilon)))
    return float(np.clip(value / float(clip), 0.0, 1.0))


def normalized_gradient_angle(
    left: np.ndarray,
    right: np.ndarray,
    *,
    epsilon: float,
) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= epsilon and right_norm <= epsilon:
        return 0.0
    if left_norm <= epsilon or right_norm <= epsilon:
        return 1.0
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    angle_degrees = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
    return float(np.clip(angle_degrees / 180.0, 0.0, 1.0))


def build_prefix_stability_features(
    diagnostics: pd.DataFrame,
    risk_config: Mapping[str, Any],
    *,
    n_folds: int,
) -> pd.DataFrame:
    expected_windows = ["full", "last512", "last256"]
    if sorted(diagnostics["window"].unique()) != sorted(expected_windows):
        raise ValueError("prefix diagnostics do not contain the fixed three windows")
    if diagnostics.duplicated(["well", "window"]).any():
        raise ValueError("duplicate well/window plane diagnostic")
    epsilon = float(risk_config["magnitude_epsilon"])
    rmse_epsilon = float(risk_config["rmse_epsilon"])
    log_clip = float(risk_config["log_ratio_clip"])
    rows: list[dict[str, Any]] = []
    for well, group in diagnostics.groupby("well", sort=True):
        lookup = group.set_index("window")
        pair_values: dict[str, list[float]] = {
            "gradient_angle_disagreement": [],
            "gradient_magnitude_log_ratio": [],
            "plane_rmse_log_ratio": [],
            "rank_ratio_absolute_gap": [],
            "condition_number_log_ratio": [],
        }
        validities = [bool(int(lookup.loc[name, "gradient_valid"])) for name in expected_windows]
        for left_name, right_name in combinations(expected_windows, 2):
            left = lookup.loc[left_name]
            right = lookup.loc[right_name]
            both_valid = bool(int(left["diagnostic_fit_valid"])) and bool(
                int(right["diagnostic_fit_valid"])
            )
            if both_valid:
                left_gradient = np.asarray([left["gradient_x"], left["gradient_y"]], dtype=float)
                right_gradient = np.asarray([right["gradient_x"], right["gradient_y"]], dtype=float)
                angle = normalized_gradient_angle(left_gradient, right_gradient, epsilon=epsilon)
                magnitude = normalized_log_ratio(
                    float(left["gradient_magnitude"]),
                    float(right["gradient_magnitude"]),
                    epsilon=epsilon,
                    clip=log_clip,
                )
                rmse_gap = normalized_log_ratio(
                    float(left["plane_rmse"]),
                    float(right["plane_rmse"]),
                    epsilon=rmse_epsilon,
                    clip=log_clip,
                )
            else:
                angle = 1.0
                magnitude = 1.0
                rmse_gap = 1.0
            rank_gap = float(
                np.clip(
                    abs(float(left["xy_rank_ratio"]) - float(right["xy_rank_ratio"])),
                    0.0,
                    1.0,
                )
            )
            condition_gap = normalized_log_ratio(
                float(left["xy_condition_number"]),
                float(right["xy_condition_number"]),
                epsilon=epsilon,
                clip=log_clip,
            )
            pair_values["gradient_angle_disagreement"].append(angle)
            pair_values["gradient_magnitude_log_ratio"].append(magnitude)
            pair_values["plane_rmse_log_ratio"].append(rmse_gap)
            pair_values["rank_ratio_absolute_gap"].append(rank_gap)
            pair_values["condition_number_log_ratio"].append(condition_gap)
        components = {key: float(max(values)) for key, values in pair_values.items()}
        components["validity_flip"] = float(len(set(validities)) > 1)
        ordered_components = [float(components[name]) for name in risk_config["components"]]
        if not np.isfinite(ordered_components).all():
            raise ValueError(f"non-finite stability component for well={well}")
        risk_score = float(np.mean(ordered_components))
        rows.append(
            {
                "well": str(well),
                "outer_fold": stable_outer_fold(str(well), n_folds),
                "full_gradient_valid": int(validities[0]),
                "valid_window_count": int(sum(validities)),
                **components,
                "stability_risk_score": risk_score,
            }
        )
    features = pd.DataFrame(rows).sort_values(["well"], kind="stable").reset_index(drop=True)
    component_values = features[list(risk_config["components"])].to_numpy(np.float64)
    if np.any(component_values < 0.0) or np.any(component_values > 1.0):
        raise ValueError("stability components escaped [0,1]")
    if not np.allclose(
        features["stability_risk_score"].to_numpy(np.float64),
        component_values.mean(axis=1),
        atol=1.0e-15,
        rtol=0.0,
    ):
        raise ValueError("stability risk is not the fixed equal-weight mean")
    return features


# %% [markdown]
# ## 6. Outcome attachment and five-fold readout helpers
#
# primary は full-valid cohort の bank-mean delta RMSE。bank-max と candidate別相関は保存するが、
# primary guardを救済しない。Spearman は rank後の Pearson として deterministic に計算する。


# %%
def spearman_correlation(x: Iterable[float], y: Iterable[float]) -> float:
    left = pd.Series(np.asarray(list(x), dtype=np.float64)).rank(method="average")
    right = pd.Series(np.asarray(list(y), dtype=np.float64)).rank(method="average")
    if left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.corr(right, method="pearson"))


def attach_outcomes_after_feature_freeze(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    candidate_long: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if any(
        token in column.lower()
        for column in features.columns
        for token in ("target", "true_tvt", "delta_rmse", "oracle", "error")
    ):
        raise ValueError("target/outcome-like column leaked into frozen stability features")
    frozen_sha = logical_frame_sha256(features, ["well"])
    merged = features.merge(outcomes, on="well", how="left", validate="one_to_one")
    if merged[outcomes.columns.difference(["well"])].isna().any().any():
        raise ValueError("candidate outcome missing after feature freeze")
    if logical_frame_sha256(features, ["well"]) != frozen_sha:
        raise AssertionError("stability feature table changed during outcome attachment")
    candidate = candidate_long.merge(
        features[["well", "outer_fold", "full_gradient_valid", "stability_risk_score"]],
        on="well",
        how="left",
        validate="many_to_one",
    )
    return merged, candidate


def build_correlation_readout(
    merged: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    primary = merged[merged["full_gradient_valid"].eq(1)].copy()
    candidate = candidate[candidate["full_gradient_valid"].eq(1)].copy()
    rows: list[dict[str, Any]] = []
    outcome_columns = {
        "gradient_bank_mean_delta_rmse_vs_scalar": "primary",
        "gradient_bank_max_delta_rmse_vs_scalar": "secondary_report_only",
    }
    for outcome, decision_role in outcome_columns.items():
        scopes = [("all", primary)] + [
            (str(fold), group) for fold, group in primary.groupby("outer_fold", sort=True)
        ]
        for fold, group in scopes:
            rows.append(
                {
                    "scope": "candidate_bank",
                    "candidate": "gradient_bank",
                    "outcome": outcome,
                    "decision_role": decision_role,
                    "outer_fold": fold,
                    "wells": int(len(group)),
                    "spearman": spearman_correlation(group["stability_risk_score"], group[outcome]),
                }
            )
    for candidate_name, candidate_group in candidate.groupby("candidate", sort=True):
        scopes = [("all", candidate_group)] + [
            (str(fold), group) for fold, group in candidate_group.groupby("outer_fold", sort=True)
        ]
        for fold, group in scopes:
            rows.append(
                {
                    "scope": "candidate",
                    "candidate": str(candidate_name),
                    "outcome": "delta_rmse_vs_scalar",
                    "decision_role": "candidate_report_only",
                    "outer_fold": fold,
                    "wells": int(len(group)),
                    "spearman": spearman_correlation(
                        group["stability_risk_score"], group["delta_rmse_vs_scalar"]
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["scope", "candidate", "outcome", "outer_fold"], kind="stable"
    )


def build_risk_quintile_readout(merged: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    primary = merged[merged["full_gradient_valid"].eq(1)].sort_values(
        ["stability_risk_score", "well"], kind="stable"
    )
    labels = [f"q{index}" for index in range(int(quantiles))]
    primary = primary.copy()
    primary["risk_quintile"] = pd.qcut(
        primary["stability_risk_score"].rank(method="first"),
        q=int(quantiles),
        labels=labels,
    ).astype(str)
    rows: list[dict[str, Any]] = []
    for quintile, group in primary.groupby("risk_quintile", sort=True, observed=True):
        rows.append(
            {
                "risk_quintile": str(quintile),
                "wells": int(len(group)),
                "risk_mean": float(group["stability_risk_score"].mean()),
                "risk_min": float(group["stability_risk_score"].min()),
                "risk_max": float(group["stability_risk_score"].max()),
                "mean_bank_delta_rmse": float(
                    group["gradient_bank_mean_delta_rmse_vs_scalar"].mean()
                ),
                "median_bank_delta_rmse": float(
                    group["gradient_bank_mean_delta_rmse_vs_scalar"].median()
                ),
                "max_bank_delta_rmse": float(
                    group["gradient_bank_mean_delta_rmse_vs_scalar"].max()
                ),
                "regressed_well_rate": float(
                    (group["gradient_bank_mean_delta_rmse_vs_scalar"] > 0.0).mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["risk_quintile"], kind="stable")


def evaluate_primary_guard(
    features: pd.DataFrame,
    correlations: pd.DataFrame,
    quintiles: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    technical = config["guards"]["technical"]
    guard = config["guards"]["primary"]
    valid = features[features["full_gradient_valid"].eq(1)]
    fold_counts = valid.groupby("outer_fold").size().sort_index()
    primary = correlations[
        correlations["outcome"].eq("gradient_bank_mean_delta_rmse_vs_scalar")
        & correlations["decision_role"].eq("primary")
    ]
    fold_rows = primary[primary["outer_fold"].ne("all")]
    pooled_row = primary[primary["outer_fold"].eq("all")]
    epsilon = float(guard["sign_epsilon"])
    positive_folds = int((fold_rows["spearman"] > epsilon).sum())
    pooled_spearman = float(pooled_row["spearman"].iloc[0])
    lowest = float(
        quintiles.loc[quintiles["risk_quintile"].eq("q0"), "mean_bank_delta_rmse"].iloc[0]
    )
    highest_label = f"q{int(config['risk']['quantiles']) - 1}"
    highest = float(
        quintiles.loc[quintiles["risk_quintile"].eq(highest_label), "mean_bank_delta_rmse"].iloc[0]
    )
    checks = {
        "rows_match": len(features) == int(technical["expected_wells"]),
        "full_valid_wells_match": len(valid) == int(technical["expected_full_valid_wells"]),
        "folds_match": len(fold_counts) == int(technical["expected_folds"]),
        "min_valid_wells_per_fold_pass": int(fold_counts.min())
        >= int(technical["min_full_valid_wells_per_fold"]),
        "positive_folds_pass": positive_folds == int(guard["required_positive_folds"]),
        "pooled_positive_pass": pooled_spearman > epsilon,
        "highest_quintile_exceeds_lowest_pass": highest > lowest,
    }
    return {
        "technical_and_primary_guard_pass": bool(all(checks.values())),
        "checks": checks,
        "positive_folds": positive_folds,
        "required_positive_folds": int(guard["required_positive_folds"]),
        "pooled_spearman": pooled_spearman,
        "lowest_quintile_mean_bank_delta_rmse": lowest,
        "highest_quintile_mean_bank_delta_rmse": highest,
        "full_valid_wells_per_fold": {
            str(int(key)): int(value) for key, value in fold_counts.items()
        },
        "decision_scope": "readout_only_separate_gate_experiment_required",
    }


def create_readout_plot(
    merged: pd.DataFrame,
    correlations: pd.DataFrame,
    quintiles: pd.DataFrame,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    primary = merged[merged["full_gradient_valid"].eq(1)]
    primary_corr = correlations[
        correlations["outcome"].eq("gradient_bank_mean_delta_rmse_vs_scalar")
        & correlations["decision_role"].eq("primary")
        & correlations["outer_fold"].ne("all")
    ].copy()
    component_columns = [str(value) for value in CONFIG["risk"]["components"]]
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    scatter = axes[0, 0].scatter(
        primary["stability_risk_score"],
        primary["gradient_bank_mean_delta_rmse_vs_scalar"],
        c=primary["outer_fold"],
        cmap="tab10",
        alpha=0.8,
        s=35,
    )
    axes[0, 0].axhline(0.0, color="black", linewidth=1)
    axes[0, 0].set_title("Full-valid wells: frozen risk vs bank mean regression")
    axes[0, 0].set_xlabel("stability risk")
    axes[0, 0].set_ylabel("mean delta RMSE vs scalar [ft]")
    figure.colorbar(scatter, ax=axes[0, 0], label="audit outer fold")

    axes[0, 1].bar(primary_corr["outer_fold"], primary_corr["spearman"])
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].set_title("Primary Spearman by fixed well fold")
    axes[0, 1].set_xlabel("outer fold")
    axes[0, 1].set_ylabel("Spearman")

    axes[1, 0].bar(quintiles["risk_quintile"], quintiles["mean_bank_delta_rmse"])
    axes[1, 0].axhline(0.0, color="black", linewidth=1)
    axes[1, 0].set_title("Frozen risk quintile mean regression")
    axes[1, 0].set_xlabel("risk quintile")
    axes[1, 0].set_ylabel("mean delta RMSE vs scalar [ft]")

    primary[component_columns].boxplot(ax=axes[1, 1], rot=35)
    axes[1, 1].set_title("Fixed normalized stability components")
    axes[1, 1].set_ylabel("component risk [0,1]")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


# %% [markdown]
# ## 7. Setup and fixed input checks
#
# 3つの exp273 kernel output、competition raw trainを解決し、期待SHA・rows・wellsを確認する。

# %%
if EXECUTE_NOTEBOOK:
    artifacts_dir = Path("/kaggle/working/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plane_path = resolve_fixed_file(CONFIG["data"]["plane_diagnostics"])
    by_well_path = resolve_fixed_file(CONFIG["data"]["by_well_metrics"])
    raw_train_dir = resolve_raw_train_dir(CONFIG["data"]["raw_train_dir_patterns"])
    shard_paths = [
        resolve_fixed_file(spec, "expected_raw_sha256") for spec in CONFIG["data"]["shards"]
    ]
    fixed_plane = pd.read_csv(plane_path)
    fixed_by_well = pd.read_csv(by_well_path)
    candidates = [str(value) for value in CONFIG["candidate_bank"]["candidates"]]
    input_overview = {
        "plane_path": str(plane_path),
        "plane_sha256": sha256_file(plane_path),
        "plane_rows": int(len(fixed_plane)),
        "by_well_path": str(by_well_path),
        "by_well_sha256": sha256_file(by_well_path),
        "by_well_rows": int(len(fixed_by_well)),
        "raw_train_dir": str(raw_train_dir),
        "shards": [str(path) for path in shard_paths],
    }
    display(input_overview)
    assert fixed_plane["well"].astype(str).nunique() == int(CONFIG["validation"]["expected_wells"])
    assert int(fixed_plane["gradient_valid"].sum()) == int(
        CONFIG["validation"]["expected_full_valid_wells"]
    )


# %% [markdown]
# ## 8. Recompute prefix diagnostics and freeze target-free risk
#
# この cell が終わるまで exp273 candidate error は feature frame に接続しない。

# %%
if EXECUTE_NOTEBOOK:
    prefix_diagnostics, raw_manifest = recompute_prefix_plane_diagnostics(
        raw_train_dir,
        fixed_plane["well"].astype(str).tolist(),
        CONFIG["plane"],
        CONFIG["plane"]["windows"],
    )
    full_parity = verify_full_plane_parity(
        prefix_diagnostics,
        fixed_plane,
        atol=float(CONFIG["plane"]["full_parity"]["absolute_tolerance"]),
        rtol=float(CONFIG["plane"]["full_parity"]["relative_tolerance"]),
    )
    stability_features = build_prefix_stability_features(
        prefix_diagnostics,
        CONFIG["risk"],
        n_folds=int(CONFIG["validation"]["n_folds"]),
    )
    frozen_feature_sha256 = logical_frame_sha256(stability_features, ["well"])
    display(full_parity)
    display(
        stability_features.groupby(["outer_fold", "full_gradient_valid"])
        .size()
        .rename("wells")
        .reset_index()
    )
    display(stability_features.describe(include="all"))


# %% [markdown]
# ## 9. Attach outcomes, evaluate guards, and save generated artifacts
#
# shard candidate RMSE parity後に outcome を接続する。PASSでもgate/inferenceは作らず、
# 別実験の設計根拠に限定する。

# %%
if EXECUTE_NOTEBOOK:
    shard_metric_parts: list[pd.DataFrame] = []
    shard_summaries: list[dict[str, Any]] = []
    for path, spec in zip(shard_paths, CONFIG["data"]["shards"], strict=True):
        metrics, summary = stream_shard_candidate_rmse(
            path,
            candidates,
            chunk_rows=int(CONFIG["runtime"]["chunk_rows"]),
        )
        if summary["rows"] != int(spec["expected_rows"]):
            raise ValueError(f"unexpected shard rows: {summary}")
        if summary["wells"] != int(spec["expected_wells"]):
            raise ValueError(f"unexpected shard wells: {summary}")
        if summary["raw_sha256"] != str(spec["expected_raw_sha256"]):
            raise ValueError(f"shard raw SHA mismatch: {summary}")
        if summary["decompressed_sha256"] != str(spec["expected_decompressed_sha256"]):
            raise ValueError(f"shard decompressed SHA mismatch: {summary}")
        shard_metric_parts.append(metrics)
        shard_summaries.append(summary)
    recomputed_candidate_metrics = pd.concat(shard_metric_parts, ignore_index=True)
    if recomputed_candidate_metrics.duplicated(["well", "candidate"]).any():
        raise ValueError("well overlaps across exp273 shards")
    technical = CONFIG["guards"]["technical"]
    if len(recomputed_candidate_metrics) != int(technical["expected_candidate_well_rows"]):
        raise ValueError("unexpected exp273 candidate/well metric row count")
    if recomputed_candidate_metrics["well"].nunique() != int(technical["expected_wells"]):
        raise ValueError("unexpected exp273 shard well union")
    outcomes, candidate_long, candidate_parity = build_candidate_outcomes(
        fixed_by_well,
        recomputed_candidate_metrics,
        candidates,
        parity_atol=float(CONFIG["guards"]["technical"]["by_well_rmse_parity_atol"]),
    )
    if len(outcomes) != int(technical["expected_wells"]):
        raise ValueError("unexpected exp273 outcome well count")
    merged, candidate_readout = attach_outcomes_after_feature_freeze(
        stability_features, outcomes, candidate_long
    )
    correlations = build_correlation_readout(merged, candidate_readout)
    quintiles = build_risk_quintile_readout(merged, int(CONFIG["risk"]["quantiles"]))
    guard = evaluate_primary_guard(stability_features, correlations, quintiles, CONFIG)

    paths = {
        "prefix_plane_diagnostics": artifacts_dir / f"{OUTPUT_PREFIX}_prefix_plane_diagnostics.csv",
        "prefix_stability_features": artifacts_dir
        / f"{OUTPUT_PREFIX}_prefix_stability_features.csv",
        "candidate_outcomes": artifacts_dir / f"{OUTPUT_PREFIX}_candidate_outcomes_by_well.csv",
        "candidate_parity": artifacts_dir / f"{OUTPUT_PREFIX}_candidate_metric_parity.csv",
        "correlations": artifacts_dir / f"{OUTPUT_PREFIX}_stability_correlations.csv",
        "quintiles": artifacts_dir / f"{OUTPUT_PREFIX}_risk_quintile_metrics.csv",
        "input_manifest": artifacts_dir / f"{OUTPUT_PREFIX}_input_manifest.csv",
        "plot": artifacts_dir / f"{OUTPUT_PREFIX}_prefix_stability_readout.png",
        "summary": artifacts_dir / f"{OUTPUT_PREFIX}_readout_summary.json",
        "reproducibility": artifacts_dir / f"{OUTPUT_PREFIX}_reproducibility_manifest.json",
    }
    prefix_diagnostics.to_csv(paths["prefix_plane_diagnostics"], index=False)
    stability_features.to_csv(paths["prefix_stability_features"], index=False)
    outcomes.to_csv(paths["candidate_outcomes"], index=False)
    candidate_parity.to_csv(paths["candidate_parity"], index=False)
    correlations.to_csv(paths["correlations"], index=False)
    quintiles.to_csv(paths["quintiles"], index=False)
    raw_manifest.to_csv(paths["input_manifest"], index=False)
    create_readout_plot(merged, correlations, quintiles, paths["plot"])

    table_specs = {
        "prefix_plane_diagnostics": (prefix_diagnostics, ["well", "window"]),
        "prefix_stability_features": (stability_features, ["well"]),
        "candidate_outcomes": (outcomes, ["well"]),
        "candidate_parity": (candidate_parity, ["candidate", "well"]),
        "correlations": (
            correlations,
            ["scope", "candidate", "outcome", "outer_fold"],
        ),
        "quintiles": (quintiles, ["risk_quintile"]),
        "input_manifest": (raw_manifest, ["well"]),
    }
    artifact_manifest: dict[str, Any] = {}
    for name, path in paths.items():
        if name in ("summary", "reproducibility"):
            continue
        artifact_manifest[name] = {
            "path": str(path),
            "file_sha256": sha256_file(path),
        }
        if name in table_specs:
            frame, sort_columns = table_specs[name]
            artifact_manifest[name]["logical_sha256"] = logical_frame_sha256(frame, sort_columns)
            artifact_manifest[name]["rows"] = int(len(frame))

    reproducibility = {
        "experiment": EXPERIMENT_NAME,
        "seed_policy": CONFIG["reproducibility"]["seed_policy"],
        "stochastic_components": CONFIG["reproducibility"]["stochastic_components"],
        "fold_policy": CONFIG["validation"]["fold_policy"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "fixed_inputs": {
            "plane_diagnostics": {
                "path": str(plane_path),
                "sha256": sha256_file(plane_path),
            },
            "by_well_metrics": {
                "path": str(by_well_path),
                "sha256": sha256_file(by_well_path),
            },
            "shards": shard_summaries,
        },
        "raw_horizontal_manifest_logical_sha256": logical_frame_sha256(raw_manifest, ["well"]),
        "frozen_stability_feature_logical_sha256": frozen_feature_sha256,
        "full_plane_parity": full_parity.to_dict(orient="records"),
        "artifacts": artifact_manifest,
        "model_sha": None,
        "prediction_sha": None,
        "submission_sha": None,
        "deterministic_anchor": False,
    }
    write_json(paths["reproducibility"], reproducibility)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": (
            "readout_guard_pass_separate_gate_design_only"
            if guard["technical_and_primary_guard_pass"]
            else "readout_guard_failed_branch_close"
        ),
        "route": "pf_beam",
        "rows": int(sum(item["rows"] for item in shard_summaries)),
        "wells": int(len(stability_features)),
        "full_valid_wells": int(stability_features["full_gradient_valid"].sum()),
        "compute_contract": compute_contract,
        "primary_guard": guard,
        "primary_pooled_spearman": float(
            correlations.loc[
                correlations["outcome"].eq("gradient_bank_mean_delta_rmse_vs_scalar")
                & correlations["outer_fold"].eq("all"),
                "spearman",
            ].iloc[0]
        ),
        "frozen_stability_feature_logical_sha256": frozen_feature_sha256,
        "reproducibility_manifest_sha256": sha256_file(paths["reproducibility"]),
        "artifacts": artifact_manifest,
        "decision": (
            "PASS supports only a separate fold-safe gate audit; no gate, inference, "
            "or submission was produced."
            if guard["technical_and_primary_guard_pass"]
            else "FAIL closes the exp273 formation-gradient branch without a rescue grid."
        ),
    }
    write_json(paths["summary"], summary)
    display(correlations)
    display(quintiles)
    display(guard)
    print(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False))
