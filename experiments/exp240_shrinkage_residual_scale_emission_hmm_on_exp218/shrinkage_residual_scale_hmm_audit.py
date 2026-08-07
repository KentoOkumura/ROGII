from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from direct_hmm_comparison import run_direct_comparison
from exact_hmm_smoother import (
    run_train_feature_cache,
    sha256_gzip_decompressed,
    sha256_path,
    to_jsonable,
)
from residual_scale_crossfit import run_crossfitted_residual_scale
from settings import ExperimentPaths, get_nested, load_config

EXPERIMENT_NAME = "exp240_shrinkage_residual_scale_emission_hmm_on_exp218"
OUTPUT_PREFIX = "exp240_shrinkage_residual_scale_emission_hmm_on_exp218"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def alpha_token(alpha: float) -> str:
    return f"a{int(round(alpha * 1000)):04d}"


def _selected_stage(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    shrinkage = dict(config.get("shrinkage") or {})
    stage_name = str(shrinkage.get("selected_stage") or "")
    stages = dict(shrinkage.get("stages") or {})
    if stage_name not in stages:
        raise ValueError(f"unknown shrinkage.selected_stage: {stage_name!r}")
    stage = dict(stages[stage_name] or {})
    if not bool(stage.get("enabled", False)):
        raise ValueError(f"selected shrinkage stage is disabled: {stage_name}")
    enabled = [name for name, value in stages.items() if bool((value or {}).get("enabled", False))]
    if enabled != [stage_name]:
        raise ValueError(
            "exp240 requires exactly one enabled stage and it must equal selected_stage; "
            f"selected={stage_name!r} enabled={enabled}"
        )
    status = str((config.get("experiment") or {}).get("status") or "")
    if stage_name != "scalar_control" and status == "implemented_scalar_control_not_run":
        raise ValueError("scalar control must complete and be recorded before shrinkage is enabled")
    return stage_name, stage


def _build_shrinkage_sidecar(
    paths: ExperimentPaths,
    scale_summary: dict[str, Any],
    *,
    alpha: float,
    scalar_sigma: float,
    sigma_floor: float,
    sigma_cap: float,
) -> tuple[Path, dict[str, Any]]:
    source_name = str((scale_summary.get("outputs") or {}).get("predictions") or "")
    source_path = paths.artifacts_dir / source_name
    if not source_path.is_file():
        raise FileNotFoundError(f"cross-fitted residual-scale output not found: {source_path}")
    frame = pd.read_csv(source_path, usecols=["id", "pred_tvt", "sigma_tvt"])
    if frame["id"].duplicated().any():
        raise ValueError("cross-fitted residual-scale sidecar has duplicate ids")
    row_sigma = pd.to_numeric(frame["sigma_tvt"], errors="raise").to_numpy(np.float64)
    # Shrink variance, not standard deviation, because sigma enters a Gaussian likelihood.
    effective = np.sqrt((1.0 - alpha) * scalar_sigma**2 + alpha * row_sigma**2)
    effective = np.clip(effective, sigma_floor, sigma_cap)
    if not np.isfinite(effective).all() or float(np.min(effective)) <= 0.0:
        raise ValueError("effective shrinkage sigma must be finite and positive")
    output = frame[["id", "pred_tvt"]].copy()
    output["sigma_tvt"] = effective
    output_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_{alpha_token(alpha)}_predictions.csv.gz"
    output.to_csv(output_path, index=False, compression="gzip")
    summary = {
        "formula": "sqrt((1-alpha)*scalar_sigma^2 + alpha*crossfitted_sigma^2)",
        "alpha": alpha,
        "scalar_sigma": scalar_sigma,
        "sigma_floor": sigma_floor,
        "sigma_cap": sigma_cap,
        "rows": int(len(output)),
        "effective_min": float(np.min(effective)),
        "effective_mean": float(np.mean(effective)),
        "effective_p90": float(np.quantile(effective, 0.90)),
        "effective_max": float(np.max(effective)),
        "output": output_path.name,
        "sha256": {
            "gzip": sha256_path(output_path),
            "decompressed": sha256_gzip_decompressed(output_path),
        },
    }
    return output_path, summary


def _emission_config_for_stage(
    config: dict[str, Any],
    paths: ExperimentPaths,
    stage: dict[str, Any],
    *,
    max_wells: int | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    base = dict(config.get("lgb_emission") or {})
    sources = dict(base.get("sources") or {})
    kind = str(stage.get("kind") or "")
    if kind == "scalar_control":
        scalar_sigma = float(stage["scalar_sigma"])
        runtime = {
            **base,
            "active_sources": ["exp218_scalar"],
            "sigma_grid": [scalar_sigma],
            "max_variants": 1,
            "sources": sources,
        }
        return runtime, None
    if kind != "variance_shrinkage":
        raise ValueError(f"unsupported exp240 stage kind: {kind!r}")

    alpha = float(stage["alpha"])
    allowed = [float(value) for value in (config.get("shrinkage", {}).get("allowed_alphas") or [])]
    if not any(math.isclose(alpha, value, abs_tol=1e-12) for value in allowed):
        raise ValueError(f"alpha={alpha} is not one of predeclared allowed_alphas={allowed}")
    if not 0.0 < alpha < 1.0:
        raise ValueError("variance-shrinkage stage alpha must be strictly between zero and one")

    scale_summary = run_crossfitted_residual_scale(max_wells=max_wells)
    guard = dict(scale_summary.get("guard") or {})
    if not bool(guard.get("passed", False)):
        raise RuntimeError("cross-fitted residual-scale guard failed; shrinkage HMM is prohibited")
    scalar_sigma = float(config["shrinkage"]["scalar_sigma"])
    scale_cfg = dict(config.get("residual_scale") or {})
    sidecar_path, sidecar_summary = _build_shrinkage_sidecar(
        paths,
        scale_summary,
        alpha=alpha,
        scalar_sigma=scalar_sigma,
        sigma_floor=float(scale_cfg["sigma_floor"]),
        sigma_cap=float(scale_cfg["sigma_cap"]),
    )
    sources["exp218_selected_shrinkage"] = {
        "description": f"exp218 center with variance-shrunk cross-fitted sigma alpha={alpha}",
        "id_column": "id",
        "prediction_column": "pred_tvt",
        "sigma_column": "sigma_tvt",
        "candidates": [str(sidecar_path)],
    }
    runtime = {
        **base,
        "active_sources": ["exp218_selected_shrinkage"],
        "sigma_grid": [],
        "max_variants": 1,
        "sources": sources,
    }
    return runtime, {"residual_scale": scale_summary, "sidecar": sidecar_summary}


def run_shrinkage_residual_scale_hmm_audit(
    *, max_wells: int | None = None, fast: bool = False
) -> dict[str, Any]:
    started = time.time()
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    config = load_config()
    stage_name, stage = _selected_stage(config)
    emission_config, scale_payload = _emission_config_for_stage(
        config, paths, stage, max_wells=max_wells
    )

    variants = len(emission_config.get("active_sources") or []) * len(
        emission_config.get("lambda_grid") or []
    )
    if variants != 1 or int(emission_config.get("max_variants", 0)) != 1:
        raise ValueError("exp240 permits exactly one HMM emission variant per Kaggle run")

    hmm_summary = run_train_feature_cache(
        root=paths.root,
        data_dir=paths.train_data_dir,
        output_dir=paths.artifacts_dir,
        hmm_config=dict(get_nested(config, "model.hmm") or {}),
        lgb_emission_config=emission_config,
        output_prefix=OUTPUT_PREFIX,
        max_wells=max_wells,
        fast=fast,
        numba_num_threads=get_nested(config, "runtime.numba_num_threads"),
        outer_workers=int(get_nested(config, "feature_cache.hmm.outer_workers") or 1),
    )
    comparison = run_direct_comparison()
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "train_side_stage_completed",
        "stage": stage_name,
        "stage_config": stage,
        "scale_payload": scale_payload,
        "hmm_summary": hmm_summary,
        "comparison_summary": comparison,
        "elapsed_seconds": round(time.time() - started, 3),
        "notes": [
            "Exactly one scalar or shrinkage HMM variant ran in this Kaggle version.",
            "No LightGBM booster, parent/control retraining, inference, or submission is allowed.",
        ],
    }
    summary_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_{stage_name}_audit_summary.json"
    write_json(summary_path, summary)
    summary["sha256"] = {"summary": sha256_path(summary_path)}
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run_shrinkage_residual_scale_hmm_audit()
