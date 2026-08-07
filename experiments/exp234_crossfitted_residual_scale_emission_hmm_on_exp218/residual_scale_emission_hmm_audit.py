from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from direct_hmm_comparison import run_direct_comparison
from exact_hmm_smoother import run_train_feature_cache, sha256_path, to_jsonable
from residual_scale_crossfit import run_crossfitted_residual_scale
from settings import ExperimentPaths, get_nested, load_config

EXPERIMENT_NAME = "exp234_crossfitted_residual_scale_emission_hmm_on_exp218"
OUTPUT_PREFIX = "exp234_crossfitted_residual_scale_emission_hmm_on_exp218"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def run_residual_scale_emission_hmm_audit(
    *,
    max_wells: int | None = None,
    fast: bool = False,
) -> dict[str, Any]:
    """Run the cross-fitted scale readout before permitting the one HMM variant."""
    started = time.time()
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    config = load_config()
    hmm_config = dict(get_nested(config, "model.hmm") or {})
    lgb_emission_config = dict(get_nested(config, "lgb_emission") or {})
    feature_cache_config = dict(get_nested(config, "feature_cache.hmm") or {})
    runtime_config = dict(config.get("runtime") or {})

    if not bool(lgb_emission_config.get("enabled", False)):
        raise ValueError("lgb_emission.enabled must be true for the residual-scale HMM audit")
    configured_variants = (
        len(lgb_emission_config.get("active_sources") or [])
        * len(lgb_emission_config.get("lambda_grid") or [])
        * len(lgb_emission_config.get("sigma_floor_grid") or [])
        * len(lgb_emission_config.get("sigma_cap_grid") or [])
    )
    if configured_variants != 1 or int(lgb_emission_config.get("max_variants", 0)) != 1:
        raise ValueError("exp234 requires exactly one configured HMM emission variant")

    scale_summary = run_crossfitted_residual_scale(max_wells=max_wells)
    guard = dict(scale_summary.get("guard") or {})
    summary_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_hmm_audit_summary.json"
    if not bool(guard.get("passed", False)):
        summary = {
            "experiment": EXPERIMENT_NAME,
            "status": "residual_scale_guard_rejected_hmm_not_run",
            "mode": "saved_exp218_oof_center_crossfitted_scale_guard",
            "residual_scale_summary": scale_summary,
            "hmm_summary": None,
            "comparison_summary": None,
            "elapsed_seconds": round(time.time() - started, 3),
            "notes": [
                "The HMM stage is intentionally skipped because the pre-HMM "
                "residual-scale guard failed.",
                "No inference or submission is allowed from this run.",
            ],
        }
        write_json(summary_path, summary)
        summary["sha256"] = {"summary": sha256_path(summary_path)}
        write_json(summary_path, summary)
        print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
        return summary

    output_prefix = str(feature_cache_config.get("output_prefix") or OUTPUT_PREFIX)
    hmm_summary = run_train_feature_cache(
        root=paths.root,
        data_dir=paths.train_data_dir,
        output_dir=paths.artifacts_dir,
        hmm_config=hmm_config,
        lgb_emission_config=lgb_emission_config,
        output_prefix=output_prefix,
        max_wells=max_wells,
        fast=fast,
        numba_num_threads=runtime_config.get("numba_num_threads"),
        outer_workers=int(feature_cache_config.get("outer_workers", 1)),
    )
    comparison_summary = run_direct_comparison()
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "residual_scale_hmm_audit_completed",
        "mode": "saved_exp218_oof_center_crossfitted_scale_single_hmm_train_side_audit",
        "residual_scale_summary": scale_summary,
        "hmm_summary": hmm_summary,
        "comparison_summary": comparison_summary,
        "elapsed_seconds": round(time.time() - started, 3),
        "notes": [
            "The HMM ran only after the residual-scale pre-HMM guard passed.",
            "This audit has one fixed HMM variant and zero LightGBM boosters.",
            "No inference or submission output is generated.",
        ],
    }
    write_json(summary_path, summary)
    summary["sha256"] = {"summary": sha256_path(summary_path)}
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run_residual_scale_emission_hmm_audit()
