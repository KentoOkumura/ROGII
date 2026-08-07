from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from direct_hmm_comparison import run_direct_comparison
from exact_hmm_smoother import run_train_feature_cache, sha256_path, to_jsonable
from settings import ExperimentPaths, get_nested, load_config


EXPERIMENT_NAME = "exp229_lgb_quantile_band_emission_hmm_on_exp148"
OUTPUT_PREFIX = "exp229_lgb_quantile_band_emission_hmm_on_exp148"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def run_quantile_band_hmm_audit(
    *,
    max_wells: int | None = None,
    fast: bool = False,
) -> dict[str, Any]:
    started = time.time()
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    config = load_config()
    hmm_config = dict(get_nested(config, "model.hmm") or {})
    lgb_emission_config = dict(get_nested(config, "lgb_emission") or {})
    feature_cache_config = dict(get_nested(config, "feature_cache.hmm") or {})
    runtime_config = dict(config.get("runtime") or {})

    if not bool(lgb_emission_config.get("enabled", False)):
        raise ValueError("lgb_emission.enabled must be true for quantile-band HMM audit")

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

    summary_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_hmm_audit_summary.json"
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "quantile_band_hmm_audit_completed",
        "mode": "quantile_band_hmm_train_side_audit",
        "hmm_summary": hmm_summary,
        "comparison_summary": comparison_summary,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(summary_path, summary)
    summary["sha256"] = {"summary": sha256_path(summary_path)}
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run_quantile_band_hmm_audit()
