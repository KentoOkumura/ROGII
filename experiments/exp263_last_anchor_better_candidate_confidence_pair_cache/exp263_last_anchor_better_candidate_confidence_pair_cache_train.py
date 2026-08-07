# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp263 last-anchor-better candidate confidence pair cache — Stage 0
#
# exp072 `last_anchor` より良い known 33 path をreference catalogとして固定し、
# family圧縮したcore 12 primitiveだけをcandidate-major OOF cacheへ変換する。
# pair/tripleのfull row tensorは保存せず、8 pairと3 named combinationはloaderが
# primitive partitionから要求時に再構成する。このnotebookはモデルを学習しない。

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Contract constants and forbidden-source guards
# 4. Input source resolution and schema preflight
# 5. Candidate and confidence inventory
# 6. Pair shortlist and formula DAG checks
# 7. Stage 0 cache generation orchestration
# 8. Generated partition and SHA checks
# 9. Virtual loader parity sample
# 10. Metrics and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from candidate_cache_builder import build_stage0_cache, resolve_input_paths
from candidate_cache_contract import (
    CORE_CANDIDATE_IDS,
    NAMED_COMBINATIONS,
    PAIR_SHORTLIST,
    RAWTEST_CORE_CANDIDATE_IDS,
    REFERENCE_CANDIDATES,
    validate_contract,
)
from candidate_cache_loader import CandidateCache, sha256_file
from IPython.display import display
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
DEBUG = os.environ.get("EXPERIMENT_DEBUG", "0") == "1"
MAX_ROWS_ENV = os.environ.get("EXPERIMENT_MAX_ROWS")
MAX_ROWS = int(MAX_ROWS_ENV) if MAX_ROWS_ENV else 20_000

paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()
output_dir = paths.artifacts_dir

print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Parent:", get_nested(config, "lineage.parent"))
print("Stage:", "stage0_oof_candidate_cache")
print("Output:", output_dir)
print("Debug:", DEBUG, "max rows:", MAX_ROWS if DEBUG else None)
print("GPU enabled:", get_nested(config, "runtime.kaggle.enable_gpu"))
print("Training variants/configs/folds/boosters:", 0, 0, 0, 0)

# %% [markdown]
# ## 3. Contract constants and forbidden-source guards

# %%
contract_counts = validate_contract()
expected_counts = {
    "reference_candidates": int(get_nested(config, "candidate_contract.reference_count")),
    "core_candidates": int(get_nested(config, "candidate_contract.core_count")),
    "rawtest_core_candidates": int(
        get_nested(config, "candidate_contract.rawtest_core_count")
    ),
    "shortlisted_pairs": int(get_nested(config, "candidate_contract.pair_count")),
    "rawtest_pairs": int(get_nested(config, "candidate_contract.rawtest_pair_count")),
    "named_triples": int(get_nested(config, "candidate_contract.named_triple_count")),
}
if contract_counts != expected_counts:
    raise ValueError(
        f"config/implementation contract mismatch: {contract_counts} != {expected_counts}"
    )

forbidden = set(get_nested(config, "candidate_contract.excluded_candidates") or [])
inventory_ids = {item.candidate_id for item in REFERENCE_CANDIDATES}
if forbidden.intersection(inventory_ids):
    raise ValueError("HMM+LGB candidates entered the exp263 inventory")

display(
    {
        "contract": contract_counts,
        "core_candidates": list(CORE_CANDIDATE_IDS),
        "rawtest_core_candidates": list(RAWTEST_CORE_CANDIDATE_IDS),
        "forbidden_candidates": sorted(forbidden),
        "row_forbidden_sources": get_nested(config, "candidate_contract.forbidden_row_sources"),
    }
)

# %% [markdown]
# ## 4. Input source resolution and schema preflight

# %%
input_config = get_nested(config, "data.inputs") or {}
search_roots = [Path(value) for value in (get_nested(config, "data.search_roots") or ["."])]
if Path("/kaggle/input").exists():
    search_roots.insert(0, Path("/kaggle/input"))


def source_preflight() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_keys = sorted(
        {item.source_key for item in REFERENCE_CANDIDATES if item.is_core and item.source_key}
    )
    for source_key in source_keys:
        try:
            resolved = resolve_input_paths(source_key, input_config, search_roots)
        except (FileNotFoundError, KeyError, ValueError) as error:
            rows.append(
                {
                    "source_key": source_key,
                    "status": "missing",
                    "resolved_files": 0,
                    "bytes": 0,
                    "paths": str(error),
                }
            )
            continue
        rows.append(
            {
                "source_key": source_key,
                "status": "resolved",
                "resolved_files": len(resolved),
                "bytes": sum(path.stat().st_size for path in resolved),
                "paths": " | ".join(str(path) for path in resolved),
            }
        )
    return pd.DataFrame(rows)


preflight = source_preflight()
display(preflight)
missing_sources = preflight.loc[preflight["status"] != "resolved", "source_key"].tolist()
if missing_sources:
    raise FileNotFoundError(f"Stage 0 source preflight failed: {missing_sources}")

# %% [markdown]
# ## 5. Candidate and confidence inventory
#
# `confidence_available` は保存済みsourceに実在するtarget-free列だけである。
# 契約上期待されてもsourceにない診断は `confidence_unavailable` としてmanifestに残し、
# 0や推測値で埋めない。

# %%
catalog_preview = pd.DataFrame(item.as_catalog_row() for item in REFERENCE_CANDIDATES)
display(
    catalog_preview[
        [
            "candidate_id",
            "family",
            "source_exp",
            "global_rmse",
            "rawtest_status",
            "cache_role",
            "confidence_available",
            "confidence_unavailable",
        ]
    ]
)

core_preview = catalog_preview[catalog_preview["is_core"]]
if len(core_preview) != 12 or core_preview["candidate_id"].nunique() != 12:
    raise ValueError("core candidate inventory is not exactly 12 unique primitives")

# %% [markdown]
# ## 6. Pair shortlist and formula DAG checks

# %%
pair_preview = pd.DataFrame(pair.as_manifest_row() for pair in PAIR_SHORTLIST)
named_preview = pd.DataFrame(
    [{"name": name, **spec} for name, spec in NAMED_COMBINATIONS.items()]
)
display(pair_preview[["pair_id", "left", "right", "deployability", "fixed_50_rmse"]])
display(named_preview[["name", "kind", "deployability", "formula"]])

if len(pair_preview) != 8 or int((pair_preview["deployability"] == "raw-test").sum()) != 5:
    raise ValueError("pair shortlist tier count mismatch")
if "blend_likpf_hmm_w500" not in NAMED_COMBINATIONS:
    raise ValueError("w500 alias is missing")

# %% [markdown]
# ## 7. Stage 0 cache generation orchestration
#
# canonical exp072を一度読み、ID/well-row/outer foldを固定する。external sourceはその
# canonical spanへjoinし、core 12のcoverageを全行確認してからpartitionを書き出す。
# outer eligibilityは各outer-valid foldを除外した4 foldだけで計算する。

# %%
stage0_enabled = bool(get_nested(config, "stage0.enabled"))
if not stage0_enabled:
    raise ValueError("Stage 0 is disabled in config")

summary = build_stage0_cache(
    config,
    output_dir,
    debug=DEBUG,
    max_rows=MAX_ROWS if DEBUG else None,
)
display(summary)

# %% [markdown]
# ## 8. Generated partition and SHA checks

# %%
manifest_path = output_dir / "cache_manifest.json"
catalog_path = output_dir / "candidate_catalog.json"
pair_path = output_dir / "pair_shortlist.csv"
eligibility_path = output_dir / "outer_fold_eligibility.csv"
readout_path = output_dir / "pair_readout.csv"

for required in [manifest_path, catalog_path, pair_path, eligibility_path, readout_path]:
    if not required.exists():
        raise FileNotFoundError(required)

manifest = json.loads(manifest_path.read_text())
catalog = json.loads(catalog_path.read_text())
eligibility = pd.read_csv(eligibility_path)
pair_readout = pd.read_csv(readout_path)

if manifest["core_candidates"] != 12 or manifest["pairs"] != 8:
    raise ValueError("generated cache manifest count mismatch")
if catalog["inventory_count"] != 33 or catalog["core_count"] != 12:
    raise ValueError("generated candidate catalog count mismatch")
if eligibility["basis"].nunique() != 1 or not eligibility["basis"].iloc[0].startswith(
    "outer_train_only"
):
    raise ValueError("outer-fold eligibility basis is not outer-train-only")

display(
    {
        "manifest_sha256": sha256_file(manifest_path),
        "catalog_sha256": sha256_file(catalog_path),
        "rows": manifest["rows"],
        "wells": manifest["wells"],
        "candidate_dtype": manifest["candidate_dtype"],
        "value_partition_count": sum(
            len(parts) for parts in manifest["candidate_value_partitions"].values()
        ),
        "confidence_partition_count": sum(
            len(parts) for parts in manifest["candidate_confidence_partitions"].values()
        ),
    }
)
display(eligibility.head(40))
display(pair_readout.head(80))

# %% [markdown]
# ## 9. Virtual loader parity sample

# %%
cache = CandidateCache(output_dir)
available = cache.list_available()
if len(available["primitive"]) != 12 or len(available["pair"]) != 8:
    raise ValueError("virtual loader inventory mismatch")

w500 = cache.materialize(
    "blend_likpf_hmm_w500", fold=0, row_slice=slice(0, 64), include_confidence=True
)
fixed = cache.materialize(
    "exp226_w500_50_50", fold=0, row_slice=slice(0, 64), include_confidence=True
)
display(w500.head(10))
display(fixed.head(10))

parity_sample = pd.read_parquet(output_dir / "small_parity_sample.parquet")
if len(parity_sample) != min(64, int((eligibility["outer_fold"] == 0).sum() > 0) * 64):
    print("Parity sample row count is lower only when debug fold 0 has fewer than 64 rows.")
display(parity_sample.head())

# %% [markdown]
# ## 10. Metrics and generated artifacts

# %%
metrics = {
    "experiment": EXPERIMENT_NAME,
    "status": summary["status"],
    "route": get_nested(config, "experiment.route"),
    "stage": "stage0",
    "cv": None,
    "public_lb": None,
    "private_lb": None,
    "rows": summary["rows"],
    "wells": summary["wells"],
    "reference_candidates": contract_counts["reference_candidates"],
    "core_candidates": contract_counts["core_candidates"],
    "rawtest_core_candidates": contract_counts["rawtest_core_candidates"],
    "shortlisted_pairs": contract_counts["shortlisted_pairs"],
    "named_triples": contract_counts["named_triples"],
    "training": {
        "variants": 0,
        "lightgbm_configs": 0,
        "fold_training": 0,
        "boosters": 0,
        "parent_control_retraining": False,
    },
    "cache_manifest_sha256": sha256_file(manifest_path),
    "candidate_catalog_sha256": sha256_file(catalog_path),
    "artifacts": summary["artifacts"],
    "model_sha": "not_applicable_no_training",
    "prediction_sha": "not_applicable_cache_only",
    "submission_sha": "not_applicable_no_submission",
}
paths.metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
print("Metrics written:", paths.metrics_path)
display(metrics)
