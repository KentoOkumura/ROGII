# %% [markdown]
# # exp253 prefix-verified bounded candidate controller — Stage 1 aggregate
#
# Four deterministic CPU well shards are validated and joined. Global RMSE and
# adoption guards are recomputed from row-level OOF predictions; shard RMSE is
# never averaged.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime, configuration, and SHA helpers
# 3. Shard discovery and integrity contract
# 4. Global metrics and adoption guard
# 5. Aggregate orchestration
# 6. Metrics and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

EXPERIMENT = "exp253_prefix_verified_bounded_candidate_controller"
PREFIX = EXPERIMENT
PACKAGE_DIR = Path.cwd()
IS_KAGGLE = Path("/kaggle/working").exists()
WORK_DIR = Path("/kaggle/working") if IS_KAGGLE else PACKAGE_DIR / "artifacts"
WORK_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## 2. Runtime, configuration, and SHA helpers

# %%
def find_config() -> Path:
    candidates = [PACKAGE_DIR / "config.yaml", Path("config.yaml")]
    candidates.extend(
        PACKAGE_DIR.parents[index] / "config.yaml"
        for index in range(min(4, len(PACKAGE_DIR.parents)))
    )
    for path in candidates:
        if path.exists() and path.stat().st_size:
            return path
    raise FileNotFoundError("config.yaml")


CONFIG_PATH = find_config()
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())


def cfg(key: str, default: Any = None) -> Any:
    value: Any = CONFIG
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def sha256_file(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as handle:  # type: ignore[arg-type]
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_token(*parts: object, length: int = 20) -> str:
    value = "::".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def stable_well_shard(well: str, shard_count: int) -> int:
    return int(
        stable_token(EXPERIMENT, "stage1_well_shard", well, length=16), 16
    ) % int(shard_count)


def find_raw_root() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        PACKAGE_DIR / "data/raw",
        Path("data/raw"),
    ]
    if Path("/kaggle/input").exists():
        candidates.extend(
            path.parent for path in Path("/kaggle/input").glob("**/sample_submission.csv")
        )
    for root in candidates:
        if (root / "train").is_dir() and (root / "test").is_dir():
            return root
    raise FileNotFoundError("ROGII raw data root")


def find_hidden_like() -> Path:
    filename = str(cfg("validation.hidden_like_assignment_file"))
    candidates = [
        PACKAGE_DIR / "inputs" / filename,
        Path("experiments/exp115_hidden_like_spatial_holdout_from_ppt/artifacts")
        / filename,
    ]
    if Path("/kaggle/input").exists():
        candidates.extend(Path("/kaggle/input").glob(f"**/{filename}"))
    valid = [path for path in candidates if path.exists() and path.stat().st_size]
    if not valid:
        raise FileNotFoundError(filename)
    return max(valid, key=lambda path: path.stat().st_size)


# %% [markdown]
# ## 3. Shard discovery and integrity contract

# %%
def discover_shard_summaries() -> dict[int, tuple[Path, dict[str, Any]]]:
    filename = f"{PREFIX}_summary.json"
    candidates = sorted(Path("/kaggle/input").glob(f"**/{filename}"))
    shards: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in candidates:
        value = json.loads(path.read_text())
        if value.get("experiment") != EXPERIMENT or not value.get("partial_stage1"):
            continue
        shard = value.get("well_shard") or {}
        index = int(shard.get("index", -1))
        if index in shards:
            raise RuntimeError(f"duplicate Stage 1 shard index {index}: {path} and {shards[index][0]}")
        shards[index] = (path, value)
    expected_count = int(cfg("audit.stage1_shard.count"))
    if set(shards) != set(range(expected_count)):
        raise RuntimeError(
            f"expected Stage 1 shards {list(range(expected_count))}, found {sorted(shards)}"
        )
    return shards


def shard_artifact(summary_path: Path, suffix: str) -> Path:
    path = summary_path.parent / f"{PREFIX}_{suffix}"
    if not path.exists() or not path.stat().st_size:
        raise FileNotFoundError(path)
    return path


def read_shard_csvs(
    shards: dict[int, tuple[Path, dict[str, Any]]],
    suffix: str,
    **kwargs: Any,
) -> pd.DataFrame:
    parts = [
        pd.read_csv(shard_artifact(shards[index][0], suffix), **kwargs)
        for index in sorted(shards)
    ]
    return pd.concat(parts, ignore_index=True)


# %% [markdown]
# ## 4. Global metrics and adoption guard

# %%
def rmse(values: np.ndarray, truth: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(truth)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values[valid] - truth[valid]))))


def metric_row(surface: str, frame: pd.DataFrame) -> dict[str, Any]:
    truth = frame["true_tvt"].to_numpy(float)
    base_score = rmse(frame["base_pred"].to_numpy(float), truth)
    controller_score = rmse(frame["controller_pred"].to_numpy(float), truth)
    return {
        "surface": surface,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "base_rmse": base_score,
        "controller_rmse": controller_score,
        "delta_rmse": controller_score - base_score,
    }


def evaluate_controller(
    oof: pd.DataFrame, hidden_like: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = oof.merge(hidden_like, left_on="well", right_on="well_id", how="left")
    rows = [metric_row("overall", frame)]
    for bucket in cfg("validation.distance_buckets"):
        lower = float(bucket["min_md_since"])
        upper = bucket.get("max_md_since")
        mask = frame["md_since"].ge(lower)
        if upper is not None:
            mask &= frame["md_since"].lt(float(upper))
        rows.append(metric_row(str(bucket["name"]), frame[mask]))
    for column, surface in [
        ("verification_like_spatial_role", "hidden_like_spatial"),
        ("verification_like_typewell_purged_role", "hidden_like_typewell_purged"),
    ]:
        rows.append(metric_row(surface, frame[frame[column].eq("valid")]))
    for fold in sorted(frame["diagnostic_fold"].unique()):
        rows.append(metric_row(f"fold_{int(fold)}", frame[frame["diagnostic_fold"].eq(fold)]))
    by_well_rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=True):
        row = metric_row("well", group)
        row["well"] = well
        by_well_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(by_well_rows)


def adoption_guard(metrics: pd.DataFrame, by_well: pd.DataFrame) -> dict[str, Any]:
    delta = metrics.set_index("surface")["delta_rmse"].to_dict()
    improved_folds = sum(
        float(delta.get(f"fold_{fold}", np.inf)) < 0
        for fold in range(int(cfg("validation.diagnostic_folds")))
    )
    worst = float(by_well["delta_rmse"].max()) if len(by_well) else float("inf")
    checks = {
        "global_improved": float(delta.get("overall", np.inf)) < 0,
        "near_nonworse": float(delta.get("000_050", np.inf)) <= 0,
        "longtail_nonworse": float(delta.get("1000_plus", np.inf)) <= 0,
        "hidden_like_spatial_nonworse": float(delta.get("hidden_like_spatial", np.inf)) <= 0,
        "hidden_like_typewell_purged_nonworse": float(
            delta.get("hidden_like_typewell_purged", np.inf)
        )
        <= 0,
        "fold_stability": improved_folds
        >= int(cfg("validation.adoption_guard.min_improved_folds")),
        "worst_well": worst
        <= float(cfg("validation.adoption_guard.max_worst_well_regression")),
    }
    required = {
        "global_improved": bool(
            cfg("validation.adoption_guard.require_global_improvement", True)
        ),
        "near_nonworse": bool(
            cfg("validation.adoption_guard.require_near_nonworse", True)
        ),
        "longtail_nonworse": bool(
            cfg("validation.adoption_guard.require_longtail_nonworse", True)
        ),
        "hidden_like_spatial_nonworse": bool(
            cfg("validation.adoption_guard.require_hidden_like_spatial_nonworse", True)
        ),
        "hidden_like_typewell_purged_nonworse": bool(
            cfg(
                "validation.adoption_guard.require_hidden_like_typewell_purged_nonworse",
                True,
            )
        ),
        "fold_stability": True,
        "worst_well": bool(
            cfg("validation.adoption_guard.require_worst_well", True)
        ),
    }
    required_checks = [name for name, is_required in required.items() if is_required]
    return {
        "checks": checks,
        "required": required,
        "required_checks": required_checks,
        "improved_folds": improved_folds,
        "worst_well_regression": worst,
        "pass": all(checks[name] for name in required_checks),
    }


# %% [markdown]
# ## 5. Aggregate orchestration

# %%
if not IS_KAGGLE and os.environ.get("ROGII_ALLOW_LOCAL", "0") != "1":
    raise RuntimeError(
        "Kaggle Notebook execution is canonical; local execution requires ROGII_ALLOW_LOCAL=1"
    )

started = time.time()
shards = discover_shard_summaries()
summary_values = [shards[index][1] for index in sorted(shards)]
if any(value.get("status") != "stage1_shard_complete" for value in summary_values):
    raise RuntimeError("all Stage 1 shard summaries must be complete")
if any(not value.get("stage0_pass") for value in summary_values):
    raise RuntimeError("a Stage 1 shard failed its technical contract")
if any(value.get("request_errors") for value in summary_values):
    raise RuntimeError("a Stage 1 shard contains request errors")

for key in ("config", "official_candidate_cache_decompressed", "base_oof_decompressed"):
    values = {value["sha256"][key] for value in summary_values}
    if len(values) != 1:
        raise RuntimeError(f"Stage 1 shard SHA mismatch for {key}: {sorted(values)}")

raw_root = find_raw_root()
raw_wells = sorted(
    path.name.split("__horizontal_well.csv")[0]
    for path in (raw_root / "train").glob("*__horizontal_well.csv")
)
shard_count = int(cfg("audit.stage1_shard.count"))
expected_by_shard = {
    index: {well for well in raw_wells if stable_well_shard(well, shard_count) == index}
    for index in range(shard_count)
}

request_manifest = read_shard_csvs(shards, "request_manifest.csv", dtype={"source_well": str})
cut_scores = read_shard_csvs(shards, "cut_scores.csv", dtype={"well": str})
well_report = read_shard_csvs(shards, "well_report.csv", dtype={"well": str})
move_report = read_shard_csvs(shards, "move_report.csv", dtype={"well": str})
oof = read_shard_csvs(
    shards,
    "oof.csv.gz",
    dtype={"id": str, "well": str, "selected_candidate": str},
)

for index, (summary_path, _summary) in sorted(shards.items()):
    shard_wells = set(
        pd.read_csv(
            shard_artifact(summary_path, "well_report.csv"), usecols=["well"], dtype={"well": str}
        )["well"]
    )
    if shard_wells != expected_by_shard[index]:
        raise RuntimeError(
            f"Stage 1 shard {index} well coverage mismatch: "
            f"expected={len(expected_by_shard[index])} actual={len(shard_wells)}"
        )

if well_report["well"].duplicated().any():
    raise RuntimeError("duplicate well rows across Stage 1 shards")
if set(well_report["well"]) != set(raw_wells):
    raise RuntimeError("Stage 1 well union does not cover all raw train wells")
if oof["id"].duplicated().any():
    raise RuntimeError("duplicate OOF ids across Stage 1 shards")
if set(oof["well"]) != set(raw_wells):
    raise RuntimeError("Stage 1 OOF union does not cover all raw train wells")

hidden_like = pd.read_csv(find_hidden_like(), dtype={"well_id": str})
metrics, by_well = evaluate_controller(oof, hidden_like)
guard = adoption_guard(metrics, by_well)

scored_ok = well_report["status"].eq("ok")
scored_fraction = float(scored_ok.mean()) if len(scored_ok) else 0.0
three_cut_fraction = (
    float(
        well_report.loc[scored_ok, "cuts"]
        .eq(len(cfg("model.prefix.cut_fractions")))
        .mean()
    )
    if scored_ok.any()
    else 0.0
)
min_candidate_count = (
    int(well_report.loc[scored_ok, "candidate_count"].min()) if scored_ok.any() else 0
)
nonfinite_predictions = int(
    (~np.isfinite(oof["controller_pred"].to_numpy(float))).sum()
)
max_alpha = float(oof["controller_alpha"].max()) if len(oof) else 0.0
max_move = float(oof["controller_move"].abs().max()) if len(oof) else 0.0
stage1_checks = {
    "four_shards_complete": len(shards) == shard_count,
    "all_raw_wells_covered": set(well_report["well"]) == set(raw_wells),
    "request_errors_zero": sum(len(value["request_errors"]) for value in summary_values)
    == 0,
    "scored_well_fraction": scored_fraction
    >= float(cfg("validation.stage0_contract.min_scored_well_fraction")),
    "three_cuts": three_cut_fraction == 1.0,
    "candidate_count": min_candidate_count
    >= int(cfg("validation.stage0_contract.min_candidates_per_well")),
    "nonfinite_zero": nonfinite_predictions == 0,
    "alpha_cap": max_alpha <= float(cfg("model.controller.alpha_cap")) + 1e-9,
    "move_cap": max_move <= float(cfg("model.controller.clip_max")) + 1e-9,
}
stage1_pass = all(stage1_checks.values())
adoption_supported = bool(stage1_pass and guard["pass"])


# %% [markdown]
# ## 6. Metrics and generated artifacts

# %%
paths = {
    "request_manifest": WORK_DIR / f"{PREFIX}_request_manifest.csv",
    "cut_scores": WORK_DIR / f"{PREFIX}_cut_scores.csv",
    "well_report": WORK_DIR / f"{PREFIX}_well_report.csv",
    "move_report": WORK_DIR / f"{PREFIX}_move_report.csv",
    "metrics": WORK_DIR / f"{PREFIX}_metrics.csv",
    "by_well": WORK_DIR / f"{PREFIX}_by_well.csv",
    "oof": WORK_DIR / f"{PREFIX}_oof.csv.gz",
    "summary": WORK_DIR / f"{PREFIX}_summary.json",
    "plot": WORK_DIR / f"{PREFIX}_distance_metrics.png",
}
request_manifest.to_csv(paths["request_manifest"], index=False)
cut_scores.to_csv(paths["cut_scores"], index=False)
well_report.to_csv(paths["well_report"], index=False)
move_report.to_csv(paths["move_report"], index=False)
metrics.to_csv(paths["metrics"], index=False)
by_well.to_csv(paths["by_well"], index=False)
oof.to_csv(paths["oof"], index=False, compression="gzip")

plot_frame = metrics[
    metrics["surface"].isin(
        [item["name"] for item in cfg("validation.distance_buckets")]
    )
].copy()
ax = plot_frame.set_index("surface")[["base_rmse", "controller_rmse"]].plot(
    kind="bar", figsize=(11, 4), title="Prefix-verified bounded controller — Stage 1"
)
ax.set_ylabel("RMSE")
plt.tight_layout()
plt.savefig(paths["plot"], dpi=140)
plt.close()

summary = {
    "experiment": EXPERIMENT,
    "status": "stage1_aggregate_complete",
    "active_mode": "stage1_full_audit",
    "route": cfg("experiment.route"),
    "runtime_seconds": float(time.time() - started),
    "selected_wells": len(raw_wells),
    "request_count": int(len(request_manifest)),
    "request_errors": [],
    "shards": {
        "count": shard_count,
        "indices": sorted(shards),
        "policy": cfg("audit.stage1_shard.policy"),
        "runtime_seconds": {
            str(index): float(shards[index][1]["runtime_seconds"])
            for index in sorted(shards)
        },
    },
    "scored_well_fraction": scored_fraction,
    "three_cut_fraction": three_cut_fraction,
    "min_candidate_count": min_candidate_count,
    "max_alpha": max_alpha,
    "max_abs_move": max_move,
    "stage1_checks": stage1_checks,
    "stage1_pass": stage1_pass,
    "adoption_guard": guard,
    "adoption_supported": adoption_supported,
    "inference_allowed": adoption_supported,
    "training_cost": {
        "variants": 1,
        "execution_shards": 4,
        "model_configs": 0,
        "folds": 0,
        "boosters": 0,
        "parent_control_retraining": False,
    },
    "sha256": {
        "config": sha256_file(CONFIG_PATH),
        "official_candidate_cache_decompressed": summary_values[0]["sha256"][
            "official_candidate_cache_decompressed"
        ],
        "base_oof_decompressed": summary_values[0]["sha256"][
            "base_oof_decompressed"
        ],
        "input_shard_summaries": {
            str(index): sha256_file(shards[index][0]) for index in sorted(shards)
        },
        "request_manifest": sha256_file(paths["request_manifest"]),
        "cut_scores": sha256_file(paths["cut_scores"]),
        "well_report": sha256_file(paths["well_report"]),
        "controller_oof_decompressed": sha256_file(paths["oof"], decompressed=True),
    },
}
paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))

print(metrics.to_string(index=False))
print(json.dumps(summary, indent=2, sort_keys=True))
