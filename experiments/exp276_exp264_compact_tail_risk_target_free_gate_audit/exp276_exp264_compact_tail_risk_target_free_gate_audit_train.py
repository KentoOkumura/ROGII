# %% [markdown]
# # exp276 exp264 compact tail-risk target-free gate audit — train
#
# hidden-safeなcorrected exp264 Stage C v6の25 compact partitionとStage D v3保存OOFを
# 固定入力にし、selector score
# dispersion、candidate divergence、top1-anchor差、confidence coverage、raw geometry/contextを
# well単位へ集約する。risk rankとq70/q80/q90は各downstream outer-train wellsだけで決め、
# outer-validのtrue TVT/error/deltaはriskを凍結した後のreadoutにだけ使う。

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Compute, leakage, and decision contract
# 3. Fixed exp264 inputs and SHA checks
# 4. Compact and raw-context aggregation helpers
# 5. Outer-fold target-free risk construction
# 6. Stage D tail-risk and fallback-gate readout
# 7. Diagnostics, metrics, and reproducibility evidence

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from IPython.display import display

EXPERIMENT_NAME = "exp276_exp264_compact_tail_risk_target_free_gate_audit"
PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path("experiments") / EXPERIMENT_NAME
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
OUTPUT_DIR = (
    Path("/kaggle/working/artifacts")
    if Path("/kaggle/working").exists()
    else PACKAGE_DIR / "artifacts"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXECUTE_NOTEBOOK = os.environ.get("EXP276_IMPORT_ONLY", "0") != "1"
STARTED_AT = time.time()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, ensure_ascii=False) + "\n")


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        to_jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def logical_frame_sha256(frame: pd.DataFrame, sort_columns: Sequence[str]) -> str:
    ordered = frame.sort_values(list(sort_columns), kind="stable").reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(ordered.columns), separators=(",", ":")).encode())
    digest.update(json.dumps([str(dtype) for dtype in ordered.dtypes]).encode())
    hashed = pd.util.hash_pandas_object(ordered, index=False, categorize=True).to_numpy(np.uint64)
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def rmse(target: Iterable[float], prediction: Iterable[float]) -> float:
    truth = np.asarray(target, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(truth - pred))))


def quantile_label(value: float) -> str:
    return f"q{int(round(100 * float(value))):02d}"


def resolve_file(patterns: Sequence[str], expected_sha256: str) -> Path:
    candidates: list[Path] = []
    for pattern in patterns:
        direct = Path(str(pattern))
        if direct.exists() and direct.is_file():
            candidates.append(direct)
    filenames = sorted({Path(str(pattern)).name for pattern in patterns})
    for root in (Path("/kaggle/input"), Path("/tmp"), PACKAGE_DIR, Path.cwd()):
        if not root.exists():
            continue
        for filename in filenames:
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
        if actual == expected_sha256:
            return path
    raise FileNotFoundError(
        f"No file matched expected SHA={expected_sha256}; checked={checked[:30]}"
    )


def resolve_raw_train_dir(patterns: Sequence[str]) -> Path:
    for pattern in patterns:
        path = Path(str(pattern))
        if path.is_dir() and next(path.glob("*__horizontal_well.csv"), None) is not None:
            return path
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for path in sorted(kaggle_input.rglob("train")):
            if path.is_dir() and next(path.glob("*__horizontal_well.csv"), None) is not None:
                return path
    raise FileNotFoundError("Could not resolve competition raw train directory")


def read_parquet_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(path)
    available = set(parquet_file.schema_arrow.names)
    missing = sorted(set(columns).difference(available))
    if missing:
        raise ValueError(f"partition is missing required columns: {missing}")
    # Read the physical file directly. ``pq.read_table(path)`` routes through the
    # dataset API and interprets hive-style parent directories such as
    # ``downstream_outer_fold=0`` as virtual partition columns. Stage C stores
    # the same column in the parquet file, so the inferred dictionary type can
    # conflict with the physical int8 field.
    return parquet_file.read(columns=list(columns)).to_pandas()


# %% [markdown]
# ## 2. Compute, leakage, and decision contract
#
# 1 audit variant / 0 model config / 0 trained fold / 0 booster。q70/q80/q90はすべて
# decision-bearingであり、結果を見た単一quantileや単一familyの救済は禁止する。

# %%
if EXECUTE_NOTEBOOK:
    compute_contract = {
        "experiment": CONFIG["experiment"]["name"],
        "route": CONFIG["experiment"]["route"],
        "stage": CONFIG["execution"]["stage"],
        "variants": int(CONFIG["execution"]["variants"]),
        "lightgbm_configs": int(CONFIG["execution"]["lightgbm_configs"]),
        "folds_trained": int(CONFIG["execution"]["folds_trained"]),
        "total_boosters": int(CONFIG["execution"]["total_boosters"]),
        "parent_control_retraining": bool(CONFIG["execution"]["parent_control_retraining"]),
        "gpu": bool(CONFIG["runtime"]["kaggle"]["enable_gpu"]),
        "internet": bool(CONFIG["runtime"]["kaggle"]["enable_internet"]),
        "inference": bool(CONFIG["execution"]["inference_enabled"]),
        "submission": bool(CONFIG["execution"]["submission_enabled"]),
    }
    display(compute_contract)
    assert compute_contract == {
        "experiment": EXPERIMENT_NAME,
        "route": "ml_model",
        "stage": "target_free_tail_risk_gate_audit",
        "variants": 1,
        "lightgbm_configs": 0,
        "folds_trained": 0,
        "total_boosters": 0,
        "parent_control_retraining": False,
        "gpu": False,
        "internet": False,
        "inference": False,
        "submission": False,
    }
    configured_quantiles = [float(value) for value in CONFIG["risk_features"]["quantiles"]]
    guard_quantiles = [
        float(value) for value in CONFIG["guards"]["tail_risk"]["decision_quantiles"]
    ]
    assert configured_quantiles == [0.70, 0.80, 0.90]
    assert configured_quantiles == guard_quantiles
    assert CONFIG["guards"]["tail_risk"]["all_quantiles_must_pass"] is True
    assert CONFIG["guards"]["tail_risk"]["family_or_single_quantile_cannot_rescue"] is True
    if not Path("/kaggle/input").exists() or not Path("/kaggle/working").exists():
        raise RuntimeError("The first full exp276 audit must run on Kaggle CPU.")
    if not bool(CONFIG["execution"]["run_approved"]):
        raise RuntimeError(
            "Kaggle audit is not approved. Set execution.run_approved=true only after approval."
        )
    print("Leakage contract")
    for rule in CONFIG["validation"]["leakage_policy"]:
        print("-", rule)


# %% [markdown]
# ## 3. Fixed exp264 inputs and SHA checks
#
# Stage C manifest/schema/25 partition、Stage D OOFをbyte SHAでfail-closed照合する。
# partitionはmanifest相対pathと個別SHAを正とし、latest kernel outputを無条件には採用しない。

# %%
def load_stage_c_contract(config: Mapping[str, Any]) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    data = config["data"]
    manifest_path = resolve_file(
        data["stage_c_manifest_patterns"], data["stage_c_expected_manifest_sha256"]
    )
    root = manifest_path.parent
    partition_manifest_path = root / str(data["stage_c_partition_manifest_filename"])
    schema_path = root / str(data["stage_c_schema_filename"])
    if sha256_file(partition_manifest_path) != data["stage_c_expected_partition_manifest_sha256"]:
        raise ValueError("Stage C partition manifest SHA mismatch")
    if sha256_file(schema_path) != data["stage_c_expected_schema_file_sha256"]:
        raise ValueError("Stage C compact schema file SHA mismatch")
    manifest = json.loads(manifest_path.read_text())
    schema = json.loads(schema_path.read_text())
    if manifest["compact_meta_schema_sha256"] != data[
        "stage_c_expected_schema_logical_sha256"
    ]:
        raise ValueError("Stage C manifest compact logical SHA mismatch")
    if schema["compact_meta_schema_sha256"] != data[
        "stage_c_expected_schema_logical_sha256"
    ]:
        raise ValueError("Stage C schema logical SHA mismatch")
    partitions = pd.read_csv(partition_manifest_path)
    technical = config["guards"]["technical"]
    if len(partitions) != int(technical["expected_stage_c_partitions"]):
        raise ValueError("unexpected Stage C partition count")
    if int(partitions["rows"].sum()) != int(technical["expected_compact_rows"]):
        raise ValueError("unexpected Stage C compact row total")
    if int(manifest["partition_count"]) != len(partitions):
        raise ValueError("Stage C JSON/CSV partition count mismatch")
    if int(manifest["rows"]) != int(partitions["rows"].sum()):
        raise ValueError("Stage C JSON/CSV row total mismatch")
    return root, partitions, schema


def selected_compact_columns(config: Mapping[str, Any]) -> list[str]:
    columns: list[str] = []
    for family, spec in config["risk_features"]["families"].items():
        if family == "geometry_context":
            continue
        columns.extend(str(column) for column in spec["compact_columns"])
    if len(columns) != len(set(columns)):
        raise ValueError("risk compact column list contains duplicates")
    forbidden = ("target", "actual_", "delta_rmse", "true_tvt")
    invalid = [
        column for column in columns if any(token in column.lower() for token in forbidden)
    ]
    if invalid:
        raise ValueError(f"risk compact schema contains forbidden label columns: {invalid}")
    return columns


if EXECUTE_NOTEBOOK:
    stage_c_root, partition_manifest, compact_schema = load_stage_c_contract(CONFIG)
    stage_d_oof_path = resolve_file(
        CONFIG["data"]["stage_d_oof_patterns"],
        CONFIG["data"]["stage_d_expected_oof_sha256"],
    )
    raw_train_dir = resolve_raw_train_dir(CONFIG["data"]["raw_train_dir_patterns"])
    compact_columns = selected_compact_columns(CONFIG)
    missing_from_schema = sorted(set(compact_columns).difference(compact_schema["features"]))
    if missing_from_schema:
        raise ValueError(f"risk columns missing from frozen compact schema: {missing_from_schema}")
    input_overview = {
        "stage_c_root": str(stage_c_root),
        "stage_c_manifest_sha256": CONFIG["data"]["stage_c_expected_manifest_sha256"],
        "stage_c_partition_count": int(len(partition_manifest)),
        "stage_c_rows": int(partition_manifest["rows"].sum()),
        "compact_schema_features": int(len(compact_schema["features"])),
        "selected_compact_risk_columns": len(compact_columns),
        "stage_d_oof_path": str(stage_d_oof_path),
        "stage_d_oof_sha256": sha256_file(stage_d_oof_path),
        "raw_train_dir": str(raw_train_dir),
    }
    display(input_overview)
    display(partition_manifest)


# %% [markdown]
# ## 4. Compact and raw-context aggregation helpers
#
# compact row signalは先頭128、先頭512、全評価区間でmean/p90へ固定集約する。
# raw geometryはpartitionの`well_row_idx`を正とし、`TVT`/`TVT_input`値を読まずに
# X/Y/Z/GRのanchor変位、step、missingnessを同じscopeへ集約する。

# %%
KEY_COLUMNS = ["id", "well", "well_row_idx", "outer_fold", "md_since"]


def scope_specs(config: Mapping[str, Any]) -> list[tuple[str, int | None]]:
    return [
        (
            str(item["name"]),
            None if item.get("max_rows") is None else int(item["max_rows"]),
        )
        for item in config["risk_features"]["scopes"]
    ]


def finite_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return {"mean": math.nan, "p90": math.nan, "end": math.nan}
    return {
        "mean": float(np.mean(finite)),
        "p90": float(np.quantile(finite, 0.90)),
        "end": float(finite[-1]),
    }


def compact_signal_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    candidate_count = int(config["risk_features"]["candidate_count"])
    for family, family_spec in config["risk_features"]["families"].items():
        if family == "geometry_context":
            continue
        transform = str(family_spec.get("transform", "identity"))
        for column in family_spec["compact_columns"]:
            specs.append(
                {
                    "family": str(family),
                    "column": str(column),
                    "transform": transform,
                    "candidate_count": candidate_count,
                    "aggregations": [str(item) for item in family_spec["aggregations"]],
                }
            )
    return specs


def transform_signal(values: pd.Series, spec: Mapping[str, Any]) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    transform = str(spec["transform"])
    if transform == "identity":
        return array
    if transform == "absolute":
        return np.abs(array)
    if transform == "candidate_count_minus_value":
        return float(spec["candidate_count"]) - array
    raise ValueError(f"unknown risk transform: {transform}")


def aggregate_compact_partition(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    required = set(KEY_COLUMNS + selected_compact_columns(config))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"compact partition missing columns: {missing}")
    ordered = frame.sort_values(["well", "well_row_idx"], kind="stable").reset_index(drop=True)
    if ordered["id"].astype(str).duplicated().any():
        raise ValueError("compact partition contains duplicate ids")
    ordered["scope_row"] = ordered.groupby("well", sort=False).cumcount()
    output = pd.DataFrame({"well": sorted(ordered["well"].astype(str).unique())})
    signal_specs = compact_signal_specs(config)
    signal_values: dict[str, Any] = {"well": ordered["well"].astype(str).to_numpy()}
    signal_metadata: list[tuple[str, Mapping[str, Any]]] = []
    for spec in signal_specs:
        signal_name = str(spec["column"]).replace("selector__", "")
        base_name = f"{spec['family']}__{signal_name}"
        signal_values[base_name] = transform_signal(ordered[spec["column"]], spec)
        signal_metadata.append((base_name, spec))
    signals = pd.DataFrame(signal_values)
    for scope_name, max_rows in scope_specs(config):
        mask = (
            np.ones(len(ordered), dtype=bool)
            if max_rows is None
            else ordered["scope_row"].lt(max_rows).to_numpy()
        )
        selected = signals.loc[mask]
        if selected.empty:
            raise ValueError(f"scope {scope_name} has no rows")
        signal_columns = [name for name, _ in signal_metadata]
        grouped = selected.groupby("well", sort=True)[signal_columns]
        means = grouped.mean()
        p90 = grouped.quantile(0.90)
        scope_frame = pd.DataFrame(index=means.index)
        for base_name, spec in signal_metadata:
            prefix = f"{spec['family']}__{scope_name}__{base_name.split('__', 1)[1]}"
            if "mean" in spec["aggregations"]:
                scope_frame[f"{prefix}__mean"] = means[base_name]
            if "p90" in spec["aggregations"]:
                scope_frame[f"{prefix}__p90"] = p90[base_name]
        scope_frame = scope_frame.reset_index()
        output = output.merge(scope_frame, on="well", how="left", validate="one_to_one")
    return output


def build_geometry_features(
    keys: pd.DataFrame,
    raw_train_dir: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    required = {"well", "well_row_idx"}
    if not required.issubset(keys.columns):
        raise ValueError("geometry key frame lacks well/well_row_idx")
    if keys[["well", "well_row_idx"]].duplicated().any():
        raise ValueError("geometry key frame contains duplicate well-row keys")
    rows: list[dict[str, Any]] = []
    raw_manifest: list[dict[str, Any]] = []
    scopes = scope_specs(config)
    for well, well_keys in keys.groupby("well", sort=True):
        well_name = str(well)
        raw_path = raw_train_dir / f"{well_name}__horizontal_well.csv"
        if not raw_path.exists() or raw_path.stat().st_size == 0:
            raise FileNotFoundError(raw_path)
        raw_manifest.append(
            {"well": well_name, "path": str(raw_path), "sha256": sha256_file(raw_path)}
        )
        required_raw_columns = {"MD", "X", "Y", "Z", "GR"}
        raw = pd.read_csv(raw_path, usecols=lambda column: column in required_raw_columns)
        missing_raw_columns = sorted(required_raw_columns.difference(raw.columns))
        if missing_raw_columns:
            raise ValueError(
                f"raw horizontal file lacks target-free context columns for well={well_name}: "
                f"{missing_raw_columns}"
            )
        positions = (
            pd.to_numeric(well_keys["well_row_idx"], errors="raise")
            .astype(np.int64)
            .sort_values(kind="stable")
            .to_numpy()
        )
        if len(positions) == 0 or positions.min() < 0 or positions.max() >= len(raw):
            raise ValueError(f"raw row-index contract failed for well={well_name}")
        anchor_position = max(int(positions[0]) - 1, 0)
        selected = raw.iloc[positions].reset_index(drop=True)
        anchor = raw.iloc[anchor_position]
        signals: dict[str, np.ndarray] = {}
        for column in ("MD", "X", "Y", "Z", "GR"):
            values = pd.to_numeric(selected.get(column), errors="coerce").to_numpy(np.float64)
            anchor_value = float(pd.to_numeric(anchor.get(column), errors="coerce"))
            signals[f"abs_anchor_delta_{column.lower()}"] = np.abs(values - anchor_value)
            previous = np.concatenate([[anchor_value], values[:-1]])
            signals[f"abs_step_{column.lower()}"] = np.abs(values - previous)
        gr_values = pd.to_numeric(selected.get("GR"), errors="coerce").to_numpy(np.float64)
        signals["gr_missing"] = (~np.isfinite(gr_values)).astype(np.float64)
        output: dict[str, Any] = {
            "well": well_name,
            "geometry_context__full__eval_length": float(len(positions)),
        }
        for scope_name, max_rows in scopes:
            stop = len(positions) if max_rows is None else min(len(positions), max_rows)
            for signal_name, values in signals.items():
                summary = finite_summary(values[:stop])
                for aggregate in ("mean", "p90", "end"):
                    output[
                        f"geometry_context__{scope_name}__{signal_name}__{aggregate}"
                    ] = summary[aggregate]
        rows.append(output)
    geometry = pd.DataFrame(rows).sort_values("well", kind="stable").reset_index(drop=True)
    return geometry, raw_manifest


def partition_path(stage_c_root: Path, relative_path: str) -> Path:
    path = stage_c_root / str(relative_path)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def aggregate_all_partitions(
    *,
    stage_c_root: Path,
    partition_manifest: pd.DataFrame,
    raw_train_dir: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    required_read_columns = KEY_COLUMNS + selected_compact_columns(config)
    aggregate_parts: list[pd.DataFrame] = []
    geometry_parts: list[pd.DataFrame] = []
    partition_evidence: list[dict[str, Any]] = []
    raw_evidence: list[dict[str, Any]] = []
    valid_manifest = partition_manifest[partition_manifest["role"].eq("valid")].copy()
    train_manifest = partition_manifest[partition_manifest["role"].eq("train")].copy()
    if len(valid_manifest) != int(config["validation"]["n_folds"]):
        raise ValueError("Stage C must contain one valid partition per downstream fold")

    # First pass: every well appears exactly once across the five valid partitions.
    for row in valid_manifest.sort_values("downstream_outer_fold").itertuples(index=False):
        path = partition_path(stage_c_root, str(row.path))
        actual_sha = sha256_file(path)
        if actual_sha != str(row.sha256):
            raise ValueError(f"Stage C partition SHA mismatch: {row.path}")
        frame = read_parquet_columns(path, required_read_columns)
        if len(frame) != int(row.rows) or frame["well"].nunique() != int(row.wells):
            raise ValueError(f"Stage C partition row/well mismatch: {row.path}")
        if set(pd.to_numeric(frame["outer_fold"], errors="raise").astype(int)) != {
            int(row.source_outer_fold)
        }:
            raise ValueError("valid partition source fold/key fold mismatch")
        compact = aggregate_compact_partition(frame, config)
        geometry, raw_manifest = build_geometry_features(
            frame[["well", "well_row_idx"]], raw_train_dir, config
        )
        compact = compact.merge(geometry, on="well", how="left", validate="one_to_one")
        compact["downstream_outer_fold"] = int(row.downstream_outer_fold)
        compact["role"] = "valid"
        compact["source_outer_fold"] = int(row.source_outer_fold)
        aggregate_parts.append(compact)
        geometry_parts.append(geometry)
        raw_evidence.extend(raw_manifest)
        partition_evidence.append(
            {
                "downstream_outer_fold": int(row.downstream_outer_fold),
                "role": "valid",
                "source_outer_fold": int(row.source_outer_fold),
                "rows": int(len(frame)),
                "wells": int(frame["well"].nunique()),
                "path": str(path),
                "sha256": actual_sha,
            }
        )
        del frame, compact

    geometry_all = pd.concat(geometry_parts, ignore_index=True)
    if geometry_all["well"].duplicated().any():
        raise ValueError("valid partitions assign a well more than once")
    expected_wells = int(config["guards"]["technical"]["expected_wells"])
    if len(geometry_all) != expected_wells:
        raise ValueError("valid partition union does not cover all wells")

    # Second pass: train partitions reuse the geometry frozen from their own valid partition.
    for row in train_manifest.sort_values(
        ["downstream_outer_fold", "source_outer_fold"], kind="stable"
    ).itertuples(index=False):
        path = partition_path(stage_c_root, str(row.path))
        actual_sha = sha256_file(path)
        if actual_sha != str(row.sha256):
            raise ValueError(f"Stage C partition SHA mismatch: {row.path}")
        frame = read_parquet_columns(path, required_read_columns)
        if len(frame) != int(row.rows) or frame["well"].nunique() != int(row.wells):
            raise ValueError(f"Stage C partition row/well mismatch: {row.path}")
        compact = aggregate_compact_partition(frame, config)
        train_geometry = geometry_all[geometry_all["well"].isin(compact["well"])].copy()
        compact = compact.merge(train_geometry, on="well", how="left", validate="one_to_one")
        compact["downstream_outer_fold"] = int(row.downstream_outer_fold)
        compact["role"] = "train"
        compact["source_outer_fold"] = int(row.source_outer_fold)
        aggregate_parts.append(compact)
        partition_evidence.append(
            {
                "downstream_outer_fold": int(row.downstream_outer_fold),
                "role": "train",
                "source_outer_fold": int(row.source_outer_fold),
                "rows": int(len(frame)),
                "wells": int(frame["well"].nunique()),
                "path": str(path),
                "sha256": actual_sha,
            }
        )
        del frame, compact

    aggregates = pd.concat(aggregate_parts, ignore_index=True)
    aggregates = aggregates.sort_values(
        ["downstream_outer_fold", "role", "source_outer_fold", "well"], kind="stable"
    ).reset_index(drop=True)
    raw_evidence = sorted(raw_evidence, key=lambda item: item["well"])
    if len(raw_evidence) != expected_wells:
        raise ValueError("raw geometry manifest must contain exactly one file per well")
    return aggregates, partition_evidence, raw_evidence


# %% [markdown]
# ## 5. Outer-fold target-free risk construction
#
# 各risk featureはouter-train medianでmissingを補い、同じouter-train empirical CDFで
# percentile化する。family内を等重み、5 familyを等重みにし、valid risk thresholdは
# train riskのq70/q80/q90だけから決める。

# %%
def risk_feature_columns(frame: pd.DataFrame) -> list[str]:
    families = (
        "score_dispersion__",
        "candidate_divergence__",
        "top1_anchor_distance__",
        "confidence_coverage__",
        "geometry_context__",
    )
    columns = [column for column in frame.columns if column.startswith(families)]
    forbidden = ("target", "actual", "error_label", "delta_rmse", "true_tvt")
    invalid = [column for column in columns if any(token in column for token in forbidden)]
    if invalid:
        raise ValueError(f"risk feature columns contain forbidden labels: {invalid}")
    return sorted(columns)


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    val = np.asarray(values, dtype=np.float64)
    if len(ref) == 0 or not np.isfinite(ref).all() or not np.isfinite(val).all():
        raise ValueError("empirical percentile requires finite non-empty arrays")
    ordered = np.sort(ref, kind="stable")
    return np.searchsorted(ordered, val, side="right").astype(np.float64) / len(ordered)


def fit_target_free_risk(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    quantiles: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    forbidden_exact = {
        "tvt",
        "target",
        "actual_tvt",
        "actual_error",
        "true_tvt",
        "delta_rmse_addonly_minus_control",
    }
    forbidden_columns = sorted(
        {
            str(column)
            for frame in (train, valid)
            for column in frame.columns
            if str(column).lower() in forbidden_exact
            or str(column).lower().startswith("actual_")
            or "delta_rmse" in str(column).lower()
        }
    )
    if forbidden_columns:
        raise ValueError(
            f"label/outcome columns cannot enter target-free risk fit: {forbidden_columns}"
        )
    if train["well"].duplicated().any() or valid["well"].duplicated().any():
        raise ValueError("risk fit expects one row per well in each role")
    overlap = set(train["well"].astype(str)).intersection(valid["well"].astype(str))
    if overlap:
        raise ValueError(f"outer train/valid well overlap: {sorted(overlap)[:10]}")
    feature_columns = risk_feature_columns(train)
    if feature_columns != risk_feature_columns(valid):
        raise ValueError("outer train/valid risk feature schema mismatch")
    family_names = sorted({column.split("__", 1)[0] for column in feature_columns})
    if family_names != [
        "candidate_divergence",
        "confidence_coverage",
        "geometry_context",
        "score_dispersion",
        "top1_anchor_distance",
    ]:
        raise ValueError(f"unexpected risk families: {family_names}")
    train_scores = train[["well"]].copy()
    valid_scores = valid[["well"]].copy()
    feature_preprocessors: dict[str, Any] = {}
    family_columns: dict[str, list[str]] = {family: [] for family in family_names}
    for column in feature_columns:
        train_values = pd.to_numeric(train[column], errors="coerce").to_numpy(np.float64)
        valid_values = pd.to_numeric(valid[column], errors="coerce").to_numpy(np.float64)
        finite_train = train_values[np.isfinite(train_values)]
        if len(finite_train) == 0:
            raise ValueError(f"risk feature is all-missing in outer train: {column}")
        median = float(np.median(finite_train))
        train_filled = np.where(np.isfinite(train_values), train_values, median)
        valid_filled = np.where(np.isfinite(valid_values), valid_values, median)
        rank_name = f"rank__{column}"
        train_scores[rank_name] = empirical_percentile(train_filled, train_filled)
        valid_scores[rank_name] = empirical_percentile(train_filled, valid_filled)
        family = column.split("__", 1)[0]
        family_columns[family].append(rank_name)
        feature_preprocessors[column] = {
            "outer_train_median": median,
            "outer_train_rows": int(len(train_filled)),
            "outer_train_sorted_values_sha256": hashlib.sha256(
                np.sort(train_filled, kind="stable").tobytes()
            ).hexdigest(),
        }
    family_score_columns: list[str] = []
    for family in family_names:
        family_score = f"risk_family__{family}"
        train_scores[family_score] = train_scores[family_columns[family]].mean(axis=1)
        valid_scores[family_score] = valid_scores[family_columns[family]].mean(axis=1)
        family_score_columns.append(family_score)
    train_scores["risk_score"] = train_scores[family_score_columns].mean(axis=1)
    valid_scores["risk_score"] = valid_scores[family_score_columns].mean(axis=1)
    thresholds: dict[str, float] = {}
    for quantile in quantiles:
        label = quantile_label(quantile)
        threshold = float(np.quantile(train_scores["risk_score"], quantile))
        thresholds[label] = threshold
        train_scores[f"risk_{label}"] = train_scores["risk_score"].ge(threshold)
        valid_scores[f"risk_{label}"] = valid_scores["risk_score"].ge(threshold)
    preprocessor = {
        "feature_columns": feature_columns,
        "family_columns": family_columns,
        "family_score_columns": family_score_columns,
        "feature_preprocessors": feature_preprocessors,
        "thresholds": thresholds,
        "fit_wells": sorted(train["well"].astype(str).tolist()),
        "valid_wells": sorted(valid["well"].astype(str).tolist()),
    }
    return train_scores, valid_scores, preprocessor


def build_outer_fold_risk(
    aggregates: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    quantiles = [float(value) for value in config["risk_features"]["quantiles"]]
    feature_outputs: list[pd.DataFrame] = []
    valid_score_outputs: list[pd.DataFrame] = []
    preprocessors: list[dict[str, Any]] = []
    expected_folds = int(config["guards"]["technical"]["expected_folds"])
    for outer_fold in range(expected_folds):
        fold = aggregates[aggregates["downstream_outer_fold"].eq(outer_fold)].copy()
        train = fold[fold["role"].eq("train")].copy()
        valid = fold[fold["role"].eq("valid")].copy()
        if train["well"].duplicated().any() or valid["well"].duplicated().any():
            raise ValueError("partition aggregation did not produce unique fold-role wells")
        train_scores, valid_scores, preprocessor = fit_target_free_risk(
            train, valid, quantiles
        )
        train_out = train.merge(train_scores, on="well", how="left", validate="one_to_one")
        valid_out = valid.merge(valid_scores, on="well", how="left", validate="one_to_one")
        feature_outputs.extend([train_out, valid_out])
        valid_score_output = valid_scores.copy()
        valid_score_output["downstream_outer_fold"] = outer_fold
        valid_score_output["role"] = "valid"
        valid_score_outputs.append(valid_score_output)
        preprocessor["downstream_outer_fold"] = outer_fold
        preprocessors.append(preprocessor)
    all_features = pd.concat(feature_outputs, ignore_index=True)
    valid_scores = pd.concat(valid_score_outputs, ignore_index=True)
    if valid_scores["well"].duplicated().any():
        raise ValueError("outer-valid risk output must cover each well exactly once")
    if len(valid_scores) != int(config["guards"]["technical"]["expected_wells"]):
        raise ValueError("outer-valid risk output has unexpected well count")
    return all_features, valid_scores, preprocessors


if EXECUTE_NOTEBOOK:
    risk_aggregates, partition_evidence, raw_evidence = aggregate_all_partitions(
        stage_c_root=stage_c_root,
        partition_manifest=partition_manifest,
        raw_train_dir=raw_train_dir,
        config=CONFIG,
    )
    risk_feature_frame, well_risk_scores, risk_preprocessors = build_outer_fold_risk(
        risk_aggregates, CONFIG
    )
    display(
        well_risk_scores.groupby("downstream_outer_fold")
        .agg(wells=("well", "size"), risk_mean=("risk_score", "mean"), risk_std=("risk_score", "std"))
    )
    display(well_risk_scores.head(20))


# %% [markdown]
# ## 6. Stage D tail-risk and fallback-gate readout
#
# risk score/thresholdを凍結した後に初めてStage D target/predictionを接続する。risk wellは
# matched controlへwell全体をfallbackし、それ以外はcompact add-onlyを維持する。

# %%
OOF_COLUMNS = [
    "id",
    "well",
    "outer_fold",
    "actual_tvt",
    "matched_control__lgb_mean__pred_tvt",
    "selector_compact_addonly__lgb_mean__pred_tvt",
]
CONTROL_COLUMN = "matched_control__lgb_mean__pred_tvt"
ADDONLY_COLUMN = "selector_compact_addonly__lgb_mean__pred_tvt"


def by_well_stage_d_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    frame = oof.copy()
    frame["control_sq_error"] = np.square(
        frame[CONTROL_COLUMN].to_numpy(np.float64) - frame["actual_tvt"].to_numpy(np.float64)
    )
    frame["addonly_sq_error"] = np.square(
        frame[ADDONLY_COLUMN].to_numpy(np.float64) - frame["actual_tvt"].to_numpy(np.float64)
    )
    grouped = (
        frame.groupby(["well", "outer_fold"], sort=True)
        .agg(
            rows=("id", "size"),
            control_mse=("control_sq_error", "mean"),
            addonly_mse=("addonly_sq_error", "mean"),
        )
        .reset_index()
    )
    grouped["matched_control_rmse"] = np.sqrt(grouped.pop("control_mse"))
    grouped["selector_compact_addonly_rmse"] = np.sqrt(grouped.pop("addonly_mse"))
    grouped["delta_rmse_addonly_minus_control"] = (
        grouped["selector_compact_addonly_rmse"] - grouped["matched_control_rmse"]
    )
    return grouped


def bad_rate_metrics(frame: pd.DataFrame, risk_column: str, threshold: float) -> dict[str, float]:
    bad = frame["delta_rmse_addonly_minus_control"].gt(float(threshold))
    risk = frame[risk_column].astype(bool)
    risk_rate = float(bad[risk].mean()) if risk.any() else math.nan
    safe_rate = float(bad[~risk].mean()) if (~risk).any() else math.nan
    overall_rate = float(bad.mean())
    recall = float((bad & risk).sum() / bad.sum()) if bad.any() else math.nan
    lift_vs_safe = risk_rate / safe_rate if safe_rate > 0 else math.inf
    lift_vs_overall = risk_rate / overall_rate if overall_rate > 0 else math.inf
    return {
        "bad_threshold_ft": float(threshold),
        "bad_wells": int(bad.sum()),
        "risk_bad_rate": risk_rate,
        "safe_bad_rate": safe_rate,
        "overall_bad_rate": overall_rate,
        "risk_bad_rate_lift_vs_safe": float(lift_vs_safe),
        "risk_bad_rate_lift_vs_overall": float(lift_vs_overall),
        "bad_well_recall": recall,
    }


def evaluate_target_free_gates(
    risk_scores: pd.DataFrame,
    oof: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    quantiles = [float(value) for value in config["risk_features"]["quantiles"]]
    bad_thresholds = [float(value) for value in config["readout"]["bad_well_thresholds_ft"]]
    risk = risk_scores.rename(columns={"downstream_outer_fold": "outer_fold"}).copy()
    if risk[["well", "outer_fold"]].duplicated().any():
        raise ValueError("risk score contains duplicate well/fold keys")
    by_well = by_well_stage_d_metrics(oof)
    joined = by_well.merge(risk, on=["well", "outer_fold"], how="left", validate="one_to_one")
    if joined["risk_score"].isna().any():
        raise ValueError("Stage D by-well metrics lack target-free risk scores")
    technical = config["guards"]["technical"]
    worsened = int(joined["delta_rmse_addonly_minus_control"].gt(0.0).sum())
    over_025 = int(joined["delta_rmse_addonly_minus_control"].gt(0.25).sum())
    if worsened != int(technical["expected_worsened_wells"]):
        raise ValueError(f"unexpected Stage D worsened-well count: {worsened}")
    if over_025 != int(technical["expected_over_0p25_wells"]):
        raise ValueError(f"unexpected Stage D over-0.25 well count: {over_025}")

    risk_columns = [f"risk_{quantile_label(value)}" for value in quantiles]
    oof_with_risk = oof.merge(
        risk[["well", "outer_fold", "risk_score", *risk_columns]],
        on=["well", "outer_fold"],
        how="left",
        validate="many_to_one",
    )
    if oof_with_risk[risk_columns + ["risk_score"]].isna().any().any():
        raise ValueError("OOF rows lack risk assignment")

    fold_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    gated_by_well_parts: list[pd.DataFrame] = []
    for quantile in quantiles:
        label = quantile_label(quantile)
        risk_column = f"risk_{label}"
        gated_column = f"gated_{label}__pred_tvt"
        oof_with_risk[gated_column] = np.where(
            oof_with_risk[risk_column].astype(bool),
            oof_with_risk[CONTROL_COLUMN],
            oof_with_risk[ADDONLY_COLUMN],
        )
        q_by_well = joined.copy()
        q_by_well["quantile"] = float(quantile)
        q_by_well["quantile_label"] = label
        q_by_well["risk_flag"] = q_by_well[risk_column].astype(bool)
        q_by_well["gated_rmse"] = np.where(
            q_by_well["risk_flag"],
            q_by_well["matched_control_rmse"],
            q_by_well["selector_compact_addonly_rmse"],
        )
        q_by_well["delta_rmse_gated_minus_control"] = (
            q_by_well["gated_rmse"] - q_by_well["matched_control_rmse"]
        )
        gated_by_well_parts.append(q_by_well)
        for outer_fold, fold_oof in oof_with_risk.groupby("outer_fold", sort=True):
            fold_wells = q_by_well[q_by_well["outer_fold"].eq(outer_fold)]
            control_rmse = rmse(fold_oof["actual_tvt"], fold_oof[CONTROL_COLUMN])
            addonly_rmse = rmse(fold_oof["actual_tvt"], fold_oof[ADDONLY_COLUMN])
            gated_rmse = rmse(fold_oof["actual_tvt"], fold_oof[gated_column])
            denominator = control_rmse - addonly_rmse
            retention = (
                (control_rmse - gated_rmse) / denominator if denominator > 0 else math.nan
            )
            common = {
                "quantile": float(quantile),
                "quantile_label": label,
                "outer_fold": int(outer_fold),
                "wells": int(len(fold_wells)),
                "risk_wells": int(fold_wells["risk_flag"].sum()),
                "risk_selection_rate": float(fold_wells["risk_flag"].mean()),
                "matched_control_rmse": control_rmse,
                "selector_compact_addonly_rmse": addonly_rmse,
                "gated_rmse": gated_rmse,
                "gated_delta_rmse_vs_control": gated_rmse - control_rmse,
                "improvement_retention_fraction": retention,
                "worst_well_delta_rmse_gated_minus_control": float(
                    fold_wells["delta_rmse_gated_minus_control"].max()
                ),
            }
            for bad_threshold in bad_thresholds:
                metrics = bad_rate_metrics(fold_wells, "risk_flag", bad_threshold)
                suffix = "gt0" if bad_threshold == 0.0 else "gt0p25"
                for key, value in metrics.items():
                    if key != "bad_threshold_ft":
                        common[f"{suffix}__{key}"] = value
            fold_rows.append(common)

        control_rmse = rmse(oof_with_risk["actual_tvt"], oof_with_risk[CONTROL_COLUMN])
        addonly_rmse = rmse(oof_with_risk["actual_tvt"], oof_with_risk[ADDONLY_COLUMN])
        gated_rmse = rmse(oof_with_risk["actual_tvt"], oof_with_risk[gated_column])
        denominator = control_rmse - addonly_rmse
        retention = (control_rmse - gated_rmse) / denominator if denominator > 0 else math.nan
        pooled = {
            "quantile": float(quantile),
            "quantile_label": label,
            "wells": int(len(q_by_well)),
            "risk_wells": int(q_by_well["risk_flag"].sum()),
            "risk_selection_rate": float(q_by_well["risk_flag"].mean()),
            "matched_control_rmse": control_rmse,
            "selector_compact_addonly_rmse": addonly_rmse,
            "gated_rmse": gated_rmse,
            "gated_delta_rmse_vs_control": gated_rmse - control_rmse,
            "improvement_retention_fraction": retention,
            "worst_well_delta_rmse_gated_minus_control": float(
                q_by_well["delta_rmse_gated_minus_control"].max()
            ),
        }
        for bad_threshold in bad_thresholds:
            metrics = bad_rate_metrics(q_by_well, "risk_flag", bad_threshold)
            suffix = "gt0" if bad_threshold == 0.0 else "gt0p25"
            for key, value in metrics.items():
                if key != "bad_threshold_ft":
                    pooled[f"{suffix}__{key}"] = value
        pooled_rows.append(pooled)

    fold_readout = pd.DataFrame(fold_rows)
    pooled_readout = pd.DataFrame(pooled_rows)
    gated_by_well = pd.concat(gated_by_well_parts, ignore_index=True)
    guard_config = config["guards"]["tail_risk"]
    guard_by_quantile: dict[str, Any] = {}
    for quantile in quantiles:
        label = quantile_label(quantile)
        fold_q = fold_readout[fold_readout["quantile_label"].eq(label)]
        pooled_q = pooled_readout[pooled_readout["quantile_label"].eq(label)].iloc[0]
        gt0_lift_folds = int((fold_q["gt0__risk_bad_rate_lift_vs_safe"] > 1.0).sum())
        gt025_lift_folds = int(
            (fold_q["gt0p25__risk_bad_rate_lift_vs_safe"] > 1.0).sum()
        )
        gated_improvement_folds = int((fold_q["gated_delta_rmse_vs_control"] < 0.0).sum())
        checks = {
            "gt0_positive_lift_folds": gt0_lift_folds,
            "gt0_positive_lift_pass": gt0_lift_folds
            == int(guard_config["required_positive_lift_folds"]),
            "gt0p25_positive_lift_folds": gt025_lift_folds,
            "gt0p25_positive_lift_pass": gt025_lift_folds
            == int(guard_config["required_positive_lift_folds"]),
            "gated_control_improvement_folds": gated_improvement_folds,
            "gated_control_improvement_pass": gated_improvement_folds
            == int(guard_config["required_gated_control_improvement_folds"]),
            "pooled_improvement_retention_fraction": float(
                pooled_q["improvement_retention_fraction"]
            ),
            "pooled_improvement_retention_pass": float(
                pooled_q["improvement_retention_fraction"]
            )
            >= float(guard_config["minimum_pooled_improvement_retention_fraction"]),
            "pooled_worst_well_delta_ft": float(
                pooled_q["worst_well_delta_rmse_gated_minus_control"]
            ),
            "pooled_worst_well_pass": float(
                pooled_q["worst_well_delta_rmse_gated_minus_control"]
            )
            <= float(guard_config["maximum_pooled_worst_well_delta_ft"]),
        }
        checks["quantile_guard_pass"] = all(
            bool(value) for key, value in checks.items() if key.endswith("_pass")
        )
        guard_by_quantile[label] = checks
    all_quantiles_pass = all(
        item["quantile_guard_pass"] for item in guard_by_quantile.values()
    )
    guard = {
        "all_quantiles_must_pass": bool(guard_config["all_quantiles_must_pass"]),
        "guard_by_quantile": guard_by_quantile,
        "target_free_tail_risk_guard_pass": bool(all_quantiles_pass),
    }
    return fold_readout, pooled_readout, gated_by_well, oof_with_risk, guard


if EXECUTE_NOTEBOOK:
    stage_d_oof = read_parquet_columns(stage_d_oof_path, OOF_COLUMNS)
    if len(stage_d_oof) != int(CONFIG["guards"]["technical"]["expected_oof_rows"]):
        raise ValueError("unexpected Stage D OOF row count")
    if stage_d_oof["id"].astype(str).duplicated().any():
        raise ValueError("Stage D OOF contains duplicate ids")
    if stage_d_oof["well"].nunique() != int(
        CONFIG["guards"]["technical"]["expected_wells"]
    ):
        raise ValueError("unexpected Stage D OOF well count")
    fold_readout, pooled_readout, gated_by_well, gated_oof, guard = (
        evaluate_target_free_gates(well_risk_scores, stage_d_oof, CONFIG)
    )
    display(fold_readout)
    display(pooled_readout)
    display(guard)


# %% [markdown]
# ## 7. Diagnostics, metrics, and reproducibility evidence

# %%
if EXECUTE_NOTEBOOK:
    import matplotlib.pyplot as plt

    by_well_base = gated_by_well[gated_by_well["quantile_label"].eq("q70")]
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    for outer_fold, fold in by_well_base.groupby("outer_fold", sort=True):
        axes[0, 0].scatter(
            fold["risk_score"],
            fold["delta_rmse_addonly_minus_control"],
            s=16,
            alpha=0.55,
            label=f"fold {outer_fold}",
        )
    axes[0, 0].axhline(0.0, color="black", linewidth=1)
    axes[0, 0].axhline(0.25, color="red", linewidth=1, linestyle="--")
    axes[0, 0].set_title("Target-free risk vs Stage D by-well delta")
    axes[0, 0].set_xlabel("risk score")
    axes[0, 0].set_ylabel("add-only RMSE - control RMSE")
    axes[0, 0].legend(fontsize=8)

    pooled_readout.plot.bar(
        x="quantile_label",
        y=[
            "gt0__risk_bad_rate_lift_vs_safe",
            "gt0p25__risk_bad_rate_lift_vs_safe",
        ],
        ax=axes[0, 1],
    )
    axes[0, 1].axhline(1.0, color="black", linewidth=1)
    axes[0, 1].set_title("Pooled bad-rate lift: risk vs safe")
    axes[0, 1].set_xlabel("fixed target-free quantile")

    pooled_readout.plot.bar(
        x="quantile_label",
        y=["gt0__bad_well_recall", "gt0p25__bad_well_recall"],
        ax=axes[1, 0],
    )
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_title("Pooled bad-well recall")
    axes[1, 0].set_xlabel("fixed target-free quantile")

    pooled_readout.plot.bar(
        x="quantile_label",
        y=["improvement_retention_fraction"],
        ax=axes[1, 1],
        legend=False,
    )
    axes[1, 1].axhline(
        float(CONFIG["guards"]["tail_risk"]["minimum_pooled_improvement_retention_fraction"]),
        color="red",
        linewidth=1,
        linestyle="--",
    )
    axes[1, 1].set_title("Stage D global improvement retained")
    axes[1, 1].set_xlabel("fixed target-free quantile")
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "target_free_gate_audit.png"
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.show()

# %%
if EXECUTE_NOTEBOOK:
    risk_feature_path = OUTPUT_DIR / "target_free_well_risk_features.parquet"
    risk_score_path = OUTPUT_DIR / "target_free_well_risk_scores.csv"
    risk_schema_path = OUTPUT_DIR / "risk_feature_schema.json"
    preprocessor_path = OUTPUT_DIR / "risk_preprocessors.json"
    fold_readout_path = OUTPUT_DIR / "risk_fold_quantile_readout.csv"
    pooled_readout_path = OUTPUT_DIR / "risk_pooled_quantile_readout.csv"
    gated_by_well_path = OUTPUT_DIR / "gated_by_well_metrics.csv"
    gated_oof_path = OUTPUT_DIR / "gated_oof_predictions.parquet"
    input_manifest_path = OUTPUT_DIR / "input_manifest.json"

    risk_feature_frame.to_parquet(risk_feature_path, index=False)
    well_risk_scores.to_csv(risk_score_path, index=False)
    fold_readout.to_csv(fold_readout_path, index=False)
    pooled_readout.to_csv(pooled_readout_path, index=False)
    gated_by_well.to_csv(gated_by_well_path, index=False)
    gated_oof.to_parquet(gated_oof_path, index=False)
    feature_columns = risk_feature_columns(risk_feature_frame)
    feature_schema = {
        "schema_version": "1.0.0",
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "families": sorted({column.split("__", 1)[0] for column in feature_columns}),
        "scope_specs": scope_specs(CONFIG),
        "family_weighting": CONFIG["risk_features"]["family_weighting"],
        "forbidden_label_columns": [
            "TVT",
            "target",
            "actual_tvt",
            "actual_error",
            "delta_rmse_addonly_minus_control",
        ],
    }
    feature_schema["feature_schema_sha256"] = sha256_json(feature_schema)
    write_json(risk_schema_path, feature_schema)
    write_json(
        preprocessor_path,
        {
            "schema_version": "1.0.0",
            "preprocessors": risk_preprocessors,
            "preprocessor_count": len(risk_preprocessors),
        },
    )
    input_manifest = {
        "stage_c_manifest_sha256": CONFIG["data"]["stage_c_expected_manifest_sha256"],
        "stage_c_partition_manifest_sha256": CONFIG["data"][
            "stage_c_expected_partition_manifest_sha256"
        ],
        "stage_c_schema_file_sha256": CONFIG["data"]["stage_c_expected_schema_file_sha256"],
        "stage_c_schema_logical_sha256": CONFIG["data"][
            "stage_c_expected_schema_logical_sha256"
        ],
        "stage_c_partitions": partition_evidence,
        "stage_d_oof_path": str(stage_d_oof_path),
        "stage_d_oof_sha256": sha256_file(stage_d_oof_path),
        "raw_horizontal_files": raw_evidence,
        "raw_horizontal_manifest_sha256": sha256_json(raw_evidence),
    }
    write_json(input_manifest_path, input_manifest)
    summary = {
        "status": "kaggle_cpu_readout_completed",
        "compute_contract": compute_contract,
        "input_counts": {
            "stage_c_partitions": len(partition_evidence),
            "stage_c_rows": int(sum(item["rows"] for item in partition_evidence)),
            "raw_horizontal_files": len(raw_evidence),
            "risk_valid_wells": int(len(well_risk_scores)),
            "stage_d_oof_rows": int(len(stage_d_oof)),
            "stage_d_wells": int(stage_d_oof["well"].nunique()),
        },
        "stage_d_anchor": {
            "matched_control_rmse": rmse(stage_d_oof["actual_tvt"], stage_d_oof[CONTROL_COLUMN]),
            "selector_compact_addonly_rmse": rmse(
                stage_d_oof["actual_tvt"], stage_d_oof[ADDONLY_COLUMN]
            ),
            "worsened_wells": int(
                by_well_stage_d_metrics(stage_d_oof)[
                    "delta_rmse_addonly_minus_control"
                ].gt(0.0).sum()
            ),
            "over_0p25_wells": int(
                by_well_stage_d_metrics(stage_d_oof)[
                    "delta_rmse_addonly_minus_control"
                ].gt(0.25).sum()
            ),
        },
        "pooled_quantile_readout": pooled_readout.to_dict(orient="records"),
        "guard": guard,
        "runtime_seconds": float(time.time() - STARTED_AT),
    }
    summary_path = OUTPUT_DIR / "audit_summary.json"
    write_json(summary_path, summary)
    output_paths = {
        path.name: path
        for path in [
            risk_feature_path,
            risk_score_path,
            risk_schema_path,
            preprocessor_path,
            fold_readout_path,
            pooled_readout_path,
            gated_by_well_path,
            gated_oof_path,
            plot_path,
            summary_path,
            input_manifest_path,
        ]
    }
    reproducibility = {
        "status": "kaggle_cpu_readout_completed",
        "seed_policy": CONFIG["reproducibility"]["seed_policy"],
        "stochastic_components": CONFIG["reproducibility"]["stochastic_components"],
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "risk_feature_schema_sha256": feature_schema["feature_schema_sha256"],
        "risk_feature_content_sha256": logical_frame_sha256(
            risk_feature_frame,
            ["downstream_outer_fold", "role", "source_outer_fold", "well"],
        ),
        "risk_score_content_sha256": logical_frame_sha256(
            well_risk_scores, ["downstream_outer_fold", "well"]
        ),
        "gated_oof_prediction_content_sha256": logical_frame_sha256(
            gated_oof, ["id"]
        ),
        "output_file_sha256": {
            name: sha256_file(path) for name, path in output_paths.items()
        },
        "model_manifest_sha256": None,
        "submission_sha256": None,
        "deterministic_anchor": False,
        "rerun_result": None,
    }
    reproducibility_path = OUTPUT_DIR / "reproducibility_manifest.json"
    write_json(reproducibility_path, reproducibility)
    required = [str(item) for item in CONFIG["artifacts"]["required"]]
    print("Generated artifacts")
    for filename in required:
        path = OUTPUT_DIR / filename
        print(f"- {filename}: exists={path.exists()} bytes={path.stat().st_size if path.exists() else 0}")
        assert path.exists() and path.stat().st_size > 0
    display(summary)
    display(reproducibility)
    if guard["target_free_tail_risk_guard_pass"]:
        print(
            "All fixed q70/q80/q90 tail-risk guards PASS. This supports a separately approved "
            "current-test port; inference and submission remain disabled in exp276."
        )
    else:
        print(
            "The fixed target-free tail-risk guard FAILS. Do not rescue it with a feature, "
            "weight, or quantile grid; keep current-test inference and submission disabled."
        )
