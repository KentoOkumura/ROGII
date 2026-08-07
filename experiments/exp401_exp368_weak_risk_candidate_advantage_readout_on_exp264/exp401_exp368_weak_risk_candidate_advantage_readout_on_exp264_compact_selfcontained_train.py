# %% [markdown]
# # exp401 exp368 weak-risk candidate-advantage readout on exp264
#
# This implementation is Stage 0 only. It does not fit a model, rerun PF/Beam,
# create TVT predictions, or create a submission. The exp368 row risk, fold,
# legal candidate domains, and their logical-content evidence are frozen before
# suffix truth is read.

# %% [markdown]
# ## Contents
# 1. Imports and fixed names
# 2. Notebook-safe configuration and path helpers
# 3. Reproducibility, hashing, and truth-access guards
# 4. Frozen scientific contract and input preflight
# 5. Target-free overlapping-block row-risk assembly
# 6. Strict-nested exp264 selector surface
# 7. Late truth attachment and candidate-advantage metrics
# 8. Technical and scientific promotion gates
# 9. Generated evidence and summaries
# 10. Setup and configuration
# 11. Stage 0 execution orchestration

# %%
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


EXPERIMENT_NAME = "exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")

BLOCK_LEDGER_FILENAME = (
    "exp368_marginalized_reliability_pf_target_free_block_ledger.csv.gz"
)
WEAK_POSTERIOR_FILENAME = (
    "exp368_marginalized_reliability_pf_target_free_weak_posterior_blocks.csv.gz"
)
STAGE_C_SCORE_FILENAME = "nested_outer_valid_candidate_score.parquet"
EXP226_TRUTH_FILENAME = (
    "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_"
    "train_oof_predictions.csv.gz"
)
HIDDEN_LIKE_FILENAME = (
    "exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv"
)

RISK_FEATURE_COLUMN = "ctx__exp368_weak_risk"
CIRCULAR_FEATURE_COLUMN = "ctx__exp368_circular_weak_risk"
TRUTH_COLUMN_TOKENS = (
    "target",
    "truth",
    "true_tvt",
    "tvt_true",
    "error",
    "abs_error",
    "bad10",
    "oracle",
)
TARGET_FREE_PREDICTED_SCORE_COLUMNS = frozenset(
    {
        "pred_abs_error",
    }
)


# %% [markdown]
# ## 2. Notebook-safe configuration and path helpers


# %%
def get_nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "experiment_summary.md").exists() and (
            candidate / "experiments"
        ).is_dir():
            return candidate
    return current


def resolve_package_dir() -> Path:
    cwd = Path.cwd()
    candidates = [
        cwd,
        cwd / "experiments" / EXPERIMENT_NAME,
        KAGGLE_WORKING_ROOT,
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(
            path.parent
            for path in sorted(KAGGLE_INPUT_ROOT.glob("**/config.yaml"))
            if path.parent.name == EXPERIMENT_NAME
        )
    for candidate in candidates:
        config_path = candidate / "config.yaml"
        if not config_path.is_file():
            continue
        loaded = yaml.safe_load(config_path.read_text()) or {}
        if get_nested(loaded, "experiment.name") == EXPERIMENT_NAME:
            return candidate
    raise FileNotFoundError(f"Could not locate config.yaml for {EXPERIMENT_NAME}")


def load_experiment_config(package_dir: Path) -> dict[str, Any]:
    config_path = package_dir / "config.yaml"
    value = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


def _existing_file(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def resolve_input(
    *,
    filename: str,
    local_candidates: Sequence[Path],
    preferred_slugs: Sequence[str],
) -> Path:
    local = _existing_file(local_candidates)
    if local is not None:
        return local
    if KAGGLE_INPUT_ROOT.exists():
        preferred_roots = [
            KAGGLE_INPUT_ROOT / slug for slug in preferred_slugs
        ] + [
            KAGGLE_INPUT_ROOT / "notebooks" / "kentookumura" / slug
            for slug in preferred_slugs
        ]
        generic_roots = [
            path for path in sorted(KAGGLE_INPUT_ROOT.iterdir()) if path.is_dir()
        ]
        seen: set[Path] = set()
        for root in [*preferred_roots, *generic_roots]:
            if root in seen or not root.exists():
                continue
            seen.add(root)
            matches = sorted(
                path
                for path in root.rglob(filename)
                if path.is_file() and path.stat().st_size > 0
            )
            if matches:
                return matches[0]
    checked = "\n".join(str(path) for path in local_candidates)
    raise FileNotFoundError(
        f"{filename} not found. Checked:\n{checked}\n"
        f"Kaggle slugs: {list(preferred_slugs)}"
    )


def resolve_all_inputs(repo_root: Path) -> dict[str, Path]:
    exp368_dir = repo_root / "experiments" / "exp368_marginalized_reliability_pf"
    exp264_dir = (
        repo_root
        / "experiments"
        / "exp264_exp263_candidate_confidence_dual_selector"
    )
    exp115_copies = [
        repo_root
        / "experiments"
        / experiment
        / "inputs"
        / HIDDEN_LIKE_FILENAME
        for experiment in (
            "exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector",
            "exp237_hmm_exp226_candidate_selector_on_exp183",
            "exp248_candidate_perturbation_augmentation_for_likelihood_ranker",
            "exp251_raw_test_safe_dual_objective_candidate_ranker",
        )
    ]
    return {
        "block_ledger": resolve_input(
            filename=BLOCK_LEDGER_FILENAME,
            local_candidates=[
                exp368_dir / "kaggle" / "output" / "train_v1" / "artifacts"
                / BLOCK_LEDGER_FILENAME,
                Path("/tmp/kaggle-output/exp368_marginalized_reliability_pf/")
                / "train_v1"
                / "artifacts"
                / BLOCK_LEDGER_FILENAME,
            ],
            preferred_slugs=["exp368-marginalized-reliability-pf-train"],
        ),
        "weak_posterior": resolve_input(
            filename=WEAK_POSTERIOR_FILENAME,
            local_candidates=[
                exp368_dir / "kaggle" / "output" / "train_v1" / "artifacts"
                / WEAK_POSTERIOR_FILENAME,
                Path("/tmp/kaggle-output/exp368_marginalized_reliability_pf/")
                / "train_v1"
                / "artifacts"
                / WEAK_POSTERIOR_FILENAME,
            ],
            preferred_slugs=["exp368-marginalized-reliability-pf-train"],
        ),
        "stage_c_score": resolve_input(
            filename=STAGE_C_SCORE_FILENAME,
            local_candidates=[
                exp264_dir / "artifacts" / "stage_c_v6" / STAGE_C_SCORE_FILENAME,
                exp264_dir
                / "kaggle"
                / "output"
                / "stage_c_v6"
                / "artifacts"
                / STAGE_C_SCORE_FILENAME,
                Path("/tmp/exp264-stage-c-v6-outer-valid/artifacts")
                / STAGE_C_SCORE_FILENAME,
            ],
            preferred_slugs=["exp264-exp263-confidence-dual-selector-train"],
        ),
        "fold_truth": resolve_input(
            filename=EXP226_TRUTH_FILENAME,
            local_candidates=[
                Path(
                    "/tmp/kaggle-output/"
                    "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_"
                    "reproduction/train_v1/artifacts"
                )
                / EXP226_TRUTH_FILENAME,
            ],
            preferred_slugs=["exp226-k16-kappa-repro-train"],
        ),
        "hidden_like": resolve_input(
            filename=HIDDEN_LIKE_FILENAME,
            local_candidates=exp115_copies,
            preferred_slugs=["exp115-hidden-like-spatial-holdout-from-ppt-train"],
        ),
    }


# %% [markdown]
# ## 3. Reproducibility, hashing, and truth-access guards


# %%
def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decompressed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def dataframe_schema_sha256(frame: pd.DataFrame) -> str:
    schema = {
        "columns": [
            {"name": str(column), "dtype": str(frame[column].dtype)}
            for column in frame.columns
        ]
    }
    return mapping_sha256(schema)


def array_bundle_sha256(columns: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(b"exp401-array-bundle-v1\n")
    for name, raw_values in columns.items():
        values = np.asarray(raw_values)
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json_bytes(list(values.shape)))
        digest.update(b"\n")
        if values.dtype.kind in {"O", "U", "S"}:
            for item in values:
                encoded = str(item).encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "little"))
                digest.update(encoded)
        else:
            contiguous = np.ascontiguousarray(values)
            digest.update(contiguous.view(np.uint8))
        digest.update(b"\n")
    return digest.hexdigest()


def write_deterministic_gzip_columns(
    columns: Mapping[str, np.ndarray],
    path: Path,
    *,
    chunk_rows: int = 100_000,
) -> dict[str, Any]:
    lengths = {len(np.asarray(values)) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError("gzip output columns have different lengths")
    rows = lengths.pop()
    path.parent.mkdir(parents=True, exist_ok=True)
    logical_digest = hashlib.sha256()
    with path.open("wb") as raw_stream:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_stream,
            mtime=0,
        ) as compressed:
            for start in range(0, rows, chunk_rows):
                stop = min(start + chunk_rows, rows)
                frame = pd.DataFrame(
                    {
                        name: np.asarray(values)[start:stop]
                        for name, values in columns.items()
                    }
                )
                text = frame.to_csv(index=False, header=start == 0)
                encoded = text.encode("utf-8")
                logical_digest.update(encoded)
                compressed.write(encoded)
    return {
        "path": str(path),
        "rows": rows,
        "raw_sha256": sha256_path(path),
        "decompressed_content_sha256": logical_digest.hexdigest(),
        "schema_sha256": mapping_sha256(
            {
                "columns": [
                    {
                        "name": str(name),
                        "dtype": str(np.asarray(values).dtype),
                    }
                    for name, values in columns.items()
                ]
            }
        ),
    }


def write_csv_with_sha(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = frame.to_csv(index=False).encode("utf-8")
    path.write_bytes(content)
    return {
        "path": str(path),
        "rows": len(frame),
        "sha256": hashlib.sha256(content).hexdigest(),
        "schema_sha256": dataframe_schema_sha256(frame),
    }


def write_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=_json_default,
    )
    path.write_text(payload + "\n")
    return sha256_path(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def contains_truth_token(column: str) -> bool:
    lowered = str(column).lower()
    if lowered in TARGET_FREE_PREDICTED_SCORE_COLUMNS:
        return False
    return any(token in lowered for token in TRUTH_COLUMN_TOKENS)


def assert_target_free_columns(columns: Iterable[str], *, stage: str) -> None:
    forbidden = sorted(column for column in columns if contains_truth_token(column))
    if forbidden:
        raise ValueError(f"{stage} contains truth/error columns: {forbidden}")


@dataclass
class TruthAccessLedger:
    frozen: bool = False
    truth_columns_read_before_freeze: int = 0
    frozen_evidence: dict[str, Any] = field(default_factory=dict)
    late_truth_columns_read: list[str] = field(default_factory=list)

    def record_target_free_projection(self, columns: Iterable[str]) -> None:
        names = [str(column) for column in columns]
        forbidden = [column for column in names if contains_truth_token(column)]
        if forbidden:
            if not self.frozen:
                self.truth_columns_read_before_freeze += len(forbidden)
            raise RuntimeError(
                f"truth/error columns requested before feature freeze: {forbidden}"
            )

    def freeze_features(self, evidence: Mapping[str, Any]) -> None:
        if self.truth_columns_read_before_freeze != 0:
            raise RuntimeError("cannot freeze after an early truth/error read")
        required = (
            "feature_schema_sha256",
            "feature_content_sha256",
            "selector_surface_content_sha256",
            "scientific_contract_sha256",
        )
        missing = [
            name
            for name in required
            if len(str(evidence.get(name, ""))) != 64
        ]
        if missing:
            raise RuntimeError(f"feature freeze evidence is incomplete: {missing}")
        self.frozen_evidence = dict(evidence)
        self.frozen = True

    def read_late_truth(self, columns: Iterable[str]) -> None:
        if not self.frozen:
            raise RuntimeError("late truth requires frozen feature/domain evidence")
        self.late_truth_columns_read.extend(str(column) for column in columns)


# %% [markdown]
# ## 4. Frozen scientific contract and input preflight


# %%
def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("unexpected experiment.name")
    if get_nested(config, "experiment.route") != "ml_model":
        raise ValueError("exp401 route must remain ml_model")
    if not bool(get_nested(config, "implementation.stage_0_implemented")):
        raise ValueError("exp401 Stage 0 implementation flag is not enabled")
    if bool(get_nested(config, "stage_1_if_stage_0_passes_and_separately_approved.enabled")):
        raise ValueError("exp401 Stage 1 must remain disabled")
    if bool(get_nested(config, "execution.run_stage_1")):
        raise ValueError("exp401 Stage 1 execution must remain disabled")
    if bool(get_nested(config, "execution.run_inference")) or bool(
        get_nested(config, "execution.create_submission")
    ):
        raise ValueError("exp401 inference/submission must remain disabled")

    candidates = [
        str(value)
        for value in get_nested(config, "candidate_contract.candidate_order")
    ]
    primary = [
        str(value)
        for value in get_nested(config, "candidate_contract.primary_domain.candidates")
    ]
    secondary = [
        str(value)
        for value in get_nested(
            config, "candidate_contract.secondary_domain.candidates"
        )
    ]
    anchor = str(get_nested(config, "candidate_contract.anchor"))
    if len(candidates) != 12 or len(set(candidates)) != 12:
        raise ValueError("candidate order must contain 12 unique candidates")
    if len(primary) != 11 or len(secondary) != 7:
        raise ValueError("legal domain sizes must remain 11 and 7")
    if anchor not in primary or anchor not in secondary:
        raise ValueError("likpf anchor must be present in both legal domains")
    if bool(
        get_nested(
            config,
            "candidate_contract.twelve_candidate_single_hard_domain_enabled",
        )
    ):
        raise ValueError("the 12-candidate single hard domain is forbidden")

    stage_0_counts = get_nested(
        config, "execution_contract.future_stage_0_if_implemented_and_approved"
    )
    expected_zero = (
        "model_configs",
        "lightgbm_configs",
        "trained_folds",
        "boosters",
        "pf_runs",
        "parent_control_retraining",
        "prediction_rows",
    )
    if any(int(stage_0_counts[name]) != 0 for name in expected_zero):
        raise ValueError("Stage 0 zero-model/zero-PF execution contract changed")

    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "ml_model",
        "stage": "stage_0_zero_booster_only",
        "parent": get_nested(config, "lineage.parent"),
        "auxiliary_source": get_nested(config, "lineage.auxiliary_source"),
        "candidate_contract": get_nested(config, "candidate_contract"),
        "features": get_nested(config, "features"),
        "stage_0": get_nested(config, "stage_0"),
        "execution_counts": stage_0_counts,
        "forbidden": get_nested(config, "forbidden"),
        "rng": "none",
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    contract["candidate_order"] = candidates
    contract["primary_domain"] = primary
    contract["secondary_domain"] = secondary
    contract["anchor"] = anchor

    if require_run_approval:
        if not bool(get_nested(config, "execution.stage_0_run_approved")):
            raise PermissionError("exp401 Stage 0 run is not approved")
        if not bool(get_nested(config, "execution.run_stage_0")):
            raise PermissionError("execution.run_stage_0 must be true for a run")
    return contract


def _require_columns(
    path: Path,
    required: Sequence[str],
    *,
    csv_compression: str | None = None,
) -> list[str]:
    header = pd.read_csv(path, nrows=0, compression=csv_compression).columns.tolist()
    missing = sorted(set(required).difference(header))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    return header


def preflight_inputs(
    paths: Mapping[str, Path],
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> dict[str, Any]:
    block_spec = get_nested(config, "data.exp368_target_free.block_ledger")
    posterior_spec = get_nested(config, "data.exp368_target_free.weak_posterior")
    score_spec = get_nested(config, "data.exp264_nested_score")
    fold_spec = get_nested(config, "data.fold_assignment")
    hidden_spec = get_nested(config, "data.hidden_like_assignment")

    block_columns = [str(value) for value in block_spec["safe_columns"]]
    posterior_columns = [str(value) for value in posterior_spec["safe_columns"]]
    score_columns = [str(value) for value in score_spec["safe_columns"]]
    fold_safe_columns = [str(value) for value in fold_spec["safe_columns"]]
    for columns in (
        block_columns,
        posterior_columns,
        score_columns,
        fold_safe_columns,
    ):
        ledger.record_target_free_projection(columns)

    block_header = _require_columns(
        paths["block_ledger"], block_columns, csv_compression="gzip"
    )
    posterior_header = _require_columns(
        paths["weak_posterior"], posterior_columns, csv_compression="gzip"
    )
    assert_target_free_columns(block_header, stage="block ledger file header")
    assert_target_free_columns(posterior_header, stage="posterior file header")
    configured_forbidden = {
        str(value) for value in posterior_spec.get("forbidden_columns", [])
    }
    present_forbidden = sorted(configured_forbidden.intersection(posterior_header))
    if present_forbidden:
        raise ValueError(
            f"weak posterior file contains forbidden columns: {present_forbidden}"
        )
    score_schema = pq.read_schema(paths["stage_c_score"])
    missing_score = sorted(set(score_columns).difference(score_schema.names))
    if missing_score:
        raise ValueError(f"Stage C score is missing columns: {missing_score}")
    _require_columns(paths["fold_truth"], fold_safe_columns, csv_compression="gzip")
    role_columns = [
        str(value) for value in hidden_spec["role_columns"].values()
    ]
    _require_columns(paths["hidden_like"], ["well_id", *role_columns])
    assert_target_free_columns(block_columns, stage="block ledger projection")
    assert_target_free_columns(posterior_columns, stage="posterior projection")
    assert_target_free_columns(score_columns, stage="Stage C score projection")
    assert_target_free_columns(fold_safe_columns, stage="fold-safe projection")

    observed = {
        "block_ledger_decompressed_sha256": decompressed_sha256(
            paths["block_ledger"]
        ),
        "weak_posterior_decompressed_sha256": decompressed_sha256(
            paths["weak_posterior"]
        ),
        "stage_c_score_artifact_sha256": sha256_path(paths["stage_c_score"]),
        "stage_c_score_schema_sha256": hashlib.sha256(
            str(score_schema).encode("utf-8")
        ).hexdigest(),
        "fold_truth_decompressed_sha256": decompressed_sha256(paths["fold_truth"]),
        "hidden_like_sha256": sha256_path(paths["hidden_like"]),
    }
    expected = {
        "block_ledger_decompressed_sha256": str(
            block_spec["expected_decompressed_content_sha256"]
        ),
        "weak_posterior_decompressed_sha256": str(
            posterior_spec["expected_decompressed_content_sha256"]
        ),
        "stage_c_score_artifact_sha256": str(
            score_spec["expected_logical_content_sha256"]
        ),
        "fold_truth_decompressed_sha256": str(
            fold_spec["expected_decompressed_content_sha256"]
        ),
        "hidden_like_sha256": str(hidden_spec["expected_sha256"]),
    }
    mismatches = {
        name: {"observed": observed[name], "expected": expected_value}
        for name, expected_value in expected.items()
        if observed[name] != expected_value
    }
    if mismatches:
        raise ValueError({"message": "frozen input SHA mismatch", **mismatches})

    score_file = pq.ParquetFile(paths["stage_c_score"])
    expected_long_rows = int(get_nested(config, "validation.expected_candidate_long_rows"))
    if score_file.metadata.num_rows != expected_long_rows:
        raise ValueError(
            f"Stage C rows={score_file.metadata.num_rows:,}, "
            f"expected={expected_long_rows:,}"
        )
    return {
        "input_sha256": observed,
        "paths": {name: str(path) for name, path in paths.items()},
        "block_header": block_header,
        "posterior_header": posterior_header,
        "score_schema_names": score_schema.names,
        "score_row_groups": score_file.metadata.num_row_groups,
        "truth_columns_read_before_freeze": ledger.truth_columns_read_before_freeze,
    }


# %% [markdown]
# ## 5. Target-free overlapping-block row-risk assembly


# %%
def load_target_free_blocks(
    paths: Mapping[str, Path],
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> pd.DataFrame:
    block_spec = get_nested(config, "data.exp368_target_free.block_ledger")
    posterior_spec = get_nested(config, "data.exp368_target_free.weak_posterior")
    block_columns = [str(value) for value in block_spec["safe_columns"]]
    posterior_columns = [str(value) for value in posterior_spec["safe_columns"]]
    ledger.record_target_free_projection(block_columns)
    ledger.record_target_free_projection(posterior_columns)
    block = pd.read_csv(
        paths["block_ledger"],
        usecols=block_columns,
        dtype={"well_id": str},
    )
    posterior = pd.read_csv(
        paths["weak_posterior"],
        usecols=posterior_columns,
        dtype={"well_id": str},
    )
    if len(block) != int(block_spec["expected_blocks"]):
        raise ValueError("block ledger row count changed")
    if len(posterior) != int(posterior_spec["expected_blocks"]):
        raise ValueError("weak posterior row count changed")
    for frame, name in ((block, "block ledger"), (posterior, "posterior")):
        if frame.duplicated(["well_id", "block_id"]).any():
            raise ValueError(f"{name} contains duplicate well/block identities")
    merged = block.merge(
        posterior,
        on=["well_id", "block_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(block):
        raise ValueError("block ledger/posterior identity mismatch")
    merged.sort_values(["well_id", "block_id"], kind="mergesort", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    numeric_columns = [
        "start_suffix_offset",
        "stop_suffix_offset_exclusive",
        "start_row_idx",
        "end_row_idx",
        "block_row_count",
        "weak_posterior_mean",
        "circular_weak_score",
        "circular_offset_blocks",
    ]
    if not np.isfinite(merged[numeric_columns].to_numpy(np.float64)).all():
        raise ValueError("target-free block inputs contain non-finite values")
    for column in ("weak_posterior_mean", "circular_weak_score"):
        if not merged[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} is outside [0, 1]")
    return merged


def aggregate_overlapping_block_risk(
    blocks: pd.DataFrame,
    *,
    block_rows: int,
    stride_rows: int,
) -> pd.DataFrame:
    required = {
        "well_id",
        "block_id",
        "start_suffix_offset",
        "stop_suffix_offset_exclusive",
        "start_row_idx",
        "end_row_idx",
        "block_row_count",
        "weak_posterior_mean",
        "circular_weak_score",
        "circular_offset_blocks",
    }
    missing = sorted(required.difference(blocks.columns))
    if missing:
        raise ValueError(f"block aggregation is missing columns: {missing}")
    assert_target_free_columns(blocks.columns, stage="row-risk block input")
    output_parts: list[pd.DataFrame] = []
    for well_id, well_blocks in blocks.groupby("well_id", sort=True):
        part = well_blocks.sort_values("block_id", kind="mergesort").reset_index(
            drop=True
        )
        starts = part["start_suffix_offset"].to_numpy(np.int64)
        stops = part["stop_suffix_offset_exclusive"].to_numpy(np.int64)
        counts = part["block_row_count"].to_numpy(np.int64)
        if len(part) == 0 or starts[0] != 0:
            raise ValueError(f"{well_id}: first suffix block must start at zero")
        n_rows = int(stops.max())
        expected_starts = np.arange(0, n_rows, stride_rows, dtype=np.int64)
        if not np.array_equal(starts, expected_starts):
            raise ValueError(f"{well_id}: block starts differ from frozen stride")
        if not np.array_equal(counts, stops - starts):
            raise ValueError(f"{well_id}: block row count differs from bounds")
        if bool(np.any(counts <= 0)) or bool(np.any(counts > block_rows)):
            raise ValueError(f"{well_id}: invalid block length")
        origins = (
            part["start_row_idx"].to_numpy(np.int64)
            - part["start_suffix_offset"].to_numpy(np.int64)
        )
        end_origins = (
            part["end_row_idx"].to_numpy(np.int64)
            - part["stop_suffix_offset_exclusive"].to_numpy(np.int64)
            + 1
        )
        if len(np.unique(origins)) != 1 or not np.array_equal(origins, end_origins):
            raise ValueError(f"{well_id}: row_idx is not contiguous with suffix offsets")
        row_origin = int(origins[0])

        real_delta = np.zeros(n_rows + 1, dtype=np.float64)
        circular_delta = np.zeros(n_rows + 1, dtype=np.float64)
        coverage_delta = np.zeros(n_rows + 1, dtype=np.int32)
        real_values = part["weak_posterior_mean"].to_numpy(np.float64)
        circular_values = part["circular_weak_score"].to_numpy(np.float64)
        np.add.at(real_delta, starts, real_values)
        np.add.at(real_delta, stops, -real_values)
        np.add.at(circular_delta, starts, circular_values)
        np.add.at(circular_delta, stops, -circular_values)
        np.add.at(coverage_delta, starts, 1)
        np.add.at(coverage_delta, stops, -1)
        coverage = np.cumsum(coverage_delta[:-1])
        if bool(np.any(coverage <= 0)):
            raise ValueError(f"{well_id}: row-risk coverage has a gap")
        real = np.cumsum(real_delta[:-1]) / coverage
        circular = np.cumsum(circular_delta[:-1]) / coverage
        offsets = part["circular_offset_blocks"].to_numpy(np.int64)
        if len(part) > 1 and (len(np.unique(offsets)) != 1 or offsets[0] == 0):
            raise ValueError(f"{well_id}: circular block offset is not stable/nonzero")
        output_parts.append(
            pd.DataFrame(
                {
                    "well_id": str(well_id),
                    "row_idx": row_origin + np.arange(n_rows, dtype=np.int64),
                    "suffix_offset": np.arange(n_rows, dtype=np.int64),
                    RISK_FEATURE_COLUMN: real.astype(np.float32),
                    CIRCULAR_FEATURE_COLUMN: circular.astype(np.float32),
                    "covering_block_count": coverage.astype(np.int8),
                }
            )
        )
    output = pd.concat(output_parts, ignore_index=True)
    output.sort_values(["well_id", "row_idx"], kind="mergesort", inplace=True)
    output.reset_index(drop=True, inplace=True)
    if output.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("row-risk output has duplicate identities")
    risk_values = output[
        [RISK_FEATURE_COLUMN, CIRCULAR_FEATURE_COLUMN]
    ].to_numpy(np.float64)
    if not np.isfinite(risk_values).all() or bool(
        np.any((risk_values < 0.0) | (risk_values > 1.0))
    ):
        raise ValueError("aggregated row risk is non-finite or outside [0, 1]")
    return output


def attach_target_free_fold(
    row_risk: pd.DataFrame,
    fold_truth_path: Path,
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> pd.DataFrame:
    fold_spec = get_nested(config, "data.fold_assignment")
    safe_columns = [str(value) for value in fold_spec["safe_columns"]]
    ledger.record_target_free_projection(safe_columns)
    fold = pd.read_csv(
        fold_truth_path,
        usecols=safe_columns,
        dtype={"well_id": str},
    )
    fold["row_idx"] = pd.to_numeric(fold["row_idx"], errors="raise").astype(np.int64)
    fold["suffix_offset"] = pd.to_numeric(
        fold["suffix_offset"], errors="raise"
    ).astype(np.int64)
    fold["fold"] = pd.to_numeric(fold["fold"], errors="raise").astype(np.int8)
    fold.sort_values(["well_id", "row_idx"], kind="mergesort", inplace=True)
    fold.reset_index(drop=True, inplace=True)
    if fold.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("fold assignment has duplicate row identities")
    if len(fold) != len(row_risk):
        raise ValueError("fold assignment/row-risk row count mismatch")
    for column in ("well_id", "row_idx", "suffix_offset"):
        if not np.array_equal(
            fold[column].to_numpy(), row_risk[column].to_numpy()
        ):
            raise ValueError(f"fold assignment differs from row-risk on {column}")
    output = row_risk.copy()
    output["fold"] = fold["fold"].to_numpy(np.int8)
    expected_folds = np.asarray(
        get_nested(config, "validation.expected_folds"), dtype=np.int8
    )
    if not np.array_equal(np.sort(output["fold"].unique()), expected_folds):
        raise ValueError("fold assignment differs from expected five folds")
    return output


def crossfit_quantile_bins(
    values: np.ndarray,
    folds: np.ndarray,
    *,
    quantiles: Sequence[float],
) -> tuple[np.ndarray, dict[int, list[float]]]:
    values = np.asarray(values, dtype=np.float64)
    folds = np.asarray(folds, dtype=np.int8)
    if len(values) != len(folds) or not np.isfinite(values).all():
        raise ValueError("crossfit quantile input is invalid")
    bins = np.full(len(values), -1, dtype=np.int8)
    boundaries: dict[int, list[float]] = {}
    for valid_fold in sorted(np.unique(folds).tolist()):
        train_mask = folds != valid_fold
        valid_mask = folds == valid_fold
        edges = np.asarray(
            np.quantile(values[train_mask], np.asarray(quantiles, dtype=np.float64)),
            dtype=np.float64,
        )
        if bool(np.any(np.diff(edges) < 0.0)):
            raise RuntimeError("quantile boundaries are not monotone")
        bins[valid_mask] = np.searchsorted(
            edges, values[valid_mask], side="right"
        ).astype(np.int8)
        boundaries[int(valid_fold)] = edges.tolist()
    if bool(np.any(bins < 0)):
        raise RuntimeError("crossfit quantile bins are incomplete")
    return bins, boundaries


def crossfit_extreme_quartiles(
    values: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, dict[int, list[float]]]:
    values = np.asarray(values, dtype=np.float64)
    folds = np.asarray(folds, dtype=np.int8)
    if len(values) != len(folds) or not np.isfinite(values).all():
        raise ValueError("crossfit quartile input is invalid")
    quartiles = np.zeros(len(values), dtype=np.int8)
    boundaries: dict[int, list[float]] = {}
    for valid_fold in sorted(np.unique(folds).tolist()):
        train_mask = folds != valid_fold
        valid_mask = folds == valid_fold
        low, high = np.quantile(values[train_mask], [0.25, 0.75]).tolist()
        if low > high:
            raise RuntimeError("crossfit quartile boundaries are reversed")
        quartiles[valid_mask & (values <= low)] = 1
        quartiles[valid_mask & (values >= high)] = 4
        boundaries[int(valid_fold)] = [float(low), float(high)]
    return quartiles, boundaries


# %% [markdown]
# ## 6. Strict-nested exp264 selector surface


# %%
def _well_row_layout(row_risk: pd.DataFrame) -> dict[str, tuple[int, int, int]]:
    layout: dict[str, tuple[int, int, int]] = {}
    for well_id, indices in row_risk.groupby("well_id", sort=False).groups.items():
        positions = np.asarray(indices, dtype=np.int64)
        if not np.array_equal(
            positions, np.arange(positions[0], positions[-1] + 1, dtype=np.int64)
        ):
            raise ValueError(f"{well_id}: row-risk positions are not contiguous")
        rows = row_risk.loc[positions, "row_idx"].to_numpy(np.int64)
        if not np.array_equal(
            rows, np.arange(rows[0], rows[0] + len(rows), dtype=np.int64)
        ):
            raise ValueError(f"{well_id}: row_idx is not contiguous")
        layout[str(well_id)] = (int(positions[0]), int(rows[0]), len(rows))
    return layout


def _positions_from_identity(
    ids: np.ndarray,
    wells: np.ndarray,
    layout: Mapping[str, tuple[int, int, int]],
) -> np.ndarray:
    ids = np.asarray(ids, dtype=str)
    wells = np.asarray(wells, dtype=str)
    row_idx = (
        pd.Series(ids, copy=False)
        .str.rsplit("_", n=1)
        .str[-1]
        .pipe(pd.to_numeric, errors="raise")
        .to_numpy(np.int64)
    )
    positions = np.full(len(ids), -1, dtype=np.int64)
    for well_id in np.unique(wells):
        if str(well_id) not in layout:
            raise ValueError(f"Stage C contains unknown well: {well_id}")
        base, origin, count = layout[str(well_id)]
        mask = wells == well_id
        local = row_idx[mask] - origin
        if bool(np.any(local < 0)) or bool(np.any(local >= count)):
            raise ValueError(f"{well_id}: Stage C row_idx is outside row-risk bounds")
        positions[mask] = base + local
    if bool(np.any(positions < 0)):
        raise RuntimeError("Stage C row positions are incomplete")
    return positions


def _allocate_surface(n_rows: int, domains: Sequence[str]) -> dict[str, np.ndarray]:
    surface: dict[str, np.ndarray] = {}
    for domain in domains:
        surface[f"{domain}__nominated_code"] = np.full(
            n_rows, -1, dtype=np.int8
        )
        for suffix in (
            "nominated_tvt",
            "nominated_pred_abs_error",
            "anchor_tvt",
            "anchor_pred_abs_error",
            "selector_margin",
        ):
            surface[f"{domain}__{suffix}"] = np.full(
                n_rows, np.nan, dtype=np.float32
            )
    return surface


def scan_strict_nested_selector_surface(
    score_path: Path,
    row_risk: pd.DataFrame,
    config: Mapping[str, Any],
    scratch_dir: Path,
) -> tuple[dict[str, np.ndarray], np.memmap, dict[str, Any]]:
    candidate_order = [
        str(value)
        for value in get_nested(config, "candidate_contract.candidate_order")
    ]
    anchor = str(get_nested(config, "candidate_contract.anchor"))
    domain_specs = {
        str(get_nested(config, "candidate_contract.primary_domain.name")): [
            str(value)
            for value in get_nested(
                config, "candidate_contract.primary_domain.candidates"
            )
        ],
        str(get_nested(config, "candidate_contract.secondary_domain.name")): [
            str(value)
            for value in get_nested(
                config, "candidate_contract.secondary_domain.candidates"
            )
        ],
    }
    anchor_code = candidate_order.index(anchor)
    domain_codes: dict[str, np.ndarray] = {}
    other_codes: dict[str, np.ndarray] = {}
    for name, candidates in domain_specs.items():
        domain_codes[name] = np.asarray(
            [candidate_order.index(candidate) for candidate in candidates],
            dtype=np.int16,
        )
        other_codes[name] = np.asarray(
            [
                candidate_order.index(candidate)
                for candidate in candidates
                if candidate != anchor
            ],
            dtype=np.int16,
        )

    n_rows = len(row_risk)
    expected_long_rows = int(get_nested(config, "validation.expected_candidate_long_rows"))
    expected_nested_model_count = int(
        get_nested(config, "data.exp264_nested_score.expected_nested_model_count")
    )
    score_columns = [
        str(value)
        for value in get_nested(config, "data.exp264_nested_score.safe_columns")
    ]
    score_file = pq.ParquetFile(score_path)
    if score_file.metadata.num_rows != expected_long_rows:
        raise ValueError("Stage C candidate-long row count changed")

    layout = _well_row_layout(row_risk)
    row_folds = row_risk["fold"].to_numpy(np.int8)
    expected_candidate_block = np.asarray(candidate_order, dtype=str)
    covered = np.zeros(n_rows, dtype=bool)
    selector_generation_folds = np.full(n_rows, -1, dtype=np.int8)
    surface = _allocate_surface(n_rows, list(domain_specs))
    scratch_dir.mkdir(parents=True, exist_ok=True)
    candidate_values_path = scratch_dir / f".{OUTPUT_PREFIX}_candidate_tvt.float32"
    candidate_values = np.memmap(
        candidate_values_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, len(candidate_order)),
    )
    processed_base_rows = 0

    for row_group_index in range(score_file.metadata.num_row_groups):
        chunk = score_file.read_row_group(
            row_group_index, columns=score_columns
        ).to_pandas()
        if len(chunk) % len(candidate_order) != 0:
            raise ValueError(f"row group {row_group_index} breaks candidate blocks")
        base_rows = len(chunk) // len(candidate_order)
        candidate_blocks = (
            chunk["candidate_id"]
            .astype(str)
            .to_numpy()
            .reshape(base_rows, len(candidate_order))
        )
        if not np.all(candidate_blocks == expected_candidate_block[None, :]):
            raise ValueError(f"row group {row_group_index} candidate order changed")
        ids = (
            chunk["id"].astype(str).to_numpy().reshape(base_rows, -1)
        )
        wells = (
            chunk["well"].astype(str).to_numpy().reshape(base_rows, -1)
        )
        folds = chunk["outer_fold"].to_numpy(np.int8).reshape(base_rows, -1)
        downstream_folds = (
            chunk["downstream_outer_fold"]
            .to_numpy(np.int8)
            .reshape(base_rows, -1)
        )
        model_counts = (
            chunk["nested_model_count"]
            .to_numpy(np.int8)
            .reshape(base_rows, -1)
        )
        if not (
            np.all(ids == ids[:, :1])
            and np.all(wells == wells[:, :1])
            and np.all(folds == folds[:, :1])
            and np.all(downstream_folds == downstream_folds[:, :1])
            and np.all(model_counts == expected_nested_model_count)
        ):
            raise ValueError(f"row group {row_group_index} nested block contract failed")
        positions = _positions_from_identity(ids[:, 0], wells[:, 0], layout)
        if len(np.unique(positions)) != len(positions):
            raise ValueError(f"row group {row_group_index} repeats base rows")
        if bool(np.any(covered[positions])):
            raise ValueError(f"row group {row_group_index} overlaps prior rows")
        if not np.array_equal(folds[:, 0], downstream_folds[:, 0]):
            raise ValueError(f"row group {row_group_index} is not outer-valid")
        selector_generation_folds[positions] = folds[:, 0]

        values = (
            chunk["candidate_tvt"]
            .to_numpy(np.float32)
            .reshape(base_rows, len(candidate_order))
        )
        error_scores = (
            chunk["pred_abs_error"]
            .to_numpy(np.float32)
            .reshape(base_rows, len(candidate_order))
        )
        probabilities = (
            chunk["p_within10"]
            .to_numpy(np.float32)
            .reshape(base_rows, len(candidate_order))
        )
        if not (
            np.isfinite(values).all()
            and np.isfinite(error_scores).all()
            and np.isfinite(probabilities).all()
            and bool(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        ):
            raise ValueError(f"row group {row_group_index} has invalid scores")
        candidate_values[positions, :] = values

        for domain_name, codes in other_codes.items():
            local_code = np.argmin(error_scores[:, codes], axis=1)
            nominated_code = codes[local_code]
            row_numbers = np.arange(base_rows, dtype=np.int64)
            nominated_score = error_scores[row_numbers, nominated_code]
            anchor_score = error_scores[:, anchor_code]
            surface[f"{domain_name}__nominated_code"][positions] = (
                nominated_code.astype(np.int8)
            )
            surface[f"{domain_name}__nominated_tvt"][positions] = values[
                row_numbers, nominated_code
            ]
            surface[f"{domain_name}__nominated_pred_abs_error"][
                positions
            ] = nominated_score
            surface[f"{domain_name}__anchor_tvt"][positions] = values[
                :, anchor_code
            ]
            surface[f"{domain_name}__anchor_pred_abs_error"][
                positions
            ] = anchor_score
            surface[f"{domain_name}__selector_margin"][positions] = (
                anchor_score - nominated_score
            )

        covered[positions] = True
        processed_base_rows += base_rows
        if (
            row_group_index == 0
            or (row_group_index + 1) % 25 == 0
            or row_group_index + 1 == score_file.metadata.num_row_groups
        ):
            print(
                "strict-nested surface "
                f"{processed_base_rows:,}/{n_rows:,} base rows",
                flush=True,
            )

    candidate_values.flush()
    if processed_base_rows != n_rows or not bool(np.all(covered)):
        raise ValueError("strict-nested selector surface does not cover every row")
    expected_folds = np.asarray(
        get_nested(config, "validation.expected_folds"), dtype=np.int8
    )
    if not np.array_equal(
        np.sort(np.unique(selector_generation_folds)),
        expected_folds,
    ):
        raise ValueError("selector generation folds differ from expected five folds")
    for name, values in surface.items():
        if name.endswith("__nominated_code"):
            if bool(np.any(values < 0)):
                raise ValueError(f"{name} is incomplete")
        elif not np.isfinite(values).all():
            raise ValueError(f"{name} is incomplete or non-finite")
    surface_sha = array_bundle_sha256(
        {
            "candidate_order": np.asarray(candidate_order, dtype=str),
            "selector_generation_outer_fold": selector_generation_folds,
            **surface,
        }
    )
    fold_mismatch = selector_generation_folds != row_folds
    audit = {
        "candidate_long_rows": expected_long_rows,
        "base_rows": processed_base_rows,
        "candidates_per_row": len(candidate_order),
        "candidate_order": candidate_order,
        "primary_domain_codes": domain_codes[
            str(get_nested(config, "candidate_contract.primary_domain.name"))
        ].tolist(),
        "secondary_domain_codes": domain_codes[
            str(get_nested(config, "candidate_contract.secondary_domain.name"))
        ].tolist(),
        "anchor_code": anchor_code,
        "covered_rows": int(covered.sum()),
        "reporting_fold_source": "exp226_saved_fold",
        "selector_generation_fold_source": (
            "exp264_outer_fold_equals_downstream_outer_fold"
        ),
        "selector_generation_vs_reporting_fold_mismatch_rows": int(
            fold_mismatch.sum()
        ),
        "selector_generation_vs_reporting_fold_mismatch_wells": int(
            row_risk.loc[fold_mismatch, "well_id"].nunique()
        ),
        "selector_surface_content_sha256": surface_sha,
        "candidate_value_storage": "temporary_float32_memmap_not_a_prediction",
        "candidate_values_path": str(candidate_values_path),
    }
    return surface, candidate_values, audit


def freeze_target_free_surface(
    row_risk: pd.DataFrame,
    surface: Mapping[str, np.ndarray],
    selector_audit: Mapping[str, Any],
    scientific_contract: Mapping[str, Any],
    artifacts_dir: Path,
    ledger: TruthAccessLedger,
) -> dict[str, Any]:
    folds = row_risk["fold"].to_numpy(np.int8)
    weak_quartile, weak_boundaries = crossfit_extreme_quartiles(
        row_risk[RISK_FEATURE_COLUMN].to_numpy(np.float64),
        folds,
    )
    row_risk = row_risk.copy()
    row_risk["crossfit_weak_quartile"] = weak_quartile

    margin_bins: dict[str, np.ndarray] = {}
    margin_boundaries: dict[str, dict[int, list[float]]] = {}
    for domain_name in ("primitive_pair_bank", "primitive_fixed_bank"):
        bins, boundaries = crossfit_quantile_bins(
            surface[f"{domain_name}__selector_margin"],
            folds,
            quantiles=np.arange(0.1, 1.0, 0.1).tolist(),
        )
        margin_bins[domain_name] = bins
        margin_boundaries[domain_name] = boundaries

    feature_columns = {
        "well_id": row_risk["well_id"].astype(str).to_numpy(),
        "row_idx": row_risk["row_idx"].to_numpy(np.int64),
        "suffix_offset": row_risk["suffix_offset"].to_numpy(np.int64),
        "fold": folds,
        RISK_FEATURE_COLUMN: row_risk[RISK_FEATURE_COLUMN].to_numpy(np.float32),
        CIRCULAR_FEATURE_COLUMN: row_risk[
            CIRCULAR_FEATURE_COLUMN
        ].to_numpy(np.float32),
        "covering_block_count": row_risk["covering_block_count"].to_numpy(np.int8),
        "crossfit_weak_quartile": row_risk[
            "crossfit_weak_quartile"
        ].to_numpy(np.int8),
        "primitive_pair_bank_margin_decile": margin_bins[
            "primitive_pair_bank"
        ],
        "primitive_fixed_bank_margin_decile": margin_bins[
            "primitive_fixed_bank"
        ],
    }
    feature_path = (
        artifacts_dir / f"{OUTPUT_PREFIX}_frozen_row_risk.csv.gz"
    )
    feature_report = write_deterministic_gzip_columns(feature_columns, feature_path)
    feature_schema_sha = feature_report["schema_sha256"]
    feature_content_sha = feature_report["decompressed_content_sha256"]
    freeze_evidence = {
        "feature_schema_sha256": feature_schema_sha,
        "feature_content_sha256": feature_content_sha,
        "selector_surface_content_sha256": selector_audit[
            "selector_surface_content_sha256"
        ],
        "scientific_contract_sha256": scientific_contract[
            "scientific_contract_sha256"
        ],
        "truth_columns_read_before_freeze": (
            ledger.truth_columns_read_before_freeze
        ),
        "weak_quartile_boundaries_by_valid_fold": weak_boundaries,
        "margin_decile_boundaries_by_valid_fold": margin_boundaries,
        "feature_output": feature_report,
    }
    ledger.freeze_features(freeze_evidence)
    return {
        "row_risk": row_risk,
        "margin_bins": margin_bins,
        "evidence": freeze_evidence,
    }


# %% [markdown]
# ## 7. Late truth attachment and candidate-advantage metrics


# %%
def load_late_truth_and_roles(
    row_risk: pd.DataFrame,
    paths: Mapping[str, Path],
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> dict[str, np.ndarray]:
    fold_spec = get_nested(config, "data.fold_assignment")
    truth_columns = [str(value) for value in fold_spec["truth_columns"]]
    ledger.read_late_truth(truth_columns)
    truth = pd.read_csv(
        paths["fold_truth"],
        usecols=truth_columns,
        dtype={"well_id": str},
    )
    truth["row_idx"] = pd.to_numeric(
        truth["row_idx"], errors="raise"
    ).astype(np.int64)
    truth["true_tvt"] = pd.to_numeric(
        truth.pop("tvt_true"), errors="raise"
    ).astype(np.float64)
    truth.sort_values(["well_id", "row_idx"], kind="mergesort", inplace=True)
    truth.reset_index(drop=True, inplace=True)
    if len(truth) != len(row_risk):
        raise ValueError("late truth/row-risk row count mismatch")
    for column in ("well_id", "row_idx"):
        if not np.array_equal(
            truth[column].to_numpy(), row_risk[column].to_numpy()
        ):
            raise ValueError(f"late truth differs from row-risk on {column}")
    if not np.isfinite(truth["true_tvt"].to_numpy(np.float64)).all():
        raise ValueError("late truth contains non-finite TVT")

    hidden_spec = get_nested(config, "data.hidden_like_assignment")
    role_columns = {
        str(scope): str(column)
        for scope, column in hidden_spec["role_columns"].items()
    }
    roles = pd.read_csv(
        paths["hidden_like"],
        usecols=["well_id", *role_columns.values()],
        dtype={"well_id": str},
    )
    if roles["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment has duplicate wells")
    roles.set_index("well_id", inplace=True)
    output: dict[str, np.ndarray] = {
        "true_tvt": truth["true_tvt"].to_numpy(np.float64)
    }
    for scope, role_column in role_columns.items():
        mapped = row_risk["well_id"].map(roles[role_column])
        if mapped.isna().any():
            raise ValueError(f"hidden-like role {scope} does not cover every well")
        output[scope] = mapped.eq("valid").to_numpy(bool)
    return output


def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("AUC labels/scores contract is invalid")
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy(np.float64)
    positive_rank_sum = float(ranks[labels].sum())
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def stratified_pairwise_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    strata: np.ndarray,
) -> tuple[float | None, int]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    strata = np.asarray(strata)
    if not (len(labels) == len(scores) == len(strata)):
        raise ValueError("stratified AUC arrays have different lengths")
    numerator = 0.0
    denominator = 0
    for stratum in np.unique(strata):
        mask = strata == stratum
        positives = int(labels[mask].sum())
        negatives = int((~labels[mask]).sum())
        if positives == 0 or negatives == 0:
            continue
        auc = roc_auc_binary(labels[mask], scores[mask])
        pairs = positives * negatives
        numerator += float(auc) * pairs
        denominator += pairs
    if denominator == 0:
        return None, 0
    return float(numerator / denominator), int(denominator)


def _domain_arrays_after_truth(
    *,
    domain_name: str,
    domain_candidates: Sequence[str],
    candidate_order: Sequence[str],
    anchor: str,
    surface: Mapping[str, np.ndarray],
    candidate_values: np.ndarray,
    truth: np.ndarray,
) -> dict[str, np.ndarray]:
    anchor_code = list(candidate_order).index(anchor)
    other_codes = np.asarray(
        [
            list(candidate_order).index(candidate)
            for candidate in domain_candidates
            if candidate != anchor
        ],
        dtype=np.int16,
    )
    anchor_tvt = np.asarray(
        surface[f"{domain_name}__anchor_tvt"], dtype=np.float64
    )
    nominated_tvt = np.asarray(
        surface[f"{domain_name}__nominated_tvt"], dtype=np.float64
    )
    anchor_error = np.abs(anchor_tvt - truth)
    nominated_error = np.abs(nominated_tvt - truth)
    oracle_error = np.min(
        np.abs(np.asarray(candidate_values[:, other_codes], dtype=np.float64) - truth[:, None]),
        axis=1,
    )
    return {
        "anchor_code": np.full(len(truth), anchor_code, dtype=np.int8),
        "nominated_code": np.asarray(
            surface[f"{domain_name}__nominated_code"], dtype=np.int8
        ),
        "anchor_abs_error": anchor_error,
        "nominated_abs_error": nominated_error,
        "oracle_other_abs_error": oracle_error,
        "nominated_recovery10": nominated_error < 10.0,
        "oracle_recoverable10": oracle_error < 10.0,
        "realized_advantage_ft": anchor_error - nominated_error,
        "selector_margin": np.asarray(
            surface[f"{domain_name}__selector_margin"], dtype=np.float64
        ),
    }


def build_scope_metrics(
    row_risk: pd.DataFrame,
    late: Mapping[str, np.ndarray],
    surface: Mapping[str, np.ndarray],
    candidate_values: np.ndarray,
    frozen: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_order = [
        str(value)
        for value in get_nested(config, "candidate_contract.candidate_order")
    ]
    anchor = str(get_nested(config, "candidate_contract.anchor"))
    domain_specs = {
        str(get_nested(config, "candidate_contract.primary_domain.name")): [
            str(value)
            for value in get_nested(
                config, "candidate_contract.primary_domain.candidates"
            )
        ],
        str(get_nested(config, "candidate_contract.secondary_domain.name")): [
            str(value)
            for value in get_nested(
                config, "candidate_contract.secondary_domain.candidates"
            )
        ],
    }
    folds = row_risk["fold"].to_numpy(np.int8)
    risk = row_risk[RISK_FEATURE_COLUMN].to_numpy(np.float64)
    circular = row_risk[CIRCULAR_FEATURE_COLUMN].to_numpy(np.float64)
    weak_quartile = row_risk["crossfit_weak_quartile"].to_numpy(np.int8)
    true_tvt = np.asarray(late["true_tvt"], dtype=np.float64)
    well_values = row_risk["well_id"].astype(str).to_numpy()
    bad_threshold = float(
        get_nested(config, "stage_0.cohort.anchor_bad10_threshold_ft")
    )
    scopes: list[tuple[str, np.ndarray]] = [
        ("overall", np.ones(len(row_risk), dtype=bool))
    ]
    for fold in get_nested(config, "validation.expected_folds"):
        scopes.append((f"fold_{int(fold)}", folds == int(fold)))
    for scope in ("hidden_like_spatial", "hidden_like_typewell_purged"):
        scopes.append((scope, np.asarray(late[scope], dtype=bool)))

    metric_rows: list[dict[str, Any]] = []
    nomination_rows: list[dict[str, Any]] = []
    for domain_name, domain_candidates in domain_specs.items():
        arrays = _domain_arrays_after_truth(
            domain_name=domain_name,
            domain_candidates=domain_candidates,
            candidate_order=candidate_order,
            anchor=anchor,
            surface=surface,
            candidate_values=candidate_values,
            truth=true_tvt,
        )
        cohort = arrays["anchor_abs_error"] >= bad_threshold
        margin_bins = np.asarray(
            frozen["margin_bins"][domain_name], dtype=np.int8
        )
        conditional_strata = folds.astype(np.int16) * 10 + margin_bins
        for scope_name, scope_mask in scopes:
            mask = cohort & scope_mask
            labels = arrays["nominated_recovery10"][mask]
            real_auc = roc_auc_binary(labels, risk[mask])
            circular_auc = roc_auc_binary(labels, circular[mask])
            oracle_auc = roc_auc_binary(
                arrays["oracle_recoverable10"][mask], risk[mask]
            )
            conditional_auc, conditional_pairs = stratified_pairwise_auc(
                labels,
                risk[mask],
                conditional_strata[mask],
            )
            q1 = mask & (weak_quartile == 1)
            q4 = mask & (weak_quartile == 4)
            q1_mean = (
                float(arrays["realized_advantage_ft"][q1].mean())
                if bool(q1.any())
                else None
            )
            q4_mean = (
                float(arrays["realized_advantage_ft"][q4].mean())
                if bool(q4.any())
                else None
            )
            metric_rows.append(
                {
                    "domain": domain_name,
                    "scope": scope_name,
                    "cohort_rows": int(mask.sum()),
                    "wells": int(pd.Series(well_values[mask]).nunique()),
                    "positive_rows": int(labels.sum()),
                    "negative_rows": int((~labels).sum()),
                    "nominated_recovery10_prevalence": (
                        float(labels.mean()) if len(labels) else None
                    ),
                    "oracle_recoverable10_prevalence": (
                        float(arrays["oracle_recoverable10"][mask].mean())
                        if bool(mask.any())
                        else None
                    ),
                    "real_nominated_recovery10_auc": real_auc,
                    "circular_nominated_recovery10_auc": circular_auc,
                    "real_minus_circular_auc": (
                        float(real_auc - circular_auc)
                        if real_auc is not None and circular_auc is not None
                        else None
                    ),
                    "oracle_recoverable10_auc": oracle_auc,
                    "margin_conditional_auc": conditional_auc,
                    "margin_conditional_pairs": conditional_pairs,
                    "q1_rows": int(q1.sum()),
                    "q4_rows": int(q4.sum()),
                    "q1_mean_realized_advantage_ft": q1_mean,
                    "q4_mean_realized_advantage_ft": q4_mean,
                    "q4_minus_q1_mean_realized_advantage_ft": (
                        float(q4_mean - q1_mean)
                        if q1_mean is not None and q4_mean is not None
                        else None
                    ),
                    "mean_realized_advantage_ft": (
                        float(arrays["realized_advantage_ft"][mask].mean())
                        if bool(mask.any())
                        else None
                    ),
                }
            )

        nominated_codes = arrays["nominated_code"]
        for code, candidate in enumerate(candidate_order):
            if candidate not in domain_candidates or candidate == anchor:
                continue
            nomination_rows.append(
                {
                    "domain": domain_name,
                    "candidate_code": code,
                    "candidate": candidate,
                    "all_rows": int(np.sum(nominated_codes == code)),
                    "bad10_cohort_rows": int(
                        np.sum((nominated_codes == code) & cohort)
                    ),
                    "bad10_recovery_rows": int(
                        np.sum(
                            (nominated_codes == code)
                            & cohort
                            & arrays["nominated_recovery10"]
                        )
                    ),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    nominations = pd.DataFrame(nomination_rows)
    return metrics, nominations


# %% [markdown]
# ## 8. Technical and scientific promotion gates


# %%
def _metric_lookup(
    metrics: pd.DataFrame, domain: str, scope: str
) -> pd.Series:
    selected = metrics.loc[
        metrics["domain"].eq(domain) & metrics["scope"].eq(scope)
    ]
    if len(selected) != 1:
        raise ValueError(f"metric row is not unique for {domain}/{scope}")
    return selected.iloc[0]


def _at_least(value: Any, threshold: float) -> bool:
    return value is not None and np.isfinite(float(value)) and float(value) >= threshold


def _strictly_above(value: Any, threshold: float) -> bool:
    return value is not None and np.isfinite(float(value)) and float(value) > threshold


def evaluate_gates(
    *,
    metrics: pd.DataFrame,
    row_risk: pd.DataFrame,
    blocks: pd.DataFrame,
    selector_audit: Mapping[str, Any],
    preflight: Mapping[str, Any],
    frozen_evidence: Mapping[str, Any],
    readout_reports: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> dict[str, Any]:
    technical = get_nested(config, "stage_0.technical_gate_all_required")
    primary_name = str(get_nested(config, "candidate_contract.primary_domain.name"))
    secondary_name = str(
        get_nested(config, "candidate_contract.secondary_domain.name")
    )
    required_scopes = [
        "overall",
        *[
            f"fold_{int(fold)}"
            for fold in get_nested(config, "validation.expected_folds")
        ],
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    ]
    minimum_class_rows = int(
        technical["minimum_positive_rows_per_required_scope"]
    )
    class_support = {
        scope: {
            "positive_rows": int(
                _metric_lookup(metrics, primary_name, scope)["positive_rows"]
            ),
            "negative_rows": int(
                _metric_lookup(metrics, primary_name, scope)["negative_rows"]
            ),
        }
        for scope in required_scopes
    }
    required_hashes = [
        *preflight["input_sha256"].values(),
        preflight["input_sha256"]["stage_c_score_schema_sha256"],
        frozen_evidence["feature_schema_sha256"],
        frozen_evidence["feature_content_sha256"],
        frozen_evidence["selector_surface_content_sha256"],
        frozen_evidence["scientific_contract_sha256"],
        *[
            str(report.get("sha256", ""))
            for report in readout_reports.values()
        ],
    ]
    feature_values = row_risk[
        [RISK_FEATURE_COLUMN, CIRCULAR_FEATURE_COLUMN]
    ].to_numpy(np.float64)
    technical_checks = {
        "expected_rows": len(row_risk) == int(technical["expected_rows"]),
        "expected_wells": row_risk["well_id"].nunique()
        == int(technical["expected_wells"]),
        "expected_blocks": len(blocks) == int(technical["expected_blocks"]),
        "expected_folds": row_risk["fold"].nunique()
        == int(technical["expected_folds"]),
        "expected_candidate_long_rows": int(
            selector_audit["candidate_long_rows"]
        )
        == int(technical["expected_candidate_long_rows"]),
        "expected_candidates_per_row": int(
            selector_audit["candidates_per_row"]
        )
        == int(technical["expected_candidates_per_row"]),
        "primary_domain_candidates": len(
            selector_audit["primary_domain_codes"]
        )
        == int(technical["primary_domain_candidates"]),
        "secondary_domain_candidates": len(
            selector_audit["secondary_domain_codes"]
        )
        == int(technical["secondary_domain_candidates"]),
        "row_coverage_fraction": int(selector_audit["covered_rows"])
        == len(row_risk),
        "feature_finite_fraction": np.isfinite(feature_values).all(),
        "feature_range": bool(
            np.all(
                (feature_values >= float(technical["feature_range"][0]))
                & (feature_values <= float(technical["feature_range"][1]))
            )
        ),
        "truth_columns_read_before_freeze": (
            ledger.truth_columns_read_before_freeze
            == int(technical["truth_columns_read_before_freeze"])
        ),
        "required_class_support": all(
            counts["positive_rows"] >= minimum_class_rows
            and counts["negative_rows"] >= minimum_class_rows
            for counts in class_support.values()
        ),
        "zero_model_pf_prediction_contract": all(
            int(technical[name]) == 0
            for name in (
                "model_configs",
                "lightgbm_configs",
                "trained_folds",
                "boosters",
                "pf_runs",
                "prediction_rows",
            )
        ),
        "input_schema_feature_and_readout_sha": all(
            len(str(value)) == 64 for value in required_hashes
        ),
    }
    technical_passed = all(bool(value) for value in technical_checks.values())

    scientific = get_nested(config, "stage_0.scientific_gate_all_required")
    primary_gate = scientific["primary_domain"]
    secondary_gate = scientific["secondary_domain"]
    primary_overall = _metric_lookup(metrics, primary_name, "overall")
    secondary_overall = _metric_lookup(metrics, secondary_name, "overall")
    primary_folds = [
        _metric_lookup(metrics, primary_name, f"fold_{int(fold)}")
        for fold in get_nested(config, "validation.expected_folds")
    ]
    hidden_spatial = _metric_lookup(
        metrics, primary_name, "hidden_like_spatial"
    )
    hidden_typewell = _metric_lookup(
        metrics, primary_name, "hidden_like_typewell_purged"
    )
    scientific_checks = {
        "primary_pooled_auc": _at_least(
            primary_overall["real_nominated_recovery10_auc"],
            float(primary_gate["minimum_pooled_nominated_recovery10_auc"]),
        ),
        "primary_auc_gain_over_circular": _at_least(
            primary_overall["real_minus_circular_auc"],
            float(primary_gate["minimum_auc_gain_over_circular"]),
        ),
        "primary_fold_auc_count": sum(
            _strictly_above(
                row["real_nominated_recovery10_auc"], 0.50
            )
            for row in primary_folds
        )
        >= int(
            primary_gate[
                "minimum_folds_with_real_auc_strictly_above_0p50"
            ]
        ),
        "primary_hidden_like_spatial_auc": _at_least(
            hidden_spatial["real_nominated_recovery10_auc"],
            float(primary_gate["minimum_hidden_like_spatial_auc"]),
        ),
        "primary_hidden_like_typewell_purged_auc": _at_least(
            hidden_typewell["real_nominated_recovery10_auc"],
            float(primary_gate["minimum_hidden_like_typewell_purged_auc"]),
        ),
        "primary_margin_conditional_auc": _at_least(
            primary_overall["margin_conditional_auc"],
            float(primary_gate["minimum_pooled_margin_conditional_auc"]),
        ),
        "primary_fold_margin_conditional_auc_count": sum(
            _strictly_above(row["margin_conditional_auc"], 0.50)
            for row in primary_folds
        )
        >= int(
            primary_gate[
                "minimum_folds_with_margin_conditional_auc_strictly_above_0p50"
            ]
        ),
        "primary_q4_minus_q1_advantage": _at_least(
            primary_overall["q4_minus_q1_mean_realized_advantage_ft"],
            float(
                primary_gate[
                    "minimum_q4_minus_q1_mean_realized_advantage_ft"
                ]
            ),
        ),
        "primary_fold_positive_q4_minus_q1_count": sum(
            _strictly_above(
                row["q4_minus_q1_mean_realized_advantage_ft"], 0.0
            )
            for row in primary_folds
        )
        >= int(
            primary_gate[
                "minimum_folds_with_positive_q4_minus_q1_advantage"
            ]
        ),
        "secondary_pooled_auc": _at_least(
            secondary_overall["real_nominated_recovery10_auc"],
            float(secondary_gate["minimum_pooled_nominated_recovery10_auc"]),
        ),
        "secondary_auc_gain_over_circular": _at_least(
            secondary_overall["real_minus_circular_auc"],
            float(secondary_gate["minimum_auc_gain_over_circular"]),
        ),
        "secondary_q4_minus_q1_advantage": _at_least(
            secondary_overall["q4_minus_q1_mean_realized_advantage_ft"],
            float(
                secondary_gate[
                    "minimum_q4_minus_q1_mean_realized_advantage_ft"
                ]
            ),
        ),
    }
    scientific_passed = all(bool(value) for value in scientific_checks.values())
    stage_0_passed = technical_passed and scientific_passed
    decision = (
        "stage_0_passed_request_separate_stage_1_approval"
        if stage_0_passed
        else "stage_0_failed_close_without_rescue"
    )
    return {
        "technical_passed": technical_passed,
        "technical_checks": technical_checks,
        "primary_required_scope_class_support": class_support,
        "scientific_passed": scientific_passed,
        "scientific_checks": scientific_checks,
        "stage_0_passed": stage_0_passed,
        "decision": decision,
    }


# %% [markdown]
# ## 9. Generated evidence and summaries


# %%
def run_stage_0(
    *,
    config: Mapping[str, Any],
    repo_root: Path,
    package_dir: Path,
    artifacts_dir: Path,
    require_run_approval: bool = True,
) -> dict[str, Any]:
    contract = validate_scientific_contract(
        config, require_run_approval=require_run_approval
    )
    ledger = TruthAccessLedger()
    paths = resolve_all_inputs(repo_root)
    preflight = preflight_inputs(paths, config, ledger)
    blocks = load_target_free_blocks(paths, config, ledger)
    active_feature = get_nested(config, "features.active")[0]
    row_risk = aggregate_overlapping_block_risk(
        blocks,
        block_rows=int(active_feature["block_rows"]),
        stride_rows=int(active_feature["stride_rows"]),
    )
    row_risk = attach_target_free_fold(
        row_risk, paths["fold_truth"], config, ledger
    )
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(row_risk) != expected_rows or row_risk["well_id"].nunique() != expected_wells:
        raise ValueError("row-risk row/well contract changed")

    surface, candidate_values, selector_audit = (
        scan_strict_nested_selector_surface(
            paths["stage_c_score"],
            row_risk,
            config,
            artifacts_dir,
        )
    )
    frozen = freeze_target_free_surface(
        row_risk,
        surface,
        selector_audit,
        contract,
        artifacts_dir,
        ledger,
    )
    late = load_late_truth_and_roles(
        frozen["row_risk"], paths, config, ledger
    )
    scope_metrics, nominations = build_scope_metrics(
        frozen["row_risk"],
        late,
        surface,
        candidate_values,
        frozen,
        config,
    )
    metric_report = write_csv_with_sha(
        scope_metrics,
        artifacts_dir / f"{OUTPUT_PREFIX}_stage_0_scope_metrics.csv",
    )
    nomination_report = write_csv_with_sha(
        nominations,
        artifacts_dir / f"{OUTPUT_PREFIX}_candidate_nomination_distribution.csv",
    )
    readout_reports = {
        "scope_metrics": metric_report,
        "candidate_nomination_distribution": nomination_report,
    }
    gates = evaluate_gates(
        metrics=scope_metrics,
        row_risk=frozen["row_risk"],
        blocks=blocks,
        selector_audit=selector_audit,
        preflight=preflight,
        frozen_evidence=frozen["evidence"],
        readout_reports=readout_reports,
        config=config,
        ledger=ledger,
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": (
            "stage_0_passed_stage_1_separate_approval_required"
            if gates["stage_0_passed"]
            else "stage_0_failed_closed"
        ),
        "stage": "stage_0_zero_booster_candidate_advantage_readout",
        "metric": "nominated_recovery10_auc",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "route": "ml_model",
        "execution_counts": {
            "diagnostic_variants": 1,
            "reporting_folds": 5,
            "model_configs": 0,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "pf_runs": 0,
            "parent_control_retraining": 0,
            "prediction_rows": 0,
            "submission_rows": 0,
        },
        "input_preflight": preflight,
        "selector_audit": selector_audit,
        "freeze": frozen["evidence"],
        "truth_access": {
            "truth_columns_read_before_freeze": (
                ledger.truth_columns_read_before_freeze
            ),
            "late_truth_columns_read": ledger.late_truth_columns_read,
        },
        "readout_outputs": readout_reports,
        "gates": gates,
        "primary_overall": _metric_lookup(
            scope_metrics, "primitive_pair_bank", "overall"
        ).to_dict(),
        "secondary_overall": _metric_lookup(
            scope_metrics, "primitive_fixed_bank", "overall"
        ).to_dict(),
        "scientific_contract_sha256": contract[
            "scientific_contract_sha256"
        ],
        "notes": (
            "Stage 0 cannot fit a selector or promote inference. Every gate must "
            "PASS before a separate Stage 1 implementation/run approval."
        ),
    }
    summary_path = artifacts_dir / f"{OUTPUT_PREFIX}_stage_0_summary.json"
    summary_sha = write_json(summary_path, summary)
    summary["stage_0_summary_sha256"] = summary_sha
    metrics_path = (
        KAGGLE_WORKING_ROOT / "metrics.json"
        if KAGGLE_WORKING_ROOT.exists()
        else package_dir / "metrics.json"
    )
    write_json(metrics_path, summary)
    candidate_values_path = Path(selector_audit["candidate_values_path"])
    del candidate_values
    if candidate_values_path.exists():
        candidate_values_path.unlink()
    return summary


# %% [markdown]
# ## 10. Setup and configuration
#
# The canonical experiment notebook remains untouched. This compact
# self-contained candidate is implementation-only until the user separately
# approves canonical notebook adoption, Kaggle packaging, and the private CPU
# Stage 0 run.


# %%
if __name__ == "__main__":
    PACKAGE_DIR = resolve_package_dir()
    REPO_ROOT = find_repo_root(PACKAGE_DIR)
    CONFIG = load_experiment_config(PACKAGE_DIR)
    ARTIFACTS_DIR = (
        KAGGLE_WORKING_ROOT if KAGGLE_WORKING_ROOT.exists() else PACKAGE_DIR
    ) / "artifacts"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT = validate_scientific_contract(CONFIG)
    print("Experiment:", EXPERIMENT_NAME)
    print("Route:", get_nested(CONFIG, "experiment.route"))
    print("Parent:", get_nested(CONFIG, "lineage.parent"))
    print("Auxiliary:", get_nested(CONFIG, "lineage.auxiliary_source"))
    print(
        "Stage 0 counts:",
        CONTRACT["execution_counts"],
    )
    print(
        "Future Stage 1 (disabled):",
        get_nested(
            CONFIG,
            "execution_contract."
            "future_stage_1_if_all_gates_pass_and_separately_approved",
        ),
    )


# %% [markdown]
# ## 11. Stage 0 execution orchestration
#
# `run_stage_0` immediately checks the separate run-approval flags. With the
# repository implementation-only config it fails closed before resolving or
# reading any experiment input.


# %%
if __name__ == "__main__":
    STAGE_0_SUMMARY = run_stage_0(
        config=CONFIG,
        repo_root=REPO_ROOT,
        package_dir=PACKAGE_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        require_run_approval=True,
    )
    print(
        json.dumps(
            STAGE_0_SUMMARY["gates"],
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    )
