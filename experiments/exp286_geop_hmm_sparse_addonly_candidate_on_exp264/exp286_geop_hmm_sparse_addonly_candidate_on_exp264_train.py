# %% [markdown]
# # exp286 geop HMM candidate added to the exp264 selector
#
# Stage 0 audited a fixed sparse gate and showed that the full `geop_hmm`
# candidate has oracle headroom while the fixed gate does not retain it. Stage B
# therefore adds the saved path as a full-coverage thirteenth primitive, with
# the same ID, availability, generic proxy, and native-confidence contract as
# the other paths. Stage B improved the original selector, so the approved
# continuation builds leakage-safe nested compact features in Stage C and then
# evaluates them in the downstream Stage D model.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Notebook-safe configuration, path, and SHA helpers
# 3. Exp263 manifest and corrected exp264 candidate-bank reconstruction
# 4. Target-free well gate construction and freeze boundary
# 5. Post-freeze exp279 candidate and truth attachment
# 6. Row, 512-block, whole-well oracle and unique-best readouts
# 7. Paired 200-well shadow runtime guard
# 8. Stage 0 orchestration, guards, and artifact persistence
# 9. Full-all-well 13-candidate selector helpers
# 10. Setup, contract preview, and Stage 0 / Stage B / Stage C execution

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.candidate_selector_pipeline import (
    read_yaml as read_candidate_contract,
    resolve_existing_path,
    resolve_exp263_cache_root,
    resolve_stage_c_artifact_root,
    sha256_file as selector_sha256_file,
    verify_exp263_root,
)
from src.geop_hmm_selector_audit import (
    resolve_geop_candidate_source,
    run_geop_hmm_stage_d_addonly,
    run_geop_hmm_selector_stage_b,
    run_geop_hmm_selector_stage_c,
    stage_d_full13_cost_contract,
    validate_full_contract,
)


# %% [markdown]
# ## 2. Notebook-safe configuration, path, and SHA helpers

# %%
EXPERIMENT_NAME = "exp286_geop_hmm_sparse_addonly_candidate_on_exp264"
OUTPUT_PREFIX = EXPERIMENT_NAME
PACKAGE_DIR = Path.cwd()


def in_notebook_runtime() -> bool:
    try:
        shell = get_ipython()  # type: ignore[name-defined]
    except NameError:
        return False
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP286_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def find_config_path() -> Path:
    direct = PACKAGE_DIR / "config.yaml"
    if direct.exists():
        return direct
    nested = (
        PACKAGE_DIR
        / "experiments"
        / EXPERIMENT_NAME
        / "config.yaml"
    )
    if nested.exists():
        return nested
    matches = sorted(PACKAGE_DIR.rglob(f"{EXPERIMENT_NAME}/config.yaml"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError("exp286 config.yaml was not found unambiguously")


def get_nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_gzip(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def frame_schema_sha256(frame: pd.DataFrame) -> str:
    schema = [{"name": str(column), "dtype": str(frame[column].dtype)} for column in frame]
    return json_sha256(schema)


def frame_content_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def expand_existing_paths(patterns: Sequence[str]) -> list[Path]:
    import glob

    found: list[Path] = []
    for raw_pattern in patterns:
        pattern = str(raw_pattern)
        path = Path(pattern)
        if path.is_absolute() and not any(token in pattern for token in "*?["):
            if path.exists():
                found.append(path)
            continue
        for match in glob.glob(pattern, recursive=True):
            candidate = Path(match)
            if candidate.exists():
                found.append(candidate)
        if not path.is_absolute():
            local = PACKAGE_DIR / path
            if local.exists():
                found.append(local)
    unique = {str(path.resolve()): path for path in found}
    return [unique[key] for key in sorted(unique)]


def resolve_file(
    patterns: Sequence[str],
    *,
    label: str,
    expected_sha256: str | None = None,
) -> Path:
    candidates = [path for path in expand_existing_paths(patterns) if path.is_file()]
    if expected_sha256:
        matching = [path for path in candidates if sha256_file(path) == expected_sha256]
        if len(matching) == 1:
            return matching[0]
        if len(matching) > 1:
            return sorted(matching, key=lambda path: len(str(path)))[0]
        if candidates:
            detail = {str(path): sha256_file(path) for path in candidates}
            raise ValueError(f"{label} SHA mismatch: {detail}")
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"{label} not found from patterns: {patterns}")
    raise ValueError(f"{label} resolved to multiple files: {candidates}")


def runtime_artifacts_dir() -> Path:
    if Path("/kaggle/working").exists():
        path = Path("/kaggle/working/artifacts")
    else:
        path = PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "artifacts"
        if not path.parent.exists():
            path = PACKAGE_DIR / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_metrics_path() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/metrics.json")
    experiment_path = PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "metrics.json"
    return experiment_path if experiment_path.parent.exists() else PACKAGE_DIR / "metrics.json"


# %% [markdown]
# ## 3. Exp263 manifest and corrected exp264 candidate-bank reconstruction

# %%
@dataclass
class CandidateBank:
    keys: pd.DataFrame
    candidate_ids: list[str]
    values: np.ndarray
    primitive_values: dict[str, np.ndarray]
    gate_row_inputs: dict[str, np.ndarray]
    manifest: dict[str, Any]
    manifest_path: Path
    input_evidence: list[dict[str, Any]]


VALUE_KEY_COLUMNS = ["id", "well", "well_row_idx", "outer_fold", "md_since"]
VALUE_READ_COLUMNS = VALUE_KEY_COLUMNS + [
    "last_known_tvt",
    "candidate_tvt",
    "candidate_available",
    "candidate_finite",
]


def _artifact_path_from_manifest(manifest_path: Path, item: Mapping[str, Any]) -> Path:
    raw = str(item["path"])
    marker = "/artifacts/"
    if marker in raw:
        relative = raw.split(marker, 1)[1]
        candidate = manifest_path.parent / relative
        if candidate.exists():
            return candidate
    direct = Path(raw)
    if direct.exists():
        return direct
    suffix = Path(raw).parts[-4:]
    for root in manifest_path.parents:
        candidate = root.joinpath(*suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"manifest partition is missing: {raw}")


def _read_manifest_partitions(
    manifest_path: Path,
    items: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str],
    label: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    evidence: list[dict[str, Any]] = []
    for item in items:
        path = _artifact_path_from_manifest(manifest_path, item)
        actual_sha = sha256_file(path)
        expected_sha = str(item.get("file_sha256", ""))
        if expected_sha and actual_sha != expected_sha:
            raise ValueError(f"{label} partition SHA mismatch: {path}")
        frame = pd.read_parquet(path, columns=list(columns))
        expected_rows = int(item.get("rows", len(frame)))
        if len(frame) != expected_rows:
            raise ValueError(f"{label} partition row mismatch: {path}")
        frames.append(frame)
        evidence.append(
            {
                "source": label,
                "path": str(path),
                "rows": len(frame),
                "file_sha256": actual_sha,
                "content_sha256": item.get("content_sha256"),
                "schema_sha256": item.get("schema_sha256"),
            }
        )
    if not frames:
        raise ValueError(f"no partitions declared for {label}")
    return pd.concat(frames, ignore_index=True), evidence


def _assert_same_keys(reference: pd.DataFrame, candidate: pd.DataFrame, label: str) -> None:
    if len(reference) != len(candidate):
        raise ValueError(f"{label} key row count mismatch")
    for column in VALUE_KEY_COLUMNS:
        left = reference[column].to_numpy()
        right = candidate[column].to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"{label} key mismatch in {column}")


def _load_confidence_fields(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    candidate_id: str,
    fields: Sequence[str],
    reference_keys: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    partitions = manifest["candidate_confidence_partitions"][candidate_id]
    first_path = _artifact_path_from_manifest(manifest_path, partitions[0])
    available = set(pd.read_parquet(first_path).columns)
    missing = set(fields) - available
    if missing:
        raise ValueError(f"{candidate_id} confidence fields missing: {sorted(missing)}")
    frame, evidence = _read_manifest_partitions(
        manifest_path,
        partitions,
        columns=VALUE_KEY_COLUMNS + list(fields),
        label=f"exp263_confidence::{candidate_id}",
    )
    _assert_same_keys(reference_keys, frame[VALUE_KEY_COLUMNS], candidate_id)
    arrays = {
        field: pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=np.float64)
        for field in fields
    }
    return arrays, evidence


def reconstruct_candidate_bank(config: Mapping[str, Any]) -> CandidateBank:
    manifest_cfg = get_nested(config, "data.exp263_manifest")
    manifest_path = resolve_file(
        manifest_cfg["patterns"],
        label="exp263 cache manifest",
        expected_sha256=str(manifest_cfg["expected_file_sha256"]),
    )
    manifest = json.loads(manifest_path.read_text())
    expected_rows = int(get_nested(config, "data.expected_rows"))
    expected_wells = int(get_nested(config, "data.expected_wells"))
    if int(manifest.get("rows", -1)) != expected_rows:
        raise ValueError("exp263 manifest row contract mismatch")
    if int(manifest.get("wells", -1)) != expected_wells:
        raise ValueError("exp263 manifest well contract mismatch")
    if manifest.get("canonical_id_sha256") != manifest_cfg["expected_canonical_id_sha256"]:
        raise ValueError("exp263 canonical ID SHA mismatch")

    primitive_ids = list(get_nested(config, "candidate_bank.primitive_candidates"))
    primitive_values: dict[str, np.ndarray] = {}
    input_evidence: list[dict[str, Any]] = [
        {
            "source": "exp263_manifest",
            "path": str(manifest_path),
            "rows": expected_rows,
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest.get("canonical_id_sha256"),
            "schema_sha256": manifest.get("generation_config_sha256"),
        }
    ]
    reference_keys: pd.DataFrame | None = None
    last_known: np.ndarray | None = None

    for candidate_id in primitive_ids:
        items = manifest["candidate_value_partitions"].get(candidate_id)
        if not items or len(items) != 5:
            raise ValueError(f"{candidate_id} must have five value partitions")
        frame, evidence = _read_manifest_partitions(
            manifest_path,
            items,
            columns=VALUE_READ_COLUMNS,
            label=f"exp263_value::{candidate_id}",
        )
        input_evidence.extend(evidence)
        if reference_keys is None:
            reference_keys = frame[VALUE_KEY_COLUMNS].copy()
            reference_keys["last_known_tvt"] = pd.to_numeric(
                frame["last_known_tvt"], errors="coerce"
            ).astype(np.float32)
            last_known = reference_keys["last_known_tvt"].to_numpy(dtype=np.float64)
        else:
            _assert_same_keys(reference_keys[VALUE_KEY_COLUMNS], frame, candidate_id)
        available = frame["candidate_available"].astype(bool).to_numpy()
        finite = frame["candidate_finite"].astype(bool).to_numpy()
        values = pd.to_numeric(frame["candidate_tvt"], errors="coerce").to_numpy(
            dtype=np.float32
        )
        values[~(available & finite & np.isfinite(values))] = np.nan
        primitive_values[candidate_id] = values

    if reference_keys is None or last_known is None:
        raise AssertionError("primitive candidate loading produced no keys")
    if len(reference_keys) != expected_rows:
        raise ValueError("candidate bank total row mismatch")
    if reference_keys["well"].nunique() != expected_wells:
        raise ValueError("candidate bank total well mismatch")
    if reference_keys["id"].duplicated().any():
        raise ValueError("candidate bank IDs must be unique")
    expected_folds = set(range(int(get_nested(config, "validation.n_folds"))))
    if set(reference_keys["outer_fold"].unique()) != expected_folds:
        raise ValueError("candidate bank outer-fold coverage mismatch")

    pair_map = get_nested(config, "candidate_bank.pair_candidates")
    reconstructed: dict[str, np.ndarray] = dict(primitive_values)
    for candidate_id, parents in pair_map.items():
        left = primitive_values[str(parents[0])].astype(np.float64)
        right = primitive_values[str(parents[1])].astype(np.float64)
        value = (0.5 * left + 0.5 * right).astype(np.float32)
        value[~(np.isfinite(left) & np.isfinite(right))] = np.nan
        reconstructed[str(candidate_id)] = value

    fixed = get_nested(config, "candidate_bank.fixed_candidate")
    fixed_value = np.zeros(expected_rows, dtype=np.float64)
    fixed_available = np.ones(expected_rows, dtype=bool)
    for parent, weight in zip(fixed["parents"], fixed["weights"], strict=True):
        parent_value = primitive_values[str(parent)].astype(np.float64)
        fixed_value += float(weight) * np.nan_to_num(parent_value, nan=0.0)
        fixed_available &= np.isfinite(parent_value)
    fixed_value[~fixed_available] = np.nan
    reconstructed[str(fixed["id"])] = fixed_value.astype(np.float32)

    ordered_ids = primitive_ids + list(pair_map) + [str(fixed["id"])]
    if len(ordered_ids) != int(get_nested(config, "candidate_bank.existing_candidate_count")):
        raise ValueError("existing candidate count mismatch")
    values = np.column_stack([reconstructed[name] for name in ordered_ids]).astype(
        np.float32,
        copy=False,
    )
    if np.any(np.sum(np.isfinite(values), axis=1) == 0):
        raise ValueError("at least one existing candidate must be available per row")

    exp226_conf, evidence = _load_confidence_fields(
        manifest,
        manifest_path,
        "exp226_k16",
        ["geometry_gr_delta"],
        reference_keys,
    )
    input_evidence.extend(evidence)
    exact_conf, evidence = _load_confidence_fields(
        manifest,
        manifest_path,
        "exact_hmm",
        ["sigma_tvt", "loglik_per_row"],
        reference_keys,
    )
    input_evidence.extend(evidence)

    gate_row_inputs = {
        "geometry_gr_delta": exp226_conf["geometry_gr_delta"],
        "exact_sigma_tvt": exact_conf["sigma_tvt"],
        "exact_loglik_per_row": exact_conf["loglik_per_row"],
        "existing_bank_std": np.nanstd(values.astype(np.float64), axis=1),
        "exp226_exact_abs": np.abs(
            primitive_values["exp226_k16"].astype(np.float64)
            - primitive_values["exact_hmm"].astype(np.float64)
        ),
        "exp226_selfgr_abs": np.abs(
            primitive_values["exp226_k16"].astype(np.float64)
            - primitive_values["selfgr_hmm_a070"].astype(np.float64)
        ),
        "last_known_tvt": last_known,
    }
    return CandidateBank(
        keys=reference_keys,
        candidate_ids=ordered_ids,
        values=values,
        primitive_values=primitive_values,
        gate_row_inputs=gate_row_inputs,
        manifest=manifest,
        manifest_path=manifest_path,
        input_evidence=input_evidence,
    )


# %% [markdown]
# ## 4. Target-free well gate construction and freeze boundary

# %%
def reject_forbidden_gate_features(
    columns: Iterable[str], forbidden_tokens: Sequence[str]
) -> None:
    violations: dict[str, list[str]] = {}
    for column in columns:
        lower = str(column).lower()
        matched = [token for token in forbidden_tokens if str(token).lower() in lower]
        if matched:
            violations[str(column)] = matched
    if violations:
        raise ValueError(f"forbidden gate features detected: {violations}")


def build_target_free_well_features(
    bank: CandidateBank,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    rows = bank.keys[["well", "well_row_idx", "outer_fold", "md_since"]].copy()
    rows["geometry_gr_delta_abs"] = np.abs(bank.gate_row_inputs["geometry_gr_delta"])
    rows["exact_sigma_tvt"] = bank.gate_row_inputs["exact_sigma_tvt"]
    rows["exact_neg_loglik_per_row"] = -bank.gate_row_inputs["exact_loglik_per_row"]
    rows["exp226_exact_abs"] = bank.gate_row_inputs["exp226_exact_abs"]
    rows["exp226_selfgr_abs"] = bank.gate_row_inputs["exp226_selfgr_abs"]
    rows["existing_bank_std"] = bank.gate_row_inputs["existing_bank_std"]
    grouped = rows.groupby("well", sort=True, observed=True)
    fold_counts = grouped["outer_fold"].nunique()
    if int(fold_counts.max()) != 1:
        raise ValueError("a well cannot span multiple outer folds")

    features = grouped.agg(
        outer_fold=("outer_fold", "first"),
        geometry_gr_delta_abs_median=("geometry_gr_delta_abs", "median"),
        known_prefix_rows=("well_row_idx", "min"),
        exact_sigma_tvt_p90=("exact_sigma_tvt", lambda values: values.quantile(0.90)),
        exact_neg_loglik_per_row_median=("exact_neg_loglik_per_row", "median"),
        exp226_exact_abs_median=("exp226_exact_abs", "median"),
        exp226_selfgr_abs_median=("exp226_selfgr_abs", "median"),
        existing_bank_std_median=("existing_bank_std", "median"),
        tail_rows=("md_since", "size"),
    ).reset_index()
    features["outer_fold"] = features["outer_fold"].astype(np.int8)
    features["known_prefix_rows"] = features["known_prefix_rows"].astype(np.int32)
    features["tail_rows"] = features["tail_rows"].astype(np.int32)

    feature_specs = list(get_nested(config, "gate.features"))
    expected = [str(spec["name"]) for spec in feature_specs]
    missing = set(expected) - set(features)
    if missing:
        raise ValueError(f"configured gate features missing: {sorted(missing)}")
    reject_forbidden_gate_features(expected, list(get_nested(config, "gate.forbidden_tokens")))
    return apply_fixed_rank_gate(features, config)


def apply_fixed_rank_gate(
    features: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    required = {"well", "outer_fold"}
    if not required.issubset(features):
        raise ValueError(f"gate frame missing columns: {sorted(required - set(features))}")
    feature_specs = list(get_nested(config, "gate.features"))
    forbidden = list(get_nested(config, "gate.forbidden_tokens"))
    feature_names = [str(spec["name"]) for spec in feature_specs]
    reject_forbidden_gate_features(feature_names, forbidden)
    output = features.copy()
    rank_columns: list[str] = []
    neutral = float(get_nested(config, "gate.missing_rank_value"))
    for spec in feature_specs:
        name = str(spec["name"])
        direction = str(spec["direction"])
        values = pd.to_numeric(output[name], errors="coerce")
        if direction == "higher":
            ranked_source = values
        elif direction == "lower":
            ranked_source = -values
        else:
            raise ValueError(f"unsupported gate direction for {name}: {direction}")
        rank_column = f"rank__{name}"
        output[rank_column] = ranked_source.groupby(output["outer_fold"]).rank(
            method="average", pct=True
        )
        output[rank_column] = output[rank_column].fillna(neutral)
        rank_columns.append(rank_column)
    output["gate_score"] = output[rank_columns].mean(axis=1)
    output["candidate_available"] = False
    cutoff = float(get_nested(config, "gate.cutoff_fraction"))
    if not 0.0 < cutoff <= 1.0:
        raise ValueError("gate cutoff_fraction must be in (0, 1]")
    for _, indices in output.groupby("outer_fold", sort=True).groups.items():
        group = output.loc[indices, ["gate_score"]]
        selected_count = math.floor(cutoff * len(group))
        if selected_count <= 0:
            raise ValueError("gate cutoff selected zero wells in a fold")
        sorted_scores = group["gate_score"].sort_values(ascending=False)
        boundary = float(sorted_scores.iloc[selected_count - 1])
        above = group["gate_score"] > boundary
        equal = group["gate_score"].eq(boundary)
        if int(above.sum() + equal.sum()) <= selected_count:
            selected_indices = group.index[above | equal]
        else:
            selected_indices = group.index[above]
        output.loc[selected_indices, "candidate_available"] = True
    output["candidate_available"] = output["candidate_available"].astype(bool)
    coverage = output.groupby("outer_fold")["candidate_available"].mean()
    if bool((coverage > cutoff + 1e-12).any()):
        raise ValueError(f"gate fold coverage exceeds cutoff: {coverage.to_dict()}")
    return output.sort_values(["outer_fold", "well"], kind="mergesort").reset_index(
        drop=True
    )


def freeze_gate_artifact(
    gate: pd.DataFrame,
    artifacts: Path,
    config: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    gate_path = artifacts / f"{OUTPUT_PREFIX}_target_free_gate_features.csv"
    gate.to_csv(gate_path, index=False, float_format="%.12g")
    contract = {
        "experiment": EXPERIMENT_NAME,
        "stage": "target_free_gate_frozen_before_truth_attachment",
        "truth_attached": False,
        "candidate_attached": False,
        "gate_name": get_nested(config, "gate.name"),
        "feature_specs": get_nested(config, "gate.features"),
        "cutoff_fraction": get_nested(config, "gate.cutoff_fraction"),
        "tie_break": get_nested(config, "gate.tie_break"),
        "rows": len(gate),
        "selected_wells": int(gate["candidate_available"].sum()),
        "feature_schema_sha256": frame_schema_sha256(gate),
        "feature_content_sha256": frame_content_sha256(gate),
        "gate_file_sha256": sha256_file(gate_path),
    }
    contract_path = artifacts / f"{OUTPUT_PREFIX}_stage0_contract.json"
    write_json(contract_path, contract)
    contract["contract_file_sha256"] = sha256_file(contract_path)
    return gate_path, contract


# %% [markdown]
# ## 5. Post-freeze exp279 candidate and truth attachment

# %%
def load_exp279_after_gate_freeze(
    bank: CandidateBank,
    config: Mapping[str, Any],
    gate_contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if gate_contract.get("truth_attached") is not False:
        raise ValueError("gate contract must prove truth_attached=false")
    if not gate_contract.get("feature_content_sha256"):
        raise ValueError("gate content SHA must be frozen before loading exp279")
    exp279_cfg = get_nested(config, "data.exp279_oof")
    path = resolve_file(
        exp279_cfg["patterns"],
        label="exp279 OOF",
        expected_sha256=str(exp279_cfg["expected_raw_gzip_sha256"]),
    )
    decompressed_sha = sha256_decompressed_gzip(path)
    if decompressed_sha != exp279_cfg["expected_decompressed_sha256"]:
        raise ValueError("exp279 decompressed content SHA mismatch")
    truth_column = str(exp279_cfg["truth_column"])
    candidate_column = str(exp279_cfg["candidate_column"])
    usecols = ["id", "well", "row_idx", "md_since", truth_column, candidate_column]
    frame = pd.read_csv(path, usecols=usecols)
    if len(frame) != len(bank.keys) or frame["id"].duplicated().any():
        raise ValueError("exp279 OOF ID inventory mismatch")
    source_index = pd.Index(frame["id"].astype(str))
    indexer = source_index.get_indexer(pd.Index(bank.keys["id"].astype(str)))
    if np.any(indexer < 0) or len(np.unique(indexer)) != len(indexer):
        raise ValueError("exp279 OOF cannot be aligned one-to-one to exp263 IDs")
    aligned = frame.iloc[indexer].reset_index(drop=True)
    if not np.array_equal(
        aligned["well"].astype(str).to_numpy(),
        bank.keys["well"].astype(str).to_numpy(),
    ):
        raise ValueError("exp279 well identity mismatch after ID alignment")
    if not np.allclose(
        aligned["md_since"].to_numpy(dtype=np.float64),
        bank.keys["md_since"].to_numpy(dtype=np.float64),
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("exp279 md_since mismatch after ID alignment")
    geop = pd.to_numeric(aligned[candidate_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    truth = pd.to_numeric(aligned[truth_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(geop).all() or not np.isfinite(truth).all():
        raise ValueError("exp279 candidate and readout truth must be finite")
    evidence = {
        "source": "exp279_oof",
        "path": str(path),
        "rows": len(frame),
        "file_sha256": sha256_file(path),
        "decompressed_content_sha256": decompressed_sha,
        "prediction_content_sha256": exp279_cfg["expected_prediction_content_sha256"],
        "truth_attachment_stage": get_nested(config, "gate.truth_attachment_stage"),
    }
    return geop, truth, evidence


def load_hidden_like_sets(
    config: Mapping[str, Any],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    hidden_cfg = get_nested(config, "data.hidden_like_assignment")
    path = resolve_file(
        hidden_cfg["patterns"],
        label="hidden-like assignment",
        expected_sha256=str(hidden_cfg["expected_file_sha256"]),
    )
    frame = pd.read_csv(path)
    well_column = str(hidden_cfg["well_column"])
    sets: dict[str, set[str]] = {}
    for scope, role_column in hidden_cfg["role_columns"].items():
        if role_column not in frame:
            raise ValueError(f"hidden-like role column missing: {role_column}")
        sets[str(scope)] = set(
            frame.loc[frame[role_column].eq("valid"), well_column].astype(str)
        )
    evidence = {
        "source": "hidden_like_assignment",
        "path": str(path),
        "rows": len(frame),
        "file_sha256": sha256_file(path),
        "content_sha256": None,
        "schema_sha256": frame_schema_sha256(frame),
    }
    return sets, evidence


# %% [markdown]
# ## 6. Row, 512-block, whole-well oracle and unique-best readouts

# %%
@dataclass
class OracleArrays:
    existing_squared_error: np.ndarray
    geop_squared_error: np.ndarray
    gate_row_available: np.ndarray
    well_codes: np.ndarray
    block_codes: np.ndarray
    well_names: np.ndarray
    well_fold: np.ndarray
    well_gate: np.ndarray


def build_oracle_arrays(
    bank: CandidateBank,
    geop: np.ndarray,
    truth: np.ndarray,
    gate: pd.DataFrame,
    block_rows: int,
) -> OracleArrays:
    n_rows, n_candidates = bank.values.shape
    squared = np.empty((n_rows, n_candidates), dtype=np.float64)
    for position in range(n_candidates):
        candidate = bank.values[:, position].astype(np.float64)
        valid = np.isfinite(candidate)
        delta = np.zeros(n_rows, dtype=np.float64)
        delta[valid] = candidate[valid] - truth[valid]
        squared[:, position] = np.inf
        squared[valid, position] = np.square(delta[valid])
    geop_squared = np.square(geop - truth)

    well_values = bank.keys["well"].astype(str).to_numpy()
    well_codes, well_names = pd.factorize(well_values, sort=True)
    if np.any(well_codes < 0):
        raise ValueError("well factorization failed")
    well_names_array = np.asarray(well_names, dtype=object)
    n_wells = len(well_names_array)
    well_fold = np.full(n_wells, -1, dtype=np.int8)
    row_fold = bank.keys["outer_fold"].to_numpy(dtype=np.int8)
    for code in range(n_wells):
        folds = np.unique(row_fold[well_codes == code])
        if len(folds) != 1:
            raise ValueError("well spans multiple outer folds")
        well_fold[code] = folds[0]

    gate_by_well = gate.set_index("well")["candidate_available"]
    if set(well_names_array) != set(gate_by_well.index.astype(str)):
        raise ValueError("gate well inventory mismatch")
    well_gate = gate_by_well.reindex(well_names_array).to_numpy(dtype=bool)
    gate_row_available = well_gate[well_codes]

    md_since = bank.keys["md_since"].to_numpy(dtype=np.float64)
    local_block = np.floor(np.maximum(md_since - 1.0, 0.0) / block_rows).astype(np.int32)
    multiplier = int(local_block.max()) + 1
    block_key = well_codes.astype(np.int64) * multiplier + local_block.astype(np.int64)
    block_codes, _ = pd.factorize(block_key, sort=True)
    return OracleArrays(
        existing_squared_error=squared,
        geop_squared_error=geop_squared,
        gate_row_available=gate_row_available,
        well_codes=well_codes.astype(np.int32),
        block_codes=block_codes.astype(np.int32),
        well_names=well_names_array,
        well_fold=well_fold,
        well_gate=well_gate,
    )


def _group_candidate_sse(
    squared_error: np.ndarray,
    group_codes: np.ndarray,
    row_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected_codes = group_codes[row_mask]
    if len(selected_codes) == 0:
        return np.empty((0, squared_error.shape[1])), np.empty(0, dtype=np.int64)
    unique_codes = np.unique(selected_codes)
    remap = np.full(int(group_codes.max()) + 1, -1, dtype=np.int64)
    remap[unique_codes] = np.arange(len(unique_codes), dtype=np.int64)
    local_codes = remap[selected_codes]
    output = np.empty((len(unique_codes), squared_error.shape[1]), dtype=np.float64)
    for position in range(squared_error.shape[1]):
        values = squared_error[row_mask, position]
        finite = np.isfinite(values)
        sums = np.bincount(
            local_codes[finite], weights=values[finite], minlength=len(unique_codes)
        ).astype(np.float64)
        counts = np.bincount(local_codes[finite], minlength=len(unique_codes))
        total_counts = np.bincount(local_codes, minlength=len(unique_codes))
        sums[counts != total_counts] = np.inf
        output[:, position] = sums
    return output, unique_codes


def oracle_metrics_for_mask(
    arrays: OracleArrays,
    row_mask: np.ndarray,
    *,
    scope: str,
    granularity: str,
) -> dict[str, Any]:
    if row_mask.dtype != bool or len(row_mask) != len(arrays.geop_squared_error):
        raise ValueError("oracle row mask contract mismatch")
    row_count = int(row_mask.sum())
    if row_count == 0:
        return {
            "scope": scope,
            "granularity": granularity,
            "rows": 0,
            "groups": 0,
            "base_oracle_rmse": None,
            "full_union_oracle_rmse": None,
            "sparse_union_oracle_rmse": None,
            "full_sse_gain": None,
            "sparse_sse_gain": None,
        }

    if granularity == "row":
        base_sse_by_group = np.min(arrays.existing_squared_error[row_mask], axis=1)
        geop_sse_by_group = arrays.geop_squared_error[row_mask]
        gate_available = arrays.gate_row_available[row_mask]
        groups = row_count
    elif granularity in {"block512", "whole_well"}:
        codes = arrays.block_codes if granularity == "block512" else arrays.well_codes
        base_group_sse, unique_codes = _group_candidate_sse(
            arrays.existing_squared_error, codes, row_mask
        )
        geop_group_sse, geop_codes = _group_candidate_sse(
            arrays.geop_squared_error[:, None], codes, row_mask
        )
        if not np.array_equal(unique_codes, geop_codes):
            raise AssertionError("oracle group identity mismatch")
        base_sse_by_group = np.min(base_group_sse, axis=1)
        geop_sse_by_group = geop_group_sse[:, 0]
        if granularity == "whole_well":
            gate_available = arrays.well_gate[unique_codes]
        else:
            first_row = np.full(len(unique_codes), len(codes), dtype=np.int64)
            selected_indices = np.flatnonzero(row_mask)
            local_code = codes[selected_indices]
            mapped = np.searchsorted(unique_codes, local_code)
            np.minimum.at(first_row, mapped, selected_indices)
            if np.any(first_row == len(codes)):
                raise AssertionError("block oracle first-row mapping failed")
            gate_available = arrays.gate_row_available[first_row]
        groups = len(unique_codes)
    else:
        raise ValueError(f"unsupported oracle granularity: {granularity}")

    if not np.isfinite(base_sse_by_group).all():
        raise ValueError("existing candidate bank has an unavailable oracle group")
    full_sse_by_group = np.minimum(base_sse_by_group, geop_sse_by_group)
    sparse_geop = np.where(gate_available, geop_sse_by_group, np.inf)
    sparse_sse_by_group = np.minimum(base_sse_by_group, sparse_geop)
    base_sse = float(base_sse_by_group.sum())
    full_sse = float(full_sse_by_group.sum())
    sparse_sse = float(sparse_sse_by_group.sum())
    return {
        "scope": scope,
        "granularity": granularity,
        "rows": row_count,
        "groups": int(groups),
        "base_oracle_rmse": math.sqrt(base_sse / row_count),
        "full_union_oracle_rmse": math.sqrt(full_sse / row_count),
        "sparse_union_oracle_rmse": math.sqrt(sparse_sse / row_count),
        "full_sse_gain": base_sse - full_sse,
        "sparse_sse_gain": base_sse - sparse_sse,
        "full_rmse_gain": math.sqrt(base_sse / row_count)
        - math.sqrt(full_sse / row_count),
        "sparse_rmse_gain": math.sqrt(base_sse / row_count)
        - math.sqrt(sparse_sse / row_count),
    }


def unique_best_for_mask(
    arrays: OracleArrays,
    row_mask: np.ndarray,
    *,
    scope: str,
    granularity: str,
    tolerance: float,
) -> dict[str, Any]:
    if granularity == "row":
        base = np.min(arrays.existing_squared_error[row_mask], axis=1)
        geop = arrays.geop_squared_error[row_mask]
        available = arrays.gate_row_available[row_mask]
        group_rows = np.ones(len(base), dtype=np.int64)
    else:
        codes = arrays.block_codes if granularity == "block512" else arrays.well_codes
        base_matrix, unique_codes = _group_candidate_sse(
            arrays.existing_squared_error, codes, row_mask
        )
        geop_matrix, _ = _group_candidate_sse(
            arrays.geop_squared_error[:, None], codes, row_mask
        )
        base = np.min(base_matrix, axis=1)
        geop = geop_matrix[:, 0]
        selected_codes = codes[row_mask]
        remap = np.full(int(codes.max()) + 1, -1, dtype=np.int64)
        remap[unique_codes] = np.arange(len(unique_codes))
        group_rows = np.bincount(
            remap[selected_codes], minlength=len(unique_codes)
        ).astype(np.int64)
        if granularity == "whole_well":
            available = arrays.well_gate[unique_codes]
        else:
            selected_indices = np.flatnonzero(row_mask)
            mapped = np.searchsorted(unique_codes, codes[selected_indices])
            first_row = np.full(len(unique_codes), len(codes), dtype=np.int64)
            np.minimum.at(first_row, mapped, selected_indices)
            if np.any(first_row == len(codes)):
                raise AssertionError("block unique-best first-row mapping failed")
            available = arrays.gate_row_available[first_row]
    unique = geop + tolerance < base
    gated_unique = unique & available
    return {
        "scope": scope,
        "granularity": granularity,
        "groups": int(len(base)),
        "unique_best_groups": int(unique.sum()),
        "unique_best_group_rate": float(unique.mean()) if len(unique) else None,
        "unique_best_rows": int(group_rows[unique].sum()),
        "gated_unique_best_groups": int(gated_unique.sum()),
        "gated_unique_best_rows": int(group_rows[gated_unique].sum()),
    }


def build_scope_masks(
    bank: CandidateBank,
    gate: pd.DataFrame,
    hidden_sets: Mapping[str, set[str]],
    n_folds: int,
) -> dict[str, np.ndarray]:
    wells = bank.keys["well"].astype(str).to_numpy()
    folds = bank.keys["outer_fold"].to_numpy(dtype=np.int8)
    masks: dict[str, np.ndarray] = {"overall": np.ones(len(bank.keys), dtype=bool)}
    for fold in range(n_folds):
        masks[f"fold_{fold}"] = folds == fold
    masks["1000_plus"] = bank.keys["md_since"].to_numpy(dtype=float) >= 1000.0
    for scope, selected_wells in hidden_sets.items():
        masks[str(scope)] = np.isin(wells, np.asarray(sorted(selected_wells), dtype=object))
    gate_wells = set(gate.loc[gate["candidate_available"], "well"].astype(str))
    masks["gate_selected"] = np.isin(wells, np.asarray(sorted(gate_wells), dtype=object))
    for fold in range(n_folds):
        masks[f"gate_selected_fold_{fold}"] = masks["gate_selected"] & (folds == fold)
    return masks


def build_by_well_oracle(
    arrays: OracleArrays,
    row_count: int,
) -> pd.DataFrame:
    mask = np.ones(row_count, dtype=bool)
    base_matrix, codes = _group_candidate_sse(
        arrays.existing_squared_error, arrays.well_codes, mask
    )
    geop_matrix, _ = _group_candidate_sse(
        arrays.geop_squared_error[:, None], arrays.well_codes, mask
    )
    rows = np.bincount(arrays.well_codes, minlength=len(codes)).astype(np.int64)
    base_sse = np.min(base_matrix, axis=1)
    geop_sse = geop_matrix[:, 0]
    full_sse = np.minimum(base_sse, geop_sse)
    sparse_sse = np.minimum(
        base_sse,
        np.where(arrays.well_gate[codes], geop_sse, np.inf),
    )
    output = pd.DataFrame(
        {
            "well": arrays.well_names[codes],
            "outer_fold": arrays.well_fold[codes],
            "rows": rows,
            "candidate_available": arrays.well_gate[codes],
            "base_oracle_rmse": np.sqrt(base_sse / rows),
            "geop_hmm_rmse": np.sqrt(geop_sse / rows),
            "full_union_oracle_rmse": np.sqrt(full_sse / rows),
            "sparse_union_oracle_rmse": np.sqrt(sparse_sse / rows),
        }
    )
    output["full_delta_rmse_vs_base"] = (
        output["full_union_oracle_rmse"] - output["base_oracle_rmse"]
    )
    output["sparse_delta_rmse_vs_base"] = (
        output["sparse_union_oracle_rmse"] - output["base_oracle_rmse"]
    )
    output["geop_unique_best"] = geop_sse + 1e-9 < base_sse
    return output.sort_values("well").reset_index(drop=True)


def evaluate_stage0_guards(
    oracle_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    gate: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    guard_cfg = get_nested(config, "validation.guard")
    granularities = list(guard_cfg["union_required_granularities"])
    pooled = oracle_metrics[oracle_metrics["scope"].eq("overall")].set_index(
        "granularity"
    )
    pooled_pass = {
        granularity: bool(
            float(pooled.loc[granularity, "full_sse_gain"])
            > float(guard_cfg["minimum_pooled_sse_gain_ft2"])
        )
        for granularity in granularities
    }
    fold_counts: dict[str, int] = {}
    fold_pass: dict[str, bool] = {}
    for granularity in granularities:
        records = oracle_metrics[
            oracle_metrics["scope"].str.match(r"^fold_[0-4]$")
            & oracle_metrics["granularity"].eq(granularity)
        ]
        count = int((records["full_sse_gain"] > 0.0).sum())
        fold_counts[granularity] = count
        fold_pass[granularity] = count >= int(guard_cfg["minimum_improved_folds"])

    full_whole_gain = float(pooled.loc["whole_well", "full_sse_gain"])
    sparse_whole_gain = float(pooled.loc["whole_well", "sparse_sse_gain"])
    retention = sparse_whole_gain / full_whole_gain if full_whole_gain > 0.0 else 0.0
    selected = oracle_metrics[
        oracle_metrics["scope"].eq("gate_selected")
        & oracle_metrics["granularity"].eq("whole_well")
    ]
    selected_gain = float(selected.iloc[0]["full_sse_gain"]) if len(selected) else 0.0
    selected_fold = oracle_metrics[
        oracle_metrics["scope"].str.match(r"^gate_selected_fold_[0-4]$")
        & oracle_metrics["granularity"].eq("whole_well")
    ]
    selected_improved_folds = int((selected_fold["full_sse_gain"] > 0.0).sum())
    coverage = gate.groupby("outer_fold")["candidate_available"].mean()
    worst_full = float(by_well["full_delta_rmse_vs_base"].max())
    worst_sparse = float(by_well["sparse_delta_rmse_vs_base"].max())
    worst_limit = float(guard_cfg["worst_well_maximum_rmse_regression_ft"])
    result = {
        "pooled_union_pass_by_granularity": pooled_pass,
        "improved_folds_by_granularity": fold_counts,
        "fold_union_pass_by_granularity": fold_pass,
        "gate_whole_well_sse_gain_retention": retention,
        "gate_retention_passed": retention
        >= float(guard_cfg["gate_minimum_full_whole_well_sse_gain_retention"]),
        "gate_selected_whole_well_sse_gain": selected_gain,
        "gate_selected_improved_folds": selected_improved_folds,
        "gate_selected_gain_passed": selected_gain > 0.0
        and selected_improved_folds
        >= int(guard_cfg["gate_minimum_selected_well_improved_folds"]),
        "gate_coverage_by_fold": {str(int(key)): float(value) for key, value in coverage.items()},
        "gate_coverage_passed": bool(
            (coverage <= float(guard_cfg["gate_maximum_fold_coverage"]) + 1e-12).all()
        ),
        "worst_well_full_delta_rmse": worst_full,
        "worst_well_sparse_delta_rmse": worst_sparse,
        "worst_well_passed": worst_full <= worst_limit and worst_sparse <= worst_limit,
    }
    result["candidate_guard_passed"] = bool(
        all(pooled_pass.values())
        and all(fold_pass.values())
        and result["gate_retention_passed"]
        and result["gate_selected_gain_passed"]
        and result["gate_coverage_passed"]
        and result["worst_well_passed"]
    )
    return result


# %% [markdown]
# ## 7. Paired 200-well shadow runtime guard

# %%
def evaluate_runtime_guard(
    manifest: pd.DataFrame | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    cfg = get_nested(config, "runtime_guard")
    if manifest is None:
        return {
            "status": "not_evaluated_missing_paired_200well_shadow_manifest",
            "passed": False,
            "reason": "A saved-OOF Stage 0 readout cannot prove hidden inference runtime.",
        }
    forbidden = {str(column).lower() for column in cfg["forbidden_columns"]}
    present_forbidden = sorted(
        column for column in manifest if str(column).lower() in forbidden
    )
    if present_forbidden:
        raise ValueError(f"runtime manifest contains forbidden columns: {present_forbidden}")
    required = set(cfg["required_columns"])
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"runtime manifest missing columns: {sorted(missing)}")
    frame = manifest.copy()
    if frame["well"].duplicated().any():
        raise ValueError("runtime manifest must contain one row per shadow well")
    expected_wells = int(cfg["shadow_wells"])
    if len(frame) != expected_wells:
        return {
            "status": "failed_shadow_well_count",
            "passed": False,
            "wells": len(frame),
            "expected_wells": expected_wells,
        }
    frame["gate_selected"] = frame["gate_selected"].astype(bool)
    timing_columns = [
        "base_seconds",
        "geop_additional_seconds",
        "selector_seconds",
        "tvt_model_seconds",
        "save_seconds",
    ]
    for column in timing_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[timing_columns].to_numpy(dtype=float)).all():
        raise ValueError("runtime manifest timing values must be finite")
    if bool((frame[timing_columns] < 0.0).any().any()):
        raise ValueError("runtime manifest timing values cannot be negative")
    selected = int(frame["gate_selected"].sum())
    selected_limit = min(
        math.floor(float(cfg["selected_fraction_max"]) * len(frame)),
        int(cfg["selected_wells_max"]),
    )
    overall_coverage = selected / len(frame)
    fold_coverage = frame.groupby("fold")["gate_selected"].mean()

    rng = np.random.default_rng(int(cfg["seed"]))
    iterations = int(cfg["bootstrap_iterations"])
    n_rows = len(frame)
    base = frame["base_seconds"].to_numpy(dtype=float)
    geop = (
        frame["geop_additional_seconds"].to_numpy(dtype=float)
        * frame["gate_selected"].to_numpy(dtype=float)
    )
    overhead = frame[
        ["selector_seconds", "tvt_model_seconds", "save_seconds"]
    ].sum(axis=1).to_numpy(dtype=float)
    geop_totals = np.empty(iterations, dtype=float)
    total_totals = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sample = rng.integers(0, n_rows, size=n_rows)
        geop_totals[iteration] = float(geop[sample].sum())
        total_totals[iteration] = float((base[sample] + geop[sample] + overhead[sample]).sum())
    geop_p50, geop_p95 = np.quantile(geop_totals, [0.50, 0.95])
    total_p50, total_p95 = np.quantile(total_totals, [0.50, 0.95])
    coverage_pass = (
        selected <= selected_limit
        and overall_coverage <= float(cfg["selected_fraction_max"]) + 1e-12
        and bool((fold_coverage <= float(cfg["per_fold_fraction_max"]) + 1e-12).all())
    )
    geop_pass = geop_p95 <= float(cfg["geop_additional_p95_seconds_max"])
    total_pass = total_p95 <= float(cfg["total_p95_seconds_max"])
    return {
        "status": "evaluated",
        "passed": bool(coverage_pass and geop_pass and total_pass),
        "wells": n_rows,
        "selected_wells": selected,
        "selected_wells_limit": selected_limit,
        "overall_coverage": overall_coverage,
        "fold_coverage": {str(key): float(value) for key, value in fold_coverage.items()},
        "coverage_passed": bool(coverage_pass),
        "bootstrap_iterations": iterations,
        "seed": int(cfg["seed"]),
        "geop_additional_p50_seconds": float(geop_p50),
        "geop_additional_p95_seconds": float(geop_p95),
        "geop_additional_passed": bool(geop_pass),
        "total_p50_seconds": float(total_p50),
        "total_p95_seconds": float(total_p95),
        "total_passed": bool(total_pass),
    }


def load_optional_runtime_manifest(config: Mapping[str, Any]) -> pd.DataFrame | None:
    runtime_cfg = get_nested(config, "data.paired_shadow_runtime_manifest")
    if not bool(runtime_cfg.get("enabled", False)):
        return None
    paths = expand_existing_paths(runtime_cfg["patterns"])
    files = [path for path in paths if path.is_file()]
    if len(files) != 1:
        raise FileNotFoundError("enabled paired shadow runtime manifest was not found uniquely")
    return pd.read_csv(files[0])


# %% [markdown]
# ## 8. Stage 0 orchestration, guards, and artifact persistence

# %%
def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    execution = get_nested(config, "execution")
    expected_execution = {
        "active_audit_variants": 1,
        "lightgbm_config_count": 0,
        "trained_fold_count": 0,
        "total_boosters": 0,
        "hmm_well_runs": 0,
        "pf_well_runs": 0,
        "parent_control_retraining": False,
        "stage_b_enabled": False,
        "stage_c_enabled": False,
        "stage_d_enabled": False,
        "gpu": False,
        "inference": False,
        "submission": False,
    }
    mismatches = {
        key: {"actual": execution.get(key), "expected": expected}
        for key, expected in expected_execution.items()
        if execution.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Stage 0 execution contract mismatch: {mismatches}")

    artifacts = runtime_artifacts_dir()
    bank = reconstruct_candidate_bank(config)
    gate = build_target_free_well_features(bank, config)
    gate_path, gate_contract = freeze_gate_artifact(gate, artifacts, config)

    # This is the intentional scientific boundary: exp279 contains readout truth
    # and is not opened until the target-free gate file and content SHA exist.
    geop, truth, exp279_evidence = load_exp279_after_gate_freeze(
        bank, config, gate_contract
    )
    hidden_sets, hidden_evidence = load_hidden_like_sets(config)

    arrays = build_oracle_arrays(
        bank,
        geop,
        truth,
        gate,
        int(get_nested(config, "validation.block_rows")),
    )
    masks = build_scope_masks(
        bank,
        gate,
        hidden_sets,
        int(get_nested(config, "validation.n_folds")),
    )
    metric_records: list[dict[str, Any]] = []
    unique_records: list[dict[str, Any]] = []
    granularities = ["row", "block512", "whole_well"]
    tolerance = float(get_nested(config, "validation.tie_tolerance_squared_ft"))
    for scope, mask in masks.items():
        for granularity in granularities:
            metric_records.append(
                oracle_metrics_for_mask(
                    arrays, mask, scope=scope, granularity=granularity
                )
            )
            unique_records.append(
                unique_best_for_mask(
                    arrays,
                    mask,
                    scope=scope,
                    granularity=granularity,
                    tolerance=tolerance,
                )
            )
    oracle_metrics = pd.DataFrame(metric_records)
    unique_metrics = pd.DataFrame(unique_records)
    by_well = build_by_well_oracle(arrays, len(bank.keys))
    guards = evaluate_stage0_guards(oracle_metrics, by_well, gate, config)

    runtime_manifest = load_optional_runtime_manifest(config)
    runtime_guard = evaluate_runtime_guard(runtime_manifest, config)

    oracle_path = artifacts / f"{OUTPUT_PREFIX}_oracle_metrics.csv"
    unique_path = artifacts / f"{OUTPUT_PREFIX}_unique_best_metrics.csv"
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well_oracle.csv"
    availability_path = artifacts / f"{OUTPUT_PREFIX}_candidate_availability.csv"
    runtime_path = artifacts / f"{OUTPUT_PREFIX}_runtime_guard.json"
    input_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"

    oracle_metrics.to_csv(oracle_path, index=False, float_format="%.12g")
    unique_metrics.to_csv(unique_path, index=False, float_format="%.12g")
    by_well.to_csv(by_well_path, index=False, float_format="%.12g")
    gate[["well", "outer_fold", "gate_score", "candidate_available"]].to_csv(
        availability_path, index=False, float_format="%.12g"
    )
    write_json(runtime_path, runtime_guard)
    input_evidence = bank.input_evidence + [exp279_evidence, hidden_evidence]
    input_manifest = pd.DataFrame(input_evidence)
    input_manifest.to_csv(input_path, index=False)

    artifact_paths = {
        "stage0_contract": artifacts / f"{OUTPUT_PREFIX}_stage0_contract.json",
        "target_free_gate": gate_path,
        "oracle_metrics": oracle_path,
        "unique_best_metrics": unique_path,
        "by_well_oracle": by_well_path,
        "candidate_availability": availability_path,
        "runtime_guard": runtime_path,
        "input_manifest": input_path,
    }
    artifact_sha = {key: sha256_file(path) for key, path in artifact_paths.items()}
    eligible_for_stage_b = bool(
        guards["candidate_guard_passed"]
        and execution.get("exp276_corrected_guard_passed") is True
        and execution.get("stage_b_enabled") is True
    )
    summary: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage0_completed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "route": get_nested(config, "experiment.route"),
        "rows": len(bank.keys),
        "wells": int(bank.keys["well"].nunique()),
        "existing_candidates": len(bank.candidate_ids),
        "union_candidates": len(bank.candidate_ids) + 1,
        "active_audit_variants": execution["active_audit_variants"],
        "lightgbm_configs": execution["lightgbm_config_count"],
        "trained_folds": execution["trained_fold_count"],
        "boosters": execution["total_boosters"],
        "hmm_well_runs": execution["hmm_well_runs"],
        "pf_well_runs": execution["pf_well_runs"],
        "parent_control_retraining": execution["parent_control_retraining"],
        "gpu": execution["gpu"],
        "inference": execution["inference"],
        "submission": execution["submission"],
        "gate_contract": gate_contract,
        "stage0_guards": guards,
        "runtime_guard": runtime_guard,
        "exp276_corrected_guard_passed": execution["exp276_corrected_guard_passed"],
        "eligible_for_stage_b": eligible_for_stage_b,
        "artifact_sha256": artifact_sha,
        "input_manifest_content_sha256": frame_content_sha256(input_manifest),
        "oracle_metrics_content_sha256": frame_content_sha256(oracle_metrics),
        "unique_best_content_sha256": frame_content_sha256(unique_metrics),
        "by_well_content_sha256": frame_content_sha256(by_well),
    }
    write_json(summary_path, summary)
    summary["artifact_sha256"]["summary"] = sha256_file(summary_path)

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage0_completed",
        "route": "ensemble",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "active_audit_variants": 1,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "pf_well_runs": 0,
        "candidate_guard_passed": guards["candidate_guard_passed"],
        "runtime_guard_passed": runtime_guard["passed"],
        "eligible_for_stage_b": eligible_for_stage_b,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
    }
    write_json(runtime_metrics_path(), metrics)
    return summary


# %% [markdown]
# ## 9. Full-all-well 13-candidate selector helpers
#
# The heavy candidate-long transformation and LightGBM fold loop stay in the
# shared selector pipeline. This notebook keeps the high-level input resolution,
# exact compute contract, candidate/confidence preview, orchestration, and
# artifact readout visible.

# %%
def find_raw_train_test_dirs() -> tuple[Path, Path]:
    roots = [
        PACKAGE_DIR / "data" / "raw",
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path("/kaggle/working/data/raw"),
    ]
    for root in roots:
        if (root / "train").is_dir() and (root / "test").is_dir():
            return root / "train", root / "test"
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = sorted(
            {
                sample.parent
                for sample in kaggle_input.rglob("sample_submission.csv")
                if (sample.parent / "train").is_dir()
                and (sample.parent / "test").is_dir()
                and any((sample.parent / "train").glob("*__horizontal_well.csv"))
                and any((sample.parent / "test").glob("*__horizontal_well.csv"))
            },
            key=lambda path: ("rogii-wellbore-geology-prediction" not in str(path), len(str(path))),
        )
        if matches:
            return matches[0] / "train", matches[0] / "test"
    raise FileNotFoundError("raw train/test directories were not found unambiguously")


def resolve_selector_support_file(patterns: Sequence[str], label: str) -> Path:
    search_roots = [Path.cwd(), Path("/kaggle/input"), Path("/tmp"), PACKAGE_DIR]
    try:
        return resolve_existing_path([str(item) for item in patterns], search_roots)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} was not found: {list(patterns)}") from exc


def selector_execution_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    execution = dict(config["execution"])
    expected = {
        "stage": "selector_outer_oof_full13",
        "active_variants": 1,
        "lightgbm_config_count": 1,
        "objectives": 2,
        "trained_fold_count": 5,
        "total_boosters": 10,
        "hmm_well_runs": 0,
        "pf_well_runs": 0,
        "parent_control_retraining": False,
        "stage_a_feature_audit_enabled": True,
        "stage_b_enabled": True,
        "stage_c_enabled": False,
        "stage_d_enabled": False,
        "gpu": False,
        "inference": False,
        "submission": False,
        "run_approved": True,
    }
    mismatches = {
        key: {"actual": execution.get(key), "expected": value}
        for key, value in expected.items()
        if execution.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Stage B execution contract mismatch: {mismatches}")
    return {
        "stage": execution["stage"],
        "active_variants": execution["active_variants"],
        "lightgbm_configs": execution["lightgbm_config_count"],
        "objectives": execution["objectives"],
        "outer_folds": execution["trained_fold_count"],
        "total_cpu_boosters": execution["total_boosters"],
        "parent_control_retraining": execution["parent_control_retraining"],
        "hmm_well_runs": execution["hmm_well_runs"],
        "pf_well_runs": execution["pf_well_runs"],
        "stage_c_models": 0,
        "stage_d_gpu_models": 0,
        "inference": execution["inference"],
        "submission": execution["submission"],
    }


def stage_c_execution_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    execution = dict(config["execution"])
    expected = {
        "stage": "nested_compact_meta_full13",
        "active_variants": 1,
        "lightgbm_config_count": 1,
        "objectives": 2,
        "trained_fold_count": 20,
        "total_boosters": 40,
        "hmm_well_runs": 0,
        "pf_well_runs": 0,
        "parent_control_retraining": False,
        "stage_a_feature_audit_enabled": True,
        "stage_b_enabled": False,
        "stage_c_enabled": True,
        "stage_d_enabled": False,
        "gpu": False,
        "inference": False,
        "submission": False,
        "run_approved": True,
    }
    mismatches = {
        key: {"actual": execution.get(key), "expected": value}
        for key, value in expected.items()
        if execution.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Stage C execution contract mismatch: {mismatches}")
    nested = dict(config["model"]["nested_downstream_stage"])
    if not bool(nested.get("enabled")):
        raise ValueError("Stage C nested_downstream_stage.enabled must be true")
    if int(nested.get("planned_cpu_selector_boosters", -1)) != 40:
        raise ValueError("Stage C must train exactly 40 CPU selector models")
    return {
        "stage": execution["stage"],
        "active_variants": 1,
        "lightgbm_configs": 1,
        "objectives": 2,
        "outer_folds": 5,
        "inner_folds": 4,
        "total_cpu_boosters": 40,
        "parent_control_retraining": False,
        "hmm_well_runs": 0,
        "pf_well_runs": 0,
        "stage_d_gpu_models": 0,
        "inference": False,
        "submission": False,
    }


def stage_d_execution_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    execution = dict(config["execution"])
    expected = {
        "stage": "downstream_tvt_addonly_full13",
        "active_variants": 1,
        "lightgbm_config_count": 3,
        "objectives": 1,
        "trained_fold_count": 5,
        "total_boosters": 15,
        "hmm_well_runs": 0,
        "pf_well_runs": 0,
        "parent_control_retraining": False,
        "stage_a_feature_audit_enabled": False,
        "stage_b_enabled": False,
        "stage_c_enabled": False,
        "stage_d_enabled": True,
        "gpu": True,
        "inference": False,
        "submission": False,
        "run_approved": True,
    }
    mismatches = {
        key: {"actual": execution.get(key), "expected": value}
        for key, value in expected.items()
        if execution.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Stage D execution contract mismatch: {mismatches}")
    cost = stage_d_full13_cost_contract(config)
    if int(cost["total_gpu_boosters"]) != 15 or bool(cost["control_retraining"]):
        raise ValueError("Stage D must train 15 add-only GPU models and no control")
    return {
        "stage": execution["stage"],
        "active_variants": 1,
        "lightgbm_configs": 3,
        "outer_folds": 5,
        "total_gpu_boosters": 15,
        "parent_control_retraining": False,
        "hmm_well_runs": 0,
        "pf_well_runs": 0,
        "inference": False,
        "submission": False,
    }
def run_selector_stage_b(config: Mapping[str, Any]) -> dict[str, Any]:
    cost = selector_execution_contract(config)
    contract_path = resolve_selector_support_file(
        [str(config["data"]["candidate_contract"])], "candidate contract"
    )
    contract = read_candidate_contract(contract_path)
    contract_evidence = validate_full_contract(contract)
    raw_train_dir, raw_test_dir = find_raw_train_test_dirs()
    search_roots = [Path("/kaggle/input"), Path("/tmp"), PACKAGE_DIR, Path.cwd()]
    cache_root = resolve_exp263_cache_root(config, search_roots)
    cache_evidence = verify_exp263_root(cache_root, config)
    geop_path = resolve_geop_candidate_source(config, search_roots)
    parent_schema_path = resolve_selector_support_file(
        config["data"]["exp251_selected_feature_schema_patterns"],
        "exp251 selected feature schema",
    )
    parent_cfg = config["data"]["parent_stage_b_v5"]
    parent_metrics_path = resolve_selector_support_file(
        parent_cfg["selector_metrics_patterns"], "exp264 Stage B v5 selector metrics"
    )
    parent_candidate_metrics_path = resolve_selector_support_file(
        parent_cfg["selector_candidate_metrics_patterns"],
        "exp264 Stage B v5 candidate metrics",
    )
    inputs = {
        "candidate_contract": str(contract_path),
        "candidate_contract_sha256": contract_evidence["candidate_contract_sha256"],
        "candidate_order": contract_evidence["candidate_order"],
        "geop_hmm_spec": contract_evidence["geop_hmm_spec"],
        "geop_hmm_native_confidence_mapping": contract["confidence_contract"][
            "geop_hmm_source_mapping"
        ],
        "exp263_cache": cache_evidence,
        "exp279_geop_source": str(geop_path),
        "exp279_geop_source_sha256": selector_sha256_file(geop_path),
        "raw_train_dir": str(raw_train_dir),
        "raw_test_dir": str(raw_test_dir),
        "parent_feature_schema": str(parent_schema_path),
        "parent_stage_b_metrics": str(parent_metrics_path),
        "parent_stage_b_candidate_metrics": str(parent_candidate_metrics_path),
    }
    print("Exact approved compute contract")
    print(json.dumps(cost, indent=2, ensure_ascii=False))
    print("Candidate and confidence input contract")
    print(json.dumps(inputs, indent=2, ensure_ascii=False))
    artifacts = runtime_artifacts_dir()
    summary = run_geop_hmm_selector_stage_b(
        config=config,
        contract=contract,
        cache_root=cache_root,
        geop_candidate_path=geop_path,
        raw_train_dir=raw_train_dir,
        raw_test_dir=raw_test_dir,
        output_dir=artifacts,
        parent_schema_path=parent_schema_path,
        parent_metrics_path=parent_metrics_path,
        parent_candidate_metrics_path=parent_candidate_metrics_path,
    )
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": summary["status"],
        "route": config["experiment"]["route"],
        "cv": summary["stage_b"]["hard_primary_oof_rmse"],
        "public_lb": None,
        "private_lb": None,
        "candidate_count": 13,
        "model_count": summary["actual_model_count"],
        "selector_addition_comparison": summary["selector_addition_comparison"],
        "candidate_score_oof_sha256": summary["stage_b"]["candidate_score_oof_sha256"],
        "compact_meta_oof_sha256": summary["stage_b"]["compact_meta_oof_sha256"],
        "model_manifest_sha256": summary["stage_b"]["model_manifest_sha256"],
        "inference": False,
        "submission": False,
    }
    write_json(runtime_metrics_path(), metrics)
    return summary


def run_selector_stage_c(config: Mapping[str, Any]) -> dict[str, Any]:
    cost = stage_c_execution_contract(config)
    contract_path = resolve_selector_support_file(
        [str(config["data"]["candidate_contract"])], "candidate contract"
    )
    contract = read_candidate_contract(contract_path)
    contract_evidence = validate_full_contract(contract)
    raw_train_dir, raw_test_dir = find_raw_train_test_dirs()
    search_roots = [Path("/kaggle/input"), Path("/tmp"), PACKAGE_DIR, Path.cwd()]
    cache_root = resolve_exp263_cache_root(config, search_roots)
    cache_evidence = verify_exp263_root(cache_root, config)
    geop_path = resolve_geop_candidate_source(config, search_roots)
    parent_schema_path = resolve_selector_support_file(
        config["data"]["exp251_selected_feature_schema_patterns"],
        "exp251 selected feature schema",
    )
    parent_cfg = config["data"]["parent_stage_c_v6"]
    parent_metrics_path = resolve_selector_support_file(
        parent_cfg["nested_selector_metrics_patterns"],
        "exp264 Stage C v6 nested selector metrics",
    )
    parent_fold_metrics_path = resolve_selector_support_file(
        parent_cfg["nested_selector_fold_metrics_patterns"],
        "exp264 Stage C v6 nested selector fold metrics",
    )
    inputs = {
        "candidate_contract": str(contract_path),
        "candidate_contract_sha256": contract_evidence["candidate_contract_sha256"],
        "candidate_order": contract_evidence["candidate_order"],
        "geop_hmm_spec": contract_evidence["geop_hmm_spec"],
        "geop_hmm_native_confidence_mapping": contract["confidence_contract"][
            "geop_hmm_source_mapping"
        ],
        "exp263_cache": cache_evidence,
        "exp279_geop_source": str(geop_path),
        "exp279_geop_source_sha256": selector_sha256_file(geop_path),
        "raw_train_dir": str(raw_train_dir),
        "raw_test_dir": str(raw_test_dir),
        "parent_feature_schema": str(parent_schema_path),
        "parent_stage_c_metrics": str(parent_metrics_path),
        "parent_stage_c_fold_metrics": str(parent_fold_metrics_path),
    }
    print("Exact approved Stage C compute contract")
    print(json.dumps(cost, indent=2, ensure_ascii=False))
    print("Candidate, confidence, and parent Stage C input contract")
    print(json.dumps(inputs, indent=2, ensure_ascii=False))
    artifacts = runtime_artifacts_dir()
    summary = run_geop_hmm_selector_stage_c(
        config=config,
        contract=contract,
        cache_root=cache_root,
        geop_candidate_path=geop_path,
        raw_train_dir=raw_train_dir,
        raw_test_dir=raw_test_dir,
        output_dir=artifacts,
        parent_schema_path=parent_schema_path,
        parent_stage_c_metrics_path=parent_metrics_path,
        parent_stage_c_fold_metrics_path=parent_fold_metrics_path,
    )
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": summary["status"],
        "route": config["experiment"]["route"],
        "cv": summary["stage_c"]["hard_primary_oof_rmse"],
        "public_lb": None,
        "private_lb": None,
        "candidate_count": 13,
        "compact_feature_count": 77,
        "model_count": summary["actual_model_count"],
        "stage_c_parent_comparison": summary["stage_c_parent_comparison"],
        "score_guard": summary["stage_c"]["score_guard"],
        "leakage_audit": summary["stage_c"]["leakage_audit"],
        "nested_selector_model_manifest_sha256": summary["stage_c"][
            "nested_selector_model_manifest_sha256"
        ],
        "nested_compact_manifest_sha256": summary["stage_c"][
            "nested_compact_manifest_sha256"
        ],
        "inference": False,
        "submission": False,
    }
    write_json(runtime_metrics_path(), metrics)
    return summary


def run_downstream_stage_d(config: Mapping[str, Any]) -> dict[str, Any]:
    cost = stage_d_execution_contract(config)
    raw_train_dir, _ = find_raw_train_test_dirs()
    search_roots = [Path("/kaggle/input"), Path("/tmp"), PACKAGE_DIR, Path.cwd()]
    stage_c_root = resolve_stage_c_artifact_root(config, search_roots)
    exp218_source_path = resolve_selector_support_file(
        config["data"]["exp218_source_patterns"], "exp218 source"
    )
    exp218_config_path = resolve_selector_support_file(
        config["data"]["exp218_config_patterns"], "exp218 config"
    )
    base_allowlist_path = resolve_selector_support_file(
        config["data"]["exp218_clean_273_allowlist_patterns"],
        "exp218 clean 273 allowlist",
    )
    hidden_assignment_path = resolve_selector_support_file(
        config["data"]["hidden_like_assignment_patterns"],
        "hidden-like assignment",
    )
    parent_cfg = config["data"]["parent_stage_d_v3"]
    parent_paths = {
        "metrics": resolve_selector_support_file(
            parent_cfg["metrics_patterns"], "parent Stage D metrics"
        ),
        "fold_metrics": resolve_selector_support_file(
            parent_cfg["fold_metrics_patterns"], "parent Stage D fold metrics"
        ),
        "bucket_metrics": resolve_selector_support_file(
            parent_cfg["bucket_metrics_patterns"], "parent Stage D bucket metrics"
        ),
        "hidden_like_metrics": resolve_selector_support_file(
            parent_cfg["hidden_like_metrics_patterns"],
            "parent Stage D hidden-like metrics",
        ),
        "by_well": resolve_selector_support_file(
            parent_cfg["by_well_patterns"], "parent Stage D by-well metrics"
        ),
    }
    inputs = {
        "stage_c_root": str(stage_c_root),
        "stage_c_nested_compact_manifest_sha256": selector_sha256_file(
            stage_c_root / "nested_compact_manifest.json"
        ),
        "exp218_source": str(exp218_source_path),
        "exp218_config": str(exp218_config_path),
        "base_feature_allowlist": str(base_allowlist_path),
        "hidden_like_assignment": str(hidden_assignment_path),
        "parent_stage_d_references": {
            name: {"path": str(path), "sha256": selector_sha256_file(path)}
            for name, path in parent_paths.items()
        },
        "raw_train_dir": str(raw_train_dir),
    }
    print("Exact approved Stage D compute contract")
    print(json.dumps(cost, indent=2, ensure_ascii=False))
    print("Stage C, downstream, and saved parent input contract")
    print(json.dumps(inputs, indent=2, ensure_ascii=False))
    metrics = run_geop_hmm_stage_d_addonly(
        config=config,
        stage_c_root=stage_c_root,
        exp218_source_path=exp218_source_path,
        exp218_config_path=exp218_config_path,
        base_feature_allowlist_path=base_allowlist_path,
        hidden_like_assignment_path=hidden_assignment_path,
        raw_train_dir=raw_train_dir,
        parent_reference_paths=parent_paths,
        output_dir=runtime_artifacts_dir(),
    )
    runtime_metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": metrics["status"],
        "route": config["experiment"]["route"],
        "cv": metrics["comparison_vs_parent12"][
            "new13_selector_compact_addonly_rmse"
        ],
        "public_lb": None,
        "private_lb": None,
        "candidate_count": 13,
        "base_feature_count": 273,
        "compact_feature_count": 77,
        "final_feature_count": 350,
        "model_count": metrics["model_count"],
        "comparison_vs_parent12": metrics["comparison_vs_parent12"],
        "parent_control_retraining": False,
        "inference": False,
        "submission": False,
    }
    write_json(runtime_metrics_path(), runtime_metrics)
    return metrics


# %% [markdown]
# ## 10. Setup, contract preview, and Stage 0 / Stage B / Stage C execution

# %%
if EXECUTE_NOTEBOOK:
    CONFIG_PATH = find_config_path()
    CONFIG = read_yaml(CONFIG_PATH)
    STAGE = str(get_nested(CONFIG, "execution.stage"))
    print("Experiment:", EXPERIMENT_NAME)
    print("Config:", CONFIG_PATH)
    print("Route:", get_nested(CONFIG, "experiment.route"))
    print("Parent:", get_nested(CONFIG, "lineage.parent"))
    print("Candidate generator parent:", get_nested(CONFIG, "lineage.candidate_generator_parent"))
    print("Stage:", STAGE)
    print("Expected artifacts:", get_nested(CONFIG, "audit.expected_artifacts"))

    if STAGE == "stage0_saved_oof_readout_only":
        RUN_SUMMARY = run_stage0(CONFIG)
    elif STAGE == "selector_outer_oof_full13":
        RUN_SUMMARY = run_selector_stage_b(CONFIG)
    elif STAGE == "nested_compact_meta_full13":
        RUN_SUMMARY = run_selector_stage_c(CONFIG)
    elif STAGE == "downstream_tvt_addonly_full13":
        RUN_SUMMARY = run_downstream_stage_d(CONFIG)
    else:
        raise ValueError(f"unsupported exp286 train stage: {STAGE}")

    print(json.dumps(RUN_SUMMARY, indent=2, ensure_ascii=False))
    if STAGE == "selector_outer_oof_full13":
        artifacts_dir = runtime_artifacts_dir()
        print("Selector parent comparison")
        print((artifacts_dir / "selector_parent_comparison.json").read_text())
        selection = pd.read_csv(artifacts_dir / "selector_selection_rate.csv")
        print("geop_hmm selection by fold/objective")
        print(selection[selection["candidate_id"].eq("geop_hmm")].to_string(index=False))
        importance = pd.read_csv(
            artifacts_dir / "feature_importance_by_objective_fold.csv"
        )
        importance_summary = (
            importance.groupby(["objective", "feature"], as_index=False)["gain_importance"]
            .mean()
            .sort_values(["objective", "gain_importance"], ascending=[True, False])
        )
        for objective in ("pred_abs_error", "p_within10"):
            print(f"Top gain features: {objective}")
            print(
                importance_summary[importance_summary["objective"].eq(objective)]
                .head(30)
                .to_string(index=False)
            )
    elif STAGE == "nested_compact_meta_full13":
        artifacts_dir = runtime_artifacts_dir()
        print("Stage C parent comparison")
        print((artifacts_dir / "stage_c_parent_comparison.json").read_text())
        print("Stage C fold comparison")
        print(
            pd.read_csv(artifacts_dir / "stage_c_parent_fold_comparison.csv").to_string(
                index=False
            )
        )
        print("Stage C score and leakage guards")
        print(json.dumps(RUN_SUMMARY["stage_c"]["score_guard"], indent=2))
        print(json.dumps(RUN_SUMMARY["stage_c"]["leakage_audit"], indent=2))
        importance = pd.read_csv(
            artifacts_dir / "nested_feature_importance_by_objective_outer_inner.csv"
        )
        importance_summary = (
            importance[importance["importance_type"].eq("gain")]
            .groupby(["objective", "feature"], as_index=False)["importance"]
            .mean()
            .sort_values(["objective", "importance"], ascending=[True, False])
        )
        for objective in ("pred_abs_error", "p_within10"):
            print(f"Stage C top gain features: {objective}")
            print(
                importance_summary[importance_summary["objective"].eq(objective)]
                .head(30)
                .to_string(index=False)
            )
    elif STAGE == "downstream_tvt_addonly_full13":
        artifacts_dir = runtime_artifacts_dir()
        print("Stage D full13 vs saved parent12 comparison")
        print(json.dumps(RUN_SUMMARY["comparison_vs_parent12"], indent=2))
        for filename in (
            "stage_d_parent_fold_comparison.csv",
            "stage_d_bucket_comparison.csv",
            "stage_d_hidden_like_comparison.csv",
        ):
            print(filename)
            print(pd.read_csv(artifacts_dir / filename).to_string(index=False))
        importance = pd.read_csv(artifacts_dir / "stage_d_feature_importance.csv")
        compact_importance = (
            importance[
                importance["importance_type"].eq("gain")
                & importance["feature"].str.startswith("selector__")
            ]
            .groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
        )
        print("Stage D compact feature mean gain importance")
        print(compact_importance.head(50).to_string(index=False))
    print("Generated artifacts:")
    for artifact in sorted(runtime_artifacts_dir().rglob("*")):
        if artifact.is_file():
            print(
                " -",
                artifact.relative_to(runtime_artifacts_dir()),
                artifact.stat().st_size,
                sha256_file(artifact),
            )
