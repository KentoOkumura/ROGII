# %% [markdown]
# # exp507 exp504 nested rank compact add-only on exp413
#
# exp504 の H512 pairwise surface を strict outer/inner nested に再生成し、
# hard-selected TVT を使わず 45 列へ圧縮する。Stage N と Stage D は同じ
# source に実装するが、config の個別 authorization がない限り実行しない。

# %% [markdown]
# ## Contents
# 1. Imports and immutable scientific contract
# 2. Notebook-safe paths, hashes, and authorization
# 3. Frozen exp504 target-free surface preflight
# 4. Pair labels, rank model, and antisymmetric prediction
# 5. Frozen 45-column compact builder
# 6. Stage N strict outer/inner nested orchestration
# 7. Exp413 final370 and Stage N partition verification
# 8. Stage D final415 LightGBM orchestration
# 9. Setup, reproducibility evidence, and fixed stop

# %% [markdown]
# ## 1. Imports and immutable scientific contract

# %%
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from IPython.display import display

from src.candidate_selector_pipeline import (
    KEY_COLUMNS,
    load_stage_d_compact_fold,
    verify_stage_c_artifact_root,
)
from src.likpf_full_replacement import (
    build_replacement_clean273_surface,
    resolve_by_patterns,
)
from src.signed_residual_meta import (
    load_signed_compact_fold,
    verify_signed_stage_s_root,
)

EXPERIMENT_NAME = "exp507_exp504_nested_rank_compact_addonly_on_exp413"
PARENT_EXPERIMENT = "exp413_scale5_likpf_full_replacement_on_exp335"
RANK_SOURCE_EXPERIMENT = "exp504_h512_regret_weighted_block_rank_selector"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")

CANDIDATE_ORDER = (
    "exp226_k16",
    "selfgr_hmm_a070",
    "likpf_mean",
    "exact_hmm",
    "pf_ancc",
    "beam_mean",
    "exp226_k16__selfgr_hmm_a070",
    "exp226_k16__exact_hmm",
    "exp226_k16__likpf_mean",
    "selfgr_hmm_a070__likpf_mean",
    "likpf_mean__exact_hmm",
    "exp226_w500_50_50",
)
ANCHOR_CANDIDATE = "exp226_w500_50_50"
ANCHOR_INDEX = CANDIDATE_ORDER.index(ANCHOR_CANDIDATE)
PAIR_LEFT, PAIR_RIGHT = np.triu_indices(len(CANDIDATE_ORDER), k=1)

BORDA_FEATURES = [f"rank_borda__{name}" for name in CANDIDATE_ORDER]
ANCHOR_PAIR_FEATURES = [
    f"rank_p_vs_anchor__{name}" for name in CANDIDATE_ORDER if name != ANCHOR_CANDIDATE
]
BORDA_SUMMARY_FEATURES = [
    "rank_borda_top1_score",
    "rank_borda_top2_score",
    "rank_borda_margin",
    "rank_borda_score_std",
    "rank_borda_entropy",
]
PROVISIONAL_FEATURES = [f"rank_provisional_is__{name}" for name in CANDIDATE_ORDER]
RANK_COMPACT_FEATURES = [
    *BORDA_FEATURES,
    *ANCHOR_PAIR_FEATURES,
    *BORDA_SUMMARY_FEATURES,
    "rank_anchor_rank",
    *PROVISIONAL_FEATURES,
    "rank_anchor_guard_fallback",
    "rank_borda_tvt_mean",
    "rank_borda_tvt_std",
    "rank_h512_relative_position",
]
BLOCK_CONSTANT_FEATURES = RANK_COMPACT_FEATURES[:42]
ROW_VARYING_FEATURES = RANK_COMPACT_FEATURES[42:]

assert len(PAIR_LEFT) == 66
assert len(RANK_COMPACT_FEATURES) == len(set(RANK_COMPACT_FEATURES)) == 45
assert len(BLOCK_CONSTANT_FEATURES) == 42
assert ROW_VARYING_FEATURES == [
    "rank_borda_tvt_mean",
    "rank_borda_tvt_std",
    "rank_h512_relative_position",
]


# %% [markdown]
# ## 2. Notebook-safe paths, hashes, and authorization

# %%
def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


ROOT = find_project_root()


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return value


def config_path() -> Path:
    candidates = [
        Path.cwd() / "config.yaml",
        ROOT / "experiments" / EXPERIMENT_NAME / "config.yaml",
        KAGGLE_WORKING_ROOT / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{EXPERIMENT_NAME} config.yaml")


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
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


def write_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(dict(value)), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def logical_frame_sha256(frame: pd.DataFrame, chunk_rows: int = 100_000) -> str:
    string_columns = list(frame.select_dtypes(include=["string"]).columns)
    normalized_dtypes = [
        "object" if column in string_columns else str(frame[column].dtype)
        for column in frame.columns
    ]
    digest = hashlib.sha256()
    digest.update("|".join(frame.columns).encode())
    digest.update("|".join(normalized_dtypes).encode())
    for start in range(0, len(frame), int(chunk_rows)):
        selected = frame.iloc[start : start + int(chunk_rows)].copy()
        for column in string_columns:
            selected[column] = selected[column].astype(object)
        hashes = pd.util.hash_pandas_object(selected, index=False, categorize=True)
        digest.update(hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def matrix_sha256(values: np.ndarray, columns: Sequence[str]) -> str:
    array = np.ascontiguousarray(values, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(sha256_json({"shape": list(array.shape), "columns": list(columns)}).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def search_roots() -> list[Path]:
    return [KAGGLE_INPUT_ROOT, KAGGLE_WORKING_ROOT, Path("/tmp"), ROOT, Path.cwd()]


def resolve_file(spec: Mapping[str, Any], *, sha_key: str = "file_sha256") -> Path:
    expected = str(spec.get(sha_key) or spec.get("sha256") or "")
    if not expected:
        raise ValueError(f"frozen SHA is missing for {spec.get('filename', spec)}")
    return resolve_by_patterns(
        [str(item) for item in spec["patterns"]], search_roots(), marker_sha256=expected
    )


def resolve_artifact_root(
    patterns: Sequence[str], *, marker: str, expected_marker_sha256: str
) -> Path:
    candidates: list[Path] = []
    for raw in patterns:
        direct = Path(raw)
        if direct.is_dir():
            candidates.append(direct)
        if not direct.is_absolute():
            for root in search_roots():
                if root.exists():
                    candidates.extend(item for item in root.glob(raw) if item.is_dir())
    for root in search_roots():
        if root.exists():
            candidates.extend(item.parent for item in root.rglob(marker))
    for candidate in dict.fromkeys(candidates):
        marker_path = candidate / marker
        if marker_path.is_file() and sha256_file(marker_path) == expected_marker_sha256:
            return candidate
    raise FileNotFoundError(f"frozen artifact root not found: marker={marker}")


def competition_data_root(config: Mapping[str, Any]) -> Path:
    local = ROOT / str(config["data"]["raw_dir"])
    if not is_kaggle_runtime():
        return local
    project = load_yaml(ROOT / "project.yml") if (ROOT / "project.yml").is_file() else {}
    slug = str(project.get("competition", {}).get("slug", ""))
    candidates = [KAGGLE_INPUT_ROOT / slug, KAGGLE_INPUT_ROOT / "competitions" / slug]
    for candidate in candidates:
        if (candidate / "train").is_dir():
            return candidate
    for candidate in KAGGLE_INPUT_ROOT.iterdir():
        if (candidate / "train").is_dir():
            return candidate
    raise FileNotFoundError("competition train directory was not found")


def validate_static_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    if config["experiment"]["name"] != EXPERIMENT_NAME:
        raise ValueError("experiment name changed")
    if config["experiment"]["route"] != "ensemble":
        raise ValueError("exp507 route must remain ensemble")
    if config["lineage"]["parent"] != PARENT_EXPERIMENT:
        raise ValueError("exp507 parent changed")
    if tuple(config["candidate_contract"]["order"]) != CANDIDATE_ORDER:
        raise ValueError("candidate order changed")
    if config["candidate_contract"]["anchor"] != ANCHOR_CANDIDATE:
        raise ValueError("anchor changed")
    if not bool(config["implementation"]["implementation_approval_received"]):
        raise ValueError("implementation approval is not recorded")
    counts = config["execution_contract"]
    observed = {
        "rank_models": int(counts["stage_n"]["new_cpu_models"]),
        "tvt_models": int(counts["stage_d"]["new_gpu_models"]),
        "total": int(counts["total_new_boosters"]),
        "control_retrains": int(counts["stage_d"]["control_retrains"]),
        "outer_retrains": int(counts["stage_n"]["outer_model_retrains"]),
    }
    expected = {
        "rank_models": 20,
        "tvt_models": 15,
        "total": 35,
        "control_retrains": 0,
        "outer_retrains": 0,
    }
    if observed != expected:
        raise ValueError(f"compute contract changed: {observed}")
    rank = config["features"]["rank_compact"]
    if (
        int(rank["feature_count"]) != 45
        or int(rank["block_constant_feature_count"]) != 42
        or int(rank["row_varying_feature_count"]) != 3
    ):
        raise ValueError("rank compact count changed")
    if int(config["technical_gate"]["expected_final_features"]) != 415:
        raise ValueError("final415 contract changed")
    return {"cost_contract": observed, "rank_schema_sha256": sha256_json(RANK_COMPACT_FEATURES)}


def require_stage_authorization(config: Mapping[str, Any], stage: str) -> None:
    if not (is_kaggle_runtime() or os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") == "1"):
        raise RuntimeError("exp507 is Kaggle-first; local execution requires explicit approval")
    runtime = config["runtime"]
    checks = {
        "selected_stage": str(runtime["selected_stage"]) == stage,
        "run_approved": bool(runtime[f"{stage}_run_approved"]),
        "implementation_approved": bool(
            config["implementation"]["implementation_approval_received"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"{stage} remains disabled: {checks}")


# %% [markdown]
# ## 3. Frozen exp504 target-free surface preflight

# %%
@dataclass
class FrozenRankSurface:
    row_metadata: pd.DataFrame
    blocks: pd.DataFrame
    candidate_values: np.memmap
    candidate_features: np.ndarray
    shared_features: np.ndarray
    block_context: np.ndarray
    pair_feature_names: list[str]
    input_evidence: dict[str, Any]


def candidate_bank_content_sha256(
    values: np.ndarray, key_sha256: str, chunk_rows: int = 100_000
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(CANDIDATE_ORDER), separators=(",", ":")).encode())
    digest.update(str(key_sha256).encode())
    for position, candidate_id in enumerate(CANDIDATE_ORDER):
        digest.update(candidate_id.encode())
        for start in range(0, len(values), chunk_rows):
            stop = min(start + chunk_rows, len(values))
            digest.update(np.asarray(values[start:stop, position], dtype="<f4").tobytes())
    return digest.hexdigest()


def load_frozen_rank_surface(config: Mapping[str, Any]) -> FrozenRankSurface:
    exp504 = config["data"]["exp504"]
    specs = exp504["artifacts"]
    paths = {name: resolve_file(spec) for name, spec in specs.items()}
    schema = json.loads(paths["block_feature_schema"].read_text(encoding="utf-8"))
    freeze = json.loads(paths["target_free_freeze"].read_text(encoding="utf-8"))
    pair_contract = json.loads(paths["pair_contract"].read_text(encoding="utf-8"))
    model_manifest = json.loads(paths["model_manifest"].read_text(encoding="utf-8"))
    if tuple(freeze["candidate_order"]) != CANDIDATE_ORDER:
        raise ValueError("exp504 target-free candidate order changed")
    if tuple(pair_contract["candidate_order"]) != CANDIDATE_ORDER:
        raise ValueError("exp504 pair contract candidate order changed")
    if pair_contract["canonical_pair_left"] != PAIR_LEFT.tolist() or pair_contract[
        "canonical_pair_right"
    ] != PAIR_RIGHT.tolist():
        raise ValueError("exp504 canonical pair order changed")
    if model_manifest["model_count"] != 5 or model_manifest[
        "model_manifest_sha256"
    ] != exp504["model_manifest_logical_sha256"]:
        raise ValueError("exp504 saved outer model manifest changed")

    row_columns = [str(item) for item in specs["row_metadata"]["allowlist_columns"]]
    row_metadata = pd.read_parquet(paths["row_metadata"], columns=row_columns)
    blocks = pd.read_parquet(paths["block_metadata"])
    expected = config["technical_gate"]
    if (
        len(row_metadata) != int(expected["expected_rows"])
        or row_metadata["well"].nunique() != int(expected["expected_wells"])
        or len(blocks) != int(expected["expected_h512_blocks"])
        or row_metadata["id"].astype(str).duplicated().any()
    ):
        raise ValueError("exp504 row/block identity contract changed")
    if not np.array_equal(blocks["h512_group"], np.arange(len(blocks))):
        raise ValueError("H512 block IDs are not dense")
    for row in blocks.itertuples(index=False):
        start, stop = int(row.row_start), int(row.row_stop_exclusive)
        part = row_metadata.iloc[start:stop]
        if (
            len(part) != int(row.row_count)
            or part["well"].nunique() != 1
            or int(part["h512_group"].iloc[0]) != int(row.h512_group)
            or not part["h512_group"].eq(int(row.h512_group)).all()
        ):
            raise ValueError(f"H512 block mapping changed: {row.h512_group}")

    n_rows = len(row_metadata)
    candidate_values = np.memmap(
        paths["candidate_bank"], mode="r", dtype="float32", shape=(n_rows, 12)
    )
    candidate_features = np.load(paths["candidate_block_features"], mmap_mode="r")
    shared_features = np.load(paths["shared_block_features"], mmap_mode="r")
    block_context = np.load(paths["block_context"], mmap_mode="r")
    expected_shapes = {
        "candidate_block_features": tuple(specs["candidate_block_features"]["shape"]),
        "shared_block_features": tuple(specs["shared_block_features"]["shape"]),
        "block_context": tuple(specs["block_context"]["shape"]),
    }
    observed_shapes = {
        "candidate_block_features": candidate_features.shape,
        "shared_block_features": shared_features.shape,
        "block_context": block_context.shape,
    }
    if observed_shapes != expected_shapes:
        raise ValueError(f"exp504 array shape changed: {observed_shapes}")
    if candidate_bank_content_sha256(
        candidate_values, specs["candidate_bank"]["key_content_sha256"]
    ) != specs["candidate_bank"]["logical_content_sha256"]:
        raise ValueError("candidate bank logical content SHA mismatch")
    if (
        schema["feature_schema_sha256"] != exp504["block_feature_schema_sha256"]
        or schema["feature_content_sha256"] != exp504["block_feature_content_sha256"]
        or len(schema["pair_features"]) != int(config["runtime"]["max_pair_feature_columns"])
    ):
        raise ValueError("exp504 block feature schema/content contract changed")
    evidence = {
        "paths": {name: str(path) for name, path in paths.items()},
        "file_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "row_allowlist": row_columns,
        "row_allowlist_logical_sha256": logical_frame_sha256(row_metadata),
        "block_metadata_logical_sha256": logical_frame_sha256(blocks),
        "candidate_bank_logical_sha256": specs["candidate_bank"]["logical_content_sha256"],
        "truth_columns_read_before_target_freeze": [],
        "saved_outer_models_retrained": 0,
    }
    if evidence["row_allowlist_logical_sha256"] != specs["row_metadata"][
        "logical_allowlist_sha256"
    ]:
        raise ValueError("exp504 row allowlist logical SHA mismatch")
    if evidence["block_metadata_logical_sha256"] != specs["block_metadata"][
        "logical_content_sha256"
    ]:
        raise ValueError("exp504 block metadata logical SHA mismatch")
    return FrozenRankSurface(
        row_metadata=row_metadata,
        blocks=blocks,
        candidate_values=candidate_values,
        candidate_features=candidate_features,
        shared_features=shared_features,
        block_context=block_context,
        pair_feature_names=[str(item) for item in schema["pair_features"]],
        input_evidence=evidence,
    )


# %% [markdown]
# ## 4. Pair labels, rank model, and antisymmetric prediction

# %%
@dataclass
class PairTargets:
    block_ids: np.ndarray
    left: np.ndarray
    right: np.ndarray
    label: np.ndarray
    sample_weight: np.ndarray
    unordered_examples: int
    logical_sha256: str


def load_truth_for_folds(
    surface: FrozenRankSurface, raw_train_dir: Path, folds: Sequence[int]
) -> np.ndarray:
    allowed = {int(item) for item in folds}
    truth = np.full(len(surface.row_metadata), np.nan, dtype=np.float32)
    subset = surface.row_metadata[surface.row_metadata["outer_fold"].isin(allowed)]
    for well, group in subset.groupby("well", sort=True):
        path = raw_train_dir / f"{well}__horizontal_well.csv"
        raw = pd.read_csv(path, usecols=["TVT"])
        indices = group["well_row_idx"].to_numpy(np.int64)
        values = pd.to_numeric(raw.iloc[indices]["TVT"], errors="raise").to_numpy(np.float32)
        truth[group.index.to_numpy(np.int64)] = values
    expected = surface.row_metadata["outer_fold"].isin(allowed).to_numpy()
    if not np.isfinite(truth[expected]).all() or np.isfinite(truth[~expected]).any():
        raise ValueError("truth access crossed the allowed fold boundary")
    return truth


def compute_block_mse(
    surface: FrozenRankSurface, truth: np.ndarray, block_ids: np.ndarray
) -> np.ndarray:
    output = np.full((len(surface.blocks), 12), np.nan, dtype=np.float64)
    for block_id in np.asarray(block_ids, dtype=np.int32):
        row = surface.blocks.iloc[int(block_id)]
        start, stop = int(row["row_start"]), int(row["row_stop_exclusive"])
        actual = truth[start:stop]
        if not np.isfinite(actual).all():
            raise ValueError(f"truth missing for training block {block_id}")
        delta = surface.candidate_values[start:stop].astype(np.float64) - actual[:, None]
        output[block_id] = np.mean(np.square(delta), axis=0)
    return output


def build_pair_targets(
    block_mse: np.ndarray,
    row_count: np.ndarray,
    block_ids: np.ndarray,
    *,
    tie_tolerance: float = 1.0e-12,
) -> PairTargets:
    selected = np.asarray(block_ids, dtype=np.int32)
    left = np.tile(PAIR_LEFT.astype(np.int8), len(selected))
    right = np.tile(PAIR_RIGHT.astype(np.int8), len(selected))
    repeated = np.repeat(selected, len(PAIR_LEFT))
    delta = block_mse[repeated, left] - block_mse[repeated, right]
    keep = np.abs(delta) > float(tie_tolerance)
    block = repeated[keep]
    canonical_left, canonical_right = left[keep], right[keep]
    label = (delta[keep] < 0.0).astype(np.int8)
    raw_weight = row_count[block].astype(np.float64) * np.log1p(np.abs(delta[keep]))
    if not np.isfinite(raw_weight).all() or np.any(raw_weight <= 0.0):
        raise ValueError("pair regret weight is invalid")
    normalized = raw_weight / raw_weight.mean()
    canonical = pd.DataFrame(
        {
            "h512_group": block,
            "candidate_left": canonical_left,
            "candidate_right": canonical_right,
            "label_left_better": label,
            "raw_weight": raw_weight,
            "normalized_weight": normalized,
        }
    )
    return PairTargets(
        block_ids=np.concatenate([block, block]).astype(np.int32),
        left=np.concatenate([canonical_left, canonical_right]).astype(np.int8),
        right=np.concatenate([canonical_right, canonical_left]).astype(np.int8),
        label=np.concatenate([label, 1 - label]).astype(np.int8),
        sample_weight=np.concatenate([normalized / 2.0, normalized / 2.0]).astype(np.float32),
        unordered_examples=len(canonical),
        logical_sha256=logical_frame_sha256(canonical),
    )


def assemble_pair_features(
    surface: FrozenRankSurface,
    block_ids: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    block_ids = np.asarray(block_ids, dtype=np.int64)
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    left_value = surface.candidate_features[block_ids, left]
    right_value = surface.candidate_features[block_ids, right]
    output = np.concatenate(
        [
            left_value - right_value,
            np.abs(left_value - right_value),
            (left_value + right_value) / np.float32(2.0),
            surface.shared_features[block_ids],
            surface.block_context[block_ids],
        ],
        axis=1,
    ).astype(np.float32)
    if output.shape[1] != len(surface.pair_feature_names) or np.isinf(output).any():
        raise ValueError("pair feature matrix contract failed")
    return output


def materialize_pair_memmap(
    surface: FrozenRankSurface, targets: PairTargets, path: Path, chunk_rows: int = 4096
) -> np.memmap:
    matrix = np.memmap(
        path,
        mode="w+",
        dtype="float32",
        shape=(len(targets.block_ids), len(surface.pair_feature_names)),
    )
    for start in range(0, len(targets.block_ids), chunk_rows):
        stop = min(start + chunk_rows, len(targets.block_ids))
        matrix[start:stop] = assemble_pair_features(
            surface,
            targets.block_ids[start:stop],
            targets.left[start:stop],
            targets.right[start:stop],
        )
    matrix.flush()
    return matrix


def make_rank_model(config: Mapping[str, Any]) -> Any:
    from lightgbm import LGBMClassifier

    model = config["rank_source_contract"]["lightgbm"]
    return LGBMClassifier(
        objective=str(model["objective"]),
        metric=str(model["metric"]),
        boosting_type=str(model["boosting_type"]),
        learning_rate=float(model["learning_rate"]),
        n_estimators=int(model["n_estimators"]),
        num_leaves=int(model["num_leaves"]),
        min_child_samples=int(model["min_child_samples"]),
        max_depth=int(model["max_depth"]),
        subsample=float(model["subsample"]),
        subsample_freq=int(model["subsample_freq"]),
        colsample_bytree=float(model["colsample_bytree"]),
        reg_alpha=float(model["reg_alpha"]),
        reg_lambda=float(model["reg_lambda"]),
        max_bin=int(model["max_bin"]),
        random_state=int(model["random_state"]),
        deterministic=bool(model["deterministic"]),
        force_col_wise=bool(model["force_col_wise"]),
        device=str(model["device"]),
        n_jobs=int(model["n_jobs"]),
        verbosity=-1,
    )


def predict_pair_probabilities(
    model: Any,
    surface: FrozenRankSurface,
    block_ids: np.ndarray,
    *,
    batch_blocks: int = 64,
) -> np.ndarray:
    block_ids = np.asarray(block_ids, dtype=np.int32)
    output = np.full((len(block_ids), len(PAIR_LEFT)), np.nan, dtype=np.float32)
    for start in range(0, len(block_ids), batch_blocks):
        batch = block_ids[start : start + batch_blocks]
        repeated = np.repeat(batch, len(PAIR_LEFT))
        left = np.tile(PAIR_LEFT.astype(np.int8), len(batch))
        right = np.tile(PAIR_RIGHT.astype(np.int8), len(batch))
        forward = assemble_pair_features(surface, repeated, left, right)
        reverse = assemble_pair_features(surface, repeated, right, left)
        probability = 0.5 * (
            model.predict_proba(forward)[:, 1] + 1.0 - model.predict_proba(reverse)[:, 1]
        )
        output[start : start + len(batch)] = probability.reshape(len(batch), -1)
    if not np.isfinite(output).all() or np.any((output < 0.0) | (output > 1.0)):
        raise ValueError("antisymmetric pair probabilities are invalid")
    return output


# %% [markdown]
# ## 5. Frozen 45-column compact builder

# %%
def pair_probability_to_rank(
    pair_probability: np.ndarray,
    *,
    tie_tolerance: float = 1.0e-12,
    guard_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    probability = np.asarray(pair_probability, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != 66:
        raise ValueError("full transient pair surface must have 66 columns")
    n_blocks = len(probability)
    win = np.full((n_blocks, 12, 12), np.nan, dtype=np.float64)
    rows = np.arange(n_blocks)
    for pair_index, (left, right) in enumerate(zip(PAIR_LEFT, PAIR_RIGHT, strict=True)):
        value = probability[:, pair_index]
        win[rows, left, right] = value
        win[rows, right, left] = 1.0 - value
    diagonal = np.arange(12)
    win[:, diagonal, diagonal] = 0.5
    borda = (win.sum(axis=2) - 0.5) / 11.0
    if not np.allclose(borda.sum(axis=1), 6.0, rtol=0.0, atol=1.0e-6):
        raise ValueError("Borda sum differs from six")
    order = np.empty((n_blocks, 12), dtype=np.int8)
    provisional = np.empty(n_blocks, dtype=np.int8)
    fallback = np.zeros(n_blocks, dtype=bool)
    for block in range(n_blocks):
        remaining = list(range(12))
        ranked: list[int] = []
        while remaining:
            maximum = max(float(borda[block, index]) for index in remaining)
            tied = [
                index
                for index in remaining
                if abs(float(borda[block, index]) - maximum) <= tie_tolerance
            ]
            tied.sort(key=lambda index: (index != ANCHOR_INDEX, index))
            ranked.extend(tied)
            remaining = [index for index in remaining if index not in tied]
        order[block] = ranked
        provisional[block] = ranked[0]
        winner = int(ranked[0])
        fallback[block] = bool(
            winner != ANCHOR_INDEX and not (win[block, winner, ANCHOR_INDEX] > guard_threshold)
        )
    anchor_probability = np.column_stack(
        [win[:, index, ANCHOR_INDEX] for index in range(12) if index != ANCHOR_INDEX]
    )
    weights = borda / borda.sum(axis=1, keepdims=True)
    entropy_terms = np.where(weights > 0.0, weights * np.log(weights), 0.0)
    return {
        "borda": borda.astype(np.float32),
        "p_vs_anchor": anchor_probability.astype(np.float32),
        "order": order,
        "provisional": provisional,
        "fallback": fallback,
        "weights": weights.astype(np.float32),
        "entropy": (-entropy_terms.sum(axis=1) / np.log(12.0)).astype(np.float32),
    }


def build_rank_compact_partition(
    *,
    surface: FrozenRankSurface,
    block_ids: np.ndarray,
    pair_probability: np.ndarray,
    downstream_outer_fold: int,
    role: str,
    held_inner_fold: int | None,
) -> pd.DataFrame:
    block_ids = np.asarray(block_ids, dtype=np.int32)
    if role not in {"train", "valid"}:
        raise ValueError(f"unexpected nested role: {role}")
    if len(block_ids) != len(pair_probability):
        raise ValueError("block/probability row count mismatch")
    rank = pair_probability_to_rank(pair_probability)
    blocks = surface.blocks.iloc[block_ids].reset_index(drop=True)
    row_positions = np.concatenate(
        [
            np.arange(int(row.row_start), int(row.row_stop_exclusive), dtype=np.int64)
            for row in blocks.itertuples(index=False)
        ]
    )
    repeated_block_position = np.repeat(
        np.arange(len(blocks), dtype=np.int32), blocks["row_count"].to_numpy(np.int32)
    )
    borda = rank["borda"]
    order = rank["order"]
    top1 = borda[np.arange(len(blocks)), order[:, 0]]
    top2 = borda[np.arange(len(blocks)), order[:, 1]]
    anchor_rank = np.argmax(order == ANCHOR_INDEX, axis=1).astype(np.float32) + 1.0
    provisional = np.eye(12, dtype=np.float32)[rank["provisional"]]
    block_constant = np.column_stack(
        [
            borda,
            rank["p_vs_anchor"],
            top1,
            top2,
            top1 - top2,
            borda.std(axis=1, ddof=0),
            rank["entropy"],
            anchor_rank,
            provisional,
            rank["fallback"].astype(np.float32),
        ]
    ).astype(np.float32)
    if block_constant.shape[1] != 42:
        raise AssertionError("block-constant rank width changed")
    values = surface.candidate_values[row_positions].astype(np.float32)
    weights = rank["weights"][repeated_block_position]
    weighted_mean = np.sum(weights * values, axis=1, dtype=np.float64).astype(np.float32)
    weighted_var = np.sum(
        weights * np.square(values - weighted_mean[:, None]), axis=1, dtype=np.float64
    )
    weighted_std = np.sqrt(np.maximum(weighted_var, 0.0)).astype(np.float32)
    relative = np.concatenate(
        [
            np.arange(int(count), dtype=np.float32) / np.float32(max(int(count) - 1, 1))
            for count in blocks["row_count"]
        ]
    )
    matrix = np.column_stack(
        [
            block_constant[repeated_block_position],
            weighted_mean,
            weighted_std,
            relative,
        ]
    ).astype(np.float32)
    if matrix.shape != (len(row_positions), 45) or not np.isfinite(matrix).all():
        raise ValueError("rank compact45 matrix shape/finite contract failed")
    frame = surface.row_metadata.iloc[row_positions][KEY_COLUMNS].reset_index(drop=True).copy()
    frame["downstream_outer_fold"] = np.int8(downstream_outer_fold)
    frame["nested_role"] = role
    frame["held_inner_fold"] = np.int8(-1 if held_inner_fold is None else held_inner_fold)
    frame["rank_model_count"] = np.int8(1)
    for position, feature in enumerate(RANK_COMPACT_FEATURES):
        frame[feature] = matrix[:, position]
    if frame["id"].astype(str).duplicated().any():
        raise ValueError("rank compact partition contains duplicate row IDs")
    return frame


def verify_saved_outer_surface(
    surface: FrozenRankSurface, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    specs = config["data"]["exp504"]["artifacts"]
    selection_path = resolve_file(specs["block_selection"])
    probability_path = resolve_file(specs["pair_probability"])
    selection = pd.read_parquet(selection_path)
    probability = pd.read_parquet(probability_path)
    if logical_frame_sha256(selection) != specs["block_selection"][
        "logical_content_sha256"
    ]:
        raise ValueError("saved block selection logical SHA mismatch")
    if logical_frame_sha256(probability) != specs["pair_probability"][
        "logical_content_sha256"
    ]:
        raise ValueError("saved pair probability logical SHA mismatch")
    expected_pair_columns = [
        f"p__{CANDIDATE_ORDER[left]}__beats__{CANDIDATE_ORDER[right]}"
        for left, right in zip(PAIR_LEFT, PAIR_RIGHT, strict=True)
    ]
    if list(probability.columns) != ["h512_group", *expected_pair_columns]:
        raise ValueError("saved exp504 pair column order changed")
    if not np.array_equal(probability["h512_group"], np.arange(len(surface.blocks))):
        raise ValueError("saved pair rows do not align with H512 blocks")
    rank = pair_probability_to_rank(probability[expected_pair_columns].to_numpy(np.float32))
    saved_borda = selection[[f"borda__{name}" for name in CANDIDATE_ORDER]].to_numpy(np.float32)
    checks = {
        "borda_max_abs": float(np.abs(saved_borda - rank["borda"]).max(initial=0.0)),
        "provisional_exact": bool(
            np.array_equal(selection["provisional_candidate_index"], rank["provisional"])
        ),
        "fallback_exact": bool(
            np.array_equal(selection["anchor_guard_fallback"], rank["fallback"])
        ),
    }
    if checks["borda_max_abs"] > 1.0e-6 or not all(
        [checks["provisional_exact"], checks["fallback_exact"]]
    ):
        raise ValueError(f"saved exp504 outer surface parity failed: {checks}")
    return selection, probability, checks


# %% [markdown]
# ## 6. Stage N strict outer/inner nested orchestration

# %%
def nested_partition_plan(folds: Sequence[int] = tuple(range(5))) -> list[dict[str, Any]]:
    ordered = [int(item) for item in folds]
    if ordered != list(range(5)):
        raise ValueError("exp507 outer fold inventory must remain 0..4")
    plan: list[dict[str, Any]] = []
    for outer_fold in ordered:
        for inner_fold in [fold for fold in ordered if fold != outer_fold]:
            plan.append(
                {
                    "downstream_outer_fold": outer_fold,
                    "source_outer_fold": inner_fold,
                    "held_inner_fold": inner_fold,
                    "role": "train",
                    "rank_training_folds": [
                        fold for fold in ordered if fold not in {outer_fold, inner_fold}
                    ],
                    "new_rank_models": 1,
                }
            )
        plan.append(
            {
                "downstream_outer_fold": outer_fold,
                "source_outer_fold": outer_fold,
                "held_inner_fold": -1,
                "role": "valid",
                "rank_training_folds": [fold for fold in ordered if fold != outer_fold],
                "new_rank_models": 0,
            }
        )
    return plan


def run_stage_n(config: Mapping[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    require_stage_authorization(config, "stage_n")
    static = validate_static_contract(config)
    surface = load_frozen_rank_surface(config)
    _, saved_probability, outer_parity = verify_saved_outer_surface(surface, config)
    raw_train_dir = competition_data_root(config) / "train"
    output_dir = KAGGLE_WORKING_ROOT / "artifacts" if is_kaggle_runtime() else (
        ROOT / "experiments" / EXPERIMENT_NAME / "artifacts"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch = KAGGLE_WORKING_ROOT / ".exp507_work" if is_kaggle_runtime() else (
        output_dir / ".exp507_work"
    )
    scratch.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "rank_models"
    partition_dir = output_dir / "rank_compact_partitions"
    model_dir.mkdir(parents=True, exist_ok=True)
    partition_dir.mkdir(parents=True, exist_ok=True)

    all_folds = list(range(5))
    plan = nested_partition_plan(all_folds)
    if len(plan) != 25 or sum(item["new_rank_models"] for item in plan) != 20:
        raise AssertionError("nested partition plan changed")
    block_fold = surface.blocks["outer_fold"].to_numpy(np.int8)
    row_count = surface.blocks["row_count"].to_numpy(np.int32)
    model_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

    for downstream_outer_fold in all_folds:
        for held_inner_fold in [fold for fold in all_folds if fold != downstream_outer_fold]:
            train_folds = [
                fold
                for fold in all_folds
                if fold not in {downstream_outer_fold, held_inner_fold}
            ]
            train_blocks = np.flatnonzero(np.isin(block_fold, train_folds)).astype(np.int32)
            valid_blocks = np.flatnonzero(block_fold == held_inner_fold).astype(np.int32)
            train_wells = set(
                surface.blocks.iloc[train_blocks]["well"].astype(str)
            )
            held_outer_wells = set(
                surface.blocks.loc[
                    surface.blocks["outer_fold"].eq(downstream_outer_fold), "well"
                ].astype(str)
            )
            held_inner_wells = set(
                surface.blocks.iloc[valid_blocks]["well"].astype(str)
            )
            leakage = {
                "downstream_outer_fold": downstream_outer_fold,
                "held_inner_fold": held_inner_fold,
                "train_folds": train_folds,
                "held_outer_train_well_overlap": len(train_wells & held_outer_wells),
                "held_inner_train_well_overlap": len(train_wells & held_inner_wells),
            }
            if leakage["held_outer_train_well_overlap"] or leakage[
                "held_inner_train_well_overlap"
            ]:
                raise ValueError(f"nested well leakage detected: {leakage}")
            leakage_rows.append(leakage)

            truth = load_truth_for_folds(surface, raw_train_dir, train_folds)
            block_mse = compute_block_mse(surface, truth, train_blocks)
            targets = build_pair_targets(block_mse, row_count, train_blocks)
            matrix_path = scratch / (
                f"pair_features_outer{downstream_outer_fold}_inner{held_inner_fold}.f32"
            )
            matrix = materialize_pair_memmap(surface, targets, matrix_path)
            model = make_rank_model(config)
            model.fit(
                matrix,
                targets.label,
                sample_weight=targets.sample_weight,
                feature_name=surface.pair_feature_names,
            )
            probability = predict_pair_probabilities(model, surface, valid_blocks)
            compact = build_rank_compact_partition(
                surface=surface,
                block_ids=valid_blocks,
                pair_probability=probability,
                downstream_outer_fold=downstream_outer_fold,
                role="train",
                held_inner_fold=held_inner_fold,
            )
            partition_path = partition_dir / (
                f"rank_compact_outer{downstream_outer_fold}_source{held_inner_fold}_train.parquet"
            )
            compact.to_parquet(partition_path, index=False)
            model_path = model_dir / (
                f"rank_outer{downstream_outer_fold}_inner{held_inner_fold}.txt"
            )
            model.booster_.save_model(str(model_path))
            model_rows.append(
                {
                    **leakage,
                    "ordered_train_examples": len(targets.block_ids),
                    "unordered_train_examples": targets.unordered_examples,
                    "pair_table_logical_sha256": targets.logical_sha256,
                    "model_path": str(model_path),
                    "model_sha256": sha256_file(model_path),
                    "trees": int(model.booster_.num_trees()),
                }
            )
            for importance_type in ("gain", "split"):
                importance = model.booster_.feature_importance(
                    importance_type=importance_type
                )
                for feature, value in zip(
                    surface.pair_feature_names, importance, strict=True
                ):
                    importance_rows.append(
                        {
                            "downstream_outer_fold": downstream_outer_fold,
                            "held_inner_fold": held_inner_fold,
                            "importance_type": importance_type,
                            "feature": feature,
                            "importance": float(value),
                        }
                    )
            partition_rows.append(
                {
                    "downstream_outer_fold": downstream_outer_fold,
                    "source_outer_fold": held_inner_fold,
                    "held_inner_fold": held_inner_fold,
                    "role": "train",
                    "rank_model_count": 1,
                    "path": str(partition_path),
                    "rows": len(compact),
                    "wells": int(compact["well"].nunique()),
                    "file_sha256": sha256_file(partition_path),
                    "logical_sha256": logical_frame_sha256(compact),
                }
            )
            del truth, block_mse, targets, matrix, model, probability, compact
            gc.collect()
            matrix_path.unlink(missing_ok=True)
            print(json.dumps(model_rows[-1], sort_keys=True), flush=True)

        valid_blocks = np.flatnonzero(block_fold == downstream_outer_fold).astype(np.int32)
        probability = saved_probability.iloc[valid_blocks, 1:].to_numpy(np.float32)
        compact = build_rank_compact_partition(
            surface=surface,
            block_ids=valid_blocks,
            pair_probability=probability,
            downstream_outer_fold=downstream_outer_fold,
            role="valid",
            held_inner_fold=None,
        )
        partition_path = partition_dir / (
            f"rank_compact_outer{downstream_outer_fold}_source{downstream_outer_fold}_valid.parquet"
        )
        compact.to_parquet(partition_path, index=False)
        partition_rows.append(
            {
                "downstream_outer_fold": downstream_outer_fold,
                "source_outer_fold": downstream_outer_fold,
                "held_inner_fold": -1,
                "role": "valid",
                "rank_model_count": 1,
                "path": str(partition_path),
                "rows": len(compact),
                "wells": int(compact["well"].nunique()),
                "file_sha256": sha256_file(partition_path),
                "logical_sha256": logical_frame_sha256(compact),
            }
        )
        del compact, probability
        gc.collect()

    row_roles = sum(int(item["rows"]) for item in partition_rows)
    forbidden_tokens = ("selected", "candidate_index", "true_tvt", "error", "oracle", "well_id")
    forbidden_features = [
        feature for feature in RANK_COMPACT_FEATURES if any(token in feature for token in forbidden_tokens)
    ]
    technical = {
        "input_sha_pass": True,
        "model_count_20": len(model_rows) == 20,
        "partition_count_25": len(partition_rows) == 25,
        "row_roles_exact": row_roles == int(config["technical_gate"]["expected_rank_compact_row_roles"]),
        "feature_count_45_unique": len(RANK_COMPACT_FEATURES) == len(set(RANK_COMPACT_FEATURES)) == 45,
        "held_outer_inner_overlap_zero": all(
            item["held_outer_train_well_overlap"] == 0
            and item["held_inner_train_well_overlap"] == 0
            for item in leakage_rows
        ),
        "forbidden_feature_count_zero": not forbidden_features,
        "outer_surface_parity": outer_parity["borda_max_abs"] <= 1.0e-6
        and outer_parity["provisional_exact"]
        and outer_parity["fallback_exact"],
    }
    if not all(technical.values()):
        raise RuntimeError(f"Stage N technical gate failed: {technical}")
    pd.DataFrame(partition_rows).to_csv(
        output_dir / "rank_compact_partition_manifest.csv", index=False
    )
    pd.DataFrame(leakage_rows).to_csv(output_dir / "rank_leakage_ledger.csv", index=False)
    importance_frame = pd.DataFrame(importance_rows)
    importance_frame.to_csv(output_dir / "rank_feature_importance.csv", index=False)
    model_manifest_sha = write_json(
        output_dir / "rank_model_manifest.json",
        {"model_count": len(model_rows), "models": model_rows},
    )
    manifest = {
        "schema_version": "1.0.0",
        "status": "stage_n_technical_pass",
        "experiment": EXPERIMENT_NAME,
        "static_contract": static,
        "feature_names": RANK_COMPACT_FEATURES,
        "feature_schema_sha256": sha256_json(RANK_COMPACT_FEATURES),
        "partition_count": len(partition_rows),
        "partition_row_roles": row_roles,
        "partition_content_sha256": sha256_json(
            [item["logical_sha256"] for item in partition_rows]
        ),
        "partitions": partition_rows,
        "model_count": len(model_rows),
        "model_manifest_file_sha256": model_manifest_sha,
        "partition_manifest_file_sha256": sha256_file(
            output_dir / "rank_compact_partition_manifest.csv"
        ),
        "leakage_ledger_file_sha256": sha256_file(
            output_dir / "rank_leakage_ledger.csv"
        ),
        "feature_importance_file_sha256": sha256_file(
            output_dir / "rank_feature_importance.csv"
        ),
        "input_evidence": surface.input_evidence,
        "outer_surface_parity": outer_parity,
        "technical_gate": technical,
        "forbidden_features": forbidden_features,
        "control_retraining_boosters": 0,
        "outer_rank_model_retraining_boosters": 0,
        "stage_d_executed": False,
        "inference_executed": False,
        "submission_generated": False,
    }
    manifest_sha = write_json(output_dir / "rank_compact_manifest.json", manifest)
    mean_importance = (
        importance_frame[importance_frame["importance_type"].eq("gain")]
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    display(mean_importance.head(100))
    top = mean_importance.head(30).sort_values("importance")
    if len(top):
        ax = top.plot.barh(
            x="feature",
            y="importance",
            figsize=(11, 11),
            legend=False,
            title="exp507 nested pair-rank mean gain importance",
        )
        ax.set_xlabel("mean gain across 20 inner rank models")
        plt.tight_layout()
        plt.savefig(output_dir / "rank_feature_importance_top30.png", dpi=140)
        plt.show()
    print(json.dumps({**manifest, "manifest_file_sha256": manifest_sha}, indent=2), flush=True)
    print("Stage N PASS. Stop here; Stage D requires separate approval and package.")
    return manifest


# %% [markdown]
# ## 7. Exp413 final370 and Stage N partition verification

# %%
def verify_rank_stage_n_root(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    spec = config["data"]["stage_n_source"]
    expected = str(spec.get("expected_manifest_sha256") or "")
    if len(expected) != 64:
        raise RuntimeError("Stage N manifest SHA is not frozen; Stage D is fail-closed")
    root = resolve_artifact_root(
        spec["root_patterns"], marker="rank_compact_manifest.json", expected_marker_sha256=expected
    )
    manifest = json.loads((root / "rank_compact_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["status"] != "stage_n_technical_pass"
        or manifest["feature_names"] != RANK_COMPACT_FEATURES
        or manifest["partition_count"] != 25
        or manifest["model_count"] != 20
        or not all(manifest["technical_gate"].values())
    ):
        raise ValueError("Stage N manifest contract failed")
    for item in manifest["partitions"]:
        path = Path(item["path"])
        if not path.is_file():
            path = root / "rank_compact_partitions" / Path(item["path"]).name
        if sha256_file(path) != item["file_sha256"]:
            raise ValueError(f"Stage N partition SHA mismatch: {path}")
        item["path"] = str(path)
    return root, manifest


def load_rank_compact_fold(
    manifest: Mapping[str, Any], downstream_outer_fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        *KEY_COLUMNS,
        "downstream_outer_fold",
        "nested_role",
        "held_inner_fold",
        "rank_model_count",
        *RANK_COMPACT_FEATURES,
    ]
    by_role: dict[str, list[pd.DataFrame]] = {"train": [], "valid": []}
    selected = [
        item
        for item in manifest["partitions"]
        if int(item["downstream_outer_fold"]) == int(downstream_outer_fold)
    ]
    for item in sorted(selected, key=lambda value: int(value["source_outer_fold"])):
        frame = pd.read_parquet(item["path"], columns=columns)
        role = str(item["role"])
        if len(frame) != int(item["rows"]) or role not in by_role:
            raise ValueError("Stage N compact partition manifest mismatch")
        by_role[role].append(frame)
    if len(by_role["train"]) != 4 or len(by_role["valid"]) != 1:
        raise ValueError("Stage D requires four rank train and one rank valid partitions")
    train = pd.concat(by_role["train"], ignore_index=True)
    valid = pd.concat(by_role["valid"], ignore_index=True)
    for role, frame in (("train", train), ("valid", valid)):
        if (
            not frame["nested_role"].eq(role).all()
            or not frame["downstream_outer_fold"].eq(downstream_outer_fold).all()
            or frame["id"].astype(str).duplicated().any()
            or not np.isfinite(frame[RANK_COMPACT_FEATURES].to_numpy(np.float32)).all()
        ):
            raise ValueError(f"Stage N {role} compact contract failed")
    return train, valid


def stage_c_verify_config(exp413: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "data": {
            "stage_c_expected_nested_selector_metrics_sha256": exp413["stage_c_metrics_sha256"],
            "stage_c_expected_nested_selector_model_manifest_sha256": exp413[
                "stage_c_model_manifest_sha256"
            ],
            "stage_c_expected_nested_compact_manifest_sha256": exp413[
                "stage_c_compact_manifest_sha256"
            ],
            "stage_c_expected_compact_meta_schema_file_sha256": exp413[
                "stage_c_schema_file_sha256"
            ],
            "stage_c_expected_compact_meta_schema_logical_sha256": exp413[
                "stage_c_schema_logical_sha256"
            ],
        }
    }


def stage_s_verify_config(exp413: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "data": {
            "stage_s_signed_selector_metrics_sha256": exp413["stage_s_metrics_sha256"],
            "stage_s_model_manifest_sha256": exp413["stage_s_model_manifest_sha256"],
            "stage_s_compact_manifest_sha256": exp413["stage_s_compact_manifest_sha256"],
            "stage_s_compact_schema_file_sha256": exp413["stage_s_schema_file_sha256"],
            "stage_s_compact_schema_logical_sha256": exp413[
                "stage_s_schema_logical_sha256"
            ],
            "stage_s_reproducibility_manifest_sha256": exp413[
                "stage_s_reproducibility_manifest_sha256"
            ],
        }
    }


def align_rank_to_parent(
    parent: pd.DataFrame, rank: pd.DataFrame, *, outer_fold: int, role: str
) -> pd.DataFrame:
    index = pd.Index(rank["id"].astype(str))
    positions = index.get_indexer(parent["id"].astype(str))
    if np.any(positions < 0) or len(np.unique(positions)) != len(parent):
        raise ValueError(f"rank/parent join is not one-to-one: fold={outer_fold} role={role}")
    aligned = rank.iloc[positions].reset_index(drop=True)
    if not parent[KEY_COLUMNS].reset_index(drop=True).equals(aligned[KEY_COLUMNS]):
        raise ValueError(f"rank/parent key mismatch: fold={outer_fold} role={role}")
    return aligned


def assemble_final415(
    *,
    base: pd.DataFrame,
    positions: np.ndarray,
    compact: pd.DataFrame,
    signed: pd.DataFrame,
    rank: pd.DataFrame,
    base_features: Sequence[str],
    compact_features: Sequence[str],
    signed_features: Sequence[str],
    chunk_columns: int,
) -> np.ndarray:
    width = len(base_features) + len(compact_features) + len(signed_features) + 45
    matrix = np.empty((len(positions), width), dtype=np.float32)
    for start in range(0, len(base_features), int(chunk_columns)):
        stop = min(start + int(chunk_columns), len(base_features))
        matrix[:, start:stop] = base[list(base_features[start:stop])].iloc[positions].to_numpy(
            np.float32, copy=True
        )
    offset = len(base_features)
    matrix[:, offset : offset + len(compact_features)] = compact[
        list(compact_features)
    ].to_numpy(np.float32, copy=False)
    offset += len(compact_features)
    matrix[:, offset : offset + len(signed_features)] = signed[
        list(signed_features)
    ].to_numpy(np.float32, copy=False)
    offset += len(signed_features)
    matrix[:, offset:] = rank[RANK_COMPACT_FEATURES].to_numpy(np.float32, copy=False)
    if width != 415 or not np.isfinite(matrix).all():
        raise ValueError("final415 matrix width/finite contract failed")
    return matrix


# %% [markdown]
# ## 8. Stage D final415 LightGBM orchestration

# %%
def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def load_saved_exp413_control(
    root: Path,
    base: pd.DataFrame,
    base_features: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    exp413 = config["data"]["exp413"]
    paths = {
        "oof": root / "stage_d_oof_predictions.parquet",
        "metrics": root / "stage_d_metrics.json",
        "model_manifest": root / "stage_d_model_manifest.json",
    }
    expected = {
        "oof": exp413["oof_prediction_sha256"],
        "metrics": exp413["stage_d_metrics_sha256"],
        "model_manifest": exp413["model_manifest_sha256"],
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != expected:
        raise ValueError(f"saved exp413 Stage D SHA mismatch: {actual}")
    manifest = json.loads(paths["model_manifest"].read_text(encoding="utf-8"))
    groups = manifest["feature_groups"]
    compact_features = [str(item) for item in groups["nested_compact"]]
    signed_features = [str(item) for item in groups["signed_compact"]]
    if (
        [str(item) for item in groups["clean_base"]] != list(base_features)
        or len(compact_features) != 74
        or len(signed_features) != 23
        or int(manifest["feature_count"]) != 370
        or int(manifest["model_count"]) != 15
    ):
        raise ValueError("saved exp413 final370 schema changed")
    prediction_column = str(exp413["stage_d_prediction_column"])
    columns = [
        "id",
        "well",
        "md_since",
        "last_known_tvt",
        "target",
        "outer_fold",
        "actual_tvt",
        prediction_column,
    ]
    control = pd.read_parquet(paths["oof"], columns=columns)
    positions = pd.Index(control["id"].astype(str)).get_indexer(base["id"].astype(str))
    if np.any(positions < 0) or len(np.unique(positions)) != len(base):
        raise ValueError("saved exp413 control/base alignment failed")
    control = control.iloc[positions].reset_index(drop=True)
    truth = base["last_known_tvt"].to_numpy(np.float32) + base["target"].to_numpy(np.float32)
    observed = rmse(truth, control[prediction_column])
    if abs(observed - float(config["promotion_gate"]["matched_control_rmse_ft"])) > 1.0e-9:
        raise ValueError("saved exp413 control RMSE changed")
    return control, compact_features, signed_features, {
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": actual,
        "rmse": observed,
        "models_retrained": 0,
    }


def evaluate_stage_d(
    *,
    config: Mapping[str, Any],
    base: pd.DataFrame,
    control: pd.DataFrame,
    prediction: np.ndarray,
    hidden_assignment_path: Path,
    technical_checks: Mapping[str, bool],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    truth = base["last_known_tvt"].to_numpy(np.float32) + base["target"].to_numpy(np.float32)
    parent = control[config["data"]["exp413"]["stage_d_prediction_column"]].to_numpy(np.float32)
    candidate = np.asarray(prediction, dtype=np.float32)
    fold = control["outer_fold"].to_numpy(np.int8)
    fold_rows: list[dict[str, Any]] = []
    for outer_fold in range(5):
        mask = fold == outer_fold
        parent_rmse, candidate_rmse = rmse(truth[mask], parent[mask]), rmse(
            truth[mask], candidate[mask]
        )
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "rows": int(mask.sum()),
                "exp413_rmse": parent_rmse,
                "exp507_rmse": candidate_rmse,
                "delta_exp507_minus_exp413": candidate_rmse - parent_rmse,
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    md_since = base["md_since"].to_numpy(np.float32)
    scope_masks = {
        "md_since_0_250": md_since <= 250.0,
        "md_since_250_1000": (md_since > 250.0) & (md_since < 1000.0),
        "md_since_1000_plus": md_since >= 1000.0,
    }
    scope_rows: list[dict[str, Any]] = []
    for scope, mask in scope_masks.items():
        p, c = rmse(truth[mask], parent[mask]), rmse(truth[mask], candidate[mask])
        scope_rows.append(
            {"scope": scope, "rows": int(mask.sum()), "exp413_rmse": p, "exp507_rmse": c, "delta": c - p}
        )
    assignment = pd.read_csv(hidden_assignment_path, dtype={"well_id": str}).set_index("well_id")
    hidden_rows: list[dict[str, Any]] = []
    for scope, column in {
        "hidden_like_spatial": "verification_like_spatial_role",
        "hidden_like_typewell_purged": "verification_like_typewell_purged_role",
    }.items():
        mask = base["well"].astype(str).map(assignment[column]).eq("valid").to_numpy()
        p, c = rmse(truth[mask], parent[mask]), rmse(truth[mask], candidate[mask])
        hidden_rows.append(
            {"scope": scope, "rows": int(mask.sum()), "exp413_rmse": p, "exp507_rmse": c, "delta": c - p}
        )
    hidden_metrics = pd.DataFrame(hidden_rows)
    by_well_rows: list[dict[str, Any]] = []
    source = pd.DataFrame({"well": base["well"].astype(str), "truth": truth, "parent": parent, "candidate": candidate})
    for well, group in source.groupby("well", sort=True):
        p, c = rmse(group["truth"], group["parent"]), rmse(group["truth"], group["candidate"])
        by_well_rows.append({"well": well, "rows": len(group), "exp413_rmse": p, "exp507_rmse": c, "delta": c - p})
    by_well = pd.DataFrame(by_well_rows)
    pooled_parent, pooled_candidate = rmse(truth, parent), rmse(truth, candidate)
    gain = pooled_parent - pooled_candidate
    all_scopes = pd.concat([pd.DataFrame(scope_rows), hidden_metrics], ignore_index=True)
    delta = by_well["delta"]
    checks = {
        "technical_all_pass": bool(technical_checks) and all(technical_checks.values()),
        "gain_at_least_0p03": gain >= float(config["promotion_gate"]["required_gain_ft"]),
        "nonworse_folds_at_least_3": int((fold_metrics["delta_exp507_minus_exp413"] <= 0).sum())
        >= int(config["promotion_gate"]["required_nonworse_folds"]),
        "all_five_scopes_within_0p02": float(all_scopes["delta"].max())
        <= float(config["promotion_gate"]["maximum_scope_delta_rmse_ft"]),
    }
    gate = {
        "exp413_rmse": pooled_parent,
        "exp507_rmse": pooled_candidate,
        "gain_ft": gain,
        "nonworse_folds": int((fold_metrics["delta_exp507_minus_exp413"] <= 0).sum()),
        "maximum_scope_delta": float(all_scopes["delta"].max()),
        "checks": checks,
        "tail": {
            "delta_median": float(delta.median()),
            "delta_p90": float(delta.quantile(0.90)),
            "delta_p95": float(delta.quantile(0.95)),
            "delta_p99": float(delta.quantile(0.99)),
            "worst_well": str(by_well.loc[delta.idxmax(), "well"]),
            "worst_delta": float(delta.max()),
            "worsened_plus_1ft": int((delta > 1.0).sum()),
            "worsened_plus_3ft": int((delta > 3.0).sum()),
            "worsened_plus_5ft": int((delta > 5.0).sum()),
        },
        "passed": bool(all(checks.values())),
        "fail_action": config["promotion_gate"]["fail_action"],
        "pass_action": config["promotion_gate"]["pass_action"],
    }
    return gate, {
        "fold": fold_metrics,
        "scope": pd.DataFrame(scope_rows),
        "hidden": hidden_metrics,
        "by_well": by_well,
    }


def run_stage_d(config: Mapping[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    require_stage_authorization(config, "stage_d")
    static = validate_static_contract(config)
    _, rank_manifest = verify_rank_stage_n_root(config)
    exp413_spec = config["data"]["exp413"]
    stage_c_root = resolve_artifact_root(
        exp413_spec["stage_c_root_patterns"],
        marker="nested_compact_manifest.json",
        expected_marker_sha256=exp413_spec["stage_c_compact_manifest_sha256"],
    )
    stage_s_root = resolve_artifact_root(
        exp413_spec["stage_s_root_patterns"],
        marker="signed_compact_manifest.json",
        expected_marker_sha256=exp413_spec["stage_s_compact_manifest_sha256"],
    )
    stage_d_root = resolve_artifact_root(
        exp413_spec["stage_d_root_patterns"],
        marker="stage_d_model_manifest.json",
        expected_marker_sha256=exp413_spec["model_manifest_sha256"],
    )
    stage_c_evidence = verify_stage_c_artifact_root(
        stage_c_root,
        stage_c_verify_config(exp413_spec),
        verify_partition_sha256=True,
        expected_compact_feature_count=74,
        require_score_guard=False,
    )
    stage_s_evidence = verify_signed_stage_s_root(
        stage_s_root,
        stage_s_verify_config(exp413_spec),
        verify_partition_sha=True,
        verify_model_sha=True,
        require_score_gate=False,
    )
    expected_fold_sha = config["data"]["fold_contract"]["expected_manifest_sha256"]
    fold_paths = [stage_c_root / "nested_fold_manifest.csv", stage_s_root / "signed_nested_fold_manifest.csv"]
    if {sha256_file(path) for path in fold_paths} != {expected_fold_sha}:
        raise ValueError("exp413 C/S fold manifests differ")

    parent_path = resolve_file(config["data"]["parent_exp413_config"], sha_key="sha256")
    parent = load_yaml(parent_path)
    data_root = competition_data_root(config)
    raw_train_dir = data_root / "train"
    frozen_prediction_path = resolve_by_patterns(
        parent["data"]["exp404_scale5_train_prediction"]["patterns"],
        search_roots(),
        marker_sha256=parent["data"]["exp404_scale5_train_prediction"]["expected_raw_sha256"],
    )
    def parent_file(section: Mapping[str, Any], patterns: str, sha: str) -> Path:
        return resolve_by_patterns(section[patterns], search_roots(), marker_sha256=section[sha])
    exp218_spec, exp145_spec = parent["data"]["exp218_source"], parent["data"]["exp145_source"]
    exp218_source = parent_file(exp218_spec, "script_patterns", "script_sha256")
    exp218_config_path = parent_file(exp218_spec, "config_patterns", "config_sha256")
    exp145_source = parent_file(exp145_spec, "script_patterns", "script_sha256")
    exp145_config = parent_file(exp145_spec, "config_patterns", "config_sha256")
    multiobs_source = parent_file(exp145_spec, "multiobs_script_patterns", "multiobs_script_sha256")
    exp099 = parent_file(parent["data"]["exp099_train_feature_cache"], "patterns", "expected_raw_sha256")
    exp111_schema = parent_file(parent["data"]["exp111_saved_models"], "schema_patterns", "schema_sha256")
    exp111_manifest = parent_file(parent["data"]["exp111_saved_models"], "manifest_patterns", "manifest_sha256")
    clean_allowlist = resolve_file(parent["data"]["clean_base_allowlist"], sha_key="sha256")
    hidden_assignment = resolve_file(parent["data"]["hidden_like_assignment"], sha_key="sha256")
    base, base_features, base_evidence, exp218, exp218_config = build_replacement_clean273_surface(
        config=parent,
        frozen_prediction_path=frozen_prediction_path,
        exp218_source_path=exp218_source,
        exp218_config_path=exp218_config_path,
        exp099_source_path=exp099,
        exp145_source_path=exp145_source,
        exp145_config_path=exp145_config,
        multiobs_source_path=multiobs_source,
        exp111_schema_path=exp111_schema,
        exp111_manifest_path=exp111_manifest,
        clean_allowlist_path=clean_allowlist,
        raw_train_dir=raw_train_dir,
    )
    required = list(dict.fromkeys(["id", "well", "target", "last_known_tvt", "md_since", *base_features]))
    base = base.loc[:, ~base.columns.duplicated()].loc[:, required].copy()
    control, compact_features, signed_features, control_evidence = load_saved_exp413_control(
        stage_d_root, base, base_features, config
    )
    if stage_c_evidence["compact_features"] != compact_features or stage_s_evidence["features"] != signed_features:
        raise ValueError("exp413 compact feature order differs from final370 manifest")
    final_features = [*base_features, *compact_features, *signed_features, *RANK_COMPACT_FEATURES]
    if len(final_features) != len(set(final_features)):
        raise ValueError("final415 schema is not unique")
    if len(final_features) != 415:
        raise ValueError("final415 feature count changed")

    output_dir = KAGGLE_WORKING_ROOT / "artifacts" if is_kaggle_runtime() else ROOT / "experiments" / EXPERIMENT_NAME / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "final415_feature_schema.json", {"features": final_features, "sha256": sha256_json(final_features)})
    mode_name = str(config["model"]["gpu_mode"])
    mode = exp218_config["model"]["training"]["modes"][mode_name]
    if not bool(mode.get("use_gpu", False)):
        raise ValueError("Stage D must inherit exp413 GPU mode")
    params_family = exp218.apply_mode_overrides(exp218.exp063_lgb_config_family(fast=False), mode)
    config_indices = [int(item) for item in config["model"]["lightgbm_config_indices"]]
    params_family = [params_family[index] for index in config_indices]
    target = base["target"].to_numpy(np.float32)
    anchor = base["last_known_tvt"].to_numpy(np.float32)
    truth = anchor + target
    n_rows = len(base)
    oof_residual = [np.full(n_rows, np.nan, dtype=np.float32) for _ in config_indices]
    model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    model_dir = output_dir / "stage_d_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    base_index = pd.Index(base["id"].astype(str))

    for outer_fold in range(5):
        compact_train, compact_valid = load_stage_d_compact_fold(
            stage_c_root=stage_c_root,
            stage_c_evidence=stage_c_evidence,
            downstream_outer_fold=outer_fold,
        )
        signed_train, signed_valid = load_signed_compact_fold(
            stage_s_evidence=stage_s_evidence, downstream_outer_fold=outer_fold
        )
        rank_train, rank_valid = load_rank_compact_fold(rank_manifest, outer_fold)
        fold_predictions: list[np.ndarray] = []
        matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for role, compact, signed, rank in (
            ("train", compact_train, signed_train, rank_train),
            ("valid", compact_valid, signed_valid, rank_valid),
        ):
            if not compact[KEY_COLUMNS].reset_index(drop=True).equals(signed[KEY_COLUMNS].reset_index(drop=True)):
                raise ValueError(f"exp413 C/S key mismatch: fold={outer_fold} role={role}")
            rank = align_rank_to_parent(compact, rank, outer_fold=outer_fold, role=role)
            positions = base_index.get_indexer(compact["id"].astype(str))
            if np.any(positions < 0) or len(np.unique(positions)) != len(compact):
                raise ValueError("final415 base join failed")
            matrix = assemble_final415(
                base=base,
                positions=positions,
                compact=compact,
                signed=signed,
                rank=rank,
                base_features=base_features,
                compact_features=compact_features,
                signed_features=signed_features,
                chunk_columns=int(config["model"]["matrix_copy_chunk_columns"]),
            )
            matrices[role] = (positions, matrix)
            matrix_rows.append(
                {"outer_fold": outer_fold, "role": role, "rows": len(matrix), "sha256": matrix_sha256(matrix, final_features)}
            )
        train_positions, x_train = matrices["train"]
        valid_positions, x_valid = matrices["valid"]
        if np.intersect1d(train_positions, valid_positions).size:
            raise ValueError(f"Stage D train/valid rows overlap: fold={outer_fold}")
        if len(np.unique(np.concatenate([train_positions, valid_positions]))) != n_rows:
            raise ValueError(f"Stage D fold does not cover every base row: fold={outer_fold}")
        if not control.iloc[valid_positions]["outer_fold"].eq(outer_fold).all():
            raise ValueError(f"Stage D valid fold differs from saved exp413: fold={outer_fold}")
        x_train_frame = pd.DataFrame(x_train, columns=final_features, copy=False)
        x_valid_frame = pd.DataFrame(x_valid, columns=final_features, copy=False)
        for array_index, (config_index, params) in enumerate(zip(config_indices, params_family, strict=True)):
            model = LGBMRegressor(**params)
            model.fit(
                x_train_frame,
                target[train_positions],
                eval_set=[(x_valid_frame, target[valid_positions])],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(int(config["model"]["early_stopping_rounds"]), verbose=False),
                    log_evaluation(int(config["model"]["log_evaluation_period"])),
                ],
            )
            best_iteration = int(model.best_iteration_ or params["n_estimators"])
            residual = model.predict(x_valid_frame, num_iteration=best_iteration).astype(np.float32)
            oof_residual[array_index][valid_positions] = residual
            fold_predictions.append(anchor[valid_positions] + residual)
            model_path = model_dir / f"rank_compact_addonly_lgb{config_index}_outer{outer_fold}.txt"
            model.booster_.save_model(str(model_path), num_iteration=best_iteration)
            model_rows.append(
                {"outer_fold": outer_fold, "config_index": config_index, "best_iteration": best_iteration, "model_sha256": sha256_file(model_path), "rmse_tvt": rmse(truth[valid_positions], fold_predictions[-1])}
            )
            gain = model.booster_.feature_importance(importance_type="gain")
            for position, (feature, value) in enumerate(
                zip(final_features, gain, strict=True)
            ):
                group = (
                    "clean273"
                    if position < 273
                    else "compact74"
                    if position < 347
                    else "signed23"
                    if position < 370
                    else "rank45"
                )
                importance_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "config_index": config_index,
                        "feature": feature,
                        "feature_group": group,
                        "importance_gain": float(value),
                    }
                )
        del compact_train, compact_valid, signed_train, signed_valid, rank_train, rank_valid, x_train, x_valid, x_train_frame, x_valid_frame
        gc.collect()

    if len(model_rows) != 15 or any(not np.isfinite(item).all() for item in oof_residual):
        raise RuntimeError("Stage D 15-model OOF is incomplete")
    mean_prediction = anchor + np.mean(np.vstack(oof_residual), axis=0).astype(np.float32)
    technical = {
        "stage_n_technical_pass": all(rank_manifest["technical_gate"].values()),
        "clean273": len(base_features) == 273,
        "compact74": len(compact_features) == 74,
        "signed23": len(signed_features) == 23,
        "rank45": len(RANK_COMPACT_FEATURES) == 45,
        "final415_unique": len(final_features) == len(set(final_features)) == 415,
        "model_count_15": len(model_rows) == 15,
        "control_retraining_zero": control_evidence["models_retrained"] == 0,
    }
    gate, tables = evaluate_stage_d(
        config=config,
        base=base,
        control=control,
        prediction=mean_prediction,
        hidden_assignment_path=hidden_assignment,
        technical_checks=technical,
    )
    prediction_frame = base[["id", "well", "md_since", "last_known_tvt", "target"]].copy()
    prediction_frame["outer_fold"] = control["outer_fold"].to_numpy(np.int8)
    prediction_frame["actual_tvt"] = truth
    prediction_frame["saved_exp413__pred_tvt"] = control[exp413_spec["stage_d_prediction_column"]].to_numpy(np.float32)
    prediction_frame["exp507__lgb_mean__pred_tvt"] = mean_prediction
    paths = {
        "oof": output_dir / "stage_d_oof_predictions.parquet",
        "fold": output_dir / "stage_d_fold_metrics.csv",
        "scope": output_dir / "stage_d_scope_metrics.csv",
        "hidden": output_dir / "stage_d_hidden_like_metrics.csv",
        "by_well": output_dir / "stage_d_by_well.csv",
        "importance": output_dir / "stage_d_feature_importance.csv",
        "model_manifest": output_dir / "stage_d_model_manifest.json",
        "metrics": output_dir / "stage_d_metrics.json",
    }
    prediction_frame.to_parquet(paths["oof"], index=False)
    for name in ("fold", "scope", "hidden", "by_well"):
        tables[name].to_csv(paths[name], index=False)
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(paths["importance"], index=False)
    write_json(paths["model_manifest"], {"model_count": 15, "models": model_rows, "feature_count": 415, "feature_schema_sha256": sha256_json(final_features)})
    metrics = {
        "status": "train_complete_gate_passed" if gate["passed"] else "train_complete_gate_failed_closed",
        "rows": n_rows,
        "wells": int(base["well"].nunique()),
        "feature_counts": {"clean": 273, "compact": 74, "signed": 23, "rank": 45, "final": 415},
        "model_count": 15,
        "cost_contract": static["cost_contract"],
        "base_evidence": base_evidence,
        "matrix_partitions": matrix_rows,
        "control": control_evidence,
        "primary_gate": gate,
        "rank_feature_readout": {
            "used_rank_features_any_model": int(
                importance[
                    importance["feature_group"].eq("rank45")
                    & importance["importance_gain"].gt(0.0)
                ]["feature"].nunique()
            ),
            "total_rank_features": 45,
            "rank_gain_importance_sum": float(
                importance.loc[
                    importance["feature_group"].eq("rank45"), "importance_gain"
                ].sum()
            ),
        },
        "inference_executed": False,
        "submission_generated": False,
    }
    write_json(paths["metrics"], metrics)
    artifact_sha = {name: sha256_file(path) for name, path in paths.items()}
    write_json(
        output_dir / "reproducibility_manifest.json",
        {
            "status": "stage_d_reproducibility_recorded",
            "deterministic_anchor": False,
            "input_stage_n_manifest_sha256": config["data"]["stage_n_source"][
                "expected_manifest_sha256"
            ],
            "final415_schema_sha256": sha256_json(final_features),
            "matrix_partitions": matrix_rows,
            "model_count": 15,
            "control_retraining_boosters": 0,
            "artifact_sha256": artifact_sha,
            "inference_executed": False,
            "submission_generated": False,
        },
    )
    mean_importance = (
        importance.groupby(["feature", "feature_group"], as_index=False)[
            "importance_gain"
        ]
        .mean()
        .sort_values("importance_gain", ascending=False)
    )
    display(tables["fold"])
    display(tables["scope"])
    display(tables["hidden"])
    display(tables["by_well"].sort_values("delta", ascending=False).head(80))
    display(mean_importance.head(100))
    top = mean_importance.head(30).sort_values("importance_gain")
    if len(top):
        ax = top.plot.barh(x="feature", y="importance_gain", figsize=(11, 11), legend=False, title="exp507 final415 mean gain importance")
        ax.set_xlabel("mean gain across 15 treatment models")
        plt.tight_layout()
        plt.savefig(output_dir / "stage_d_feature_importance_top30.png", dpi=140)
        plt.show()
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    if gate["passed"]:
        print("Stage D PASS. Stop; inference implementation/run requires separate approval.")
    else:
        print(config["promotion_gate"]["fail_action"])
    return metrics


# %% [markdown]
# ## 9. Setup, reproducibility evidence, and fixed stop
#
# Import/test は実行しない。Kaggle package でも selected_stage と個別 run approval の
# 両方が一致しなければ fail closed する。Stage N は 20 CPU boosters、Stage D は
# 15 GPU boosters、保存 control / exp504 outer model の再学習は常に 0。

# %%
CONFIG = load_yaml(config_path())
STATIC_CONTRACT = validate_static_contract(CONFIG)

if os.environ.get("EXP507_IMPORT_ONLY", "0") != "1":
    selected_stage = str(CONFIG["runtime"]["selected_stage"])
    print("Experiment:", EXPERIMENT_NAME)
    print("Route:", CONFIG["experiment"]["route"])
    print("Parent:", CONFIG["lineage"]["parent"])
    print("Selected stage:", selected_stage)
    print("Rank compact features:", len(RANK_COMPACT_FEATURES))
    print("Cost contract:", STATIC_CONTRACT["cost_contract"])
    if selected_stage == "stage_n":
        run_stage_n(CONFIG)
    elif selected_stage == "stage_d":
        run_stage_d(CONFIG)
    else:
        raise ValueError(f"unknown exp507 selected_stage: {selected_stage}")
