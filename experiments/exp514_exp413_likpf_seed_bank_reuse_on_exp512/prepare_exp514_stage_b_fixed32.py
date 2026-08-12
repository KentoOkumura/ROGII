from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments/exp514_exp413_likpf_seed_bank_reuse_on_exp512"
STAGE_A_SOURCE = (
    EXP_DIR / "exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_a_fixed32.py"
)
FULL_CANDIDATE_SOURCE = (
    EXP_DIR
    / "exp514_exp413_likpf_seed_bank_reuse_on_exp512_compact_selfcontained_inference.py"
)
OUTPUT_SOURCE = (
    EXP_DIR
    / "exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_b_fixed32.py"
)

EXPECTED_STAGE_A_SHA256 = (
    "89129ad85c129145e635633741e08ff5e058a365c344a4a4bdbdc77190ab3873"
)
EXPECTED_FULL_CANDIDATE_SHA256 = (
    "8b1616dd289672339bfba82050e25b1c678a00dcb89b17ad6de60892c4171634"
)
EXPECTED_SELECTION_SHA256 = (
    "86157959105b896271f53c841b27f5f7246db6c4f199773b0151ed75d36ae58b"
)
EXPECTED_HIDDEN_ASSIGNMENT_SHA256 = (
    "5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_slice(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[start_index:end_index].rstrip() + "\n"


def extract_functions(source: str, names: tuple[str, ...]) -> str:
    tree = ast.parse(source)
    found: dict[str, str] = {}
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            start_line = min(
                [node.lineno]
                + [decorator.lineno for decorator in getattr(node, "decorator_list", [])]
            )
            found[node.name] = "".join(lines[start_line - 1 : node.end_lineno]).rstrip()
    missing = sorted(set(names) - set(found))
    if missing:
        raise RuntimeError(f"missing source functions: {missing}")
    return "\n\n\n".join(found[name] for name in names) + "\n"


HEADER = r'''# ---
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
# # exp514 Stage B fixed32 paired scientific screening
#
# This notebook reuses the exact target-free Stage A 32-well selection. It
# compares the frozen parent SP45 PCG64 likelihood-PF bank with the shared
# exp413 stable-seed bank, while keeping the selector, common Beam path, hold,
# and branch hedge fixed. Predictions are written and content-hashed before
# suffix TVT or reporting folds are joined. This is a small screening audit,
# not a 200-well generalization proof, hidden inference, or submission run.

# %% [markdown]
# ## Contents
# 1. Imports, source identity, configuration, and input checks
# 2. Shared exp413 likelihood-PF producer and Stage A selection
# 3. Frozen parent SP45 PF, Beam, selector, and branch hedge
# 4. Target-free paired prediction generation and freeze
# 5. Post-freeze truth join, fixed-scope metrics, and all-AND gate
# 6. Reproducibility report and non-submission outputs

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from joblib import Parallel, delayed
from numba import njit
from scipy.signal import savgol_filter

EXPERIMENT_NAME = "exp514_exp413_likpf_seed_bank_reuse_on_exp512"
STAGE_B_SOURCE_FILENAME = (
    "exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_b_fixed32.py"
)
STAGE_A_SOURCE_SHA256 = "__STAGE_A_SHA256__"
FULL_CANDIDATE_SOURCE_SHA256 = "__FULL_CANDIDATE_SHA256__"
EXP073_REPLAY_SOURCE_SHA256 = (
    "4af212a8a1c83e36cdcc0bc912942a62df1fbc94ca67fd75789171afaa1a647e"
)
STAGE_B_GENERATOR_SHA256 = "__GENERATOR_SHA256__"
EXPECTED_SELECTION_SHA256 = "__SELECTION_SHA256__"
HIDDEN_ASSIGNMENT_SHA256 = "__HIDDEN_ASSIGNMENT_SHA256__"
HIDDEN_ASSIGNMENT_PATH = Path(
    "inputs/exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv"
)
EXPECTED_STAGE_A_WELLS = (
    "9314ff13", "e2fc7745", "c472c0b5", "2fa01aa6",
    "54753541", "e25f1537", "ff0aea78", "c6c96179",
    "9283ae69", "35b3ef6a", "e14d641e", "a9c9b150",
    "5693dde2", "529e88ca", "a76db406", "272abef3",
    "2c0c4a4e", "ae069086", "47222616", "d63196fe",
    "230eaaa3", "1aaf1da0", "44441e54", "137d1e44",
    "ffefef30", "e5c92e59", "16e4a047", "ce55ba43",
    "c36625df", "a85e4bc3", "f88ddb26", "8ac2f237",
)

STAGE_B_WELLS = 32
STAGE_B_N_JOBS = 4
STAGE_B_PARTICLES = 500
STAGE_B_SEEDS = 128
STAGE_B_VARIANTS = (
    "legacy_sp45_control",
    "shared_exp413_bank_candidate",
)
STAGE_B_REPORTING_FOLDS = 5
HIGH_MISSING_FRACTION_THRESHOLD = 0.30
SUFFIX_1000_FT = 1000.0
POOLED_MAX_REGRESSION_FT = 0.02
FOLD_MAX_REGRESSION_FT = 0.02
REQUIRED_NONWORSE_FOLDS = 4
FIXED_SCOPE_MAX_REGRESSION_FT = 0.05
BY_WELL_DELTA_P95_MAX_FT = 0.25
WORST_WELL_DELTA_MAX_FT = 5.0
FIXED_SCOPES = (
    "raw_gr_observed",
    "raw_gr_missing",
    "high_missing_fraction",
    "suffix_1000_plus",
    "hidden_like_spatial",
    "hidden_like_typewell_purged",
)

# Frozen exp512 profile and selector settings.
SUBMISSION_PROFILE = "vp_balanced_modelpkg_005"
SELECTOR_PF_RETURN_STD = False
RUN_BIMODAL_DETECTOR = False
RUN_BIMODAL_SELECTOR_HEDGE = False

# Frozen post-selector branch hedge from exp512.
BRANCH_HEDGE_STRENGTH = 0.60
BRANCH_HEDGE_MIN_MASS = 0.25
BRANCH_HEDGE_SEPARATION_LOW = 4.0
BRANCH_HEDGE_SEPARATION_HIGH = 40.0
BRANCH_HEDGE_CAP_FT = 2.0
BRANCH_HEDGE_SKIP_EXISTING = False


def resolve_competition_data_root() -> Path:
    candidates = (
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    )
    matches = [
        root.resolve()
        for root in candidates
        if (root / "train").is_dir()
        and (root / "test").is_dir()
        and (root / "sample_submission.csv").is_file()
    ]
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise RuntimeError(f"expected one competition data root, got {unique}")
    return unique[0]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_content_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def runtime_source_sha256() -> str:
    path = Path.cwd() / STAGE_B_SOURCE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Stage B bootstrap source missing: {path}")
    return file_sha256(path)


COMPETITION_DATA_ROOT = resolve_competition_data_root()
WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = WORKING_DIR / "exp514_stage_b_fixed32_report.json"
METRICS_PATH = WORKING_DIR / "metrics.json"
FROZEN_PREDICTION_PATH = (
    WORKING_DIR / "exp514_stage_b_fixed32_predictions_frozen.csv.gz"
)
SCORED_PREDICTION_PATH = (
    WORKING_DIR / "exp514_stage_b_fixed32_predictions_scored.csv.gz"
)

print("experiment:", EXPERIMENT_NAME)
print("stage: fixed32 paired scientific screening")
print("competition data root:", COMPETITION_DATA_ROOT)
print("Stage A source SHA256:", STAGE_A_SOURCE_SHA256)
print("full candidate source SHA256:", FULL_CANDIDATE_SOURCE_SHA256)
print("Stage B generator SHA256:", STAGE_B_GENERATOR_SHA256)
print("Stage A selection SHA256:", EXPECTED_SELECTION_SHA256)
print("execution inventory:", {
    "active_scientific_variants": 2,
    "legacy_sp45_well_bank_generations": 32,
    "shared_candidate_well_bank_generations": 32,
    "total_well_bank_generations": 64,
    "particles_per_bank": 500,
    "seeds_per_bank": 128,
    "lightgbm_configs": 0,
    "trained_folds": 0,
    "new_boosters": 0,
    "parent_control_retraining": 0,
})
'''


SELECTOR_CONSTANTS = r'''# %% [markdown]
# ## 3. Frozen parent SP45 PF, Beam, selector, and branch hedge

# %%
SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
SELECTOR_BIN_VARIANTS = {
    0: "pf_scale_5_hold_0.2",
    1: "pf_scale_3_hold_0.15",
    2: "pf_scale_12_beam_0.2_hold_0.15",
    3: "pf_scale_5_hold_0.15",
    4: "pf_scale_5_beam_0.05_hold_0.05",
    5: "pf_scale_12_beam_0.2_hold_0.05",
}
SELECTOR_GLOBAL_VARIANT = "pf_scale_8_hold_0.2"
SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2),
    (10, 8.0, 64.0, 2),
    (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4),
    (25, 6.0, 50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2),
    (10, 50.0, 400.0, 0),
]


def _bimodal_selector_weight(base, beam, hw=None, tw=None):
    del base, beam, hw, tw
    raise RuntimeError("bimodal selector is disabled in frozen exp512 profile")
'''


ORCHESTRATION = r'''# %%
def apply_branch_hedge(prediction, branch_stats):
    prediction = np.asarray(prediction, dtype=np.float64).copy()
    info = {
        "reason": "not_qualified",
        "shift": 0.0,
        "moved_rows": 0,
    }
    try:
        center_low = float(branch_stats["center_low"])
        center_high = float(branch_stats["center_high"])
        mass_low = float(branch_stats["mass_low"])
        mass_high = float(branch_stats["mass_high"])
        weighted_center = float(branch_stats["weighted_center"])
        separation = abs(center_high - center_low)
        minor_mass = min(mass_low, mass_high)
        info.update(
            center_low=center_low,
            center_high=center_high,
            mass_low=mass_low,
            mass_high=mass_high,
            separation=separation,
            weighted_center=weighted_center,
        )
        if BRANCH_HEDGE_SKIP_EXISTING:
            raise RuntimeError("Stage B does not support an existing-route skip set")
        if minor_mass < BRANCH_HEDGE_MIN_MASS:
            info["reason"] = "skip_minor_mass"
        elif not (
            BRANCH_HEDGE_SEPARATION_LOW
            <= separation
            <= BRANCH_HEDGE_SEPARATION_HIGH
        ):
            info["reason"] = "skip_separation"
        else:
            target = 0.5 * (center_low + center_high)
            shift = float(
                np.clip(
                    BRANCH_HEDGE_STRENGTH * (target - weighted_center),
                    -BRANCH_HEDGE_CAP_FT,
                    BRANCH_HEDGE_CAP_FT,
                )
            )
            evaluation_index = np.asarray(
                branch_stats.get("eval_rows", []), dtype=np.int64
            )
            if (
                abs(shift) >= 0.01
                and len(evaluation_index) > 0
                and int(evaluation_index.max()) < len(prediction)
            ):
                prediction[evaluation_index] += shift
                info.update(
                    reason="applied",
                    shift=shift,
                    moved_rows=int(len(evaluation_index)),
                )
            else:
                info["reason"] = "skip_zero_or_missing_rows"
    except Exception as exc:
        raise RuntimeError(f"branch hedge failed closed: {exc!r}") from exc
    if not np.isfinite(prediction).all():
        raise RuntimeError("branch hedge produced non-finite predictions")
    return prediction, info


def load_stage_b_raw_well(well, split="train"):
    split_root = COMPETITION_DATA_ROOT / str(split)
    horizontal = pd.read_csv(
        split_root / f"{well}__horizontal_well.csv",
        usecols=["MD", "Z", "GR", "TVT_input"],
    )
    typewell = pd.read_csv(
        split_root / f"{well}__typewell.csv",
        usecols=lambda name: name in {"TVT", "GR"},
    )
    if list(horizontal.columns) != ["MD", "Z", "GR", "TVT_input"]:
        raise ValueError(f"truth-free horizontal schema mismatch for {well}")
    if {"TVT", "GR"} - set(typewell.columns):
        raise ValueError(f"typewell schema mismatch for {well}")
    return horizontal, typewell


def target_free_input_sha256(wells):
    records = []
    for well in wells:
        horizontal, typewell = load_stage_b_raw_well(well, "train")
        records.append(
            {
                "well": str(well),
                "horizontal_sha256": frame_content_sha256(horizontal),
                "typewell_sha256": frame_content_sha256(typewell),
            }
        )
    return _shared_json_sha(records), records


def load_hidden_like_roles(wells):
    if not HIDDEN_ASSIGNMENT_PATH.is_file():
        raise FileNotFoundError(
            f"hidden-like assignment bootstrap missing: {HIDDEN_ASSIGNMENT_PATH}"
        )
    observed_sha = file_sha256(HIDDEN_ASSIGNMENT_PATH)
    if observed_sha != HIDDEN_ASSIGNMENT_SHA256:
        raise RuntimeError(
            "hidden-like assignment SHA drift: "
            f"expected {HIDDEN_ASSIGNMENT_SHA256}, got {observed_sha}"
        )
    frame = pd.read_csv(HIDDEN_ASSIGNMENT_PATH)
    required = {
        "well_id",
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"hidden-like assignment columns missing: {sorted(missing)}")
    frame["well_id"] = frame["well_id"].astype(str)
    selected = frame[frame["well_id"].isin(set(map(str, wells)))].copy()
    if selected["well_id"].duplicated().any() or set(selected["well_id"]) != set(wells):
        raise ValueError("hidden-like role coverage is not one-to-one for fixed32 wells")
    return selected.set_index("well_id"), observed_sha


def run_legacy_control_one_well(well):
    started = time.time()
    horizontal, typewell = load_stage_b_raw_well(well, "train")
    branch_stats = {}
    pf_by_scale = run_pf_lik_ensemble_scales(
        horizontal,
        typewell,
        scales=SELECTOR_SCALES,
        n_particles=STAGE_B_PARTICLES,
        n_seeds=STAGE_B_SEEDS,
        branch_stats=branch_stats,
    )
    beam = run_beam_ensemble(horizontal, typewell)
    required = {f"pf_scale_{scale:g}" for scale in SELECTOR_SCALES} | {"pf_mean"}
    if set(pf_by_scale) != required:
        raise RuntimeError(f"legacy SP45 aggregate schema mismatch for {well}")
    if not branch_stats or branch_stats.get("error"):
        raise RuntimeError(f"legacy SP45 branch summary failed for {well}: {branch_stats}")
    return {
        "well": str(well),
        "pf_by_scale": pf_by_scale,
        "branch_stats": branch_stats,
        "beam": beam,
        "elapsed_seconds": round(time.time() - started, 6),
    }


def stage_b_reporting_fold_map(wells):
    ordered = sorted(
        map(str, wells),
        key=lambda well: hashlib.sha256(
            f"exp514::stage_b_fold::{well}".encode("utf-8")
        ).hexdigest(),
    )
    records = [
        {"well": well, "fold": int(rank % STAGE_B_REPORTING_FOLDS)}
        for rank, well in enumerate(ordered)
    ]
    fold_counts = pd.Series([record["fold"] for record in records]).value_counts()
    if set(fold_counts.index) != set(range(STAGE_B_REPORTING_FOLDS)):
        raise RuntimeError("Stage B reporting fold assignment contains an empty fold")
    return (
        {record["well"]: record["fold"] for record in records},
        records,
        _shared_json_sha(records),
    )


def freeze_prediction_frame(frame):
    if "target_tvt" in frame.columns or "TVT" in frame.columns:
        raise RuntimeError("truth leaked into the Stage B prediction freeze frame")
    if frame["id"].duplicated().any() or not np.isfinite(
        frame[[
            "control_before_branch_tvt",
            "candidate_before_branch_tvt",
            "control_tvt",
            "candidate_tvt",
        ]].to_numpy(np.float64)
    ).all():
        raise RuntimeError("Stage B prediction freeze frame failed ID/finite checks")
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    content_sha = hashlib.sha256(payload).hexdigest()
    FROZEN_PREDICTION_PATH.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    return content_sha, hashlib.sha256(FROZEN_PREDICTION_PATH.read_bytes()).hexdigest()


PREDICTIONS_FROZEN = False
TRUTH_READ_COUNT = 0


def read_stage_b_truth_after_freeze(wells):
    global TRUTH_READ_COUNT
    if not PREDICTIONS_FROZEN or not FROZEN_PREDICTION_PATH.is_file():
        raise RuntimeError("Stage B truth read attempted before prediction freeze")
    records = []
    split_root = COMPETITION_DATA_ROOT / "train"
    for well in wells:
        truth = pd.read_csv(
            split_root / f"{well}__horizontal_well.csv",
            usecols=["TVT"],
        )
        TRUTH_READ_COUNT += 1
        for row_index, value in enumerate(
            pd.to_numeric(truth["TVT"], errors="coerce").to_numpy(np.float64)
        ):
            records.append(
                {"id": f"{well}_{row_index}", "target_tvt": float(value)}
            )
    return pd.DataFrame(records)


def metric_record(frame, *, label):
    if len(frame) == 0:
        raise RuntimeError(f"Stage B metric scope is empty: {label}")
    target = frame["target_tvt"].to_numpy(np.float64)
    control = frame["control_tvt"].to_numpy(np.float64)
    candidate = frame["candidate_tvt"].to_numpy(np.float64)
    if not np.isfinite(np.column_stack([target, control, candidate])).all():
        raise RuntimeError(f"Stage B metric scope is non-finite: {label}")
    control_rmse = float(np.sqrt(np.mean((control - target) ** 2)))
    candidate_rmse = float(np.sqrt(np.mean((candidate - target) ** 2)))
    return {
        "label": str(label),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "control_rmse": control_rmse,
        "candidate_rmse": candidate_rmse,
        "delta_candidate_minus_control": candidate_rmse - control_rmse,
    }


def metric_bundle(scored, *, control_column, candidate_column):
    # Do not rename onto the existing post-branch columns.  The pre-branch
    # scoring call receives a frame that already contains ``control_tvt`` and
    # ``candidate_tvt``; pandas permits duplicate labels and would then return
    # an (n, 2) frame instead of the one-dimensional metric vector.
    renamed = scored.copy()
    renamed["control_tvt"] = scored[control_column].to_numpy()
    renamed["candidate_tvt"] = scored[candidate_column].to_numpy()
    overall = metric_record(renamed, label="pooled")
    folds = [
        metric_record(renamed[renamed["fold"] == fold], label=f"fold_{fold}")
        for fold in range(STAGE_B_REPORTING_FOLDS)
    ]
    scope_masks = {
        "raw_gr_observed": renamed["raw_gr_observed"].astype(bool),
        "raw_gr_missing": ~renamed["raw_gr_observed"].astype(bool),
        "high_missing_fraction": (
            renamed["well_raw_gr_missing_fraction"]
            >= HIGH_MISSING_FRACTION_THRESHOLD
        ),
        "suffix_1000_plus": renamed["md_since_last_known"] >= SUFFIX_1000_FT,
        "hidden_like_spatial": renamed["hidden_like_spatial"].astype(bool),
        "hidden_like_typewell_purged": renamed[
            "hidden_like_typewell_purged"
        ].astype(bool),
    }
    if tuple(scope_masks) != FIXED_SCOPES:
        raise RuntimeError("Stage B fixed scope order/schema drifted")
    scopes = [
        metric_record(renamed[mask], label=name)
        for name, mask in scope_masks.items()
    ]
    by_well = [
        metric_record(group, label=str(well))
        for well, group in renamed.groupby("well", sort=True)
    ]
    deltas = np.asarray(
        [record["delta_candidate_minus_control"] for record in by_well],
        dtype=np.float64,
    )
    return {
        "overall": overall,
        "folds": folds,
        "fixed_scopes": scopes,
        "by_well": by_well,
        "by_well_delta_p95": float(np.quantile(deltas, 0.95)),
        "worst_well_delta": float(np.max(deltas)),
    }


# %% [markdown]
# ## 4. Target-free paired prediction generation and freeze

# %%
STAGE_B_STARTED = time.time()
wells, selection, selection_sha = select_shared_likpf_stage_a_wells(
    COMPETITION_DATA_ROOT,
    split="train",
    count=STAGE_B_WELLS,
)
if selection_sha != EXPECTED_SELECTION_SHA256:
    raise RuntimeError(
        f"Stage A selection SHA drift: expected {EXPECTED_SELECTION_SHA256}, got {selection_sha}"
    )
if tuple(wells) != EXPECTED_STAGE_A_WELLS:
    raise RuntimeError("Stage B fixed32 well order differs from Stage A evidence")
selection_by_well = {str(record["well"]): record for record in selection}

hidden_roles, hidden_assignment_sha = load_hidden_like_roles(wells)
target_free_input_sha, target_free_input_records = target_free_input_sha256(wells)
stage_b_source_sha = runtime_source_sha256()

shared_started = time.time()
shared_bank, shared_parallel_report = materialize_shared_likpf_bank(
    wells,
    "train",
    load_stage_b_raw_well,
    n_jobs=STAGE_B_N_JOBS,
    particles=STAGE_B_PARTICLES,
    seeds=STAGE_B_SEEDS,
)
shared_seconds = time.time() - shared_started

legacy_started = time.time()
legacy_records = Parallel(
    n_jobs=STAGE_B_N_JOBS,
    backend="threading",
)(delayed(run_legacy_control_one_well)(well) for well in wells)
legacy_seconds = time.time() - legacy_started
legacy_bank = {record["well"]: record for record in legacy_records}
if set(legacy_bank) != set(wells):
    raise RuntimeError("legacy SP45 bank well coverage mismatch")

prediction_records = []
control_branch_records = []
candidate_branch_records = []
for well in wells:
    horizontal, typewell = load_stage_b_raw_well(well, "train")
    evaluation_mask = horizontal["TVT_input"].isna().to_numpy()
    evaluation_index = np.flatnonzero(evaluation_mask)
    if not np.array_equal(
        evaluation_index,
        np.asarray(shared_bank[well]["evaluation_index"], dtype=np.int64),
    ):
        raise RuntimeError(f"shared evaluation row order mismatch for {well}")
    known = horizontal.loc[~evaluation_mask]
    if len(known) == 0:
        raise RuntimeError(f"Stage B well has no known prefix: {well}")
    last_known_tvt = float(known.iloc[-1]["TVT_input"])
    last_known_md = float(known.iloc[-1]["MD"])
    selector_code, selector_variant, n_eval, z_span = selector_well_code(horizontal)

    legacy = legacy_bank[well]
    control_before_branch, control_selector_info = apply_selector_variant(
        selector_variant,
        legacy["pf_by_scale"],
        legacy["beam"],
        last_known_tvt,
        hw=horizontal,
        tw=typewell,
        return_info=True,
    )
    candidate_pf_by_scale, candidate_branch_stats = shared_likpf_sp45_adapter(
        shared_bank[well]
    )
    candidate_before_branch, candidate_selector_info = apply_selector_variant(
        selector_variant,
        candidate_pf_by_scale,
        legacy["beam"],
        last_known_tvt,
        hw=horizontal,
        tw=typewell,
        return_info=True,
    )
    if control_selector_info["bimodal_active"] or candidate_selector_info["bimodal_active"]:
        raise RuntimeError("frozen exp512 profile unexpectedly activated bimodal selector")
    control_after_branch, control_branch_info = apply_branch_hedge(
        control_before_branch, legacy["branch_stats"]
    )
    candidate_after_branch, candidate_branch_info = apply_branch_hedge(
        candidate_before_branch, candidate_branch_stats
    )
    control_branch_records.append({"well": well, **control_branch_info})
    candidate_branch_records.append({"well": well, **candidate_branch_info})

    spatial_valid = (
        str(hidden_roles.loc[well, "verification_like_spatial_role"]) == "valid"
    )
    typewell_valid = (
        str(hidden_roles.loc[well, "verification_like_typewell_purged_role"])
        == "valid"
    )
    missing_fraction = float(
        selection_by_well[well]["raw_gr_missing_fraction"]
    )
    for row_index in evaluation_index:
        prediction_records.append(
            {
                "id": f"{well}_{int(row_index)}",
                "well": well,
                "well_row_index": int(row_index),
                "selector_code": int(selector_code),
                "selector_variant": str(selector_variant),
                "selector_n_eval": int(n_eval),
                "selector_z_span": float(z_span),
                "md_since_last_known": float(horizontal.iloc[row_index]["MD"])
                - last_known_md,
                "raw_gr_observed": bool(
                    pd.notna(horizontal.iloc[row_index]["GR"])
                ),
                "well_raw_gr_missing_fraction": missing_fraction,
                "hidden_like_spatial": bool(spatial_valid),
                "hidden_like_typewell_purged": bool(typewell_valid),
                "control_before_branch_tvt": float(
                    control_before_branch[row_index]
                ),
                "candidate_before_branch_tvt": float(
                    candidate_before_branch[row_index]
                ),
                "control_tvt": float(control_after_branch[row_index]),
                "candidate_tvt": float(candidate_after_branch[row_index]),
            }
        )

# The exp413 adapter is consumed once only to close the shared producer ledger;
# it does not train or run a model in this Stage B screening.
exp413_contract_frame = shared_likpf_exp413_adapter(shared_bank, wells)
shared_manifest = finalize_shared_likpf_manifest(shared_bank, wells)
if len(exp413_contract_frame) != sum(
    int(record["audit"]["evaluation_rows"]) for record in shared_bank.values()
):
    raise RuntimeError("exp413 adapter row count mismatch")

prediction_frame = pd.DataFrame(prediction_records).sort_values(
    ["well", "well_row_index"], kind="stable"
).reset_index(drop=True)
prediction_content_sha, prediction_gzip_sha = freeze_prediction_frame(
    prediction_frame
)
PREDICTIONS_FROZEN = True
PREDICTIONS_FROZEN_AT_SECONDS = time.time() - STAGE_B_STARTED
print(
    "predictions frozen before truth/fold join:",
    prediction_frame.shape,
    prediction_content_sha,
    flush=True,
)


# %% [markdown]
# ## 5. Post-freeze truth join, fixed-scope metrics, and all-AND gate

# %%
fold_map, fold_records, fold_assignment_sha = stage_b_reporting_fold_map(wells)
truth_by_id = read_stage_b_truth_after_freeze(wells)
scored = prediction_frame.merge(truth_by_id, on="id", how="left", validate="one_to_one")
scored = scored[scored["target_tvt"].notna()].copy()
if len(scored) != len(prediction_frame):
    raise RuntimeError("Stage B suffix truth join lost or duplicated prediction rows")
scored["fold"] = scored["well"].map(fold_map)
if scored["fold"].isna().any():
    raise RuntimeError("Stage B reporting fold join is incomplete")
scored["fold"] = scored["fold"].astype(int)

pre_branch_metrics = metric_bundle(
    scored,
    control_column="control_before_branch_tvt",
    candidate_column="candidate_before_branch_tvt",
)
post_branch_metrics = metric_bundle(
    scored,
    control_column="control_tvt",
    candidate_column="candidate_tvt",
)

pooled_gate = (
    post_branch_metrics["overall"]["delta_candidate_minus_control"]
    <= POOLED_MAX_REGRESSION_FT
)
fold_gate_rows = [
    {
        **record,
        "nonworse": bool(
            record["delta_candidate_minus_control"] <= FOLD_MAX_REGRESSION_FT
        ),
    }
    for record in post_branch_metrics["folds"]
]
nonworse_fold_count = sum(bool(record["nonworse"]) for record in fold_gate_rows)
fold_gate = nonworse_fold_count >= REQUIRED_NONWORSE_FOLDS
scope_gate_rows = [
    {
        **record,
        "nonworse": bool(
            record["delta_candidate_minus_control"]
            <= FIXED_SCOPE_MAX_REGRESSION_FT
        ),
    }
    for record in post_branch_metrics["fixed_scopes"]
]
scope_gate = all(bool(record["nonworse"]) for record in scope_gate_rows)
by_well_p95_gate = (
    post_branch_metrics["by_well_delta_p95"] <= BY_WELL_DELTA_P95_MAX_FT
)
worst_well_gate = (
    post_branch_metrics["worst_well_delta"] <= WORST_WELL_DELTA_MAX_FT
)
all_folds_nonempty = all(record["rows"] > 0 for record in fold_gate_rows)
all_scopes_nonempty = all(record["rows"] > 0 for record in scope_gate_rows)
gate = {
    "pooled_delta_le_0p02": bool(pooled_gate),
    "at_least_4_of_5_folds_delta_le_0p02": bool(fold_gate),
    "all_fixed_scopes_delta_le_0p05": bool(scope_gate),
    "by_well_delta_p95_le_0p25": bool(by_well_p95_gate),
    "worst_well_delta_le_5p0": bool(worst_well_gate),
    "all_reporting_folds_nonempty": bool(all_folds_nonempty),
    "all_fixed_scopes_nonempty": bool(all_scopes_nonempty),
}
all_and_gate_passed = all(gate.values())

scored_payload = scored.to_csv(index=False, lineterminator="\n").encode("utf-8")
scored_content_sha = hashlib.sha256(scored_payload).hexdigest()
SCORED_PREDICTION_PATH.write_bytes(
    gzip.compress(scored_payload, compresslevel=9, mtime=0)
)


# %% [markdown]
# ## 6. Reproducibility report and non-submission outputs

# %%
report = {
    "experiment": EXPERIMENT_NAME,
    "stage": "stage_b_fixed32_paired_scientific_screening",
    "evidence_role": "small_screening_not_200well_generalization_proof",
    "status": "PASS" if all_and_gate_passed else "FAIL",
    "all_and_gate_passed": bool(all_and_gate_passed),
    "truth_read_policy": "after_control_and_candidate_prediction_content_freeze",
    "predictions_frozen_before_truth": True,
    "predictions_frozen_at_seconds": round(PREDICTIONS_FROZEN_AT_SECONDS, 6),
    "truth_read_count": int(TRUTH_READ_COUNT),
    "reporting_fold_joined_after_prediction_freeze": True,
    "selection": {
        "wells": len(wells),
        "well_ids": wells,
        "selection_sha256": selection_sha,
        "reused_stage_a_selection_exact": True,
        "reselection_performed": False,
    },
    "source_identity": {
        "stage_b_source_sha256": stage_b_source_sha,
        "stage_b_generator_sha256": STAGE_B_GENERATOR_SHA256,
        "stage_a_source_sha256": STAGE_A_SOURCE_SHA256,
        "full_candidate_source_sha256": FULL_CANDIDATE_SOURCE_SHA256,
        "exp073_replay_source_sha256": EXP073_REPLAY_SOURCE_SHA256,
        "hidden_assignment_sha256": hidden_assignment_sha,
        "target_free_input_content_sha256": target_free_input_sha,
    },
    "prediction_identity": {
        "frozen_prediction_path": str(FROZEN_PREDICTION_PATH),
        "frozen_prediction_rows": int(len(prediction_frame)),
        "frozen_prediction_content_sha256": prediction_content_sha,
        "frozen_prediction_gzip_sha256": prediction_gzip_sha,
        "scored_prediction_path": str(SCORED_PREDICTION_PATH),
        "scored_prediction_content_sha256": scored_content_sha,
        "id_order_sha256": _shared_json_sha(prediction_frame["id"].tolist()),
        "prediction_schema_sha256": _shared_json_sha(
            prediction_frame.columns.tolist()
        ),
    },
    "fold_assignment": {
        "method": "stable_sha256_rank_round_robin_5fold",
        "sha256": fold_assignment_sha,
        "records": fold_records,
    },
    "variants": list(STAGE_B_VARIANTS),
    "frozen_parent_contract": {
        "profile": SUBMISSION_PROFILE,
        "selector_bimodal_detector": False,
        "common_beam_computed_once_per_well": True,
        "branch_hedge": {
            "strength": BRANCH_HEDGE_STRENGTH,
            "min_mass": BRANCH_HEDGE_MIN_MASS,
            "separation_low": BRANCH_HEDGE_SEPARATION_LOW,
            "separation_high": BRANCH_HEDGE_SEPARATION_HIGH,
            "cap_ft": BRANCH_HEDGE_CAP_FT,
            "skip_existing": BRANCH_HEDGE_SKIP_EXISTING,
        },
    },
    "execution_count": {
        "active_scientific_variants": 2,
        "legacy_sp45_well_bank_generations": len(wells),
        "shared_candidate_well_bank_generations": len(wells),
        "total_well_bank_generations": 2 * len(wells),
        "particles_per_bank": STAGE_B_PARTICLES,
        "seeds_per_bank": STAGE_B_SEEDS,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "new_boosters": 0,
        "parent_control_retraining": 0,
        "inference_time_booster_training": 0,
    },
    "runtime": {
        "shared_seconds": round(shared_seconds, 6),
        "legacy_and_common_beam_seconds": round(legacy_seconds, 6),
        "shared_parallel_report": shared_parallel_report,
        "legacy_per_well_seconds": {
            record["well"]: record["elapsed_seconds"]
            for record in legacy_records
        },
        "total_seconds": round(time.time() - STAGE_B_STARTED, 6),
    },
    "shared_manifest": shared_manifest,
    "branch_summary_identity": {
        "legacy_sha256": _shared_json_sha(
            [
                {"well": well, **legacy_bank[well]["branch_stats"]}
                for well in sorted(wells)
            ]
        ),
        "candidate_sha256": shared_manifest["branch_summary_sha256"],
        "control_hedge_reason_counts": pd.Series(
            [record["reason"] for record in control_branch_records]
        ).value_counts().sort_index().to_dict(),
        "candidate_hedge_reason_counts": pd.Series(
            [record["reason"] for record in candidate_branch_records]
        ).value_counts().sort_index().to_dict(),
    },
    "metrics": {
        "before_branch_hedge": pre_branch_metrics,
        "after_branch_hedge_primary": post_branch_metrics,
    },
    "gate_thresholds": {
        "pooled_max_regression_ft": POOLED_MAX_REGRESSION_FT,
        "fold_max_regression_ft": FOLD_MAX_REGRESSION_FT,
        "required_nonworse_folds": REQUIRED_NONWORSE_FOLDS,
        "fixed_scope_max_regression_ft": FIXED_SCOPE_MAX_REGRESSION_FT,
        "by_well_delta_p95_max_ft": BY_WELL_DELTA_P95_MAX_FT,
        "worst_well_delta_max_ft": WORST_WELL_DELTA_MAX_FT,
    },
    "gate": gate,
    "fold_gate_rows": fold_gate_rows,
    "fixed_scope_gate_rows": scope_gate_rows,
    "nonworse_fold_count": int(nonworse_fold_count),
    "submission_file_generated": False,
    "external_submission_performed": False,
    "stage_c_or_hidden_executed": False,
    "deterministic_anchor": False,
}

report_payload = json.dumps(
    report,
    indent=2,
    sort_keys=True,
    ensure_ascii=False,
    allow_nan=False,
) + "\n"
REPORT_PATH.write_text(report_payload, encoding="utf-8")
METRICS_PATH.write_text(report_payload, encoding="utf-8")

print("Stage B primary pooled metrics:")
display(pd.DataFrame([post_branch_metrics["overall"]]))
print("Stage B fold gate:")
display(pd.DataFrame(fold_gate_rows))
print("Stage B fixed-scope gate:")
display(pd.DataFrame(scope_gate_rows))
print("Stage B gate:", gate)
print("Stage B nonworse folds:", nonworse_fold_count, "/", STAGE_B_REPORTING_FOLDS)
print("Stage B all-AND status:", report["status"])
print("Stage B report path:", REPORT_PATH)
print("Stage B metrics path:", METRICS_PATH)
print("submission file generated: False")
print("external submission performed: False")
'''


def build_source() -> str:
    if sha256(STAGE_A_SOURCE) != EXPECTED_STAGE_A_SHA256:
        raise RuntimeError("Stage A source SHA drifted; refusing Stage B generation")
    if sha256(FULL_CANDIDATE_SOURCE) != EXPECTED_FULL_CANDIDATE_SHA256:
        raise RuntimeError("full candidate source SHA drifted; refusing Stage B generation")

    stage_a_source = STAGE_A_SOURCE.read_text(encoding="utf-8")
    full_source = FULL_CANDIDATE_SOURCE.read_text(encoding="utf-8")
    shared_runtime = source_slice(
        stage_a_source,
        "SHARED_LIKPF_SCALES =",
        "def run_shared_likpf_stage_a(",
    )
    selector_source = source_slice(
        full_source,
        "SELECTOR_N_EVAL_THRESHOLD =",
        "# %% [markdown]\n# ## 4. Ridge/PF anchor and shared deterministic candidate surface",
    )
    selector_functions = extract_functions(
        selector_source,
        (
            "run_particle_filter",
            "run_pf_lik_ensemble_scales",
            "beam_search",
            "run_beam_ensemble",
            "selector_well_code",
            "parse_selector_variant",
            "apply_selector_variant",
        ),
    )
    generator_sha = sha256(Path(__file__))
    header = (
        HEADER.replace("__STAGE_A_SHA256__", EXPECTED_STAGE_A_SHA256)
        .replace("__FULL_CANDIDATE_SHA256__", EXPECTED_FULL_CANDIDATE_SHA256)
        .replace("__GENERATOR_SHA256__", generator_sha)
        .replace("__SELECTION_SHA256__", EXPECTED_SELECTION_SHA256)
        .replace("__HIDDEN_ASSIGNMENT_SHA256__", EXPECTED_HIDDEN_ASSIGNMENT_SHA256)
    )
    return (
        header.rstrip()
        + "\n\n# %% [markdown]\n"
        + "# ## 2. Shared exp413 likelihood-PF producer and Stage A selection\n\n"
        + "# %%\n"
        + shared_runtime.rstrip()
        + "\n\n"
        + SELECTOR_CONSTANTS.rstrip()
        + "\n\n"
        + selector_functions.rstrip()
        + "\n\n"
        + ORCHESTRATION.lstrip()
    )


def main() -> None:
    source = build_source()
    ast.parse(source)
    OUTPUT_SOURCE.write_text(source, encoding="utf-8")
    print(f"wrote {OUTPUT_SOURCE.relative_to(ROOT)}")
    print(f"lines={len(source.splitlines())}")
    print(f"sha256={hashlib.sha256(source.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
