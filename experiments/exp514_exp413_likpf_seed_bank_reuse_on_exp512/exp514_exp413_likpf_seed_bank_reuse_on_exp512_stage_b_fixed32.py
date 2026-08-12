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
STAGE_A_SOURCE_SHA256 = "89129ad85c129145e635633741e08ff5e058a365c344a4a4bdbdc77190ab3873"
FULL_CANDIDATE_SOURCE_SHA256 = "8b1616dd289672339bfba82050e25b1c678a00dcb89b17ad6de60892c4171634"
EXP073_REPLAY_SOURCE_SHA256 = (
    "4af212a8a1c83e36cdcc0bc912942a62df1fbc94ca67fd75789171afaa1a647e"
)
STAGE_B_GENERATOR_SHA256 = "6825f2928684fb3d2d126afe7f15cfca79ef9154716db43a12a32e83d03030d1"
EXPECTED_SELECTION_SHA256 = "86157959105b896271f53c841b27f5f7246db6c4f199773b0151ed75d36ae58b"
HIDDEN_ASSIGNMENT_SHA256 = "5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597"
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

# %% [markdown]
# ## 2. Shared exp413 likelihood-PF producer and Stage A selection

# %%
SHARED_LIKPF_SCALES = (3.0, 5.0, 8.0, 12.0)
SHARED_LIKPF_PARTICLES = 500
SHARED_LIKPF_SEEDS = 128
SHARED_LIKPF_BRANCH_SCALE = 5.0
SHARED_LIKPF_N_JOBS = 4
SHARED_LIKPF_SEED_NAMESPACE = "SHA256(likpf::<split>::<well>)"


def shared_likpf_stable_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo + 1


@njit(cache=True)
def _shared_likpf_interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t


@njit(cache=True, nogil=True)
def _shared_pf_lik_allseeds(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, n_seeds, seed_base,
                            MOM, VN, PN, RP, RR, RESAMP, init_spr):
    n = len(md_v); preds = np.empty((n_seeds, n)); liks = np.empty(n_seeds); tmax = vmin + len(gg)*step
    for s in range(n_seeds):
        np.random.seed(seed_base + s)
        pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
        for j in range(N):
            pos[j] = ls + init_spr*np.random.randn(); rate[j] = ir + 0.01*np.random.randn()
        log_lik = 0.0; prev_md = md_v[0] - 1.0
        for i in range(n):
            dm = md_v[i] - prev_md
            if dm < 1.0: dm = 1.0
            for j in range(N):
                rate[j] = MOM*rate[j] + VN*np.random.randn(); pos[j] += rate[j]*dm + PN*np.random.randn()
                tvt_j = pos[j] - z_v[i]
                if tvt_j < vmin-100.: tvt_j = vmin-100.
                if tvt_j > tmax+100.: tvt_j = tmax+100.
                pos[j] = tvt_j + z_v[i]
            avg_lk = 0.0
            for j in range(N):
                eg = _shared_likpf_interp1(gg, pos[j]-z_v[i], vmin, step); d = (gr_v[i]-eg)/gs; dd = d*d
                if dd > 600.: dd = 600.
                lk = np.exp(-0.5*dd)
                if lk < 1e-300: lk = 1e-300
                avg_lk += w[j]*lk; w[j] = w[j]*lk
            if avg_lk < 1e-300: avg_lk = 1e-300
            log_lik += np.log(avg_lk)
            ws = 0.0
            for j in range(N): ws += w[j]
            if ws > 0.0:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
            neff = 0.0
            for j in range(N): neff += w[j]*w[j]
            neff = 1.0/neff
            if neff < RESAMP*N:
                cum = np.empty(N); c = 0.0
                for j in range(N): c += w[j]; cum[j] = c
                u0 = np.random.uniform(0., 1./N); newpos = np.empty(N); newrate = np.empty(N); ci = 0
                for j in range(N):
                    u = u0 + j/N
                    while ci < N-1 and cum[ci] < u: ci += 1
                    newpos[j] = pos[ci] + RP*np.random.randn(); newrate[j] = rate[ci] + RR*np.random.randn()
                for j in range(N): pos[j] = newpos[j]; rate[j] = newrate[j]; w[j] = 1./N
            est = 0.0
            for j in range(N): est += w[j]*(pos[j]-z_v[i])
            preds[s, i] = est; prev_md = md_v[i]
        liks[s] = log_lik
    return preds, liks


def _shared_likpf_grid(typewell_tvt, typewell_gr, step=0.2):
    tmin = float(typewell_tvt.min()); tmax = float(typewell_tvt.max())
    tvt_g = np.arange(tmin, tmax+step, step)
    return np.interp(tvt_g, typewell_tvt, typewell_gr).astype(np.float64), float(tmin), float(step)


def _shared_array_sha(values) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _shared_json_sha(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _shared_branch_summary(predictions, centered_log_likelihoods, evaluation_index):
    if predictions.shape[0] < 4 or predictions.shape[1] < 10:
        raise ValueError("shared likelihood-PF branch summary requires >=4 seeds and >=10 rows")
    seed_weight = np.exp(centered_log_likelihoods / SHARED_LIKPF_BRANCH_SCALE)
    seed_weight /= float(seed_weight.sum())
    level = np.nanmedian(predictions, axis=1)
    valid = np.isfinite(level) & np.isfinite(seed_weight) & (seed_weight > 0)
    level = level[valid]
    seed_weight = seed_weight[valid]
    if len(level) < 4:
        raise ValueError("shared likelihood-PF branch summary has fewer than four valid seeds")
    seed_weight /= float(seed_weight.sum())
    order = np.argsort(level)
    values = level[order]
    weights = seed_weight[order]
    cumulative_weight = np.cumsum(weights)
    cumulative_value = np.cumsum(weights * values)
    cumulative_square = np.cumsum(weights * values * values)
    total_weight = float(cumulative_weight[-1])
    total_value = float(cumulative_value[-1])
    total_square = float(cumulative_square[-1])
    best = None
    for cut in range(1, len(values)):
        low_weight = float(cumulative_weight[cut - 1])
        high_weight = total_weight - low_weight
        if low_weight < 0.05 or high_weight < 0.05:
            continue
        low_value = float(cumulative_value[cut - 1])
        high_value = total_value - low_value
        low_sse = float(
            cumulative_square[cut - 1] - low_value * low_value / low_weight
        )
        high_sse = float(
            total_square
            - cumulative_square[cut - 1]
            - high_value * high_value / high_weight
        )
        score = max(0.0, low_sse) + max(0.0, high_sse)
        if best is None or score < best[0]:
            best = (
                score,
                low_weight,
                high_weight,
                low_value / low_weight,
                high_value / high_weight,
            )
    if best is None:
        raise ValueError("shared likelihood-PF branch split has no valid mass partition")
    _, mass_low, mass_high, center_low, center_high = best
    return {
        "center_low": float(center_low),
        "center_high": float(center_high),
        "mass_low": float(mass_low),
        "mass_high": float(mass_high),
        "weighted_center": float(np.sum(seed_weight * level)),
        "eval_rows": np.asarray(evaluation_index, dtype=int).tolist(),
        "seed_count": int(len(level)),
    }


def _shared_likpf_one_well(
    well,
    split,
    loader,
    *,
    particles=SHARED_LIKPF_PARTICLES,
    seeds=SHARED_LIKPF_SEEDS,
    scales=SHARED_LIKPF_SCALES,
):
    started = time.time()
    horizontal, typewell = loader(str(well), str(split))
    horizontal = horizontal.copy()
    typewell = typewell.sort_values("TVT").copy()
    required_horizontal = {"MD", "Z", "GR", "TVT_input"}
    if missing := required_horizontal - set(horizontal.columns):
        raise ValueError(f"shared likelihood-PF horizontal columns missing: {sorted(missing)}")
    if {"TVT", "GR"} - set(typewell.columns):
        raise ValueError("shared likelihood-PF typewell requires TVT and GR")
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    evaluation_mask = ~known_mask
    known = horizontal.loc[known_mask]
    evaluation = horizontal.loc[evaluation_mask]
    if len(known) < 10 or len(evaluation) < 10:
        raise ValueError(f"shared likelihood-PF ineligible well: {well}")
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = (
        typewell["GR"]
        .fillna(float(typewell["GR"].mean()))
        .to_numpy(np.float64)
    )
    last = known.iloc[-1]
    level_start = float(last["TVT_input"]) + float(last["Z"])
    typewell_at_known = np.interp(
        known["TVT_input"].to_numpy(np.float64),
        typewell_tvt,
        typewell_gr,
    )
    gr_sigma = float(
        np.clip(
            np.nanstd(
                known["GR"].fillna(0).to_numpy(np.float64) - typewell_at_known
            ),
            10.0,
            60.0,
        )
    )
    tail = known.tail(30)
    delta_tvt = np.diff(tail["TVT_input"].to_numpy(np.float64))
    delta_z = np.diff(tail["Z"].to_numpy(np.float64))
    delta_md = np.diff(tail["MD"].to_numpy(np.float64))
    positive = delta_md > 0
    initial_rate = (
        float(np.median((delta_tvt + delta_z)[positive] / delta_md[positive]))
        if int(positive.sum()) >= 3
        else 0.0
    )
    gr_grid, grid_min, grid_step = _shared_likpf_grid(typewell_tvt, typewell_gr)
    interpolated_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(float(typewell_gr.mean()))
        .to_numpy(np.float64)
    )
    evaluation_index = evaluation.index.to_numpy(np.int64)
    seed_base = shared_likpf_stable_seed("likpf", split, well)
    core_started = time.time()
    predictions, log_likelihoods = _shared_pf_lik_allseeds(
        evaluation["MD"].to_numpy(np.float64),
        evaluation["Z"].to_numpy(np.float64),
        interpolated_gr[evaluation_index],
        gr_grid,
        grid_min,
        grid_step,
        gr_sigma,
        level_start,
        initial_rate,
        int(particles),
        int(seeds),
        int(seed_base),
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )
    core_seconds = time.time() - core_started
    if predictions.shape != (int(seeds), len(evaluation)):
        raise ValueError(f"shared likelihood-PF raw bank shape mismatch for {well}")
    if not np.isfinite(predictions).all() or not np.isfinite(log_likelihoods).all():
        raise ValueError(f"shared likelihood-PF raw bank is non-finite for {well}")
    centered = log_likelihoods - float(log_likelihoods.max())
    suffix_aggregates = {}
    for scale in scales:
        weights = np.exp(centered / float(scale))
        weights /= float(weights.sum())
        suffix_aggregates[f"pf_scale_{float(scale):g}"] = (
            weights[:, None] * predictions
        ).sum(axis=0)
    suffix_aggregates["pf_mean"] = predictions.mean(axis=0)
    branch_summary = _shared_branch_summary(predictions, centered, evaluation_index)

    raw_tvt_input = pd.to_numeric(
        horizontal["TVT_input"], errors="coerce"
    ).to_numpy(np.float64)
    sp45_full = {}
    for name, suffix in suffix_aggregates.items():
        full = raw_tvt_input.copy()
        full[evaluation_index] = np.asarray(suffix, dtype=np.float64)
        if not np.array_equal(full[known_mask], raw_tvt_input[known_mask]):
            raise ValueError(f"shared likelihood-PF known-prefix parity failed for {well}")
        if not np.isfinite(full).all():
            raise ValueError(f"shared likelihood-PF full adapter is non-finite for {well}")
        sp45_full[name] = full

    expected_ids = [f"{well}_{int(index)}" for index in evaluation_index]
    scale5 = suffix_aggregates["pf_scale_5"].astype(np.float32)
    arithmetic_mean = suffix_aggregates["pf_mean"].astype(np.float32)
    exp413_frame = pd.DataFrame(
        {
            "id": expected_ids,
            "likpf_scale_5": scale5,
            "likpf_mean": arithmetic_mean,
        }
    )
    if exp413_frame["id"].duplicated().any():
        raise ValueError(f"shared likelihood-PF duplicate exp413 IDs for {well}")
    scale_content_sha = {
        name: _shared_array_sha(np.asarray(values, dtype=np.float64))
        for name, values in suffix_aggregates.items()
    }
    del predictions, log_likelihoods, centered
    return {
        "well": str(well),
        "split": str(split),
        "seed_base": int(seed_base),
        "row_index": horizontal.index.to_numpy(np.int64),
        "evaluation_index": evaluation_index,
        "known_mask": known_mask,
        "sp45_full": sp45_full,
        "exp413_frame": exp413_frame,
        "branch_summary": branch_summary,
        "scale_content_sha256": scale_content_sha,
        "audit": {
            "rows": int(len(horizontal)),
            "known_rows": int(known_mask.sum()),
            "evaluation_rows": int(evaluation_mask.sum()),
            "particles": int(particles),
            "seeds": int(seeds),
            "scales": [float(value) for value in scales],
            "gr_sigma": gr_sigma,
            "kernel_dtype": "float64",
            "exp413_consumer_dtype": "float32",
            "raw_seed_bank_retained": False,
            "core_seconds": round(core_seconds, 6),
            "elapsed_seconds": round(time.time() - started, 6),
        },
        "ledger": {
            "producer_calls": 1,
            "core_calls": 1,
            "sp45_consumer_hits": 0,
            "exp413_consumer_hits": 0,
            "legacy_sp45_bank_calls": 0,
            "exp413_duplicate_bank_calls": 0,
            "fallback_calls": 0,
        },
    }


def materialize_shared_likpf_bank(
    wells,
    split,
    loader,
    *,
    n_jobs=SHARED_LIKPF_N_JOBS,
    particles=SHARED_LIKPF_PARTICLES,
    seeds=SHARED_LIKPF_SEEDS,
):
    ordered_wells = sorted(str(well) for well in wells)
    if not ordered_wells or len(set(ordered_wells)) != len(ordered_wells):
        raise ValueError("shared likelihood-PF requires unique nonempty wells")
    effective_n_jobs = min(max(1, int(n_jobs)), len(ordered_wells))
    started = time.time()
    results = Parallel(n_jobs=effective_n_jobs, backend="threading")(
        delayed(_shared_likpf_one_well)(
            well,
            split,
            loader,
            particles=int(particles),
            seeds=int(seeds),
        )
        for well in ordered_wells
    )
    bank = {record["well"]: record for record in results}
    if list(bank) != ordered_wells:
        raise RuntimeError("shared likelihood-PF merge order changed")
    report = {
        "requested_n_jobs": int(n_jobs),
        "effective_n_jobs": effective_n_jobs,
        "backend": "threading",
        "wells": len(ordered_wells),
        "elapsed_seconds": round(time.time() - started, 6),
    }
    return bank, report


def shared_likpf_sp45_adapter(record):
    ledger = record["ledger"]
    ledger["sp45_consumer_hits"] += 1
    if ledger["sp45_consumer_hits"] != 1:
        raise RuntimeError(f"SP45 consumed shared bank more than once: {record['well']}")
    required = {f"pf_scale_{scale:g}" for scale in SHARED_LIKPF_SCALES}
    required.add("pf_mean")
    if set(record["sp45_full"]) != required:
        raise ValueError(f"SP45 shared adapter schema mismatch: {record['well']}")
    return (
        {name: values.copy() for name, values in record["sp45_full"].items()},
        dict(record["branch_summary"]),
    )


def shared_likpf_exp413_adapter(bank, wells):
    frames = []
    for well in [str(value) for value in wells]:
        if well not in bank:
            raise KeyError(f"exp413 shared likelihood-PF bank is missing well {well}")
        record = bank[well]
        record["ledger"]["exp413_consumer_hits"] += 1
        if record["ledger"]["exp413_consumer_hits"] != 1:
            raise RuntimeError(f"exp413 consumed shared bank more than once: {well}")
        frames.append(record["exp413_frame"].copy(deep=True))
    frame = pd.concat(frames, ignore_index=True)
    expected_columns = ["id", "likpf_scale_5", "likpf_mean"]
    if list(frame.columns) != expected_columns or frame["id"].duplicated().any():
        raise ValueError("exp413 shared likelihood-PF adapter schema/ID contract failed")
    return frame


def finalize_shared_likpf_manifest(bank, expected_wells):
    ordered_wells = sorted(str(value) for value in expected_wells)
    if sorted(bank) != ordered_wells:
        raise ValueError("shared likelihood-PF manifest well set mismatch")
    records = []
    branch_records = []
    ledger_records = []
    for well in ordered_wells:
        record = bank[well]
        if any(key in record for key in ("predictions", "log_likelihoods", "raw_bank")):
            raise RuntimeError(f"raw likelihood-PF bank leaked beyond well scope: {well}")
        ledger = {"well": well, **record["ledger"]}
        expected_ledger = {
            "producer_calls": 1,
            "core_calls": 1,
            "sp45_consumer_hits": 1,
            "exp413_consumer_hits": 1,
            "legacy_sp45_bank_calls": 0,
            "exp413_duplicate_bank_calls": 0,
            "fallback_calls": 0,
        }
        if record["ledger"] != expected_ledger:
            raise RuntimeError(f"shared likelihood-PF ledger failed for {well}: {ledger}")
        records.append(
            {
                "well": well,
                "split": record["split"],
                "seed_base": record["seed_base"],
                "scale_content_sha256": record["scale_content_sha256"],
                "audit": record["audit"],
            }
        )
        branch_records.append({"well": well, **record["branch_summary"]})
        ledger_records.append(ledger)
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "producer_id": "exp413_x1p0_stable_seed_bank",
        "source_sha256": EXP073_REPLAY_SOURCE_SHA256,
        "seed_namespace": SHARED_LIKPF_SEED_NAMESPACE,
        "wells": len(ordered_wells),
        "records": records,
        "branch_summary_sha256": _shared_json_sha(branch_records),
        "generation_ledger_sha256": _shared_json_sha(ledger_records),
        "aggregate_content_sha256": _shared_json_sha(
            [record["scale_content_sha256"] for record in records]
        ),
        "raw_seed_bank_retained": False,
        "all_contracts_passed": True,
    }
    return manifest


def select_shared_likpf_stage_a_wells(data_root, *, split="train", count=32):
    split_root = Path(data_root) / split
    records = []
    for horizontal_path in sorted(split_root.glob("*__horizontal_well.csv")):
        well = horizontal_path.name.split("__", 1)[0]
        typewell_path = split_root / f"{well}__typewell.csv"
        if not typewell_path.is_file():
            continue
        horizontal = pd.read_csv(
            horizontal_path,
            usecols=lambda name: name in {"MD", "Z", "GR", "TVT_input"},
        )
        evaluation = horizontal["TVT_input"].isna()
        known = ~evaluation
        if int(evaluation.sum()) < 10 or int(known.sum()) < 10:
            continue
        evaluation_z = pd.to_numeric(
            horizontal.loc[evaluation, "Z"], errors="coerce"
        ).to_numpy(np.float64)
        if len(evaluation_z) == 0 or not np.isfinite(evaluation_z).any():
            continue
        records.append(
            {
                "well": well,
                "evaluation_rows": int(evaluation.sum()),
                "raw_gr_missing_fraction": float(horizontal["GR"].isna().mean()),
                "evaluation_z_span": float(np.nanmax(evaluation_z) - np.nanmin(evaluation_z)),
                "stable_sha256": hashlib.sha256(
                    f"exp514::stage_a::{split}::{well}".encode("utf-8")
                ).hexdigest(),
            }
        )
    if len(records) < int(count):
        raise ValueError(f"Stage A needs {count} eligible wells, found {len(records)}")
    frame = pd.DataFrame(records)
    for column in (
        "evaluation_rows",
        "raw_gr_missing_fraction",
        "evaluation_z_span",
    ):
        rank = frame[column].rank(method="first").to_numpy(np.int64) - 1
        frame[f"{column}_stratum"] = np.minimum(3, 4 * rank // len(frame))
    stratum_columns = [
        "evaluation_rows_stratum",
        "raw_gr_missing_fraction_stratum",
        "evaluation_z_span_stratum",
    ]
    groups = []
    for key, group in frame.groupby(stratum_columns, sort=True):
        groups.append((tuple(int(value) for value in key), group.sort_values("stable_sha256")))
    selected = []
    offset = 0
    while len(selected) < int(count):
        progressed = False
        for _, group in groups:
            if offset < len(group):
                selected.append(group.iloc[offset].to_dict())
                progressed = True
                if len(selected) == int(count):
                    break
        if not progressed:
            break
        offset += 1
    if len(selected) != int(count):
        raise RuntimeError("Stage A target-free stratified selection did not reach count")
    selection_sha = _shared_json_sha(selected)
    return [str(record["well"]) for record in selected], selected, selection_sha

# %% [markdown]
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

def run_particle_filter(hw, tw, n_particles=500, seed=42):
    tw_s   = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)

    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy(), 0.0

    last     = kn.iloc[-1]
    last_tvt = float(last['TVT_input'])
    last_Z   = float(last['Z'])
    last_MD  = float(last['MD'])

    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn['GR'].fillna(0).values - tw_at_k), 10., 60.))

    tail = kn.tail(30)
    dt = np.diff(tail['TVT_input'].values)
    dz = np.diff(tail['Z'].values)
    dm = np.diff(tail['MD'].values)
    m  = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    N   = n_particles
    rng = np.random.default_rng(seed)
    ls   = last_tvt + last_Z
    pos  = ls + 4.5 * rng.standard_normal(N)  # sp45 patch (sel15 vb best)
    rate = ir + 0.01 * rng.standard_normal(N)
    w    = np.ones(N) / N

    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5

    md_v = ev['MD'].values.astype(float)
    z_v  = ev['Z'].values.astype(float)
    # Interpolate GR gaps before tracking
    gr_interp = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]

    out_vals = hw['TVT_input'].values.astype(float).copy()
    res = np.empty(len(ev))
    prev_MD = last_MD
    log_lik = 0.0

    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos  = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos   = tvt_p + z_v[i]

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d  = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d**2, 600.))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        n_eff = 1.0 / (w**2).sum()
        if n_eff < RESAMP * N:
            cum = np.cumsum(w)
            u0  = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos  = pos[idx]  + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w    = np.ones(N) / N

        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]

    out_vals[list(ev.index)] = res
    return out_vals, log_lik


def run_pf_lik_ensemble_scales(hw, tw, scales=SELECTOR_SCALES, n_particles=500, n_seeds=128, branch_stats=None):
    preds = []
    liks = []
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)
        preds.append(p)
        liks.append(ll)
    pred_arr = np.stack(preds, 0)
    liks = np.array(liks)
    liks_n = liks - liks.max()
    out = {}
    for scale in scales:
        weights = np.exp(liks_n / float(scale))
        weights /= weights.sum()
        out[f'pf_scale_{scale:g}'] = (weights[:, None] * pred_arr).sum(0)
    out['pf_mean'] = pred_arr.mean(0)
    if bool(globals().get('SELECTOR_PF_RETURN_STD', False)):
        out['pf_seed_std'] = pred_arr.std(0)
    if branch_stats is not None:
        try:
            eval_mask = pd.to_numeric(hw['TVT_input'], errors='coerce').isna().to_numpy()
            if int(eval_mask.sum()) >= 10:
                seed_weight = np.exp(liks_n / 5.0)
                seed_weight = seed_weight / max(float(seed_weight.sum()), 1e-12)
                level = np.nanmedian(pred_arr[:, eval_mask], axis=1)
                valid = np.isfinite(level) & np.isfinite(seed_weight) & (seed_weight > 0)
                level = level[valid]
                seed_weight = seed_weight[valid]
                seed_weight = seed_weight / max(float(seed_weight.sum()), 1e-12)
                if len(level) >= 4:
                    order = np.argsort(level)
                    x = level[order]
                    w = seed_weight[order]
                    cw = np.cumsum(w)
                    cx = np.cumsum(w * x)
                    cx2 = np.cumsum(w * x * x)
                    total_w, total_x, total_x2 = float(cw[-1]), float(cx[-1]), float(cx2[-1])
                    best = None
                    for cut in range(1, len(x)):
                        wl = float(cw[cut - 1])
                        wr = total_w - wl
                        if wl < 0.05 or wr < 0.05:
                            continue
                        xl = float(cx[cut - 1])
                        xr = total_x - xl
                        ssel = float(cx2[cut - 1] - xl * xl / wl)
                        sser = float(total_x2 - cx2[cut - 1] - xr * xr / wr)
                        score = max(0.0, ssel) + max(0.0, sser)
                        if best is None or score < best[0]:
                            best = (score, wl, wr, xl / wl, xr / wr)
                    if best is not None:
                        _, mass_low, mass_high, center_low, center_high = best
                        branch_stats.update(
                            center_low=float(center_low),
                            center_high=float(center_high),
                            mass_low=float(mass_low),
                            mass_high=float(mass_high),
                            weighted_center=float(np.sum(seed_weight * level)),
                            eval_rows=np.flatnonzero(eval_mask).astype(int).tolist(),
                            seed_count=int(len(level)),
                        )
        except Exception as exc:
            branch_stats['error'] = repr(exc)
    return out


def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    n  = len(hgr)
    nt = len(tw_tvt)
    if n == 0:
        return np.array([last_tvt])

    if r > 0 and n > max(3, 2 * r + 1):
        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
        sgr = savgol_filter(hgr, win, min(2, win - 1))
    else:
        sgr = hgr.copy()

    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))

    MOVES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    MC    = mc * np.array([2., 1., 0., 1., 2.])

    bidx  = np.full(bs, si, dtype=np.int64)
    bcost = np.full(bs, np.inf)
    bcost[0] = 0.
    bn = 1

    result = np.zeros(n)

    for step in range(n):
        gv = sgr[step]
        ni = bidx[:bn, None] + MOVES[None, :]
        ci = np.clip(ni, 0, nt - 1)
        valid = (ni >= 0) & (ni < nt)

        gr_e = (gv - tw_gr[ci])**2 / es
        tot  = bcost[:bn, None] + gr_e + MC[None, :]
        tot  = np.where(valid, tot, np.inf)

        ni_f  = ni.flatten()
        tot_f = tot.flatten()
        vf    = valid.flatten()
        ni_f  = ni_f[vf]
        tot_f = tot_f[vf]

        order = np.argsort(tot_f)
        ni_s  = ni_f[order]
        tot_s = tot_f[order]

        _, first = np.unique(ni_s, return_index=True)
        ni_u  = ni_s[first]
        tot_u = tot_s[first]

        kept = min(bs, len(ni_u))
        top  = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]
        top  = top[np.argsort(tot_u[top])]

        bidx[:kept]  = ni_u[top]
        bcost[:kept] = tot_u[top]
        if kept < bs:
            bidx[kept:]  = bidx[kept - 1]
            bcost[kept:] = np.inf
        bn = kept

        result[step] = tw_tvt[bidx[0]]

    return result


def run_beam_ensemble(hw, tw):
    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy()

    last_tvt = float(kn.iloc[-1]['TVT_input'])
    tw_s  = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)

    gr_all = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)
    hgr    = gr_all[ev.index]

    beam_results = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
                    for (bs, mc, es, r) in BEAM_CONFIGS]

    beam_mean = np.stack(beam_results, 0).mean(0)

    out = hw['TVT_input'].values.astype(float).copy()
    out[list(ev.index)] = beam_mean
    return out


def selector_well_code(hw):
    eval_mask = hw['TVT_input'].isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, 'Z'].values.astype(float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side='right'))
    code = n_bin + 2 * z_bin
    variant = SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)
    return code, variant, n_eval, z_span


def parse_selector_variant(name):
    parts = name.split('_')
    scale = float(parts[2])
    beam_weight = 0.0
    hold_weight = 0.0
    if 'beam' in parts:
        beam_weight = float(parts[parts.index('beam') + 1])
    if 'hold' in parts:
        hold_weight = float(parts[parts.index('hold') + 1])
    return scale, beam_weight, hold_weight


def apply_selector_variant(name, pf_by_scale, tvt_beam, last_known_tvt, hw=None, tw=None, return_info=False):
    scale, beam_weight, hold_weight = parse_selector_variant(name)
    base = pf_by_scale.get(f'pf_scale_{scale:g}')
    if base is None:
        base = pf_by_scale[SELECTOR_GLOBAL_VARIANT.split('_beam_')[0].split('_hold_')[0]]
    base = np.asarray(base, dtype=float)
    tvt_beam = np.asarray(tvt_beam, dtype=float)
    info = {
        'bimodal_active': False,
        'base_beam_weight': float(beam_weight),
        'effective_beam_weight': float(beam_weight),
        'delta_star': 0.0,
        'p_base': np.nan,
        'p_eff': np.nan,
        'prefix_trust': np.nan,
        'temperature': np.nan,
        'rho1': np.nan,
        'n_eff': np.nan,
        'heel_calibrated': False,
        'heel_rows': 0,
        'heel_alpha': np.nan,
        'heel_beta': np.nan,
        'heel_rmse_raw': np.nan,
        'heel_rmse_calibrated': np.nan,
        'heel_denoised': False,
    }
    if bool(globals().get('RUN_BIMODAL_DETECTOR', globals().get('RUN_BIMODAL_SELECTOR_HEDGE', False))):
        hedge = _bimodal_selector_weight(base, tvt_beam, hw=hw, tw=tw)
        if hedge is not None:
            info.update(hedge)
            info['bimodal_active'] = True
            info['effective_beam_weight'] = float(beam_weight)
            pred = base + float(hedge['delta_star'])
        else:
            pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam
    else:
        pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    if return_info:
        return pred, info
    return pred

# %%
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
