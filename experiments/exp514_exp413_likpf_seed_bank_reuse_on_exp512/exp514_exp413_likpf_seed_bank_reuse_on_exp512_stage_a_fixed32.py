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
# # exp514 Stage A fixed32 technical and determinism audit
#
# This notebook selects 32 train-like wells using target-free raw attributes,
# evaluates only the shared exp413 likelihood-PF producer/adapters, and writes
# technical SHA/ledger evidence. It does not read suffix TVT, fit a model,
# create a submission, or run the full exp512 inference path.

# %% [markdown]
# ## Contents
# 1. Imports, source identity, and dynamic input checks
# 2. Shared exp413 likelihood-PF producer, adapters, and Stage A audit
# 3. Fixed32 execution and reproducibility outputs

# %%
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from joblib import Parallel, delayed
from numba import njit

EXPERIMENT_NAME = "exp514_exp413_likpf_seed_bank_reuse_on_exp512"
EXP512_PARENT_SOURCE_SHA256 = "16982879716918811dfa9915c4862d45836bd9360efafbaee41046c3e1b6240f"
EXP073_REPLAY_SOURCE_SHA256 = "4af212a8a1c83e36cdcc0bc912942a62df1fbc94ca67fd75789171afaa1a647e"
EXP514_GENERATOR_SHA256 = "f7380e31597a7095af318163ca887f71b0a63539886640a37f9c53800fdbaf41"


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


COMPETITION_DATA_ROOT = resolve_competition_data_root()
print("experiment:", EXPERIMENT_NAME)
print("competition data root:", COMPETITION_DATA_ROOT)
print("parent source SHA256:", EXP512_PARENT_SOURCE_SHA256)
print("exp073 replay source SHA256:", EXP073_REPLAY_SOURCE_SHA256)

# %% [markdown]
# ## 2. Shared exp413 likelihood-PF producer, adapters, and Stage A audit
#
# The only candidate likelihood-PF producer below is source-identical to the
# SHA-pinned exp073/exp413 x1.0 kernel. It materializes only aggregates and
# branch summaries. Raw 128-seed trajectories and log-likelihoods remain local
# to one well and are released before the result enters the process-wide bank.

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


def run_shared_likpf_stage_a(
    data_root,
    *,
    split="train",
    count=32,
    thread_counts=(1, 4),
    reruns=2,
    particles=SHARED_LIKPF_PARTICLES,
    seeds=SHARED_LIKPF_SEEDS,
):
    wells, selection, selection_sha = select_shared_likpf_stage_a_wells(
        data_root,
        split=split,
        count=count,
    )

    def loader(well, selected_split):
        root = Path(data_root) / selected_split
        return (
            pd.read_csv(root / f"{well}__horizontal_well.csv"),
            pd.read_csv(root / f"{well}__typewell.csv"),
        )

    run_records = []
    reference_signature = None
    for thread_count in thread_counts:
        for rerun in range(int(reruns)):
            bank, parallel_report = materialize_shared_likpf_bank(
                wells,
                split,
                loader,
                n_jobs=int(thread_count),
                particles=int(particles),
                seeds=int(seeds),
            )
            for well in wells:
                shared_likpf_sp45_adapter(bank[well])
            shared_likpf_exp413_adapter(bank, wells)
            manifest = finalize_shared_likpf_manifest(bank, wells)
            signature = {
                "aggregate_content_sha256": manifest["aggregate_content_sha256"],
                "branch_summary_sha256": manifest["branch_summary_sha256"],
                "generation_ledger_sha256": manifest["generation_ledger_sha256"],
            }
            if reference_signature is None:
                reference_signature = signature
            elif signature != reference_signature:
                raise RuntimeError(
                    "Stage A thread/rerun shared likelihood-PF content parity failed"
                )
            run_records.append(
                {
                    "thread_count": int(thread_count),
                    "rerun": int(rerun),
                    "signature": signature,
                    "parallel_report": parallel_report,
                }
            )
    return {
        "stage": "fixed32_technical_determinism",
        "truth_read": False,
        "source_sha256": EXP073_REPLAY_SOURCE_SHA256,
        "selection_sha256": selection_sha,
        "selection": selection,
        "wells": wells,
        "particles": int(particles),
        "seeds": int(seeds),
        "thread_counts": [int(value) for value in thread_counts],
        "reruns": int(reruns),
        "runs": run_records,
        "exp413_scale5_source_contract": "source_identical_exp073_x1p0_stable_seed",
        "all_and_gate_passed": True,
    }


def _warm_shared_likpf_kernel():
    started = time.time()
    md = np.linspace(1.0, 12.0, 12)
    zeros = np.zeros(12)
    gr = np.full(12, 50.0)
    grid = np.linspace(45.0, 55.0, 64)
    _shared_pf_lik_allseeds(
        md,
        zeros,
        gr,
        grid,
        45.0,
        0.2,
        20.0,
        50.0,
        0.0,
        16,
        4,
        1,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )
    return round(time.time() - started, 6)


SHARED_LIKPF_JIT_WARMUP_SECONDS = _warm_shared_likpf_kernel()
print("shared exp413 likelihood-PF kernel compiled:", SHARED_LIKPF_JIT_WARMUP_SECONDS)

# %% [markdown]
# ## 3. Fixed32 execution and reproducibility outputs

# %%
STAGE_A_STARTED_AT = time.time()
stage_a_report = run_shared_likpf_stage_a(
    COMPETITION_DATA_ROOT,
    split="train",
    count=32,
    thread_counts=(1, 4),
    reruns=2,
    particles=SHARED_LIKPF_PARTICLES,
    seeds=SHARED_LIKPF_SEEDS,
)
stage_a_report["experiment"] = EXPERIMENT_NAME
stage_a_report["route"] = "ensemble"
stage_a_report["elapsed_seconds"] = round(time.time() - STAGE_A_STARTED_AT, 6)
stage_a_report["new_booster_training_count"] = 0
stage_a_report["parent_control_retraining_count"] = 0
stage_a_report["submission_file_generated"] = False
stage_a_report["external_submission_performed"] = False
stage_a_report["source_identity"] = {
    "parent_exp512_candidate_sha256": EXP512_PARENT_SOURCE_SHA256,
    "exp073_replay_source_sha256": EXP073_REPLAY_SOURCE_SHA256,
    "generator_sha256": EXP514_GENERATOR_SHA256,
}

WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = WORKING_DIR / "exp514_stage_a_fixed32_report.json"
METRICS_PATH = WORKING_DIR / "metrics.json"
payload = json.dumps(stage_a_report, indent=2, sort_keys=True, default=str) + "\n"
REPORT_PATH.write_text(payload, encoding="utf-8")
METRICS_PATH.write_text(payload, encoding="utf-8")
print("Stage A report:", REPORT_PATH)
print("submission.csv generated: False")
print("external submission performed: False")
display(pd.DataFrame(stage_a_report["selection"]).head(32))
display(pd.DataFrame(stage_a_report["runs"]))
display(stage_a_report)
