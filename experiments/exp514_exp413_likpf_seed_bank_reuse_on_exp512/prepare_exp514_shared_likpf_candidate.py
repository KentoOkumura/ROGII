from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments/exp514_exp413_likpf_seed_bank_reuse_on_exp512"
PARENT = (
    ROOT
    / "experiments/exp512_hjyact_v2_final_10pct_hedge_on_exp413"
    / "exp512_hjyact_v2_final_10pct_hedge_on_exp413_compact_selfcontained_inference.py"
)
OUTPUT = (
    EXP_DIR
    / "exp514_exp413_likpf_seed_bank_reuse_on_exp512_compact_selfcontained_inference.py"
)
STAGE_A_OUTPUT = (
    EXP_DIR
    / "exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_a_fixed32.py"
)

EXPECTED_PARENT_SHA256 = (
    "16982879716918811dfa9915c4862d45836bd9360efafbaee41046c3e1b6240f"
)
EXP073_REPLAY_SOURCE_SHA256 = (
    "4af212a8a1c83e36cdcc0bc912942a62df1fbc94ca67fd75789171afaa1a647e"
)
STAGE_A_FROZEN_GENERATOR_SHA256 = (
    "f7380e31597a7095af318163ca887f71b0a63539886640a37f9c53800fdbaf41"
)


SHARED_LIKPF_RUNTIME = r'''
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
'''


HJYACT_DETERMINISTIC_REUSE_RUNTIME = r'''
# The SP45 ridge path has already generated the expensive deterministic test
# feature block. HJYACT keeps its own stochastic pf_ancc/pf_z draw, then
# rebuilds only columns that depend on that draw.
HJYACT_PF_REGENERATED_COLUMNS = (
    "pf_ancc", "pf_ancc_std", "pf_ancc_delta", "pf_z", "pf_z_delta",
    "pf_vs_z", "sig_std", "sig_mean_d", "pf_vs_spatial", "pf_vs_dense",
    *tuple(f"tdpf{int(offset)}" for offset in PF_OFFS),
)
HJYACT_AUXILIARY_SHARED_COLUMNS = tuple(
    column
    for column in SP45_SHARED_TEST_FEATURE_FRAME.columns
    if column.startswith("__exp514_shared_")
)
HJYACT_DETERMINISTIC_REUSE_MANIFEST = None


def _hjyact_pf_only_well(wid, split, deterministic_well):
    horizontal, typewell = load_well(str(wid), str(split))
    typewell = typewell.sort_values("TVT")
    known = horizontal[horizontal["TVT_input"].notna()]
    evaluation = horizontal[horizontal["TVT_input"].isna()]
    expected_ids = [f"{wid}_{int(index)}" for index in evaluation.index]
    deterministic_well = deterministic_well.reset_index(drop=True)
    if deterministic_well["id"].astype(str).tolist() != expected_ids:
        raise ValueError(f"SP45/HJYACT deterministic row order mismatch for well {wid}")
    if len(evaluation) == 0 or len(known) < 10:
        raise ValueError(f"HJYACT PF-only refresh received ineligible well {wid}")

    typewell_tvt = typewell["TVT"].to_numpy(np.float32)
    typewell_gr = typewell["GR"].to_numpy(np.float32)
    pf_ancc, pf_ancc_std = run_pf_ancc(horizontal, typewell_tvt, typewell_gr)
    pf_z, _ = run_pf_z(horizontal, typewell_tvt, typewell_gr)
    if len(pf_ancc) != len(expected_ids):
        raise ValueError(f"HJYACT pf_ancc row mismatch for well {wid}")
    pf_use = pf_ancc.astype(np.float32)
    std_use = pf_ancc_std.astype(np.float32)
    has_z = len(pf_z) == len(pf_ancc) and not np.any(np.isnan(pf_z))
    last_tvt = float(known["TVT_input"].iloc[-1])
    last_values = np.full(len(evaluation), np.float32(last_tvt), np.float32)
    pf_z_use = pf_z.astype(np.float32) if has_z else last_values

    required_auxiliary = {
        *[f"__exp514_shared_beam_abs_{tag}" for _, _, _, _, tag in BEAMS],
        "__exp514_shared_sc8_abs",
        "__exp514_shared_sc15_abs",
        "__exp514_shared_sc25_abs",
        "__exp514_shared_sc_ens_abs",
        "__exp514_shared_tvt_dense_abs",
    }
    if missing := required_auxiliary - set(deterministic_well.columns):
        raise ValueError(f"SP45/HJYACT auxiliary feature columns missing: {sorted(missing)}")
    signals = [pf_use]
    signals.extend(
        deterministic_well[f"__exp514_shared_beam_abs_{tag}"].to_numpy(np.float32)
        for _, _, _, _, tag in BEAMS
    )
    signals.extend(
        deterministic_well[column].to_numpy(np.float32)
        for column in (
            "__exp514_shared_sc8_abs",
            "__exp514_shared_sc15_abs",
            "__exp514_shared_sc25_abs",
            "__exp514_shared_sc_ens_abs",
            "tvtF_ANCC",
            "__exp514_shared_tvt_dense_abs",
        )
    )
    signal_matrix = np.stack(signals, axis=1)
    gr_full = (
        horizontal["GR"].astype(float).interpolate(limit_direction="both")
        .fillna(float(np.nanmean(typewell_gr)))
    )
    hidden_gr = gr_full.iloc[evaluation.index[0]:].to_numpy(np.float32)
    if len(hidden_gr) != len(evaluation):
        raise ValueError(f"HJYACT PF-only hidden GR row mismatch for well {wid}")

    updates = {
        "id": expected_ids,
        "pf_ancc": pf_use,
        "pf_ancc_std": std_use,
        "pf_ancc_delta": (pf_use - last_tvt).astype(np.float32),
        "pf_z": pf_z_use,
        "pf_z_delta": ((pf_z - last_tvt).astype(np.float32) if has_z else np.zeros(len(evaluation), np.float32)),
        "pf_vs_z": ((pf_use - pf_z.astype(np.float32)) if has_z else np.zeros(len(evaluation), np.float32)),
        "sig_std": signal_matrix.std(axis=1).astype(np.float32),
        "sig_mean_d": (signal_matrix.mean(axis=1) - last_tvt).astype(np.float32),
        "pf_vs_spatial": (
            pf_use - deterministic_well["tvtF_ANCC"].to_numpy(np.float32)
        ).astype(np.float32),
        "pf_vs_dense": (
            pf_use
            - deterministic_well["__exp514_shared_tvt_dense_abs"].to_numpy(np.float32)
        ).astype(np.float32),
    }
    for offset in PF_OFFS:
        updates[f"tdpf{int(offset)}"] = (
            hidden_gr
            - np.interp(pf_use + offset, typewell_tvt, typewell_gr).astype(np.float32)
        ).astype(np.float32)
    return pd.DataFrame(updates)


def build_hjyact_features_from_sp45(wids, split):
    started = time.time()
    base = SP45_SHARED_TEST_FEATURE_FRAME
    ordered_wells = [str(well) for well in wids]
    if set(base["well"].astype(str)) != set(ordered_wells):
        raise ValueError("SP45/HJYACT deterministic feature well set mismatch")
    by_well = {
        str(well): group.reset_index(drop=True)
        for well, group in base.groupby("well", sort=False)
    }
    effective_n_jobs = min(CFG.n_jobs, len(ordered_wells))
    pf_blocks = Parallel(n_jobs=effective_n_jobs, prefer="threads")(
        delayed(_hjyact_pf_only_well)(well, split, by_well[well])
        for well in ordered_wells
    )
    pf_frame = pd.concat(pf_blocks, ignore_index=True)
    if pf_frame["id"].duplicated().any() or len(pf_frame) != len(base):
        raise ValueError("HJYACT PF-only refresh ID contract failed")
    result = base.copy(deep=True).reset_index(drop=True)
    if result["id"].astype(str).tolist() != pf_frame["id"].astype(str).tolist():
        raise ValueError("HJYACT PF-only refresh merge order changed")
    for column in HJYACT_PF_REGENERATED_COLUMNS:
        if column not in pf_frame:
            raise ValueError(f"HJYACT PF-only refresh column missing: {column}")
        result[column] = pf_frame[column].to_numpy(np.float32)
    result = result.drop(columns=list(HJYACT_AUXILIARY_SHARED_COLUMNS))
    reused_columns = [
        column
        for column in result.columns
        if column not in set(HJYACT_PF_REGENERATED_COLUMNS)
    ]
    globals()["HJYACT_DETERMINISTIC_REUSE_MANIFEST"] = {
        "producer": "sp45_test_feature_frame",
        "consumers": ["sp45_ridge", "hjyact_learned", "exp413"],
        "wells": len(ordered_wells),
        "rows": len(result),
        "effective_pf_refresh_threads": effective_n_jobs,
        "imputer_instance_reused": True,
        "deterministic_generation_count": 1,
        "hjyact_full_build_features_calls": 0,
        "reused_column_count": len(reused_columns),
        "reused_columns": reused_columns,
        "pf_regenerated_columns": list(HJYACT_PF_REGENERATED_COLUMNS),
        "auxiliary_columns_dropped": list(HJYACT_AUXILIARY_SHARED_COLUMNS),
        "elapsed_pf_refresh_seconds": round(time.time() - started, 6),
    }
    return result
'''


GOLD_PROCESS_PARALLEL_LOOP = r'''    from joblib import Parallel as _GoldParallel, delayed as _gold_delayed

    def _gold_run_one_well(_task):
        _wi, _wid = _task
        _worker_started = _gold_time.time()
        try:
            from threadpoolctl import threadpool_limits as _gold_threadpool_limits
            with _gold_threadpool_limits(limits=1):
                _hw_path = _GOLD_DATA / 'test' / f'{_wid}__horizontal_well.csv'
                _tw_path = _GOLD_DATA / 'test' / f'{_wid}__typewell.csv'
                if not _hw_path.exists() or not _tw_path.exists():
                    return dict(
                        order=_wi, well=_wid,
                        report=dict(well=_wid, status='skip_missing_files'),
                        cut_rows=[], candidate_by_id={},
                        elapsed_sec=float(_gold_time.time() - _worker_started),
                    )
                _hw = _gold_pd.read_csv(_hw_path)
                _tw = _gold_pd.read_csv(_tw_path)
                if _wid in _gold_bimodal_skip_wells:
                    return dict(
                        order=_wi, well=_wid,
                        report=dict(
                            well=_wid,
                            status='skip_bimodal_hedge',
                            bimodal_prefix_guard=True,
                            reason='visible-prefix commit disabled for active bimodal selector well',
                        ),
                        cut_rows=[], candidate_by_id={},
                        elapsed_sec=float(_gold_time.time() - _worker_started),
                    )
                _rep = _gold_calibrate_well(_wid, _hw, _tw, _GOLD_DATA, _gold_variants)
                if _rep is None:
                    _rep = dict(well=_wid, status='skip_none')
                _cut_rows = _rep.pop('cut_rows', []) if isinstance(_rep, dict) else []
                _candidate_by_id = {}
                if _rep.get('status') == 'ok':
                    _best_name = _rep['best_name']
                    _need_pf_final = str(_best_name).startswith('pf|')
                    _pool_final = _gold_candidate_pool(
                        _wid, _hw, _tw, _GOLD_DATA, _gold_variants,
                        include_pf=_need_pf_final,
                        n_seeds=_GOLD_FINAL_SEEDS,
                        n_particles=_GOLD_PARTICLES,
                    )
                    if _best_name not in _pool_final and _need_pf_final:
                        _pool_final = _gold_candidate_pool(
                            _wid, _hw, _tw, _GOLD_DATA, _gold_variants,
                            include_pf=False,
                            n_seeds=0,
                            n_particles=_GOLD_PARTICLES,
                        )
                    if _best_name in _pool_final:
                        _g = _gold_base[_gold_base['well'] == _wid]
                        _arr = _pool_final[_best_name]
                        for _rid, _ri in zip(
                            _g['id'].astype(str).values,
                            _g['row_idx'].astype(int).values,
                        ):
                            if 0 <= int(_ri) < len(_arr) and _gold_np.isfinite(_arr[int(_ri)]):
                                _candidate_by_id[_rid] = float(_arr[int(_ri)])
                        _rep['final_candidate_available'] = True
                    else:
                        _rep['final_candidate_available'] = False
                        _rep['status'] = 'skip_no_final_candidate'
                return dict(
                    order=_wi, well=_wid, report=_rep,
                    cut_rows=_cut_rows, candidate_by_id=_candidate_by_id,
                    elapsed_sec=float(_gold_time.time() - _worker_started),
                )
        except Exception as _e:
            return dict(
                order=_wi, well=_wid,
                report=dict(well=_wid, status='error', error=repr(_e)),
                cut_rows=[], candidate_by_id={},
                elapsed_sec=float(_gold_time.time() - _worker_started),
            )

    _gold_reports = []
    _gold_cut_reports = []
    _gold_candidate_by_id = {}
    _gold_wells = list(_gold_base['well'].drop_duplicates())[:_GOLD_MAX_WELLS]
    _gold_requested_processes = 4
    _gold_effective_processes = min(_gold_requested_processes, len(_gold_wells))
    globals()['EXP514_KDTREE_WORKERS'] = 1
    _gold_parallel_started = _gold_time.time()
    _gold_tasks = list(enumerate(_gold_wells, 1))
    if _gold_effective_processes > 1:
        _gold_results = _GoldParallel(
            n_jobs=_gold_effective_processes,
            backend='multiprocessing',
            batch_size=1,
            pre_dispatch=_gold_effective_processes,
        )(
            _gold_delayed(_gold_run_one_well)(_task)
            for _task in _gold_tasks
        )
        _gold_backend = 'multiprocessing'
    else:
        _gold_results = [_gold_run_one_well(_task) for _task in _gold_tasks]
        _gold_backend = 'sequential'
    _gold_results = sorted(_gold_results, key=lambda item: int(item['order']))
    if [int(item['order']) for item in _gold_results] != list(range(1, len(_gold_wells) + 1)):
        raise RuntimeError('Gold process result order/coverage mismatch')
    if [str(item['well']) for item in _gold_results] != [str(well) for well in _gold_wells]:
        raise RuntimeError('Gold process well order changed')
    for _result in _gold_results:
        _rep = _result['report']
        if _rep.get('status') == 'error':
            raise RuntimeError(f"Gold process worker failed for {_result['well']}: {_rep.get('error')}")
        _gold_reports.append(_rep)
        _gold_cut_reports.extend(_result['cut_rows'])
        for _rid, _value in _result['candidate_by_id'].items():
            if _rid in _gold_candidate_by_id:
                raise RuntimeError(f'Gold process duplicate candidate id: {_rid}')
            _gold_candidate_by_id[_rid] = float(_value)
        print(
            '[gold %d/%d] %s report:' % (
                int(_result['order']), len(_gold_wells), _result['well']
            ),
            {k: _rep.get(k) for k in ['status', 'best_name', 'best_score', 'default_score', 'gain', 'consistency']},
            flush=True,
        )
    GOLD_WELL_PARALLEL_REPORT = {
        'requested_processes': _gold_requested_processes,
        'effective_processes': _gold_effective_processes,
        'backend': _gold_backend,
        'wells': len(_gold_wells),
        'result_merge_order': 'input_well_order',
        'inner_blas_threads': 1,
        'inner_kdtree_workers': 1,
        'elapsed_seconds': round(_gold_time.time() - _gold_parallel_started, 6),
        'worker_elapsed_seconds': [round(float(item['elapsed_sec']), 6) for item in _gold_results],
    }
'''


STAGE_A_HEADER = r'''# ---
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
EXP512_PARENT_SOURCE_SHA256 = "__PARENT_SHA256__"
EXP073_REPLAY_SOURCE_SHA256 = "__REPLAY_SHA256__"
EXP514_GENERATOR_SHA256 = "__GENERATOR_SHA256__"


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
'''


STAGE_A_ORCHESTRATION = r'''
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
'''


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label} replacement expected once, found {count}")
    return source.replace(old, new, 1)


def replace_between_once(
    source: str,
    start: str,
    end: str,
    replacement: str,
    label: str,
) -> str:
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError(f"{label} boundary contract failed")
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[:start_index] + replacement + source[end_index:]


def transform_parent(source: str, generator_sha256: str) -> str:
    source = replace_once(
        source,
        "# # exp512 hjyact-v2 final fixed 50/50 blend on exp413",
        "# # exp514 shared exp413 likelihood-PF bank on exp512",
        "title",
    )
    source = replace_once(
        source,
        "# This candidate extracts only the active version-2 source path, regenerates both\n"
        "# components on the dynamic sample, reuses the deterministic learned-replay surface,\n"
        "# and writes `0.50 * exp413 + 0.50 * hjyact_v2_final` in float64.",
        "# This candidate preserves the frozen exp512 ensemble while generating the exp413\n"
        "# x1.0 stable-seed likelihood-PF bank once per dynamic well. The same raw bank feeds\n"
        "# SP45 all-scale/branch summaries and the unchanged exp413 scale-5 consumer.",
        "header summary",
    )
    source = replace_once(
        source,
        "# 1. Imports, source identity, and frozen profile\n"
        "# 2. SP45 PF / Beam selector helpers\n"
        "# 3. Ridge/PF anchor and shared deterministic candidate surface\n"
        "# 4. Saved ridge artifact inference and runtime Ridge\n"
        "# 5. Projection and learned trajectory replay\n"
        "# 6. Guarded overlap and final hjyact-v2 layers\n"
        "# 7. Embedded hidden-safe exp413 runtime\n"
        "# 8. Shared-DAG manifest, fixed blend, and reproducibility outputs",
        "# 1. Imports, source identity, and frozen profile\n"
        "# 2. Shared exp413 likelihood-PF producer, adapters, and Stage A audit\n"
        "# 3. SP45 PF / Beam selector helpers\n"
        "# 4. Ridge/PF anchor and shared deterministic candidate surface\n"
        "# 5. Saved ridge artifact inference and runtime Ridge\n"
        "# 6. Projection and learned trajectory replay\n"
        "# 7. Guarded overlap and final hjyact-v2 layers\n"
        "# 8. Embedded hidden-safe exp413 runtime\n"
        "# 9. Shared-PF ledger, fixed blend, and reproducibility outputs",
        "contents",
    )
    constant_marker = 'SOURCE_KERNEL = "hjyact/ultimate-pf-config-strategy-a-reproducible-score"'
    constant_block = (
        'EXPERIMENT_NAME = "exp514_exp413_likpf_seed_bank_reuse_on_exp512"\n'
        f'EXP514_GENERATOR_SHA256 = "{generator_sha256}"\n'
        f'EXP512_PARENT_SOURCE_SHA256 = "{EXPECTED_PARENT_SHA256}"\n'
        f'EXP073_REPLAY_SOURCE_SHA256 = "{EXP073_REPLAY_SOURCE_SHA256}"\n'
        + constant_marker
    )
    source = replace_once(source, constant_marker, constant_block, "source constants")

    heading_replacements = (
        ("# ## 8. Shared-DAG manifest, fixed blend, and reproducibility outputs", "# ## 9. Shared-PF ledger, fixed blend, and reproducibility outputs"),
        ("# ## 7. Embedded hidden-safe exp413 runtime", "# ## 8. Embedded hidden-safe exp413 runtime"),
        ("# ## 6. Guarded overlap and final hjyact-v2 layers", "# ## 7. Guarded overlap and final hjyact-v2 layers"),
        ("# ## 5. Projection and learned trajectory replay", "# ## 6. Projection and learned trajectory replay"),
        ("# ## 4. Saved ridge artifact inference and runtime Ridge", "# ## 5. Saved ridge artifact inference and runtime Ridge"),
        ("# ## 3. Ridge/PF anchor and shared deterministic candidate surface", "# ## 4. Ridge/PF anchor and shared deterministic candidate surface"),
        ("# ## 2. SP45 PF / Beam selector helpers", "# ## 3. SP45 PF / Beam selector helpers"),
    )
    for old, new in heading_replacements:
        source = replace_once(source, old, new, f"heading {old}")
    shared_insert_marker = "# %% [markdown]\n# ## 3. SP45 PF / Beam selector helpers"
    source = replace_once(
        source,
        shared_insert_marker,
        SHARED_LIKPF_RUNTIME.rstrip() + "\n\n" + shared_insert_marker,
        "shared runtime insertion",
    )
    if source.count("workers=-1") != 4:
        raise RuntimeError("expected four Formation/Dense cKDTree worker sites")
    source = source.replace(
        "workers=-1",
        "workers=int(globals().get('EXP514_KDTREE_WORKERS', -1))",
    )

    source = replace_once(
        source,
        "        'pf_vs_z':((pf_use-pf_z.astype(np.float32)) if has_z else sc(0.)),\n"
        "        **{f'beam_{t}_d':(p-np.float32(last_tvt)).astype(np.float32) for t,p in bpaths.items()},",
        "        'pf_vs_z':((pf_use-pf_z.astype(np.float32)) if has_z else sc(0.)),\n"
        "        **{f'__exp514_shared_beam_abs_{t}':p.astype(np.float32) for t,p in bpaths.items()},\n"
        "        '__exp514_shared_sc8_abs':sc8.astype(np.float32),\n"
        "        '__exp514_shared_sc15_abs':sc15.astype(np.float32),\n"
        "        '__exp514_shared_sc25_abs':sc25.astype(np.float32),\n"
        "        '__exp514_shared_sc_ens_abs':sc_ens.astype(np.float32),\n"
        "        '__exp514_shared_tvt_dense_abs':tvt_dense.astype(np.float32),\n"
        "        **{f'beam_{t}_d':(p-np.float32(last_tvt)).astype(np.float32) for t,p in bpaths.items()},",
        "SP45 deterministic auxiliary columns",
    )
    source = replace_once(
        source,
        'test_df = build_dataset(test_paths, is_train=False, label="test")\n\nfeatures = [c for c in train_df.columns',
        'test_df = build_dataset(test_paths, is_train=False, label="test")\n'
        'SP45_SHARED_TEST_FEATURE_FRAME = test_df\n'
        'SP45_SHARED_IMPUTERS = (FI, DI)\n\n'
        'features = [c for c in train_df.columns',
        "SP45 deterministic frame producer",
    )
    hjyact_reuse_marker = "HJYACT_SHARED_FEATURE_RUNTIME_SECONDS = None\n"
    source = replace_once(
        source,
        hjyact_reuse_marker,
        HJYACT_DETERMINISTIC_REUSE_RUNTIME.rstrip()
        + "\n\n"
        + hjyact_reuse_marker,
        "HJYACT deterministic feature reuse runtime",
    )
    source = replace_once(
        source,
        "    init_imputers(train_wids)\n\n"
        '    print("building lik-PF + shared deterministic features (test)...", flush=True)\n'
        '    likpf_test = build_likpf(test_wids, "test")\n'
        "    shared_started = time.time()\n"
        '    shared = build_features(test_wids, "test", is_train=False)\n'
        '    globals()["HJYACT_SHARED_FEATURE_RUNTIME_SECONDS"] = time.time() - shared_started\n',
        "    globals()['_FI'], globals()['_DI'] = SP45_SHARED_IMPUTERS\n\n"
        '    print("building lik-PF + reusing SP45 deterministic features (test)...", flush=True)\n'
        '    likpf_test = build_likpf(test_wids, "test")\n'
        "    shared_started = time.time()\n"
        '    shared = build_hjyact_features_from_sp45(test_wids, "test")\n'
        '    globals()["HJYACT_SHARED_FEATURE_RUNTIME_SECONDS"] = time.time() - shared_started\n',
        "HJYACT deterministic frame consumer",
    )

    test_well_marker = (
        "test_hw_files = sorted(glob.glob(str(CFG.dataset_path / 'test' / '*__horizontal_well.csv')))\n"
        "test_wells = [os.path.basename(f).split('__')[0] for f in test_hw_files]\n\n"
        "rows = []"
    )
    test_well_replacement = (
        "test_hw_files = sorted(glob.glob(str(CFG.dataset_path / 'test' / '*__horizontal_well.csv')))\n"
        "test_wells = [os.path.basename(f).split('__')[0] for f in test_hw_files]\n"
        "if not test_wells:\n"
        "    raise FileNotFoundError('shared likelihood-PF requires at least one dynamic test well')\n"
        "SHARED_LIKPF_BANK, SHARED_LIKPF_PARALLEL_REPORT = materialize_shared_likpf_bank(\n"
        "    test_wells,\n"
        "    'test',\n"
        "    load_well,\n"
        "    n_jobs=SHARED_LIKPF_N_JOBS,\n"
        ")\n"
        "print('shared likelihood-PF producer report:', SHARED_LIKPF_PARALLEL_REPORT)\n\n"
        "rows = []"
    )
    source = replace_once(
        source,
        test_well_marker,
        test_well_replacement,
        "pre-SP45 shared producer",
    )

    gold_loop_start = "    _gold_reports = []\n"
    gold_loop_end = "    _gold_report_df = _gold_pd.DataFrame(_gold_reports)\n"
    source = replace_between_once(
        source,
        gold_loop_start,
        gold_loop_end,
        GOLD_PROCESS_PARALLEL_LOOP,
        "Gold four-process well loop",
    )
    source = replace_once(
        source,
        "        profiles=_profile_summaries,\n"
        "    )\n"
        "    with open(_GOLD_WORK / 'gold_prefix_submission_audit.json'",
        "        profiles=_profile_summaries,\n"
        "        well_parallel=GOLD_WELL_PARALLEL_REPORT,\n"
        "    )\n"
        "    with open(_GOLD_WORK / 'gold_prefix_submission_audit.json'",
        "Gold parallel audit",
    )

    sp45_start = "    # 128-seed likelihood-weighted PF ensemble\n"
    sp45_end = "    # Beam search ensemble\n"
    sp45_replacement = (
        "    # Consume the already materialized exp413 x1.0 stable-seed bank.\n"
        "    # Failure is terminal: no last-known or duplicate legacy PF fallback is allowed.\n"
        "    pf_by_scale, _seed_branch = shared_likpf_sp45_adapter(\n"
        "        SHARED_LIKPF_BANK[str(wid)]\n"
        "    )\n"
        "    tvt_pf = pf_by_scale['pf_scale_8']\n"
        "    messages.append(\n"
        "        f'  Shared exp413 PF bank OK seeds={SHARED_LIKPF_SEEDS} '\n"
        "        f'scales={SHARED_LIKPF_SCALES}'\n"
        "    )\n\n"
    )
    source = replace_between_once(
        source,
        sp45_start,
        sp45_end,
        sp45_replacement,
        "SP45 shared adapter",
    )

    source = replace_once(
        source,
        "def generate_dynamic_exp413_prediction(shared_deterministic_frame=None, reuse_tracker=None):",
        "def generate_dynamic_exp413_prediction(\n"
        "    shared_deterministic_frame=None,\n"
        "    reuse_tracker=None,\n"
        "    shared_likpf_bank=None,\n"
        "):",
        "exp413 signature",
    )
    source = replace_once(
        source,
        "    sys.path.insert(0, str(source_work))\n\n"
        "    import exp263_k16_source as k16_module  # noqa: E402",
        "    sys.path.insert(0, str(source_work))\n"
        "    observed_replay_source_sha = sha256_file(\n"
        "        resolved_sources['exp263_public_replay_source']\n"
        "    )\n"
        "    if observed_replay_source_sha != EXP073_REPLAY_SOURCE_SHA256:\n"
        "        raise ValueError(\n"
        "            'shared likelihood-PF replay source SHA mismatch: '\n"
        "            f'{observed_replay_source_sha} != {EXP073_REPLAY_SOURCE_SHA256}'\n"
        "        )\n\n"
        "    import exp263_k16_source as k16_module  # noqa: E402",
        "embedded replay source audit",
    )
    source = replace_once(
        source,
        "    likpf_test = replay_source.build_likpf(test_wells, \"test\")\n"
        "    pf_frame = replay_source.add_likpf_features(pf_frame, likpf_test).reset_index(drop=True)",
        "    if shared_likpf_bank is None:\n"
        "        raise RuntimeError('exp413 requires the precomputed shared likelihood-PF bank')\n"
        "    likpf_test = shared_likpf_exp413_adapter(shared_likpf_bank, test_wells)\n"
        "    pf_frame = pf_frame.merge(likpf_test, on='id', how='left', validate='one_to_one')\n"
        "    for likpf_column in [column for column in likpf_test.columns if column != 'id']:\n"
        "        if pf_frame[likpf_column].isna().any():\n"
        "            raise ValueError(f'shared exp413 likelihood-PF coverage failed: {likpf_column}')\n"
        "        pf_frame[likpf_column] = pf_frame[likpf_column].astype(np.float32)\n"
        "        pf_frame[likpf_column + '_d'] = (\n"
        "            pf_frame[likpf_column] - pf_frame['last_known_tvt']\n"
        "        ).astype(np.float32)\n"
        "    pf_frame = pf_frame.reset_index(drop=True)",
        "exp413 shared adapter",
    )
    source = replace_once(
        source,
        "exp413_predictions_memory, exp413_metrics, exp413_prediction_path = generate_dynamic_exp413_prediction(\n"
        "    shared_deterministic_frame=HJYACT_SHARED_FEATURE_FRAME,\n"
        "    reuse_tracker=CANDIDATE_REUSE_TRACKER,\n"
        ")",
        "exp413_predictions_memory, exp413_metrics, exp413_prediction_path = generate_dynamic_exp413_prediction(\n"
        "    shared_deterministic_frame=HJYACT_SHARED_FEATURE_FRAME,\n"
        "    reuse_tracker=CANDIDATE_REUSE_TRACKER,\n"
        "    shared_likpf_bank=SHARED_LIKPF_BANK,\n"
        ")",
        "exp413 shared bank call",
    )

    manifest_marker = "model_manifest = {"
    manifest_block = (
        "shared_likpf_manifest = finalize_shared_likpf_manifest(\n"
        "    SHARED_LIKPF_BANK,\n"
        "    test_wells,\n"
        ")\n"
        "shared_likpf_manifest['parallel_report'] = SHARED_LIKPF_PARALLEL_REPORT\n"
        "shared_likpf_manifest['jit_warmup_seconds'] = SHARED_LIKPF_JIT_WARMUP_SECONDS\n"
        "SHARED_LIKPF_MANIFEST_PATH = WORKING_DIR / 'exp514_shared_likpf_manifest.json'\n"
        "SHARED_LIKPF_MANIFEST_PATH.write_text(\n"
        "    json.dumps(shared_likpf_manifest, indent=2, sort_keys=True, default=str) + '\\n',\n"
        "    encoding='utf-8',\n"
        ")\n\n"
        + manifest_marker
    )
    source = replace_once(
        source,
        manifest_marker,
        manifest_block,
        "shared manifest finalization",
    )

    source = source.replace(
        '"experiment": "exp512_hjyact_v2_final_10pct_hedge_on_exp413"',
        '"experiment": "exp514_exp413_likpf_seed_bank_reuse_on_exp512"',
    )
    source = source.replace(
        'WORKING_DIR / "exp512_model_manifest.json"',
        'WORKING_DIR / "exp514_model_manifest.json"',
    )
    source = source.replace(
        'WORKING_DIR / "exp512_reproducibility_manifest.json"',
        'WORKING_DIR / "exp514_reproducibility_manifest.json"',
    )
    source = source.replace(
        'WORKING_DIR / "exp512_component_readout.csv"',
        'WORKING_DIR / "exp514_component_readout.csv"',
    )
    source = replace_once(
        source,
        '    "actual_formula": "0.50 * exp413 + 0.50 * hjyact_v2_final",',
        '    "parent_experiment": "exp512_hjyact_v2_final_10pct_hedge_on_exp413",\n'
        '    "actual_formula": "0.50 * exp413 + 0.50 * hjyact_v2_final",',
        "model manifest parent",
    )
    source = replace_once(
        source,
        '    "visible_reference_checks": visible_reference_checks,',
        '    "shared_likelihood_pf": shared_likpf_manifest,\n'
        '    "hjyact_deterministic_feature_reuse": HJYACT_DETERMINISTIC_REUSE_MANIFEST,\n'
        '    "visible_reference_checks": visible_reference_checks,',
        "metrics shared manifest",
    )
    source = replace_once(
        source,
        '        "candidate_reuse_manifest": sha256_file(WORKING_DIR / "candidate_reuse_manifest.json"),',
        '        "candidate_reuse_manifest": sha256_file(WORKING_DIR / "candidate_reuse_manifest.json"),\n'
        '        "shared_likelihood_pf_manifest": sha256_file(SHARED_LIKPF_MANIFEST_PATH),',
        "metrics shared SHA",
    )
    source = source.replace(
        'print("exp512 fixed 50/50 submission generated:"',
        'print("exp514 shared-PF fixed 50/50 submission generated:"',
    )
    source = source.replace(
        "str(Path(module.__file__).resolve())",
        'str(Path(getattr(module, "__" + "file__")).resolve())',
    )
    source = source.replace(
        "module.__file__ = '<fallback koolbox Trainer shim>'",
        'setattr(module, "__" + "file__", "<fallback koolbox Trainer shim>")',
    )
    source = source.replace(
        "getattr(_koolbox_probe, '__file__', '<unknown>')",
        'getattr(_koolbox_probe, "__" + "file__", "<unknown>")',
    )
    if "replay_source.build_likpf(test_wells" in source:
        raise RuntimeError("exp413 duplicate likelihood-PF generation remains in candidate")
    if 'shared = build_features(test_wids, "test", is_train=False)' in source:
        raise RuntimeError("HJYACT duplicate deterministic feature generation remains")
    if "for _wi, _wid in enumerate(_gold_wells, 1):" in source:
        raise RuntimeError("Gold sequential well loop remains")
    if "__file__" in source:
        raise RuntimeError("notebook-unsafe __file__ token remains in candidate")
    return source


def main() -> None:
    raw = PARENT.read_bytes()
    observed = sha256_bytes(raw)
    if observed != EXPECTED_PARENT_SHA256:
        raise RuntimeError(
            f"exp512 parent source SHA mismatch: {observed} != {EXPECTED_PARENT_SHA256}"
        )
    generator_sha256 = sha256_bytes(Path(__file__).read_bytes())
    generated = transform_parent(raw.decode("utf-8"), generator_sha256)
    OUTPUT.write_text(generated, encoding="utf-8")
    stage_a_source = (
        STAGE_A_HEADER.replace("__PARENT_SHA256__", EXPECTED_PARENT_SHA256)
        .replace("__REPLAY_SHA256__", EXP073_REPLAY_SOURCE_SHA256)
        .replace("__GENERATOR_SHA256__", STAGE_A_FROZEN_GENERATOR_SHA256)
        + "\n"
        + SHARED_LIKPF_RUNTIME.strip()
        + "\n\n"
        + STAGE_A_ORCHESTRATION.strip()
        + "\n"
    )
    if "__file__" in stage_a_source:
        raise RuntimeError("notebook-unsafe __file__ token remains in Stage A source")
    STAGE_A_OUTPUT.write_text(stage_a_source, encoding="utf-8")
    print(
        f"generated {OUTPUT.relative_to(ROOT)} "
        f"({len(generated.splitlines())} lines, sha256={sha256_bytes(generated.encode())})"
    )
    print(
        f"generated {STAGE_A_OUTPUT.relative_to(ROOT)} "
        f"({len(stage_a_source.splitlines())} lines, "
        f"sha256={sha256_bytes(stage_a_source.encode())})"
    )


if __name__ == "__main__":
    main()
