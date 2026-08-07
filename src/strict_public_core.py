from __future__ import annotations

import gc
import gzip
import hashlib
import importlib.util
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

FORMATIONS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
META_COLUMNS = {"id", "well", "target", "outer_fold", "actual_tvt"}
STALE_EXP072_LIKPF_COLUMNS = ("likpf_mean_d",)
LEARNED_LIKPF_ABSOLUTE_COLUMNS = (
    "likpf_scale_3",
    "likpf_scale_5",
    "likpf_scale_8",
    "likpf_scale_12",
    "likpf_mean",
)
LEARNED_LIKPF_COLUMNS = LEARNED_LIKPF_ABSOLUTE_COLUMNS + tuple(
    f"{name}_d" for name in LEARNED_LIKPF_ABSOLUTE_COLUMNS
)
SELECTOR_VARIANTS = (
    "pf_scale_5_hold_0p20",
    "pf_scale_3_hold_0p15",
    "pf_scale_12_beam_0p20_hold_0p15",
    "pf_scale_5_hold_0p15",
    "pf_scale_5_beam_0p05_hold_0p05",
    "pf_scale_12_beam_0p20_hold_0p05",
    "pf_scale_8_hold_0p20",
)
SELECTOR_BEAM_CONFIGS = (
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
)
SPATIAL_COLUMNS = (
    *(
        f"{prefix}_{formation}"
        for formation in FORMATIONS
        for prefix in ("tvtF", "tvtFw", "tvtF50", "bw", "bww", "bw50", "bw_early", "bw_mid")
    ),
    *(f"frm_rmse_{formation}" for formation in FORMATIONS),
    "form_mean_d",
    "form_std_d",
    "form_rng_d",
    "spatial_ancc_d",
    "spatial_knn_dist",
    "dense_ancc",
    "dense_std",
    "dense_dist",
    "tvt_dense_d",
    "tvt_densew_d",
    "tvt_dense50_d",
    "dense_rmse",
    "dense_bias",
    "dense_nb_std",
    "pf_vs_spatial",
    "pf_vs_dense",
    "spatial_vs_dense",
    "beam_vs_spatial",
    "sig_std",
    "sig_mean_d",
)


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prediction_sha256(ids: Sequence[str] | pd.Series, values: np.ndarray, label: str) -> str:
    digest = hashlib.sha256(label.encode("utf-8"))
    for raw_id in pd.Series(ids, dtype=str).to_numpy():
        digest.update(str(raw_id).encode("utf-8"))
        digest.update(b"\0")
    digest.update(np.asarray(values, dtype=np.float32).tobytes())
    return digest.hexdigest()


def stable_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    return value % modulo + 1


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
        + "\n"
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(true - pred))))


def find_artifact(
    filename: str,
    *,
    explicit: str | Path | None = None,
    roots: Iterable[str | Path] = (),
) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for root in roots:
        root_path = Path(root)
        candidates.extend((root_path / filename, root_path / "artifacts" / filename))
    candidates.extend((Path.cwd() / filename, Path.cwd() / "artifacts" / filename))
    if Path("/kaggle/input").exists():
        candidates.extend(Path("/kaggle/input").glob(f"**/{filename}"))
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(
        f"Could not resolve non-empty artifact {filename}; checked={list(seen)[:120]}"
    )


def find_competition_train_dir(explicit: str | Path | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        (
            Path("data/raw/train"),
            Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction/train"),
            Path("/kaggle/input/rogii-wellbore-geology-prediction/train"),
        )
    )
    for candidate in candidates:
        if candidate.is_dir() and next(candidate.glob("*__horizontal_well.csv"), None):
            return candidate
    raise FileNotFoundError(f"Could not resolve raw train directory from {candidates}")


def load_public_replay_module(explicit: str | Path | None = None) -> ModuleType:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(
        Path("experiments/exp072_exp063_full_replay_feature_cache/public_notebook_replay_audit.py")
    )
    if Path("/kaggle/input").exists():
        candidates.extend(Path("/kaggle/input").glob("**/public_notebook_replay_audit.py"))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location("exp497_public_replay_runtime", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        required = (
            "_grid",
            "_pf_lik_allseeds",
            "beam_search",
            "FormationPlaneKNN",
            "DenseANCCImputer",
            "seg_b_well",
        )
        if all(hasattr(module, name) for name in required):
            module.__dict__["_exp497_source_path"] = str(candidate)
            return module
    raise FileNotFoundError("No compatible public_notebook_replay_audit.py was found")


def load_parent_oof(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_rows: int = 3_783_989,
    expected_wells: int = 773,
) -> pd.DataFrame:
    path = Path(path)
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha256:
        raise ValueError(f"Parent OOF SHA mismatch: {observed_sha} != {expected_sha256}")
    columns = [
        "id",
        "well",
        "md_since",
        "last_known_tvt",
        "target",
        "outer_fold",
        "actual_tvt",
        "scale5_x1p0_full_replacement__lgb_mean__pred_tvt",
    ]
    frame = pd.read_parquet(path, columns=columns)
    if len(frame) != expected_rows or frame["well"].nunique() != expected_wells:
        raise ValueError("Parent OOF row/well contract mismatch")
    fold_count = frame.groupby("well", sort=False)["outer_fold"].nunique()
    if not fold_count.eq(1).all() or sorted(frame["outer_fold"].unique()) != list(range(5)):
        raise ValueError("Parent OOF does not provide one outer fold per well")
    return frame


def outer_fold_well_map(parent_oof: pd.DataFrame) -> pd.DataFrame:
    return (
        parent_oof[["well", "outer_fold"]]
        .drop_duplicates()
        .sort_values("well")
        .reset_index(drop=True)
    )


def load_exp072_base_shard(
    cache_path: str | Path,
    wells: set[str],
    *,
    expected_sha256: str,
    expected_base_feature_count: int = 195,
    chunksize: int = 100_000,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    cache_path = Path(cache_path)
    observed_sha = sha256_file(cache_path)
    if observed_sha != expected_sha256:
        raise ValueError(f"exp072 cache SHA mismatch: {observed_sha} != {expected_sha256}")
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        cache_path,
        dtype={"id": str, "well": str},
        chunksize=chunksize,
    ):
        part = chunk[chunk["well"].isin(wells)].copy()
        if len(part):
            parts.append(part)
    if not parts:
        raise ValueError("exp072 shard selection produced zero rows")
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.drop(columns=list(STALE_EXP072_LIKPF_COLUMNS), errors="raise")
    feature_columns = [name for name in frame.columns if name not in {"id", "well", "target"}]
    if len(feature_columns) != expected_base_feature_count:
        raise ValueError(
            f"Expected {expected_base_feature_count} public-core base features, "
            f"got {len(feature_columns)}"
        )
    if frame["well"].nunique() != len(wells):
        missing = sorted(wells - set(frame["well"]))
        raise ValueError(f"exp072 shard is missing wells: {missing[:20]}")
    for column in ["target", *feature_columns]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    if not np.isfinite(frame[["target", *feature_columns]].to_numpy(np.float32)).all():
        raise ValueError("exp072 base shard contains non-finite numeric values")
    return (
        frame,
        feature_columns,
        {
            "path": str(cache_path),
            "sha256": observed_sha,
            "rows": len(frame),
            "wells": frame["well"].nunique(),
            "base_feature_count": len(feature_columns),
        },
    )


def _likelihood_pf_bank(
    runtime: ModuleType,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    gr_sigma_multiplier: float,
    seed_base: int,
    particles: int = 500,
    seeds: int = 128,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    tw = typewell.sort_values("TVT")
    tw_tvt = tw["TVT"].to_numpy(float)
    tw_gr = tw["GR"].fillna(tw["GR"].mean()).to_numpy(float)
    known = horizontal[horizontal["TVT_input"].notna()]
    evaluation = horizontal[horizontal["TVT_input"].isna()]
    if evaluation.empty or known.empty:
        raise ValueError("Likelihood PF requires known prefix and evaluation rows")
    last = known.iloc[-1]
    last_level = float(last["TVT_input"]) + float(last["Z"])
    tw_at_known = np.interp(known["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    gr_sigma = float(
        np.clip(
            np.nanstd(known["GR"].fillna(0).to_numpy(float) - tw_at_known),
            10.0,
            60.0,
        )
        * gr_sigma_multiplier
    )
    tail = known.tail(30)
    dt = np.diff(tail["TVT_input"].to_numpy(float))
    dz = np.diff(tail["Z"].to_numpy(float))
    dm = np.diff(tail["MD"].to_numpy(float))
    valid = dm > 0
    initial_rate = float(np.median((dt + dz)[valid] / dm[valid])) if valid.sum() >= 3 else 0.0
    grid, grid_min, grid_step = runtime._grid(tw_tvt, tw_gr)
    gr_values = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(float(np.nanmean(tw_gr)))
        .to_numpy(float)[evaluation.index]
    )
    predictions, log_likelihoods = runtime._pf_lik_allseeds(
        evaluation["MD"].to_numpy(float),
        evaluation["Z"].to_numpy(float),
        gr_values,
        grid,
        grid_min,
        grid_step,
        gr_sigma,
        last_level,
        initial_rate,
        particles,
        seeds,
        seed_base,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )
    centered = log_likelihoods - np.max(log_likelihoods)
    outputs: dict[str, np.ndarray] = {}
    for scale in (3.0, 5.0, 8.0, 12.0):
        weights = np.exp(centered / scale)
        weights /= weights.sum()
        outputs[f"scale_{scale:g}"] = (weights[:, None] * predictions).sum(0).astype(np.float32)
    outputs["mean"] = predictions.mean(0).astype(np.float32)
    return outputs, evaluation.index.to_numpy(), gr_sigma


def _selector_beam_mean(
    runtime: ModuleType,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    tw = typewell.sort_values("TVT")
    tw_tvt = tw["TVT"].to_numpy(np.float32)
    tw_gr = tw["GR"].fillna(tw["GR"].mean()).to_numpy(np.float32)
    known = horizontal[horizontal["TVT_input"].notna()]
    evaluation = horizontal[horizontal["TVT_input"].isna()]
    last_tvt = float(known.iloc[-1]["TVT_input"])
    full_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(float(np.nanmean(tw_gr)))
        .to_numpy(np.float32)
    )
    eval_gr = full_gr[evaluation.index]
    paths = [
        runtime.beam_search(eval_gr, tw_tvt, tw_gr, last_tvt, bs, mc, es, radius)
        for bs, mc, es, radius in SELECTOR_BEAM_CONFIGS
    ]
    return np.stack(paths, axis=0).mean(axis=0).astype(np.float32), evaluation.index.to_numpy()


def _selector_candidate_frame(
    *,
    well: str,
    row_indices: np.ndarray,
    last_tvt: float,
    selector_bank: Mapping[str, np.ndarray],
    beam_mean: np.ndarray,
) -> pd.DataFrame:
    base = {
        "pf_scale_3": selector_bank["scale_3"],
        "pf_scale_5": selector_bank["scale_5"],
        "pf_scale_8": selector_bank["scale_8"],
        "pf_scale_12": selector_bank["scale_12"],
    }
    result: dict[str, Any] = {
        "id": [f"{well}_{int(index)}" for index in row_indices],
        "well": well,
        "selector_beam_mean": beam_mean,
    }
    for name in SELECTOR_VARIANTS:
        tokens = name.split("_")
        scale = tokens[2]
        path = np.asarray(base[f"pf_scale_{scale}"], dtype=np.float32)
        beam_weight = 0.0
        hold_weight = 0.0
        if "beam" in tokens:
            beam_weight = float(tokens[tokens.index("beam") + 1].replace("p", "."))
        if "hold" in tokens:
            hold_weight = float(tokens[tokens.index("hold") + 1].replace("p", "."))
        blended = (1.0 - beam_weight) * path + beam_weight * beam_mean
        blended = (1.0 - hold_weight) * blended + hold_weight * last_tvt
        result[f"selector__{name}"] = blended.astype(np.float32)
    return pd.DataFrame(result)


def build_physical_well(
    runtime: ModuleType,
    train_dir: str | Path,
    well: str,
    *,
    particles: int = 500,
    seeds: int = 128,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_dir = Path(train_dir)
    horizontal = pd.read_csv(train_dir / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(train_dir / f"{well}__typewell.csv")
    known = horizontal[horizontal["TVT_input"].notna()]
    evaluation = horizontal[horizontal["TVT_input"].isna()]
    if len(known) < 10 or evaluation.empty:
        raise ValueError(f"Invalid known/evaluation rows for well {well}")
    selector_bank, selector_indices, selector_sigma = _likelihood_pf_bank(
        runtime,
        horizontal,
        typewell,
        gr_sigma_multiplier=1.0,
        seed_base=stable_seed("exp497", "stage_p", "selector_likpf", well),
        particles=particles,
        seeds=seeds,
    )
    learned_bank, learned_indices, learned_sigma = _likelihood_pf_bank(
        runtime,
        horizontal,
        typewell,
        gr_sigma_multiplier=1.3,
        seed_base=stable_seed("exp497", "stage_p", "learned_likpf", well),
        particles=particles,
        seeds=seeds,
    )
    beam_mean, beam_indices = _selector_beam_mean(runtime, horizontal, typewell)
    if not (
        np.array_equal(selector_indices, learned_indices)
        and np.array_equal(selector_indices, beam_indices)
    ):
        raise ValueError(f"Physical row indices disagree for well {well}")
    last_tvt = float(known.iloc[-1]["TVT_input"])
    candidates = _selector_candidate_frame(
        well=well,
        row_indices=selector_indices,
        last_tvt=last_tvt,
        selector_bank=selector_bank,
        beam_mean=beam_mean,
    )
    for key, values in learned_bank.items():
        name = f"likpf_{key}"
        candidates[name] = np.asarray(values, dtype=np.float32)
        candidates[f"{name}_d"] = (np.asarray(values) - last_tvt).astype(np.float32)
    z_eval = evaluation["Z"].to_numpy(float)
    metadata = {
        "well": well,
        "rows": len(evaluation),
        "n_eval": len(evaluation),
        "z_span": float(np.nanmax(z_eval) - np.nanmin(z_eval)),
        "last_known_tvt": last_tvt,
        "selector_gr_sigma": selector_sigma,
        "learned_gr_sigma": learned_sigma,
        "selector_seed_base": stable_seed("exp497", "stage_p", "selector_likpf", well),
        "learned_seed_base": stable_seed("exp497", "stage_p", "learned_likpf", well),
    }
    return candidates, metadata


def _spatial_well_patch(
    runtime: ModuleType,
    formation_imputer: Any,
    dense_imputer: Any,
    train_dir: Path,
    well: str,
    base_rows: pd.DataFrame,
) -> pd.DataFrame:
    horizontal = pd.read_csv(train_dir / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(train_dir / f"{well}__typewell.csv").sort_values("TVT")
    known = horizontal[horizontal["TVT_input"].notna()]
    evaluation = horizontal[horizontal["TVT_input"].isna()]
    expected_ids = pd.Series(
        [f"{well}_{int(index)}" for index in evaluation.index],
        dtype=str,
        name="id",
    )
    base_rows = _align_base_rows_to_expected_ids(base_rows, expected_ids, well)
    last_tvt = float(known.iloc[-1]["TVT_input"])
    known_tvt = known["TVT_input"].to_numpy(np.float32)
    known_z = known["Z"].to_numpy(np.float32)
    eval_z = evaluation["Z"].to_numpy(np.float32)
    xy_eval = evaluation[["X", "Y"]].to_numpy(np.float64)
    xy_known = known[["X", "Y"]].to_numpy(np.float64)
    formation_eval, knn_distance = formation_imputer.impute(xy_eval, self_wid=well)
    formation_known, _ = formation_imputer.impute(xy_known, self_wid=well)
    values: dict[str, Any] = {"id": expected_ids}
    formation_paths: list[np.ndarray] = []
    formation_rmse: dict[str, float] = {}
    for index, formation in enumerate(FORMATIONS):
        b_full, b_early, b_mid, b_late, b_wls = runtime.seg_b_well(
            known_tvt, known_z, formation_known[:, index]
        )
        path = (-eval_z + formation_eval[:, index] + b_full).astype(np.float32)
        values[f"tvtF_{formation}"] = path
        values[f"tvtFw_{formation}"] = (-eval_z + formation_eval[:, index] + b_wls).astype(
            np.float32
        )
        values[f"tvtF50_{formation}"] = (-eval_z + formation_eval[:, index] + b_late).astype(
            np.float32
        )
        values[f"bw_{formation}"] = np.float32(b_full)
        values[f"bww_{formation}"] = np.float32(b_wls)
        values[f"bw50_{formation}"] = np.float32(b_late)
        values[f"bw_early_{formation}"] = np.float32(b_early)
        values[f"bw_mid_{formation}"] = np.float32(b_mid)
        formation_rmse[formation] = float(
            np.sqrt(np.mean(np.square(known_tvt - (-known_z + formation_known[:, index] + b_full))))
        )
        values[f"frm_rmse_{formation}"] = np.float32(formation_rmse[formation])
        formation_paths.append(path)
    formation_matrix = np.stack(formation_paths, axis=1)
    values["form_mean_d"] = (formation_matrix.mean(1) - last_tvt).astype(np.float32)
    values["form_std_d"] = formation_matrix.std(1).astype(np.float32)
    values["form_rng_d"] = (formation_matrix.max(1) - formation_matrix.min(1)).astype(np.float32)
    tw_tvt = typewell["TVT"].to_numpy(float)
    tw_gr = typewell["GR"].to_numpy(float)
    values["spatial_ancc_d"] = (
        formation_eval[:, 0] - np.float32(np.interp(last_tvt, tw_tvt, tw_gr))
    ).astype(np.float32)
    values["spatial_knn_dist"] = knn_distance.astype(np.float32)
    dense_eval, dense_std, dense_distance = dense_imputer.impute(xy_eval, self_wid=well)
    dense_known, dense_known_std, _ = dense_imputer.impute(xy_known, self_wid=well)
    b_values = known_tvt + known_z - dense_known
    _, _, _, b_late, b_wls = runtime.seg_b_well(known_tvt, known_z, dense_known)
    b_full = float(np.median(b_values))
    dense_path = (-eval_z + dense_eval + b_full).astype(np.float32)
    dense_wls_path = (-eval_z + dense_eval + b_wls).astype(np.float32)
    dense_late_path = (-eval_z + dense_eval + b_late).astype(np.float32)
    dense_residual = known_tvt + known_z - dense_known
    values.update(
        {
            "dense_ancc": dense_eval.astype(np.float32),
            "dense_std": dense_std.astype(np.float32),
            "dense_dist": dense_distance.astype(np.float32),
            "tvt_dense_d": (dense_path - last_tvt).astype(np.float32),
            "tvt_densew_d": (dense_wls_path - last_tvt).astype(np.float32),
            "tvt_dense50_d": (dense_late_path - last_tvt).astype(np.float32),
            "dense_rmse": np.float32(np.sqrt(np.mean(np.square(dense_residual)))),
            "dense_bias": np.float32(np.mean(dense_residual)),
            "dense_nb_std": np.float32(np.mean(dense_known_std)),
        }
    )
    pf = base_rows["pf_ancc"].to_numpy(np.float32)
    beam_cons = last_tvt + base_rows["beam_cons_d"].to_numpy(np.float32)
    sc8 = last_tvt + base_rows["sc8_d"].to_numpy(np.float32)
    sc15 = last_tvt + base_rows["sc15_d"].to_numpy(np.float32)
    sc25 = last_tvt + base_rows["sc25_d"].to_numpy(np.float32)
    sc_ens = last_tvt + base_rows["sc_ens_d"].to_numpy(np.float32)
    beam_paths = [
        last_tvt + base_rows[f"beam_{tag}_d"].to_numpy(np.float32)
        for tag in ("cons", "loose", "vcons", "sm5", "vloose", "mid", "stiff")
    ]
    values["pf_vs_spatial"] = (pf - formation_paths[0]).astype(np.float32)
    values["pf_vs_dense"] = (pf - dense_path).astype(np.float32)
    values["spatial_vs_dense"] = (formation_paths[0] - dense_path).astype(np.float32)
    values["beam_vs_spatial"] = (beam_cons - formation_paths[0]).astype(np.float32)
    signal_matrix = np.stack(
        [pf, *beam_paths, sc8, sc15, sc25, sc_ens, formation_paths[0], dense_path],
        axis=1,
    )
    values["sig_std"] = signal_matrix.std(1).astype(np.float32)
    values["sig_mean_d"] = (signal_matrix.mean(1) - last_tvt).astype(np.float32)
    result = pd.DataFrame(values)
    if set(result.columns) != {"id", *SPATIAL_COLUMNS}:
        missing = sorted(set(SPATIAL_COLUMNS) - set(result.columns))
        extra = sorted(set(result.columns) - {"id", *SPATIAL_COLUMNS})
        raise ValueError(f"Spatial patch schema mismatch: missing={missing} extra={extra}")
    return result


def _align_base_rows_to_expected_ids(
    base_rows: pd.DataFrame,
    expected_ids: pd.Series,
    well: str,
) -> pd.DataFrame:
    expected = expected_ids.astype(str).reset_index(drop=True)
    actual = base_rows["id"].astype(str).reset_index(drop=True)
    if expected.duplicated().any() or actual.duplicated().any():
        raise ValueError(f"Spatial patch duplicate ids for {well}")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if len(expected) != len(actual) or missing or extra:
        raise ValueError(
            f"Spatial patch id set mismatch for {well}: missing={missing[:3]} extra={extra[:3]}"
        )
    aligned = base_rows.assign(id=base_rows["id"].astype(str)).set_index("id").loc[expected]
    aligned = aligned.reset_index()
    if not np.array_equal(expected.to_numpy(), aligned["id"].to_numpy()):
        raise ValueError(f"Spatial patch id reindex failed for {well}")
    return aligned


def build_spatial_surface(
    runtime: ModuleType,
    train_dir: str | Path,
    base_frame: pd.DataFrame,
    fold_wells: pd.DataFrame,
    outer_fold: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_dir = Path(train_dir)
    pool_wells = sorted(fold_wells.loc[fold_wells["outer_fold"].ne(outer_fold), "well"].astype(str))
    all_wells = sorted(fold_wells["well"].astype(str))
    formation_imputer = runtime.FormationPlaneKNN(pool_wells, train_dir)
    dense_imputer = runtime.DenseANCCImputer(pool_wells, train_dir)
    parts: list[pd.DataFrame] = []
    for well in all_wells:
        well_rows = base_frame.loc[base_frame["well"].eq(well)].reset_index(drop=True)
        if well_rows.empty:
            continue
        parts.append(
            _spatial_well_patch(
                runtime,
                formation_imputer,
                dense_imputer,
                train_dir,
                well,
                well_rows,
            )
        )
    surface = pd.concat(parts, ignore_index=True)
    return surface, {
        "outer_fold": outer_fold,
        "pool_wells": len(pool_wells),
        "query_wells": surface["id"].str[:8].nunique(),
        "rows": len(surface),
        "pool_sha256": sha256_json(pool_wells),
    }


def run_stage_p_shard(
    *,
    shard_fold: int,
    output_dir: str | Path,
    exp072_cache_path: str | Path,
    exp072_cache_sha256: str,
    parent_oof_path: str | Path,
    parent_oof_sha256: str,
    train_dir: str | Path,
    public_runtime_path: str | Path | None = None,
    particles: int = 500,
    seeds: int = 128,
) -> dict[str, Any]:
    if shard_fold not in range(5):
        raise ValueError("shard_fold must be 0..4")
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = load_public_replay_module(public_runtime_path)
    parent = load_parent_oof(
        parent_oof_path,
        expected_sha256=parent_oof_sha256,
    )
    fold_wells = outer_fold_well_map(parent)
    shard_wells = set(fold_wells.loc[fold_wells["outer_fold"].eq(shard_fold), "well"].astype(str))
    base, base_features, base_meta = load_exp072_base_shard(
        exp072_cache_path,
        shard_wells,
        expected_sha256=exp072_cache_sha256,
    )
    base = base.merge(fold_wells, on="well", how="left", validate="many_to_one")
    if not base["outer_fold"].eq(shard_fold).all():
        raise ValueError("Stage P base shard contains an unexpected outer fold")
    physical_parts: list[pd.DataFrame] = []
    well_rows: list[dict[str, Any]] = []
    for index, well in enumerate(sorted(shard_wells), start=1):
        physical, metadata = build_physical_well(
            runtime,
            train_dir,
            well,
            particles=particles,
            seeds=seeds,
        )
        physical_parts.append(physical)
        well_rows.append(metadata)
        print(
            json.dumps(
                {
                    "event": "stage_p_well_complete",
                    "shard_fold": shard_fold,
                    "index": index,
                    "total": len(shard_wells),
                    "well": well,
                    "rows": metadata["rows"],
                }
            ),
            flush=True,
        )
    physical = pd.concat(physical_parts, ignore_index=True)
    feature_shard = base.merge(physical, on=["id", "well"], how="left", validate="one_to_one")
    required = [*LEARNED_LIKPF_COLUMNS, *(f"selector__{v}" for v in SELECTOR_VARIANTS)]
    if feature_shard[required].isna().any().any():
        raise ValueError("Stage P physical merge produced missing values")
    feature_columns = [name for name in feature_shard.columns if name not in META_COLUMNS]
    if len(base_features) != 195 or len(feature_columns) != 213:
        raise ValueError(
            f"Stage P feature count mismatch: base={len(base_features)} "
            f"total={len(feature_columns)}"
        )
    feature_path = output_dir / f"stage_p_fold{shard_fold}_physical_features.parquet"
    well_path = output_dir / f"stage_p_fold{shard_fold}_well_metadata.parquet"
    schema_path = output_dir / f"stage_p_fold{shard_fold}_feature_schema.csv"
    feature_shard.to_parquet(feature_path, index=False, compression="zstd")
    pd.DataFrame(well_rows).to_parquet(well_path, index=False, compression="zstd")
    pd.DataFrame(
        {
            "feature_index": np.arange(len(feature_columns), dtype=np.int16),
            "feature": feature_columns,
        }
    ).to_csv(schema_path, index=False)
    summary = {
        "experiment": "exp497_strict_public_core_fold_safe_ensemble_on_exp413",
        "stage": "stage_p_physical_shard",
        "status": "complete",
        "shard_fold": shard_fold,
        "rows": len(feature_shard),
        "wells": len(shard_wells),
        "base_feature_count": len(base_features),
        "learned_feature_count": len(base_features) + len(LEARNED_LIKPF_COLUMNS),
        "stage_p_feature_count_including_selector_candidates": len(feature_columns),
        "physical_inventory": {
            "likelihood_pf_seed_banks": 2 * len(shard_wells),
            "seed_well_runs": 2 * len(shard_wells) * seeds,
            "particle_starts": 2 * len(shard_wells) * seeds * particles,
            "selector_beam_well_runs": len(shard_wells) * len(SELECTOR_BEAM_CONFIGS),
            "reused_exp072_pf_ancc_well_runs": len(shard_wells),
            "reused_exp072_pf_z_well_runs": len(shard_wells),
            "reused_exp072_learned_beam_well_runs": len(shard_wells) * 7,
            "reused_exp072_ncc_well_window_runs": len(shard_wells) * 3,
        },
        "inputs": {
            "exp072": base_meta,
            "parent_oof": {
                "path": str(parent_oof_path),
                "sha256": parent_oof_sha256,
            },
            "public_runtime": {
                "path": runtime.__dict__["_exp497_source_path"],
                "sha256": sha256_file(runtime.__dict__["_exp497_source_path"]),
            },
        },
        "outputs": {
            "features": feature_path.name,
            "well_metadata": well_path.name,
            "schema": schema_path.name,
        },
        "sha256": {
            "features": sha256_file(feature_path),
            "well_metadata": sha256_file(well_path),
            "schema": sha256_file(schema_path),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    summary_path = output_dir / f"stage_p_fold{shard_fold}_summary.json"
    write_json(summary_path, summary)
    summary["sha256"]["summary"] = sha256_file(summary_path)
    print(json.dumps(summary, indent=2, default=_json_default), flush=True)
    return summary


def load_stage_p_union(
    feature_paths: Sequence[str | Path],
    summary_paths: Sequence[str | Path],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if len(feature_paths) != 5 or len(summary_paths) != 5:
        raise ValueError("Stage P union requires exactly five feature and five summary files")
    summaries: list[dict[str, Any]] = []
    parts: list[pd.DataFrame] = []
    for feature_path, summary_path in zip(feature_paths, summary_paths, strict=True):
        feature_path = Path(feature_path)
        summary_path = Path(summary_path)
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete" or summary.get("stage") != "stage_p_physical_shard":
            raise ValueError(f"Invalid Stage P summary: {summary_path}")
        expected_sha = str(summary["sha256"]["features"])
        observed_sha = sha256_file(feature_path)
        if observed_sha != expected_sha:
            raise ValueError(f"Stage P feature SHA mismatch for {feature_path}")
        part = pd.read_parquet(feature_path)
        if len(part) != int(summary["rows"]):
            raise ValueError(f"Stage P row mismatch for {feature_path}")
        summaries.append(summary)
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    if len(frame) != 3_783_989 or frame["well"].nunique() != 773:
        raise ValueError("Stage P union row/well contract mismatch")
    if frame["id"].duplicated().any():
        raise ValueError("Stage P union contains duplicate ids")
    if sorted(frame["outer_fold"].unique()) != list(range(5)):
        raise ValueError("Stage P union does not cover outer folds 0..4")
    return frame, summaries


def base_and_learned_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    selector_columns = {"selector_beam_mean", *(f"selector__{v}" for v in SELECTOR_VARIANTS)}
    base = [
        name
        for name in frame.columns
        if name not in META_COLUMNS
        and name not in selector_columns
        and name not in LEARNED_LIKPF_COLUMNS
    ]
    learned = [*base, *LEARNED_LIKPF_COLUMNS]
    if len(base) != 195 or len(learned) != 205:
        raise ValueError(f"Model feature schema mismatch: base={len(base)} learned={len(learned)}")
    return base, learned


def fit_selector_policy(
    well_metadata: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    train_wells: set[str],
    *,
    minimum_bin_wells: int = 40,
) -> dict[str, Any]:
    metadata = well_metadata[well_metadata["well"].isin(train_wells)].copy()
    if metadata["well"].nunique() != len(train_wells):
        missing = sorted(train_wells - set(metadata["well"]))
        raise ValueError(f"Selector metadata is missing train wells: {missing[:20]}")
    n_eval_threshold = float(metadata["n_eval"].median())
    z_thresholds = [float(value) for value in metadata["z_span"].quantile([1 / 3, 2 / 3])]
    metadata["selector_code"] = metadata["n_eval"].gt(n_eval_threshold).astype(
        np.int8
    ) + 2 * np.searchsorted(
        np.asarray(z_thresholds, dtype=float),
        metadata["z_span"].to_numpy(float),
        side="right",
    ).astype(np.int8)
    truth = candidate_rows["last_known_tvt"].to_numpy(np.float32) + candidate_rows[
        "target"
    ].to_numpy(np.float32)
    score_rows: list[dict[str, Any]] = []
    row_train = candidate_rows["well"].isin(train_wells).to_numpy()
    for variant in SELECTOR_VARIANTS:
        prediction = candidate_rows[f"selector__{variant}"].to_numpy(np.float32)
        error2 = np.square(prediction - truth)
        score_rows.append(
            {
                "scope": "global",
                "selector_code": -1,
                "variant": variant,
                "rows": int(row_train.sum()),
                "wells": len(train_wells),
                "rmse": float(np.sqrt(np.mean(error2[row_train]))),
            }
        )
    global_scores = pd.DataFrame(score_rows)
    global_variant = str(global_scores.sort_values(["rmse", "variant"]).iloc[0]["variant"])
    code_by_well = metadata.set_index("well")["selector_code"]
    mapping: dict[int, str] = {}
    for code in range(6):
        code_wells = set(code_by_well[code_by_well.eq(code)].index.astype(str))
        if len(code_wells) < minimum_bin_wells:
            mapping[code] = global_variant
            continue
        mask = candidate_rows["well"].isin(code_wells).to_numpy()
        local: list[tuple[float, str]] = []
        for variant in SELECTOR_VARIANTS:
            prediction = candidate_rows[f"selector__{variant}"].to_numpy(np.float32)
            local_rmse = float(np.sqrt(np.mean(np.square(prediction[mask] - truth[mask]))))
            local.append((local_rmse, variant))
            score_rows.append(
                {
                    "scope": "bin",
                    "selector_code": code,
                    "variant": variant,
                    "rows": int(mask.sum()),
                    "wells": len(code_wells),
                    "rmse": local_rmse,
                }
            )
        mapping[code] = min(local)[1]
    return {
        "n_eval_threshold": n_eval_threshold,
        "z_span_thresholds": z_thresholds,
        "minimum_bin_wells": minimum_bin_wells,
        "global_variant": global_variant,
        "mapping": {str(code): variant for code, variant in mapping.items()},
        "train_wells": len(train_wells),
        "train_well_sha256": sha256_json(sorted(train_wells)),
        "score_rows": score_rows,
    }


def apply_selector_policy(
    frame: pd.DataFrame,
    well_metadata: pd.DataFrame,
    policy: Mapping[str, Any],
) -> np.ndarray:
    metadata = well_metadata.set_index("well")
    if not metadata.index.is_unique:
        raise ValueError("Selector metadata contains duplicate wells")
    n_eval = frame["well"].map(metadata["n_eval"]).to_numpy(float)
    z_span = frame["well"].map(metadata["z_span"]).to_numpy(float)
    if not np.isfinite(n_eval).all() or not np.isfinite(z_span).all():
        raise ValueError("Selector application has missing well metadata")
    code = (n_eval > float(policy["n_eval_threshold"])).astype(np.int8) + 2 * np.searchsorted(
        np.asarray(policy["z_span_thresholds"], dtype=float),
        z_span,
        side="right",
    ).astype(np.int8)
    prediction = np.empty(len(frame), dtype=np.float32)
    mapping = {int(key): str(value) for key, value in policy["mapping"].items()}
    for value in range(6):
        mask = code == value
        variant = mapping.get(value, str(policy["global_variant"]))
        prediction[mask] = frame.loc[mask, f"selector__{variant}"].to_numpy(np.float32)
    if not np.isfinite(prediction).all():
        raise ValueError("Selector prediction contains non-finite values")
    return prediction


def _lgb_configs() -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "verbose": -1,
        "n_jobs": -1,
        "max_bin": 255,
        "device_type": "gpu",
        "gpu_use_dp": False,
    }
    return [
        {
            **base,
            "num_leaves": 255,
            "min_child_samples": 15,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 3.0,
            "reg_alpha": 0.05,
            "learning_rate": 0.02,
            "n_estimators": 5000,
            "seed": 123,
        },
        {
            **base,
            "num_leaves": 64,
            "min_child_samples": 40,
            "subsample": 0.47,
            "subsample_freq": 1,
            "colsample_bytree": 0.39,
            "reg_lambda": 95.75,
            "reg_alpha": 10.8,
            "min_child_weight": 0.24,
            "learning_rate": 0.0093,
            "n_estimators": 10000,
            "random_state": 0,
        },
        {
            **base,
            "num_leaves": 64,
            "min_child_samples": 40,
            "subsample": 0.47,
            "subsample_freq": 1,
            "colsample_bytree": 0.39,
            "reg_lambda": 95.75,
            "reg_alpha": 10.8,
            "min_child_weight": 0.24,
            "learning_rate": 0.0093,
            "n_estimators": 10000,
            "random_state": 29,
        },
    ]


def _cat_configs() -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "iterations": 8000,
        "depth": 7,
        "l2_leaf_reg": 2.0,
        "min_data_in_leaf": 15,
        "border_count": 254,
        "loss_function": "RMSE",
        "task_type": "GPU",
        "devices": "0",
        "od_type": "Iter",
        "od_wait": 300,
        "verbose": 0,
    }
    return [
        {**base, "learning_rate": 0.020, "random_seed": 7},
        {**base, "learning_rate": 0.030, "random_seed": 123},
    ]


def training_inventory() -> dict[str, int]:
    lgb = 2 * 5 * 4 * len(_lgb_configs())
    cat = 2 * 5 * 4 * len(_cat_configs())
    return {
        "scientific_variants": 1,
        "branches": 2,
        "outer_folds": 5,
        "inner_folds": 4,
        "lightgbm_configs_per_branch": len(_lgb_configs()),
        "catboost_configs_per_branch": len(_cat_configs()),
        "planned_lightgbm_boosters": lgb,
        "planned_catboost_boosters": cat,
        "planned_total_boosters": lgb + cat,
        "planned_ridge_models": 10,
        "exp413_retraining": 0,
    }


def _fit_config_inner_oof(
    *,
    branch: str,
    config_name: str,
    config_kind: str,
    params: Mapping[str, Any],
    frame: pd.DataFrame,
    features: list[str],
    outer_train_indices: np.ndarray,
    outer_valid_indices: np.ndarray,
    inner_fold: np.ndarray,
    target_delta: np.ndarray,
    outer_fold: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray]:
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    x_train = frame.iloc[outer_train_indices][features].to_numpy(np.float32)
    x_valid = frame.iloc[outer_valid_indices][features].to_numpy(np.float32)
    y_train = target_delta[outer_train_indices]
    oof = np.empty(len(outer_train_indices), dtype=np.float32)
    valid_predictions: list[np.ndarray] = []
    manifest: list[dict[str, Any]] = []
    importances: list[np.ndarray] = []
    for inner in range(4):
        train_mask = inner_fold != inner
        validation_mask = inner_fold == inner
        if config_kind == "lightgbm":
            model = LGBMRegressor(**dict(params))
            model.fit(
                x_train[train_mask],
                y_train[train_mask],
                eval_set=[(x_train[validation_mask], y_train[validation_mask])],
                eval_metric="rmse",
                callbacks=[early_stopping(250, verbose=False), log_evaluation(0)],
            )
            best_iteration = int(model.best_iteration_)
            oof[validation_mask] = model.predict(
                x_train[validation_mask], num_iteration=best_iteration
            ).astype(np.float32)
            valid_predictions.append(
                model.predict(x_valid, num_iteration=best_iteration).astype(np.float32)
            )
            importances.append(model.booster_.feature_importance(importance_type="gain"))
        elif config_kind == "catboost":
            model = CatBoostRegressor(**dict(params))
            model.fit(
                x_train[train_mask],
                y_train[train_mask],
                eval_set=(x_train[validation_mask], y_train[validation_mask]),
                early_stopping_rounds=250,
                use_best_model=True,
            )
            best_iteration = int(model.get_best_iteration()) + 1
            oof[validation_mask] = model.predict(x_train[validation_mask]).astype(np.float32)
            valid_predictions.append(model.predict(x_valid).astype(np.float32))
            importances.append(np.asarray(model.get_feature_importance(), dtype=np.float64))
        else:
            raise ValueError(f"Unknown config kind: {config_kind}")
        fold_rmse = rmse(y_train[validation_mask], oof[validation_mask])
        row = {
            "branch": branch,
            "config": config_name,
            "kind": config_kind,
            "outer_fold": outer_fold,
            "inner_fold": inner,
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
            "best_iteration": best_iteration,
            "rmse_delta": fold_rmse,
        }
        manifest.append(row)
        print(json.dumps({"event": "model_complete", **row}), flush=True)
    return (
        oof,
        np.mean(np.stack(valid_predictions, axis=0), axis=0).astype(np.float32),
        manifest,
        np.mean(np.stack(importances, axis=0), axis=0),
    )


def _fit_branch(
    *,
    branch: str,
    frame: pd.DataFrame,
    features: list[str],
    outer_train_indices: np.ndarray,
    outer_valid_indices: np.ndarray,
    inner_fold: np.ndarray,
    target_delta: np.ndarray,
    outer_fold: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    from sklearn.linear_model import Ridge

    config_specs: list[tuple[str, str, Mapping[str, Any]]] = [
        (f"lgb{index}", "lightgbm", params) for index, params in enumerate(_lgb_configs())
    ] + [(f"cat{index}", "catboost", params) for index, params in enumerate(_cat_configs())]
    train_columns: list[np.ndarray] = []
    valid_columns: list[np.ndarray] = []
    manifest: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for name, kind, params in config_specs:
        train_prediction, valid_prediction, rows, importance = _fit_config_inner_oof(
            branch=branch,
            config_name=name,
            config_kind=kind,
            params=params,
            frame=frame,
            features=features,
            outer_train_indices=outer_train_indices,
            outer_valid_indices=outer_valid_indices,
            inner_fold=inner_fold,
            target_delta=target_delta,
            outer_fold=outer_fold,
        )
        train_columns.append(train_prediction)
        valid_columns.append(valid_prediction)
        manifest.extend(rows)
        importance_rows.extend(
            {
                "branch": branch,
                "outer_fold": outer_fold,
                "config": name,
                "feature": feature,
                "importance": float(value),
            }
            for feature, value in zip(features, importance, strict=True)
        )
    train_matrix = np.column_stack(train_columns).astype(np.float32)
    valid_matrix = np.column_stack(valid_columns).astype(np.float32)
    target_train = target_delta[outer_train_indices]
    ridge = Ridge(alpha=1.66, positive=True, fit_intercept=True, tol=0.0005)
    ridge.fit(train_matrix, target_train)
    train_stack = ridge.predict(train_matrix).astype(np.float32)
    valid_stack = ridge.predict(valid_matrix).astype(np.float32)
    ridge_meta = {
        "branch": branch,
        "outer_fold": outer_fold,
        "alpha": 1.66,
        "positive": True,
        "coef": ridge.coef_.astype(float).tolist(),
        "intercept": float(ridge.intercept_),
        "train_rmse_delta": rmse(target_train, train_stack),
    }
    return train_stack, valid_stack, manifest, pd.DataFrame(importance_rows), ridge_meta


def fit_convex_weight(
    truth: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bounds: tuple[float, float],
) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    direction = right - left
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-12:
        return float(bounds[0])
    weight = float(np.dot(truth - left, direction) / denominator)
    return float(np.clip(weight, bounds[0], bounds[1]))


def robust_u_projection(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    degree: int = 3,
    iterations: int = 4,
) -> np.ndarray:
    output = np.asarray(prediction, dtype=np.float64).copy()
    work = frame[["well", "md_since", "z"]].reset_index(drop=True)
    for _, indices in work.groupby("well", sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        if len(positions) < degree + 2:
            continue
        md = work.loc[positions, "md_since"].to_numpy(np.float64)
        scale = max(float(np.nanmax(md)), 1e-6)
        normalized = md / scale
        u_values = output[positions] + work.loc[positions, "z"].to_numpy(np.float64)
        coefficients = np.polyfit(normalized, u_values, degree)
        for _ in range(iterations):
            residual = u_values - np.polyval(coefficients, normalized)
            robust_scale = np.median(np.abs(residual)) * 1.4826 + 1e-6
            weights = 1.0 / (1.0 + np.square(residual / (2.0 * robust_scale)))
            coefficients = np.polyfit(normalized, u_values, degree, w=weights)
        fitted_u = np.polyval(coefficients, normalized)
        output[positions] = fitted_u - work.loc[positions, "z"].to_numpy(np.float64)
    if not np.isfinite(output).all():
        raise ValueError("Robust U projection produced non-finite values")
    return output.astype(np.float32)


def savgol_by_well(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    window: int = 61,
    polyorder: int = 3,
) -> np.ndarray:
    from scipy.signal import savgol_filter

    output = np.asarray(prediction, dtype=np.float64).copy()
    work = frame[["well"]].reset_index(drop=True)
    for _, indices in work.groupby("well", sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        local_window = min(window, len(positions))
        if local_window % 2 == 0:
            local_window -= 1
        if local_window >= polyorder + 2:
            output[positions] = savgol_filter(output[positions], local_window, polyorder)
    return output.astype(np.float32)


def _inner_fold_for_outer_train(frame: pd.DataFrame, outer_train_indices: np.ndarray) -> np.ndarray:
    from sklearn.model_selection import GroupKFold

    groups = frame.iloc[outer_train_indices]["well"].astype(str).to_numpy()
    inner = np.full(len(outer_train_indices), -1, dtype=np.int8)
    splitter = GroupKFold(n_splits=4)
    dummy = np.zeros(len(groups), dtype=np.int8)
    for fold, (_, valid) in enumerate(splitter.split(dummy, groups=groups)):
        inner[valid] = fold
    if np.any(inner < 0):
        raise ValueError("Inner fold assignment is incomplete")
    return inner


def run_stage_m_outer(
    *,
    outer_fold: int,
    output_dir: str | Path,
    stage_p_feature_paths: Sequence[str | Path],
    stage_p_summary_paths: Sequence[str | Path],
    parent_oof_path: str | Path,
    parent_oof_sha256: str,
    train_dir: str | Path,
    public_runtime_path: str | Path | None = None,
) -> dict[str, Any]:
    if outer_fold not in range(5):
        raise ValueError("outer_fold must be 0..4")
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = training_inventory()
    if inventory["planned_total_boosters"] != 200 or inventory["exp413_retraining"] != 0:
        raise ValueError("Frozen training inventory changed")
    frame, stage_p_summaries = load_stage_p_union(stage_p_feature_paths, stage_p_summary_paths)
    parent = load_parent_oof(parent_oof_path, expected_sha256=parent_oof_sha256)
    join = parent[["id", "well", "outer_fold", "actual_tvt"]]
    frame = frame.merge(
        join,
        on=["id", "well", "outer_fold"],
        how="left",
        validate="one_to_one",
    )
    if frame["actual_tvt"].isna().any():
        raise ValueError("Stage M parent truth/fold join produced missing values")
    frame = frame.sort_values(["outer_fold", "well", "md_since", "id"]).reset_index(drop=True)
    base_features, learned_features = base_and_learned_feature_columns(frame)
    runtime = load_public_replay_module(public_runtime_path)
    fold_wells = outer_fold_well_map(parent)
    spatial_surface, spatial_meta = build_spatial_surface(
        runtime,
        train_dir,
        frame,
        fold_wells,
        outer_fold,
    )
    patch = spatial_surface.set_index("id")
    for column in SPATIAL_COLUMNS:
        frame[column] = frame["id"].map(patch[column]).astype(np.float32)
    if frame[list(SPATIAL_COLUMNS)].isna().any().any():
        raise ValueError("Fold-safe spatial patch produced missing values")
    base_features, learned_features = base_and_learned_feature_columns(frame)
    valid_mask = frame["outer_fold"].eq(outer_fold).to_numpy()
    outer_train_indices = np.flatnonzero(~valid_mask)
    outer_valid_indices = np.flatnonzero(valid_mask)
    inner_fold = _inner_fold_for_outer_train(frame, outer_train_indices)
    target_delta = frame["target"].to_numpy(np.float32)
    train_wells = set(frame.iloc[outer_train_indices]["well"].astype(str))
    well_metadata = pd.DataFrame(
        [
            {
                "well": str(summary_well["well"]),
                "n_eval": int(summary_well["n_eval"]),
                "z_span": float(summary_well["z_span"]),
            }
            for path in stage_p_summary_paths
            for summary_well in pd.read_parquet(
                find_artifact(
                    f"stage_p_fold{json.loads(Path(path).read_text())['shard_fold']}_well_metadata.parquet",
                    roots=[Path(path).parent],
                )
            ).to_dict("records")
        ]
    )
    if well_metadata["well"].nunique() != 773:
        raise ValueError("Stage P well metadata union is incomplete")
    selector_train = np.empty(len(outer_train_indices), dtype=np.float32)
    selector_policy_rows: list[dict[str, Any]] = []
    train_frame = frame.iloc[outer_train_indices].reset_index(drop=True)
    for inner in range(4):
        fit_wells = set(train_frame.loc[inner_fold != inner, "well"].astype(str))
        policy = fit_selector_policy(well_metadata, train_frame, fit_wells)
        selector_train[inner_fold == inner] = apply_selector_policy(
            train_frame.loc[inner_fold == inner], well_metadata, policy
        )
        selector_policy_rows.append({"role": "inner_oof", "inner_fold": inner, **policy})
    outer_policy = fit_selector_policy(well_metadata, frame, train_wells)
    selector_valid = apply_selector_policy(
        frame.iloc[outer_valid_indices], well_metadata, outer_policy
    )
    selector_policy_rows.append({"role": "outer_valid", "inner_fold": None, **outer_policy})
    sp45_train_delta, sp45_valid_delta, sp45_models, sp45_importance, sp45_ridge = _fit_branch(
        branch="sp45_residual",
        frame=frame,
        features=base_features,
        outer_train_indices=outer_train_indices,
        outer_valid_indices=outer_valid_indices,
        inner_fold=inner_fold,
        target_delta=target_delta,
        outer_fold=outer_fold,
    )
    (
        learned_train_delta,
        learned_valid_delta,
        learned_models,
        learned_importance,
        learned_ridge,
    ) = _fit_branch(
        branch="learned_trajectory",
        frame=frame,
        features=learned_features,
        outer_train_indices=outer_train_indices,
        outer_valid_indices=outer_valid_indices,
        inner_fold=inner_fold,
        target_delta=target_delta,
        outer_fold=outer_fold,
    )
    train_rows = frame.iloc[outer_train_indices].reset_index(drop=True)
    valid_rows = frame.iloc[outer_valid_indices].reset_index(drop=True)
    train_truth = train_rows["actual_tvt"].to_numpy(np.float32)
    last_train = train_rows["last_known_tvt"].to_numpy(np.float32)
    last_valid = valid_rows["last_known_tvt"].to_numpy(np.float32)
    sp45_model_train = last_train + sp45_train_delta
    sp45_model_valid = last_valid + sp45_valid_delta
    sp45_model_weight = fit_convex_weight(
        train_truth,
        selector_train,
        sp45_model_train,
        (0.15, 0.45),
    )
    sp45_raw_train = (
        (1.0 - sp45_model_weight) * selector_train + sp45_model_weight * sp45_model_train
    ).astype(np.float32)
    sp45_raw_valid = (
        (1.0 - sp45_model_weight) * selector_valid + sp45_model_weight * sp45_model_valid
    ).astype(np.float32)
    projected_train_full = robust_u_projection(train_rows, sp45_raw_train, degree=3)
    projected_valid_full = robust_u_projection(valid_rows, sp45_raw_valid, degree=3)
    projection_weight = fit_convex_weight(
        train_truth,
        sp45_raw_train,
        projected_train_full,
        (0.50, 1.00),
    )
    projected_train = (
        (1.0 - projection_weight) * sp45_raw_train + projection_weight * projected_train_full
    ).astype(np.float32)
    projected_valid = (
        (1.0 - projection_weight) * sp45_raw_valid + projection_weight * projected_valid_full
    ).astype(np.float32)
    train_warmup = 1.0 - np.exp(
        -np.maximum(train_rows["md_since"].to_numpy(np.float32), 0.0) / 85.0
    )
    valid_warmup = 1.0 - np.exp(
        -np.maximum(valid_rows["md_since"].to_numpy(np.float32), 0.0) / 85.0
    )
    learned_model_train = last_train + train_warmup * learned_train_delta
    learned_model_valid = last_valid + valid_warmup * learned_valid_delta
    likpf_train = train_rows["likpf_scale_5"].to_numpy(np.float32)
    likpf_valid = valid_rows["likpf_scale_5"].to_numpy(np.float32)
    learned_model_weight = fit_convex_weight(
        train_truth,
        likpf_train,
        learned_model_train,
        (0.50, 0.80),
    )
    learned_train = (
        (1.0 - learned_model_weight) * likpf_train + learned_model_weight * learned_model_train
    ).astype(np.float32)
    learned_valid = (
        (1.0 - learned_model_weight) * likpf_valid + learned_model_weight * learned_model_valid
    ).astype(np.float32)
    learned_train = savgol_by_well(train_rows, learned_train, window=61, polyorder=3)
    learned_valid = savgol_by_well(valid_rows, learned_valid, window=61, polyorder=3)
    projected_weight = fit_convex_weight(
        train_truth,
        learned_train,
        projected_train,
        (0.50, 0.80),
    )
    public_core_valid = (
        projected_weight * projected_valid + (1.0 - projected_weight) * learned_valid
    ).astype(np.float32)
    frozen_prediction = pd.DataFrame(
        {
            "id": valid_rows["id"].astype(str),
            "well": valid_rows["well"].astype(str),
            "outer_fold": np.int8(outer_fold),
            "physical_selector_oof": selector_valid,
            "sp45_residual_model_oof": sp45_model_valid,
            "sp45_raw_oof": sp45_raw_valid,
            "projected_sp45_oof": projected_valid,
            "learned_trajectory_oof": learned_valid,
            "strict_public_core_oof": public_core_valid,
        }
    )
    frozen_sha = prediction_sha256(
        frozen_prediction["id"],
        frozen_prediction["strict_public_core_oof"].to_numpy(np.float32),
        f"exp497:strict_public_core:outer{outer_fold}:before_truth_attach",
    )
    prediction_path = output_dir / f"stage_m_outer{outer_fold}_predictions.parquet"
    frozen_prediction.to_parquet(prediction_path, index=False, compression="zstd")
    truth = valid_rows[["id", "actual_tvt"]]
    metrics_frame = frozen_prediction.merge(truth, on="id", how="left", validate="one_to_one")
    component_columns = [
        "physical_selector_oof",
        "sp45_residual_model_oof",
        "sp45_raw_oof",
        "projected_sp45_oof",
        "learned_trajectory_oof",
        "strict_public_core_oof",
    ]
    metrics = {
        column: rmse(metrics_frame["actual_tvt"], metrics_frame[column])
        for column in component_columns
    }
    model_manifest = {
        "inventory": inventory,
        "outer_fold": outer_fold,
        "base_feature_count": len(base_features),
        "learned_feature_count": len(learned_features),
        "base_feature_sha256": sha256_json(base_features),
        "learned_feature_sha256": sha256_json(learned_features),
        "sp45_models": sp45_models,
        "learned_models": learned_models,
        "sp45_ridge": sp45_ridge,
        "learned_ridge": learned_ridge,
        "fitted_model_count": len(sp45_models) + len(learned_models),
    }
    if model_manifest["fitted_model_count"] != 40:
        raise ValueError("Outer shard did not fit exactly 40 boosters")
    weights = {
        "sp45_model_weight": sp45_model_weight,
        "projection_weight": projection_weight,
        "learned_model_weight": learned_model_weight,
        "projected_sp45_weight": projected_weight,
    }
    model_manifest_path = output_dir / f"stage_m_outer{outer_fold}_model_manifest.json"
    weights_path = output_dir / f"stage_m_outer{outer_fold}_weights.json"
    selector_path = output_dir / f"stage_m_outer{outer_fold}_selector_policies.json"
    importance_path = output_dir / f"stage_m_outer{outer_fold}_feature_importance.parquet"
    spatial_path = output_dir / f"stage_m_outer{outer_fold}_spatial_audit.json"
    write_json(model_manifest_path, model_manifest)
    write_json(weights_path, weights)
    write_json(selector_path, selector_policy_rows)
    pd.concat([sp45_importance, learned_importance], ignore_index=True).to_parquet(
        importance_path, index=False, compression="zstd"
    )
    write_json(spatial_path, spatial_meta)
    summary = {
        "experiment": "exp497_strict_public_core_fold_safe_ensemble_on_exp413",
        "stage": "stage_m_outer_fold",
        "status": "complete",
        "outer_fold": outer_fold,
        "rows": len(frozen_prediction),
        "wells": frozen_prediction["well"].nunique(),
        "metrics": metrics,
        "weights": weights,
        "frozen_prediction_sha256": frozen_sha,
        "inventory": inventory,
        "fitted_boosters": model_manifest["fitted_model_count"],
        "spatial_audit": spatial_meta,
        "stage_p_summary_sha256": [sha256_file(path) for path in stage_p_summary_paths],
        "outputs": {
            "predictions": prediction_path.name,
            "model_manifest": model_manifest_path.name,
            "weights": weights_path.name,
            "selector_policies": selector_path.name,
            "feature_importance": importance_path.name,
            "spatial_audit": spatial_path.name,
        },
        "sha256": {
            "predictions": sha256_file(prediction_path),
            "model_manifest": sha256_file(model_manifest_path),
            "weights": sha256_file(weights_path),
            "selector_policies": sha256_file(selector_path),
            "feature_importance": sha256_file(importance_path),
            "spatial_audit": sha256_file(spatial_path),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    summary_path = output_dir / f"stage_m_outer{outer_fold}_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, default=_json_default), flush=True)
    return summary


def _metric_row(
    scope: str,
    frame: pd.DataFrame,
    mask: np.ndarray,
    parent_column: str,
    candidate_column: str,
) -> dict[str, Any]:
    truth = frame.loc[mask, "actual_tvt"].to_numpy(np.float32)
    parent = frame.loc[mask, parent_column].to_numpy(np.float32)
    candidate = frame.loc[mask, candidate_column].to_numpy(np.float32)
    parent_rmse = rmse(truth, parent)
    candidate_rmse = rmse(truth, candidate)
    return {
        "scope": scope,
        "rows": int(mask.sum()),
        "wells": int(frame.loc[mask, "well"].nunique()),
        "exp413_rmse": parent_rmse,
        "exp497_rmse": candidate_rmse,
        "delta_rmse_exp497_minus_exp413": candidate_rmse - parent_rmse,
    }


def run_stage_e(
    *,
    output_dir: str | Path,
    stage_m_prediction_paths: Sequence[str | Path],
    stage_m_summary_paths: Sequence[str | Path],
    parent_oof_path: str | Path,
    parent_oof_sha256: str,
    hidden_like_assignment_path: str | Path,
    hidden_like_assignment_sha256: str,
    public_core_weight_bounds: tuple[float, float] = (0.0, 0.30),
) -> dict[str, Any]:
    started = time.time()
    if len(stage_m_prediction_paths) != 5 or len(stage_m_summary_paths) != 5:
        raise ValueError("Stage E requires exactly five Stage M shards")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    stage_m_summaries: list[dict[str, Any]] = []
    for prediction_path, summary_path in zip(
        stage_m_prediction_paths, stage_m_summary_paths, strict=True
    ):
        prediction_path = Path(prediction_path)
        summary_path = Path(summary_path)
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete" or summary.get("stage") != "stage_m_outer_fold":
            raise ValueError(f"Invalid Stage M summary: {summary_path}")
        if sha256_file(prediction_path) != str(summary["sha256"]["predictions"]):
            raise ValueError(f"Stage M prediction SHA mismatch: {prediction_path}")
        part = pd.read_parquet(prediction_path)
        frozen_sha = prediction_sha256(
            part["id"],
            part["strict_public_core_oof"].to_numpy(np.float32),
            f"exp497:strict_public_core:outer{int(summary['outer_fold'])}:before_truth_attach",
        )
        if frozen_sha != str(summary["frozen_prediction_sha256"]):
            raise ValueError(f"Stage M frozen prediction mismatch: {prediction_path}")
        parts.append(part)
        stage_m_summaries.append(summary)
    public_core = pd.concat(parts, ignore_index=True)
    if len(public_core) != 3_783_989 or public_core["well"].nunique() != 773:
        raise ValueError("Stage M union row/well contract mismatch")
    if public_core["id"].duplicated().any():
        raise ValueError("Stage M union contains duplicate ids")
    parent = load_parent_oof(parent_oof_path, expected_sha256=parent_oof_sha256)
    parent_column = "scale5_x1p0_full_replacement__lgb_mean__pred_tvt"
    parent_subset = parent[["id", "well", "outer_fold", "md_since", "actual_tvt", parent_column]]
    frame = public_core.merge(
        parent_subset,
        on=["id", "well", "outer_fold"],
        how="left",
        validate="one_to_one",
    )
    if frame[["actual_tvt", parent_column]].isna().any().any():
        raise ValueError("Stage E parent join produced missing values")
    frame = frame.sort_values(["outer_fold", "well", "id"]).reset_index(drop=True)
    truth = frame["actual_tvt"].to_numpy(np.float32)
    parent_prediction = frame[parent_column].to_numpy(np.float32)
    public_prediction = frame["strict_public_core_oof"].to_numpy(np.float32)
    crossfit = np.empty(len(frame), dtype=np.float32)
    weight_rows: list[dict[str, Any]] = []
    for held_fold in range(5):
        fit_mask = frame["outer_fold"].ne(held_fold).to_numpy()
        apply_mask = ~fit_mask
        weight = fit_convex_weight(
            truth[fit_mask],
            parent_prediction[fit_mask],
            public_prediction[fit_mask],
            public_core_weight_bounds,
        )
        crossfit[apply_mask] = (
            (1.0 - weight) * parent_prediction[apply_mask] + weight * public_prediction[apply_mask]
        ).astype(np.float32)
        weight_rows.append(
            {
                "meta_fold": held_fold,
                "fit_rows": int(fit_mask.sum()),
                "apply_rows": int(apply_mask.sum()),
                "public_core_weight": weight,
                "fit_rmse": rmse(
                    truth[fit_mask],
                    (1.0 - weight) * parent_prediction[fit_mask]
                    + weight * public_prediction[fit_mask],
                ),
            }
        )
    frame["exp413_oof"] = parent_prediction
    frame["exp497_crossfit_blend_oof"] = crossfit
    weights = pd.DataFrame(weight_rows)
    fold_rows = [
        _metric_row(
            f"outer_fold_{fold}",
            frame,
            frame["outer_fold"].eq(fold).to_numpy(),
            "exp413_oof",
            "exp497_crossfit_blend_oof",
        )
        for fold in range(5)
    ]
    pooled = _metric_row(
        "pooled",
        frame,
        np.ones(len(frame), dtype=bool),
        "exp413_oof",
        "exp497_crossfit_blend_oof",
    )
    md = frame["md_since"].to_numpy(np.float32)
    scope_masks = {
        "md_since_0_250": md <= 250.0,
        "md_since_250_1000": (md > 250.0) & (md < 1000.0),
        "md_since_1000_plus": md >= 1000.0,
    }
    scope_rows = [
        _metric_row(
            scope,
            frame,
            mask,
            "exp413_oof",
            "exp497_crossfit_blend_oof",
        )
        for scope, mask in scope_masks.items()
    ]
    hidden_path = Path(hidden_like_assignment_path)
    if sha256_file(hidden_path) != hidden_like_assignment_sha256:
        raise ValueError("Hidden-like assignment SHA mismatch")
    assignment = pd.read_csv(hidden_path, dtype={"well_id": str}).set_index("well_id")
    hidden_scopes = {
        "hidden_like_spatial": "verification_like_spatial_role",
        "hidden_like_typewell_purged": "verification_like_typewell_purged_role",
    }
    hidden_rows: list[dict[str, Any]] = []
    for scope, column in hidden_scopes.items():
        mask = frame["well"].map(assignment[column]).eq("valid").to_numpy()
        hidden_rows.append(
            _metric_row(
                scope,
                frame,
                mask,
                "exp413_oof",
                "exp497_crossfit_blend_oof",
            )
        )
    well_rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=True):
        parent_rmse = rmse(group["actual_tvt"], group["exp413_oof"])
        candidate_rmse = rmse(group["actual_tvt"], group["exp497_crossfit_blend_oof"])
        well_rows.append(
            {
                "well": str(well),
                "rows": len(group),
                "exp413_rmse": parent_rmse,
                "exp497_rmse": candidate_rmse,
                "delta_rmse_exp497_minus_exp413": candidate_rmse - parent_rmse,
            }
        )
    by_well = pd.DataFrame(well_rows)
    delta = by_well["delta_rmse_exp497_minus_exp413"].to_numpy(float)
    fixed_scope = pd.DataFrame([*scope_rows, *hidden_rows])
    checks = {
        "pooled_gain_min_0p03": float(pooled["exp413_rmse"] - pooled["exp497_rmse"]) >= 0.03,
        "nonworse_folds_5_of_5": sum(
            row["delta_rmse_exp497_minus_exp413"] <= 0.0 for row in fold_rows
        )
        == 5,
        "all_fixed_scopes_nonworse": bool(
            fixed_scope["delta_rmse_exp497_minus_exp413"].le(0.0).all()
        ),
        "by_well_p95_delta_le_0p25": float(np.quantile(delta, 0.95)) <= 0.25,
        "worst_well_delta_le_0p25": float(np.max(delta)) <= 0.25,
        "positive_public_core_weight_5_of_5": bool(weights["public_core_weight"].gt(0.0).all()),
        "public_core_weight_cap_0p30": bool(weights["public_core_weight"].le(0.30 + 1e-12).all()),
        "technical_model_count_200": sum(
            int(summary["fitted_boosters"]) for summary in stage_m_summaries
        )
        == 200,
        "parent_retraining_zero": all(
            int(summary["inventory"]["exp413_retraining"]) == 0 for summary in stage_m_summaries
        ),
    }
    passed = all(checks.values())
    frame["selected_oof"] = frame["exp497_crossfit_blend_oof"] if passed else frame["exp413_oof"]
    component_path = output_dir / "component_oof.parquet"
    weights_path = output_dir / "meta_fold_weights.csv"
    fold_path = output_dir / "fold_metrics.csv"
    scope_path = output_dir / "scope_metrics.csv"
    hidden_metrics_path = output_dir / "hidden_like_metrics.csv"
    by_well_path = output_dir / "by_well_metrics.csv"
    gate_path = output_dir / "promotion_gate.json"
    reproducibility_path = output_dir / "reproducibility_manifest.json"
    frame.to_parquet(component_path, index=False, compression="zstd")
    weights.to_csv(weights_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    pd.DataFrame(scope_rows).to_csv(scope_path, index=False)
    pd.DataFrame(hidden_rows).to_csv(hidden_metrics_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    gate = {
        "passed": passed,
        "checks": checks,
        "pooled": pooled,
        "by_well_delta_p95": float(np.quantile(delta, 0.95)),
        "worst_well": str(
            by_well.sort_values("delta_rmse_exp497_minus_exp413", ascending=False).iloc[0]["well"]
        ),
        "worst_well_delta": float(np.max(delta)),
        "pass_action": "qualify_same_exp_inference_for_separate_approval",
        "fail_action": "select_exp413_and_close_without_same_oof_rescue",
    }
    write_json(gate_path, gate)
    reproducibility = {
        "parent_oof_sha256": parent_oof_sha256,
        "hidden_like_assignment_sha256": hidden_like_assignment_sha256,
        "stage_m_summary_sha256": [sha256_file(path) for path in stage_m_summary_paths],
        "strict_public_core_oof_sha256": prediction_sha256(
            frame["id"], frame["strict_public_core_oof"].to_numpy(np.float32), "exp497:public_core"
        ),
        "crossfit_blend_oof_sha256": prediction_sha256(
            frame["id"], frame["exp497_crossfit_blend_oof"].to_numpy(np.float32), "exp497:crossfit"
        ),
        "selected_oof_sha256": prediction_sha256(
            frame["id"], frame["selected_oof"].to_numpy(np.float32), "exp497:selected"
        ),
        "deterministic_anchor": False,
    }
    write_json(reproducibility_path, reproducibility)
    summary = {
        "experiment": "exp497_strict_public_core_fold_safe_ensemble_on_exp413",
        "stage": "stage_e_meta_blend_and_gate",
        "status": "complete_gate_passed" if passed else "complete_gate_failed",
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "pooled": pooled,
        "weights": weight_rows,
        "promotion_gate": gate,
        "selected_prediction": "exp497_crossfit_blend_oof" if passed else "exp413_oof",
        "inference_generated": False,
        "submission_generated": False,
        "elapsed_seconds": round(time.time() - started, 3),
        "outputs": {
            "component_oof": component_path.name,
            "meta_fold_weights": weights_path.name,
            "fold_metrics": fold_path.name,
            "scope_metrics": scope_path.name,
            "hidden_like_metrics": hidden_metrics_path.name,
            "by_well_metrics": by_well_path.name,
            "promotion_gate": gate_path.name,
            "reproducibility_manifest": reproducibility_path.name,
        },
    }
    summary["sha256"] = {
        name: sha256_file(output_dir / filename) for name, filename in summary["outputs"].items()
    }
    summary_path = output_dir / "stage_e_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, default=_json_default), flush=True)
    return summary


def load_stage_p_well_metadata_union(
    stage_p_summary_paths: Sequence[str | Path],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if len(stage_p_summary_paths) != 5:
        raise ValueError("Stage I requires exactly five Stage P summaries")
    summaries: list[dict[str, Any]] = []
    parts: list[pd.DataFrame] = []
    for summary_path_raw in stage_p_summary_paths:
        summary_path = Path(summary_path_raw)
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete" or summary.get("stage") != "stage_p_physical_shard":
            raise ValueError(f"Invalid Stage P summary: {summary_path}")
        fold = int(summary["shard_fold"])
        metadata_path = find_artifact(
            f"stage_p_fold{fold}_well_metadata.parquet",
            roots=[summary_path.parent],
        )
        if sha256_file(metadata_path) != str(summary["sha256"]["well_metadata"]):
            raise ValueError(f"Stage P well metadata SHA mismatch: {metadata_path}")
        part = pd.read_parquet(metadata_path)
        if part["well"].nunique() != int(summary["wells"]):
            raise ValueError(f"Stage P well metadata count mismatch: {metadata_path}")
        summaries.append(summary)
        parts.append(part)
    metadata = pd.concat(parts, ignore_index=True)
    metadata["well"] = metadata["well"].astype(str)
    if len(metadata) != 773 or metadata["well"].nunique() != 773:
        raise ValueError("Stage P well metadata union must contain 773 unique wells")
    if metadata["well"].duplicated().any():
        raise ValueError("Stage P well metadata union contains duplicate wells")
    return metadata.sort_values("well").reset_index(drop=True), summaries


def build_stage_i_test_feature_frame(
    *,
    competition_data_dir: str | Path,
    output_dir: str | Path,
    public_runtime_path: str | Path | None = None,
    particles: int = 500,
    seeds: int = 128,
    n_jobs: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    started = time.time()
    competition_data_dir = Path(competition_data_dir)
    test_dir = competition_data_dir / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing current raw test directory: {test_dir}")
    runtime = load_public_replay_module(public_runtime_path)
    required = ("configure_public_runtime", "build_replay_test_frame")
    if not all(hasattr(runtime, name) for name in required):
        raise AttributeError(f"Public runtime is missing Stage I helpers: {required}")
    runtime.configure_public_runtime(
        data_dir=competition_data_dir,
        output_dir=output_dir,
        n_jobs=n_jobs,
        pf_seeds=seeds,
        pf_particles=particles,
        fast=False,
        use_gpu="auto",
        n_train_wells=0,
    )
    base_test, replay_meta = runtime.build_replay_test_frame()
    base_test["id"] = base_test["id"].astype(str)
    base_test["well"] = base_test["well"].astype(str)
    base_test = base_test.drop(columns=list(LEARNED_LIKPF_COLUMNS), errors="ignore")
    test_wells = sorted(base_test["well"].unique())
    if not test_wells or base_test["id"].duplicated().any():
        raise ValueError("Stage I public replay test frame has invalid identity")
    physical_parts: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    for index, well in enumerate(test_wells, start=1):
        physical, metadata = build_physical_well(
            runtime,
            test_dir,
            well,
            particles=particles,
            seeds=seeds,
        )
        physical_parts.append(physical)
        metadata_rows.append(metadata)
        print(
            json.dumps(
                {
                    "event": "stage_i_test_physical_well_complete",
                    "index": index,
                    "total": len(test_wells),
                    "well": well,
                    "rows": int(metadata["rows"]),
                }
            ),
            flush=True,
        )
    physical = pd.concat(physical_parts, ignore_index=True)
    test_frame = base_test.merge(physical, on=["id", "well"], how="left", validate="one_to_one")
    required_columns = [
        *LEARNED_LIKPF_COLUMNS,
        *(f"selector__{variant}" for variant in SELECTOR_VARIANTS),
    ]
    if test_frame[required_columns].isna().any().any():
        raise ValueError("Stage I physical merge produced missing values")
    test_metadata = pd.DataFrame(metadata_rows)
    test_metadata["well"] = test_metadata["well"].astype(str)
    for name in ("_FI", "_DI"):
        if name in runtime.__dict__:
            runtime.__dict__[name] = None
    gc.collect()
    summary = {
        "rows": len(test_frame),
        "wells": len(test_wells),
        "particles": particles,
        "seeds": seeds,
        "replay_meta": replay_meta,
        "public_runtime_path": runtime.__dict__["_exp497_source_path"],
        "public_runtime_sha256": sha256_file(runtime.__dict__["_exp497_source_path"]),
        "selector_seed_bases": {
            str(row["well"]): int(row["selector_seed_base"]) for row in metadata_rows
        },
        "learned_seed_bases": {
            str(row["well"]): int(row["learned_seed_base"]) for row in metadata_rows
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    return test_frame, test_metadata, summary


def _fit_stage_i_branch(
    *,
    branch: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    features: list[str],
    inner_fold: np.ndarray,
    target_delta: np.ndarray,
    model_output_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    from catboost import CatBoostRegressor
    from lightgbm import Booster, LGBMRegressor, early_stopping, log_evaluation
    from sklearn.linear_model import Ridge

    model_output_dir = Path(model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    x_train = train_frame[features].to_numpy(np.float32)
    x_test = test_frame[features].to_numpy(np.float32)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError(f"Stage I {branch} feature matrix contains non-finite values")
    config_specs: list[tuple[str, str, Mapping[str, Any]]] = [
        (f"lgb{index}", "lightgbm", params)
        for index, params in enumerate(_lgb_configs())
    ] + [
        (f"cat{index}", "catboost", params)
        for index, params in enumerate(_cat_configs())
    ]
    train_columns: list[np.ndarray] = []
    test_columns: list[np.ndarray] = []
    manifest: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for config_name, config_kind, params in config_specs:
        oof = np.empty(len(train_frame), dtype=np.float32)
        test_predictions: list[np.ndarray] = []
        importances: list[np.ndarray] = []
        for inner in range(4):
            train_mask = inner_fold != inner
            validation_mask = inner_fold == inner
            if config_kind == "lightgbm":
                model = LGBMRegressor(**dict(params))
                model.fit(
                    x_train[train_mask],
                    target_delta[train_mask],
                    eval_set=[(x_train[validation_mask], target_delta[validation_mask])],
                    eval_metric="rmse",
                    callbacks=[early_stopping(250, verbose=False), log_evaluation(0)],
                )
                best_iteration = int(model.best_iteration_)
                oof[validation_mask] = model.predict(
                    x_train[validation_mask], num_iteration=best_iteration
                ).astype(np.float32)
                test_prediction = model.predict(
                    x_test, num_iteration=best_iteration
                ).astype(np.float32)
                test_predictions.append(test_prediction)
                importances.append(model.booster_.feature_importance(importance_type="gain"))
                model_path = (
                    model_output_dir / f"{branch}__{config_name}__inner{inner}.txt"
                )
                model.booster_.save_model(
                    str(model_path),
                    num_iteration=best_iteration,
                )
            else:
                model = CatBoostRegressor(**dict(params))
                model.fit(
                    x_train[train_mask],
                    target_delta[train_mask],
                    eval_set=(x_train[validation_mask], target_delta[validation_mask]),
                    early_stopping_rounds=250,
                    use_best_model=True,
                )
                best_iteration = int(model.get_best_iteration()) + 1
                oof[validation_mask] = model.predict(x_train[validation_mask]).astype(
                    np.float32
                )
                test_prediction = model.predict(x_test).astype(np.float32)
                test_predictions.append(test_prediction)
                importances.append(np.asarray(model.get_feature_importance(), dtype=np.float64))
                model_path = (
                    model_output_dir / f"{branch}__{config_name}__inner{inner}.cbm"
                )
                model.save_model(str(model_path), format="cbm")
            if config_kind == "lightgbm":
                reloaded_model = Booster(model_file=str(model_path))
                reloaded_prediction = reloaded_model.predict(
                    x_test, num_iteration=best_iteration
                ).astype(np.float32)
            else:
                reloaded_model = CatBoostRegressor()
                reloaded_model.load_model(str(model_path), format="cbm")
                reloaded_prediction = reloaded_model.predict(x_test).astype(np.float32)
            serialization_max_abs = float(
                np.max(
                    np.abs(
                        reloaded_prediction.astype(np.float64)
                        - test_prediction.astype(np.float64)
                    )
                )
            )
            if serialization_max_abs > 1e-5:
                raise ValueError(
                    f"Stage I serialized {branch}/{config_name}/inner{inner} "
                    f"prediction drift is {serialization_max_abs}"
                )
            if not model_path.is_file() or model_path.stat().st_size <= 0:
                raise ValueError(f"Stage I model serialization failed: {model_path}")
            row = {
                "branch": branch,
                "config": config_name,
                "kind": config_kind,
                "inner_fold": inner,
                "train_rows": int(train_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "best_iteration": best_iteration,
                "rmse_delta": rmse(target_delta[validation_mask], oof[validation_mask]),
                "model_file": model_path.relative_to(model_output_dir.parent).as_posix(),
                "model_sha256": sha256_file(model_path),
                "model_bytes": model_path.stat().st_size,
                "serialization_test_rows": len(test_frame),
                "serialization_max_abs": serialization_max_abs,
            }
            manifest.append(row)
            print(json.dumps({"event": "stage_i_model_complete", **row}), flush=True)
            del model, reloaded_model, reloaded_prediction
            gc.collect()
        train_columns.append(oof)
        test_columns.append(np.mean(np.stack(test_predictions), axis=0).astype(np.float32))
        mean_importance = np.mean(np.stack(importances), axis=0)
        importance_rows.extend(
            {
                "branch": branch,
                "config": config_name,
                "feature": feature,
                "importance": float(value),
            }
            for feature, value in zip(features, mean_importance, strict=True)
        )
    train_matrix = np.column_stack(train_columns).astype(np.float32)
    test_matrix = np.column_stack(test_columns).astype(np.float32)
    ridge = Ridge(alpha=1.66, positive=True, fit_intercept=True, tol=0.0005)
    ridge.fit(train_matrix, target_delta)
    train_stack = ridge.predict(train_matrix).astype(np.float32)
    test_stack = ridge.predict(test_matrix).astype(np.float32)
    ridge_meta = {
        "branch": branch,
        "alpha": 1.66,
        "positive": True,
        "coef": ridge.coef_.astype(float).tolist(),
        "intercept": float(ridge.intercept_),
        "train_oof_rmse_delta": rmse(target_delta, train_stack),
    }
    del x_train, x_test, train_matrix, test_matrix, train_columns, test_columns
    gc.collect()
    return train_stack, test_stack, manifest, pd.DataFrame(importance_rows), ridge_meta


def validate_stage_i_serialized_model_manifest(
    output_dir: str | Path,
    model_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if len(model_rows) != 40:
        raise ValueError("Stage I serialized model manifest must contain 40 rows")
    kinds = pd.Series([str(row["kind"]) for row in model_rows], dtype=str).value_counts()
    if int(kinds.get("lightgbm", 0)) != 24 or int(kinds.get("catboost", 0)) != 16:
        raise ValueError("Stage I serialized models must be 24 LightGBM and 16 CatBoost")
    paths: list[str] = []
    sha_rows: list[dict[str, str]] = []
    for row in model_rows:
        relative = Path(str(row["model_file"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe Stage I model path: {relative}")
        if str(row["kind"]) == "lightgbm" and relative.suffix != ".txt":
            raise ValueError(f"Unexpected LightGBM model suffix: {relative}")
        if str(row["kind"]) == "catboost" and relative.suffix != ".cbm":
            raise ValueError(f"Unexpected CatBoost model suffix: {relative}")
        path = output_dir / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Missing Stage I serialized model: {path}")
        observed_sha = sha256_file(path)
        if observed_sha != str(row["model_sha256"]):
            raise ValueError(f"Stage I serialized model SHA mismatch: {relative}")
        if int(row["model_bytes"]) != path.stat().st_size:
            raise ValueError(f"Stage I serialized model byte-size mismatch: {relative}")
        if float(row["serialization_max_abs"]) > 1e-5:
            raise ValueError(f"Stage I serialized model parity failed: {relative}")
        paths.append(relative.as_posix())
        sha_rows.append({"model_file": relative.as_posix(), "model_sha256": observed_sha})
    if len(set(paths)) != 40:
        raise ValueError("Stage I serialized model paths are not unique")
    return {
        "serialized_model_count": 40,
        "serialized_lightgbm_count": 24,
        "serialized_catboost_count": 16,
        "serialized_model_bytes": int(
            sum((output_dir / Path(path)).stat().st_size for path in paths)
        ),
        "serialized_model_set_sha256": sha256_json(sha_rows),
    }


STAGE_I_STACK_CONFIGS = ("lgb0", "lgb1", "lgb2", "cat0", "cat1")
STAGE_I_ARTIFACT_FILES = {
    "model_manifest": "stage_i_full_fit_model_manifest.json",
    "ridge_weights": "stage_i_ridge_weights.json",
    "weights": "stage_i_weights.json",
    "selector_policy": "stage_i_selector_policy.json",
    "feature_schema": "stage_i_feature_schema.csv",
    "reproducibility_manifest": "stage_i_reproducibility_manifest.json",
}


def apply_stage_i_ridge_stack(
    config_predictions: np.ndarray,
    ridge_meta: Mapping[str, Any],
) -> np.ndarray:
    matrix = np.asarray(config_predictions, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(STAGE_I_STACK_CONFIGS):
        raise ValueError("Stage I Ridge stack must contain five config columns")
    coefficients = np.asarray(ridge_meta.get("coef", []), dtype=np.float32)
    intercept = float(ridge_meta.get("intercept", np.nan))
    if (
        coefficients.shape != (len(STAGE_I_STACK_CONFIGS),)
        or not np.isfinite(coefficients).all()
        or not np.isfinite(intercept)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Stage I Ridge stack metadata or predictions are invalid")
    prediction = matrix @ coefficients + np.float32(intercept)
    if not np.isfinite(prediction).all():
        raise ValueError("Stage I Ridge stack produced non-finite predictions")
    return prediction.astype(np.float32)


def load_stage_i_saved_inference_artifacts(
    artifact_dir: str | Path,
    *,
    expected_sha256: Mapping[str, str],
    expected_model_set_sha256: str,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    paths = {name: artifact_dir / filename for name, filename in STAGE_I_ARTIFACT_FILES.items()}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage I saved inference artifacts are missing: {missing}")
    if set(expected_sha256) != set(STAGE_I_ARTIFACT_FILES):
        raise ValueError("Stage I expected artifact SHA map is incomplete")
    observed_sha256 = {name: sha256_file(path) for name, path in paths.items()}
    mismatched = {
        name: {"expected": str(expected_sha256[name]), "observed": observed_sha256[name]}
        for name in paths
        if observed_sha256[name] != str(expected_sha256[name])
    }
    if mismatched:
        raise ValueError(f"Stage I saved artifact SHA mismatch: {mismatched}")

    model_manifest = json.loads(paths["model_manifest"].read_text())
    ridge_weights = json.loads(paths["ridge_weights"].read_text())
    internal_weights = json.loads(paths["weights"].read_text())
    selector_rows = json.loads(paths["selector_policy"].read_text())
    reproducibility = json.loads(paths["reproducibility_manifest"].read_text())
    model_rows = [
        *model_manifest.get("sp45_models", []),
        *model_manifest.get("learned_models", []),
    ]
    serialization = validate_stage_i_serialized_model_manifest(artifact_dir, model_rows)
    if (
        serialization["serialized_model_set_sha256"] != str(expected_model_set_sha256)
        or model_manifest.get("serialized_model_set_sha256") != str(expected_model_set_sha256)
        or reproducibility.get("serialized_model_set_sha256") != str(expected_model_set_sha256)
    ):
        raise ValueError("Stage I serialized model-set SHA contract failed")
    if (
        int(model_manifest.get("fitted_boosters", -1)) != 40
        or int(model_manifest.get("fitted_ridge_models", -1)) != 2
        or int(ridge_weights.get("ridge_model_count", -1)) != 2
    ):
        raise ValueError("Stage I saved model inventory contract failed")
    for branch, manifest_key in (
        ("sp45_residual", "sp45_ridge"),
        ("learned_trajectory", "learned_ridge"),
    ):
        if sha256_json(ridge_weights.get(branch)) != sha256_json(
            model_manifest.get(manifest_key)
        ):
            raise ValueError(f"Stage I saved Ridge metadata mismatch: {branch}")

    schema = pd.read_csv(paths["feature_schema"])
    if list(schema.columns) != ["branch", "feature_index", "feature"]:
        raise ValueError("Stage I feature schema columns changed")
    feature_lists: dict[str, list[str]] = {}
    for branch, manifest_key in (
        ("sp45_residual", "base_features"),
        ("learned_trajectory", "learned_features"),
    ):
        branch_schema = schema.loc[schema["branch"].eq(branch)].sort_values("feature_index")
        if branch_schema["feature_index"].tolist() != list(range(len(branch_schema))):
            raise ValueError(f"Stage I feature indices are not contiguous: {branch}")
        features = branch_schema["feature"].astype(str).tolist()
        if features != [str(value) for value in model_manifest.get(manifest_key, [])]:
            raise ValueError(f"Stage I feature schema differs from manifest: {branch}")
        feature_lists[branch] = features

    if not isinstance(selector_rows, list):
        raise ValueError("Stage I selector policy must be a list")
    deployment_rows = [row for row in selector_rows if row.get("role") == "current_test"]
    if len(deployment_rows) != 1 or deployment_rows[0].get("inner_fold") is not None:
        raise ValueError("Stage I deployment selector policy is not unique")
    deployment_policy = {
        key: value
        for key, value in deployment_rows[0].items()
        if key not in {"role", "inner_fold"}
    }
    weights = np.asarray(internal_weights.get("meta_fold_weights", []), dtype=np.float64)
    deployment_weight = float(internal_weights.get("deployment_weight", np.nan))
    required_internal_weights = (
        "sp45_model_weight",
        "projection_weight",
        "learned_model_weight",
        "projected_sp45_weight",
    )
    if (
        weights.shape != (5,)
        or not np.isfinite(weights).all()
        or not np.isfinite(deployment_weight)
        or abs(float(np.median(weights)) - deployment_weight) > 1e-15
        or any(
            not np.isfinite(float(internal_weights.get(name, np.nan)))
            for name in required_internal_weights
        )
    ):
        raise ValueError("Stage I saved internal weight contract failed")

    return {
        "artifact_dir": artifact_dir,
        "paths": paths,
        "sha256": observed_sha256,
        "model_manifest": model_manifest,
        "model_rows": model_rows,
        "ridge_weights": ridge_weights,
        "internal_weights": internal_weights,
        "deployment_policy": deployment_policy,
        "feature_lists": feature_lists,
        "serialization": serialization,
        "reproducibility": reproducibility,
    }


def _predict_stage_i_saved_branch(
    *,
    branch: str,
    artifact_dir: str | Path,
    model_rows: Sequence[Mapping[str, Any]],
    ridge_meta: Mapping[str, Any],
    test_frame: pd.DataFrame,
    features: Sequence[str],
    prediction_chunk_size: int = 250_000,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from catboost import CatBoostRegressor
    from lightgbm import Booster

    if prediction_chunk_size <= 0:
        raise ValueError("Stage I prediction chunk size must be positive")
    branch_rows = [row for row in model_rows if str(row.get("branch")) == branch]
    if len(branch_rows) != 20:
        raise ValueError(f"Stage I saved branch must contain 20 models: {branch}")
    feature_names = [str(value) for value in features]
    missing = sorted(set(feature_names) - set(test_frame.columns))
    if missing:
        raise ValueError(f"Stage I saved {branch} test features are missing: {missing[:20]}")
    matrix = test_frame[feature_names].to_numpy(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Stage I saved {branch} feature matrix contains non-finite values")

    artifact_dir = Path(artifact_dir)
    config_predictions: list[np.ndarray] = []
    audit_rows: list[dict[str, Any]] = []
    for config_name in STAGE_I_STACK_CONFIGS:
        rows = sorted(
            [row for row in branch_rows if str(row.get("config")) == config_name],
            key=lambda row: int(row["inner_fold"]),
        )
        expected_kind = "lightgbm" if config_name.startswith("lgb") else "catboost"
        if (
            len(rows) != 4
            or [int(row["inner_fold"]) for row in rows] != list(range(4))
            or any(str(row.get("kind")) != expected_kind for row in rows)
        ):
            raise ValueError(f"Stage I saved config coverage mismatch: {branch}/{config_name}")
        config_sum = np.zeros(len(test_frame), dtype=np.float64)
        for row in rows:
            model_path = artifact_dir / str(row["model_file"])
            prediction = np.empty(len(test_frame), dtype=np.float32)
            if expected_kind == "lightgbm":
                model = Booster(model_file=str(model_path))
            else:
                model = CatBoostRegressor()
                model.load_model(str(model_path), format="cbm")
            for start in range(0, len(test_frame), prediction_chunk_size):
                stop = min(start + prediction_chunk_size, len(test_frame))
                if expected_kind == "lightgbm":
                    values = model.predict(
                        matrix[start:stop],
                        num_iteration=int(row["best_iteration"]),
                    )
                else:
                    values = model.predict(matrix[start:stop])
                prediction[start:stop] = np.asarray(values, dtype=np.float32)
            if not np.isfinite(prediction).all():
                raise ValueError(f"Stage I saved model produced non-finite values: {model_path}")
            config_sum += prediction.astype(np.float64)
            audit_rows.append(
                {
                    "branch": branch,
                    "config": config_name,
                    "inner_fold": int(row["inner_fold"]),
                    "kind": expected_kind,
                    "model_file": str(row["model_file"]),
                    "model_sha256": str(row["model_sha256"]),
                    "best_iteration": int(row["best_iteration"]),
                    "prediction_rows": len(prediction),
                }
            )
            del model, prediction
            gc.collect()
        config_predictions.append((config_sum / 4.0).astype(np.float32))
    stacked = np.column_stack(config_predictions).astype(np.float32)
    prediction = apply_stage_i_ridge_stack(stacked, ridge_meta)
    del matrix, stacked, config_predictions
    gc.collect()
    return prediction, audit_rows


def validate_stage_i_visible_parity(
    *,
    strict_public_core_max_abs: float,
    blend_max_abs: float,
    strict_public_core_tolerance: float,
    blend_tolerance: float,
    exp413_max_abs: float | None = None,
) -> dict[str, Any]:
    values = {
        "strict_public_core_max_abs": float(strict_public_core_max_abs),
        "blend_max_abs": float(blend_max_abs),
        "strict_public_core_tolerance": float(strict_public_core_tolerance),
        "blend_tolerance": float(blend_tolerance),
    }
    if exp413_max_abs is not None:
        values["exp413_max_abs"] = float(exp413_max_abs)
    if (
        not np.isfinite(list(values.values())).all()
        or values["strict_public_core_max_abs"] < 0.0
        or values["blend_max_abs"] < 0.0
        or values["strict_public_core_tolerance"] <= 0.0
        or values["blend_tolerance"] <= 0.0
        or values.get("exp413_max_abs", 0.0) < 0.0
    ):
        raise ValueError(f"Invalid Stage I visible parity values: {values}")
    values["passed"] = bool(
        values["strict_public_core_max_abs"]
        <= values["strict_public_core_tolerance"]
        and values["blend_max_abs"] <= values["blend_tolerance"]
    )
    if not values["passed"]:
        raise ValueError(f"Stage I visible saved-model parity failed: {values}")
    return values


def run_stage_i_saved_model_inference(
    *,
    output_dir: str | Path,
    artifact_dir: str | Path,
    expected_artifact_sha256: Mapping[str, str],
    expected_model_set_sha256: str,
    test_frame: pd.DataFrame,
    test_well_metadata: pd.DataFrame,
    sample_submission_path: str | Path,
    exp413_prediction_frame: pd.DataFrame,
    exp413_runtime_metrics: Mapping[str, Any],
    test_feature_summary: Mapping[str, Any],
    submission_output_path: str | Path,
    prediction_chunk_size: int = 250_000,
    visible_strict_public_core_max_abs: float = 2e-3,
    visible_blend_max_abs: float = 2e-2,
) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(artifact_dir)
    artifacts = load_stage_i_saved_inference_artifacts(
        artifact_dir,
        expected_sha256=expected_artifact_sha256,
        expected_model_set_sha256=expected_model_set_sha256,
    )
    test_frame = test_frame.copy()
    test_frame["id"] = test_frame["id"].astype(str)
    test_frame["well"] = test_frame["well"].astype(str)
    test_frame = test_frame.sort_values(["well", "md_since", "id"]).reset_index(drop=True)
    test_well_metadata = test_well_metadata.copy()
    test_well_metadata["well"] = test_well_metadata["well"].astype(str)
    if (
        test_frame.empty
        or test_frame["id"].duplicated().any()
        or test_well_metadata["well"].duplicated().any()
        or set(test_well_metadata["well"]) != set(test_frame["well"])
    ):
        raise ValueError("Stage I saved-model test identity contract failed")

    selector_test = apply_selector_policy(
        test_frame,
        test_well_metadata,
        artifacts["deployment_policy"],
    )
    sp45_test_delta, sp45_model_audit = _predict_stage_i_saved_branch(
        branch="sp45_residual",
        artifact_dir=artifact_dir,
        model_rows=artifacts["model_rows"],
        ridge_meta=artifacts["ridge_weights"]["sp45_residual"],
        test_frame=test_frame,
        features=artifacts["feature_lists"]["sp45_residual"],
        prediction_chunk_size=prediction_chunk_size,
    )
    learned_test_delta, learned_model_audit = _predict_stage_i_saved_branch(
        branch="learned_trajectory",
        artifact_dir=artifact_dir,
        model_rows=artifacts["model_rows"],
        ridge_meta=artifacts["ridge_weights"]["learned_trajectory"],
        test_frame=test_frame,
        features=artifacts["feature_lists"]["learned_trajectory"],
        prediction_chunk_size=prediction_chunk_size,
    )
    internal_weights = artifacts["internal_weights"]
    last_test = test_frame["last_known_tvt"].to_numpy(np.float32)
    sp45_model_test = last_test + sp45_test_delta
    sp45_model_weight = float(internal_weights["sp45_model_weight"])
    sp45_raw_test = (
        (1.0 - sp45_model_weight) * selector_test
        + sp45_model_weight * sp45_model_test
    ).astype(np.float32)
    projected_test_full = robust_u_projection(test_frame, sp45_raw_test, degree=3)
    projection_weight = float(internal_weights["projection_weight"])
    projected_test = (
        (1.0 - projection_weight) * sp45_raw_test
        + projection_weight * projected_test_full
    ).astype(np.float32)
    warmup = 1.0 - np.exp(
        -np.maximum(test_frame["md_since"].to_numpy(np.float32), 0.0) / 85.0
    )
    learned_model_test = last_test + warmup * learned_test_delta
    learned_model_weight = float(internal_weights["learned_model_weight"])
    learned_test = (
        (1.0 - learned_model_weight) * test_frame["likpf_scale_5"].to_numpy(np.float32)
        + learned_model_weight * learned_model_test
    ).astype(np.float32)
    learned_test = savgol_by_well(test_frame, learned_test, window=61, polyorder=3)
    projected_sp45_weight = float(internal_weights["projected_sp45_weight"])
    public_core_test = (
        projected_sp45_weight * projected_test
        + (1.0 - projected_sp45_weight) * learned_test
    ).astype(np.float32)

    exp413 = exp413_prediction_frame.copy()
    required_exp413 = {"id", "well", "pred_tvt"}
    missing_exp413 = sorted(required_exp413 - set(exp413.columns))
    if missing_exp413:
        raise ValueError(f"Dynamic exp413 prediction columns are missing: {missing_exp413}")
    exp413 = exp413[["id", "well", "pred_tvt"]].rename(
        columns={"pred_tvt": "exp413_pred_tvt"}
    )
    exp413["id"] = exp413["id"].astype(str)
    exp413["well"] = exp413["well"].astype(str)
    if (
        exp413["id"].duplicated().any()
        or len(exp413) != len(test_frame)
        or set(exp413["id"]) != set(test_frame["id"])
        or not np.isfinite(exp413["exp413_pred_tvt"].to_numpy(np.float64)).all()
    ):
        raise ValueError("Dynamic exp413/exp497 hidden identity contract failed")
    if (
        int(exp413_runtime_metrics.get("booster_training_count", -1)) != 0
        or int(exp413_runtime_metrics.get("parent_selector_model_count", -1)) != 40
        or int(exp413_runtime_metrics.get("signed_selector_model_count", -1)) != 20
        or int(exp413_runtime_metrics.get("tvt_model_count", -1)) != 15
        or bool(exp413_runtime_metrics.get("external_submission_performed", True))
    ):
        raise ValueError("Dynamic exp413 saved-model runtime contract failed")

    components = pd.DataFrame(
        {
            "id": test_frame["id"],
            "well": test_frame["well"],
            "last_known_tvt": last_test,
            "physical_selector_pred_tvt": selector_test,
            "sp45_residual_model_pred_tvt": sp45_model_test,
            "sp45_raw_pred_tvt": sp45_raw_test,
            "projected_sp45_pred_tvt": projected_test,
            "learned_trajectory_pred_tvt": learned_test,
            "strict_public_core_pred_tvt": public_core_test,
        }
    ).merge(exp413, on=["id", "well"], how="left", validate="one_to_one")
    if components["exp413_pred_tvt"].isna().any():
        raise ValueError("Dynamic exp413 join produced missing predictions")
    deployment_weight = float(internal_weights["deployment_weight"])
    components["exp497_blend_pred_tvt"] = (
        (1.0 - deployment_weight) * components["exp413_pred_tvt"].to_numpy(np.float32)
        + deployment_weight * public_core_test
    ).astype(np.float32)

    sample = pd.read_csv(sample_submission_path, dtype={"id": str})
    if (
        list(sample.columns) != ["id", "tvt"]
        or sample.empty
        or sample["id"].duplicated().any()
        or len(sample) != len(components)
        or set(sample["id"]) != set(components["id"])
    ):
        raise ValueError("Stage I saved-model sample identity contract failed")
    predictions = sample[["id"]].merge(components, on="id", how="left", validate="one_to_one")
    if not predictions["id"].equals(sample["id"]):
        raise ValueError("Stage I saved-model prediction order differs from sample")
    numeric_columns = [column for column in predictions if column not in {"id", "well"}]
    if predictions[numeric_columns].isna().any().any() or not np.isfinite(
        predictions[numeric_columns].to_numpy(np.float64)
    ).all():
        raise ValueError("Stage I saved-model predictions contain missing/non-finite values")

    visible_reference_path = artifact_dir / "stage_i_current_test_predictions.csv.gz"
    visible_parity: dict[str, Any] = {"applicable": False}
    if visible_reference_path.is_file():
        reference = pd.read_csv(visible_reference_path, dtype={"id": str})
        if (
            len(reference) == len(predictions)
            and not reference["id"].duplicated().any()
            and set(reference["id"]) == set(predictions["id"])
        ):
            comparison = predictions[[
                "id",
                "strict_public_core_pred_tvt",
                "exp497_blend_pred_tvt",
            ]].merge(
                reference[[
                    "id",
                    "strict_public_core_pred_tvt",
                    "exp497_blend_pred_tvt",
                ]],
                on="id",
                validate="one_to_one",
                suffixes=("_saved", "_reference"),
            )
            strict_max = float(
                np.max(
                    np.abs(
                        comparison["strict_public_core_pred_tvt_saved"].to_numpy(np.float64)
                        - comparison["strict_public_core_pred_tvt_reference"].to_numpy(np.float64)
                    )
                )
            )
            blend_max = float(
                np.max(
                    np.abs(
                        comparison["exp497_blend_pred_tvt_saved"].to_numpy(np.float64)
                        - comparison["exp497_blend_pred_tvt_reference"].to_numpy(np.float64)
                    )
                )
            )
            exp413_max = None
            if "exp413_pred_tvt" in reference.columns:
                exp413_comparison = predictions[["id", "exp413_pred_tvt"]].merge(
                    reference[["id", "exp413_pred_tvt"]],
                    on="id",
                    validate="one_to_one",
                    suffixes=("_saved", "_reference"),
                )
                exp413_max = float(
                    np.max(
                        np.abs(
                            exp413_comparison["exp413_pred_tvt_saved"].to_numpy(
                                np.float64
                            )
                            - exp413_comparison["exp413_pred_tvt_reference"].to_numpy(
                                np.float64
                            )
                        )
                    )
                )
            visible_parity = {
                "applicable": True,
                **validate_stage_i_visible_parity(
                    strict_public_core_max_abs=strict_max,
                    blend_max_abs=blend_max,
                    strict_public_core_tolerance=visible_strict_public_core_max_abs,
                    blend_tolerance=visible_blend_max_abs,
                    exp413_max_abs=exp413_max,
                ),
            }

    prediction_path = output_dir / "exp497_saved_model_predictions.csv.gz"
    model_audit_path = output_dir / "exp497_saved_model_audit.csv"
    summary_path = output_dir / "exp497_saved_model_inference_summary.json"
    reproducibility_path = output_dir / "exp497_saved_model_reproducibility.json"
    submission_output_path = Path(submission_output_path)
    submission_output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = predictions[["id", "exp497_blend_pred_tvt"]].rename(
        columns={"exp497_blend_pred_tvt": "tvt"}
    )
    predictions.to_csv(prediction_path, index=False, compression="gzip")
    pd.DataFrame([*sp45_model_audit, *learned_model_audit]).to_csv(
        model_audit_path,
        index=False,
    )
    submission.to_csv(submission_output_path, index=False)
    if not submission_output_path.is_file():
        raise RuntimeError("Stage I saved-model submission.csv was not written")

    reproducibility = {
        "artifact_sha256": artifacts["sha256"],
        "serialized_model_set_sha256": expected_model_set_sha256,
        "strict_public_core_prediction_sha256": prediction_sha256(
            predictions["id"],
            predictions["strict_public_core_pred_tvt"].to_numpy(np.float32),
            "exp497:saved_model_inference:strict_public_core",
        ),
        "blend_prediction_sha256": prediction_sha256(
            predictions["id"],
            predictions["exp497_blend_pred_tvt"].to_numpy(np.float32),
            "exp497:saved_model_inference:blend",
        ),
        "prediction_file_decompressed_sha256": sha256_gzip_decompressed(prediction_path),
        "submission_sha256": sha256_file(submission_output_path),
        "test_feature_summary": dict(test_feature_summary),
        "exp413_runtime": dict(exp413_runtime_metrics),
        "visible_parity": visible_parity,
        "deterministic_anchor": False,
        "deterministic_anchor_note": "requires a same-source hidden rerun SHA match",
    }
    write_json(reproducibility_path, reproducibility)
    summary = {
        "experiment": "exp497_strict_public_core_fold_safe_ensemble_on_exp413",
        "stage": "stage_i_saved_model_hidden_safe_inference",
        "status": "complete",
        "train_gate_passed": False,
        "selected_train_anchor": "exp413",
        "rows": len(predictions),
        "wells": predictions["well"].nunique(),
        "fitted_boosters": 0,
        "loaded_exp497_boosters": 40,
        "loaded_exp497_ridge_models": 2,
        "loaded_exp413_boosters": 75,
        "exp413_retraining": 0,
        "exp497_retraining": 0,
        "submission_generated": True,
        "external_submission_performed": False,
        "deployment_weight": deployment_weight,
        "prediction_stats": {
            "mean": float(submission["tvt"].mean()),
            "std": float(submission["tvt"].std()),
            "min": float(submission["tvt"].min()),
            "max": float(submission["tvt"].max()),
        },
        "visible_parity": visible_parity,
        "outputs": {
            "predictions": prediction_path.name,
            "model_audit": model_audit_path.name,
            "submission": str(submission_output_path),
            "reproducibility": reproducibility_path.name,
        },
        "reproducibility": reproducibility,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, default=_json_default), flush=True)
    return summary


def run_stage_i_current_test(
    *,
    output_dir: str | Path,
    stage_p_feature_paths: Sequence[str | Path],
    stage_p_summary_paths: Sequence[str | Path],
    test_frame: pd.DataFrame,
    test_well_metadata: pd.DataFrame,
    sample_submission_path: str | Path,
    exp413_prediction_path: str | Path,
    exp413_prediction_sha256: str,
    exp413_prediction_decompressed_sha256: str,
    meta_fold_weights: Sequence[float],
    deployment_weight: float,
    test_feature_summary: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame, stage_p_summaries = load_stage_p_union(
        stage_p_feature_paths,
        stage_p_summary_paths,
    )
    train_metadata, _ = load_stage_p_well_metadata_union(stage_p_summary_paths)
    train_frame = train_frame.sort_values(["well", "md_since", "id"]).reset_index(drop=True)
    test_frame = test_frame.copy()
    test_frame["id"] = test_frame["id"].astype(str)
    test_frame["well"] = test_frame["well"].astype(str)
    test_frame = test_frame.sort_values(["well", "md_since", "id"]).reset_index(drop=True)
    test_well_metadata = test_well_metadata.copy()
    test_well_metadata["well"] = test_well_metadata["well"].astype(str)
    if (
        test_well_metadata["well"].duplicated().any()
        or test_well_metadata["well"].nunique() != test_frame["well"].nunique()
        or set(test_well_metadata["well"]) != set(test_frame["well"])
    ):
        raise ValueError("Stage I current-test well metadata contract failed")
    base_features, learned_features = base_and_learned_feature_columns(train_frame)
    missing_test = sorted(set(learned_features) - set(test_frame.columns))
    if missing_test:
        raise ValueError(f"Stage I test feature schema is missing columns: {missing_test[:20]}")
    if not np.isfinite(test_frame[learned_features].to_numpy(np.float32)).all():
        raise ValueError("Stage I test features contain non-finite values")
    all_train_indices = np.arange(len(train_frame), dtype=np.int64)
    inner_fold = _inner_fold_for_outer_train(train_frame, all_train_indices)
    target_delta = train_frame["target"].to_numpy(np.float32)
    train_truth = (
        train_frame["last_known_tvt"].to_numpy(np.float32) + target_delta
    ).astype(np.float32)
    train_wells = set(train_frame["well"].astype(str))
    selector_train = np.empty(len(train_frame), dtype=np.float32)
    selector_policy_rows: list[dict[str, Any]] = []
    for inner in range(4):
        fit_wells = set(train_frame.loc[inner_fold != inner, "well"].astype(str))
        policy = fit_selector_policy(train_metadata, train_frame, fit_wells)
        selector_train[inner_fold == inner] = apply_selector_policy(
            train_frame.loc[inner_fold == inner],
            train_metadata,
            policy,
        )
        selector_policy_rows.append({"role": "inner_oof", "inner_fold": inner, **policy})
    deployment_policy = fit_selector_policy(train_metadata, train_frame, train_wells)
    selector_test = apply_selector_policy(test_frame, test_well_metadata, deployment_policy)
    selector_policy_rows.append(
        {"role": "current_test", "inner_fold": None, **deployment_policy}
    )
    model_output_dir = output_dir / "stage_i_models"
    (
        sp45_train_delta,
        sp45_test_delta,
        sp45_models,
        sp45_importance,
        sp45_ridge,
    ) = _fit_stage_i_branch(
        branch="sp45_residual",
        train_frame=train_frame,
        test_frame=test_frame,
        features=base_features,
        inner_fold=inner_fold,
        target_delta=target_delta,
        model_output_dir=model_output_dir,
    )
    (
        learned_train_delta,
        learned_test_delta,
        learned_models,
        learned_importance,
        learned_ridge,
    ) = _fit_stage_i_branch(
        branch="learned_trajectory",
        train_frame=train_frame,
        test_frame=test_frame,
        features=learned_features,
        inner_fold=inner_fold,
        target_delta=target_delta,
        model_output_dir=model_output_dir,
    )
    last_train = train_frame["last_known_tvt"].to_numpy(np.float32)
    last_test = test_frame["last_known_tvt"].to_numpy(np.float32)
    sp45_model_train = last_train + sp45_train_delta
    sp45_model_test = last_test + sp45_test_delta
    sp45_model_weight = fit_convex_weight(
        train_truth,
        selector_train,
        sp45_model_train,
        (0.15, 0.45),
    )
    sp45_raw_train = (
        (1.0 - sp45_model_weight) * selector_train
        + sp45_model_weight * sp45_model_train
    ).astype(np.float32)
    sp45_raw_test = (
        (1.0 - sp45_model_weight) * selector_test
        + sp45_model_weight * sp45_model_test
    ).astype(np.float32)
    projected_train_full = robust_u_projection(train_frame, sp45_raw_train, degree=3)
    projected_test_full = robust_u_projection(test_frame, sp45_raw_test, degree=3)
    projection_weight = fit_convex_weight(
        train_truth,
        sp45_raw_train,
        projected_train_full,
        (0.50, 1.00),
    )
    projected_train = (
        (1.0 - projection_weight) * sp45_raw_train
        + projection_weight * projected_train_full
    ).astype(np.float32)
    projected_test = (
        (1.0 - projection_weight) * sp45_raw_test
        + projection_weight * projected_test_full
    ).astype(np.float32)
    train_warmup = 1.0 - np.exp(
        -np.maximum(train_frame["md_since"].to_numpy(np.float32), 0.0) / 85.0
    )
    test_warmup = 1.0 - np.exp(
        -np.maximum(test_frame["md_since"].to_numpy(np.float32), 0.0) / 85.0
    )
    learned_model_train = last_train + train_warmup * learned_train_delta
    learned_model_test = last_test + test_warmup * learned_test_delta
    likpf_train = train_frame["likpf_scale_5"].to_numpy(np.float32)
    likpf_test = test_frame["likpf_scale_5"].to_numpy(np.float32)
    learned_model_weight = fit_convex_weight(
        train_truth,
        likpf_train,
        learned_model_train,
        (0.50, 0.80),
    )
    learned_train = (
        (1.0 - learned_model_weight) * likpf_train
        + learned_model_weight * learned_model_train
    ).astype(np.float32)
    learned_test = (
        (1.0 - learned_model_weight) * likpf_test
        + learned_model_weight * learned_model_test
    ).astype(np.float32)
    learned_train = savgol_by_well(train_frame, learned_train, window=61, polyorder=3)
    learned_test = savgol_by_well(test_frame, learned_test, window=61, polyorder=3)
    projected_sp45_weight = fit_convex_weight(
        train_truth,
        learned_train,
        projected_train,
        (0.50, 0.80),
    )
    public_core_test = (
        projected_sp45_weight * projected_test
        + (1.0 - projected_sp45_weight) * learned_test
    ).astype(np.float32)
    weights = np.asarray(meta_fold_weights, dtype=np.float64)
    if len(weights) != 5 or not np.isfinite(weights).all():
        raise ValueError("Stage I requires five finite Stage E meta-fold weights")
    expected_deployment_weight = float(np.median(weights))
    if abs(expected_deployment_weight - float(deployment_weight)) > 1e-15:
        raise ValueError("Stage I deployment weight does not equal the meta-fold median")
    exp413_prediction_path = Path(exp413_prediction_path)
    if sha256_file(exp413_prediction_path) != exp413_prediction_sha256:
        raise ValueError("exp413 current-test prediction file SHA mismatch")
    if sha256_gzip_decompressed(exp413_prediction_path) != exp413_prediction_decompressed_sha256:
        raise ValueError("exp413 current-test decompressed prediction SHA mismatch")
    exp413 = pd.read_csv(
        exp413_prediction_path,
        usecols=["id", "well", "pred_tvt"],
        dtype={"id": str, "well": str},
    ).rename(columns={"pred_tvt": "exp413_pred_tvt"})
    if (
        exp413["id"].duplicated().any()
        or len(exp413) != len(test_frame)
        or set(exp413["id"]) != set(test_frame["id"])
    ):
        raise ValueError("Stage I exp413/current-test identity contract failed")
    components = pd.DataFrame(
        {
            "id": test_frame["id"].astype(str),
            "well": test_frame["well"].astype(str),
            "last_known_tvt": last_test,
            "physical_selector_pred_tvt": selector_test,
            "sp45_residual_model_pred_tvt": sp45_model_test,
            "sp45_raw_pred_tvt": sp45_raw_test,
            "projected_sp45_pred_tvt": projected_test,
            "learned_trajectory_pred_tvt": learned_test,
            "strict_public_core_pred_tvt": public_core_test,
        }
    )
    components = components.merge(exp413, on=["id", "well"], how="left", validate="one_to_one")
    if components["exp413_pred_tvt"].isna().any():
        raise ValueError("Stage I exp413 prediction join produced missing values")
    components["exp497_blend_pred_tvt"] = (
        (1.0 - deployment_weight) * components["exp413_pred_tvt"].to_numpy(np.float32)
        + deployment_weight * public_core_test
    ).astype(np.float32)
    sample = pd.read_csv(sample_submission_path, dtype={"id": str})
    if sample["id"].duplicated().any() or list(sample.columns) != ["id", "tvt"]:
        raise ValueError("Unexpected sample submission identity/column contract")
    predictions = sample[["id"]].merge(components, on="id", how="left", validate="one_to_one")
    if len(predictions) != len(sample) or not predictions["id"].equals(sample["id"]):
        raise ValueError("Stage I prediction row/order contract failed")
    numeric_columns = [name for name in predictions.columns if name not in {"id", "well"}]
    if predictions[numeric_columns].isna().any().any() or not np.isfinite(
        predictions[numeric_columns].to_numpy(np.float64)
    ).all():
        raise ValueError("Stage I predictions contain missing/non-finite values")
    model_manifest = {
        "mode": "full_train_inner4_current_test_prediction_only",
        "sp45_models": sp45_models,
        "learned_models": learned_models,
        "sp45_ridge": sp45_ridge,
        "learned_ridge": learned_ridge,
        "fitted_boosters": len(sp45_models) + len(learned_models),
        "fitted_ridge_models": 2,
        "exp413_retraining": 0,
        "exp413_reinference": 0,
        "base_features": base_features,
        "learned_features": learned_features,
        "base_feature_sha256": sha256_json(base_features),
        "learned_feature_sha256": sha256_json(learned_features),
    }
    if model_manifest["fitted_boosters"] != 40:
        raise ValueError("Stage I must fit exactly 40 candidate boosters")
    kinds = pd.Series(
        [row["kind"] for row in [*sp45_models, *learned_models]],
        dtype=str,
    ).value_counts()
    if int(kinds.get("lightgbm", 0)) != 24 or int(kinds.get("catboost", 0)) != 16:
        raise ValueError("Stage I must fit exactly 24 LightGBM and 16 CatBoost boosters")
    serialization = validate_stage_i_serialized_model_manifest(
        output_dir,
        [*sp45_models, *learned_models],
    )
    model_manifest.update(serialization)
    internal_weights = {
        "sp45_model_weight": sp45_model_weight,
        "projection_weight": projection_weight,
        "learned_model_weight": learned_model_weight,
        "projected_sp45_weight": projected_sp45_weight,
        "meta_fold_weights": weights.tolist(),
        "deployment_weight": deployment_weight,
    }
    prediction_path = output_dir / "stage_i_current_test_predictions.csv.gz"
    model_manifest_path = output_dir / "stage_i_full_fit_model_manifest.json"
    ridge_weights_path = output_dir / "stage_i_ridge_weights.json"
    weights_path = output_dir / "stage_i_weights.json"
    selector_path = output_dir / "stage_i_selector_policy.json"
    feature_schema_path = output_dir / "stage_i_feature_schema.csv"
    importance_path = output_dir / "stage_i_feature_importance.parquet"
    reproducibility_path = output_dir / "stage_i_reproducibility_manifest.json"
    predictions.to_csv(prediction_path, index=False, compression="gzip")
    write_json(model_manifest_path, model_manifest)
    write_json(
        ridge_weights_path,
        {
            "sp45_residual": sp45_ridge,
            "learned_trajectory": learned_ridge,
            "ridge_model_count": 2,
        },
    )
    write_json(weights_path, internal_weights)
    write_json(selector_path, selector_policy_rows)
    pd.DataFrame(
        [
            *(
                {"branch": "sp45_residual", "feature_index": index, "feature": feature}
                for index, feature in enumerate(base_features)
            ),
            *(
                {"branch": "learned_trajectory", "feature_index": index, "feature": feature}
                for index, feature in enumerate(learned_features)
            ),
        ]
    ).to_csv(feature_schema_path, index=False)
    pd.concat([sp45_importance, learned_importance], ignore_index=True).to_parquet(
        importance_path,
        index=False,
        compression="zstd",
    )
    reproducibility = {
        "stage_p_feature_sha256": [summary["sha256"]["features"] for summary in stage_p_summaries],
        "stage_p_summary_sha256": [sha256_file(path) for path in stage_p_summary_paths],
        "exp413_prediction_file_sha256": exp413_prediction_sha256,
        "exp413_prediction_decompressed_sha256": exp413_prediction_decompressed_sha256,
        "strict_public_core_prediction_sha256": prediction_sha256(
            predictions["id"],
            predictions["strict_public_core_pred_tvt"].to_numpy(np.float32),
            "exp497:stage_i:strict_public_core",
        ),
        "blend_prediction_sha256": prediction_sha256(
            predictions["id"],
            predictions["exp497_blend_pred_tvt"].to_numpy(np.float32),
            "exp497:stage_i:blend",
        ),
        "prediction_file_decompressed_sha256": sha256_gzip_decompressed(prediction_path),
        "serialized_model_count": serialization["serialized_model_count"],
        "serialized_model_bytes": serialization["serialized_model_bytes"],
        "serialized_model_set_sha256": serialization["serialized_model_set_sha256"],
        "ridge_weights_sha256": sha256_file(ridge_weights_path),
        "test_feature_summary": dict(test_feature_summary),
        "deterministic_anchor": False,
    }
    write_json(reproducibility_path, reproducibility)
    outputs = {
        "predictions": prediction_path.name,
        "model_manifest": model_manifest_path.name,
        "ridge_weights": ridge_weights_path.name,
        "weights": weights_path.name,
        "selector_policy": selector_path.name,
        "feature_schema": feature_schema_path.name,
        "feature_importance": importance_path.name,
        "reproducibility_manifest": reproducibility_path.name,
    }
    summary = {
        "experiment": "exp497_strict_public_core_fold_safe_ensemble_on_exp413",
        "stage": "stage_i_current_test_prediction_only_override",
        "status": "complete",
        "train_gate_passed": False,
        "selected_train_anchor": "exp413",
        "diagnostic_prediction": "exp497_blend_pred_tvt",
        "rows": len(predictions),
        "wells": predictions["well"].nunique(),
        "fitted_boosters": model_manifest["fitted_boosters"],
        "fitted_ridge_models": 2,
        **serialization,
        "exp413_retraining": 0,
        "exp413_reinference": 0,
        "submission_generated": False,
        "external_submission_performed": False,
        "deployment_weight": deployment_weight,
        "prediction_stats": {
            "mean": float(predictions["exp497_blend_pred_tvt"].mean()),
            "std": float(predictions["exp497_blend_pred_tvt"].std()),
            "min": float(predictions["exp497_blend_pred_tvt"].min()),
            "max": float(predictions["exp497_blend_pred_tvt"].max()),
            "changed_rows_vs_exp413": int(
                np.count_nonzero(
                    predictions["exp497_blend_pred_tvt"].to_numpy(np.float32)
                    != predictions["exp413_pred_tvt"].to_numpy(np.float32)
                )
            ),
        },
        "internal_weights": internal_weights,
        "reproducibility": reproducibility,
        "outputs": outputs,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    summary["sha256"] = {
        name: sha256_file(output_dir / filename) for name, filename in outputs.items()
    }
    summary_path = output_dir / "stage_i_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, default=_json_default), flush=True)
    return summary
