from __future__ import annotations

import gzip
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config

OUTPUT_PREFIX = "exp140_z_slope_posthoc_correction_on_pfbeam_candidates"
EXP072_ARTIFACTS = Path("experiments") / "exp072_exp063_full_replay_feature_cache" / "artifacts"
EXP072_TRAIN_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
EXP072_FEATURE_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    source_columns: tuple[str, ...]
    transform: str
    role: str


@dataclass(frozen=True)
class VariantSpec:
    name: str
    base_candidate: str
    alpha: float
    clip_ft: float
    z_abs_min: float
    aux_mode: str
    disagreement_min: float
    near_guard_ft: float
    fade_end_ft: float


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if pd.isna(value) and not isinstance(value, str):
        return None
    return value


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_artifact(filename: str, explicit_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            EXP072_ARTIFACTS / filename,
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
        ]
    )
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked:\n{checked}")


def numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"required column is missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)


def row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce").to_numpy()
    if np.isnan(values).any():
        bad = ids[pd.isna(extracted)].head(5).tolist()
        raise ValueError(f"Could not recover row index from ids, examples={bad}")
    return values.astype(np.int32)


def float_tag(value: float) -> str:
    text = f"{value:.5g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def parse_candidate_specs(config: dict[str, Any]) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for raw in get_nested(config, "audit.candidates") or []:
        source_columns = raw.get("source_columns")
        if source_columns is None:
            source_columns = [raw["source_column"]]
        specs.append(
            CandidateSpec(
                name=str(raw["name"]),
                source_columns=tuple(str(column) for column in source_columns),
                transform=str(raw["transform"]),
                role=str(raw.get("role", "candidate")),
            )
        )
    if not specs:
        raise ValueError("audit.candidates must define at least one candidate")
    return specs


def resolve_source_columns(header: list[str], specs: list[CandidateSpec]) -> dict[str, str]:
    header_set = set(header)
    resolved: dict[str, str] = {}
    missing: dict[str, list[str]] = {}
    for spec in specs:
        match = next((column for column in spec.source_columns if column in header_set), None)
        if match is None:
            missing[spec.name] = list(spec.source_columns)
        else:
            resolved[spec.name] = match
    if missing:
        detail = "; ".join(f"{name}: {columns}" for name, columns in missing.items())
        raise ValueError(f"feature cache is missing candidate source columns: {detail}")
    return resolved


def read_feature_cache(
    config: dict[str, Any],
    specs: list[CandidateSpec],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    explicit = get_nested(config, "data.exp072_train_feature_cache_local")
    source = find_artifact(EXP072_TRAIN_FEATURES, explicit)
    header = pd.read_csv(source, nrows=0).columns.tolist()
    resolved_sources = resolve_source_columns(header, specs)
    required = {"id", "well", "target", "last_known_tvt", "md_since"}
    optional = {"eval_len", "last_anchor_tvt", "pf_ancc_std", "likpf_mean_d"}
    usecols = sorted(required | optional.intersection(header) | set(resolved_sources.values()))
    max_rows = get_nested(config, "audit.max_rows")
    frame = pd.read_csv(
        source,
        usecols=usecols,
        nrows=None if max_rows in {None, "null"} else int(max_rows),
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    max_wells = get_nested(config, "audit.max_wells")
    if max_wells not in {None, "null"}:
        keep_wells = sorted(frame["well"].unique().tolist())[: int(max_wells)]
        frame = frame[frame["well"].isin(keep_wells)].reset_index(drop=True)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    frame["row_idx"] = row_indices_from_ids(frame["id"])
    schema_path: Path | None = None
    try:
        schema_path = find_artifact(
            EXP072_FEATURE_SCHEMA,
            get_nested(config, "data.exp072_feature_schema_local"),
        )
    except FileNotFoundError:
        schema_path = None
    metadata = {
        "source": str(source),
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": (
            sha256_path(source, decompressed=True) if source.suffix == ".gz" else None
        ),
        "schema": str(schema_path) if schema_path else None,
        "schema_sha256": sha256_path(schema_path) if schema_path else None,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns": list(frame.columns),
        "resolved_sources": resolved_sources,
    }
    return frame, metadata, resolved_sources


def materialize_candidates(
    frame: pd.DataFrame,
    specs: list[CandidateSpec],
    resolved_sources: dict[str, str],
) -> list[str]:
    frame["true_tvt"] = numeric_array(frame, "last_known_tvt") + numeric_array(frame, "target")
    last_known = numeric_array(frame, "last_known_tvt")
    names: list[str] = []
    for spec in specs:
        values = numeric_array(frame, resolved_sources[spec.name])
        source_column = resolved_sources[spec.name]
        if spec.transform == "absolute":
            pred = values
        elif spec.transform == "base_plus_delta":
            pred = last_known + values if source_column.endswith("_d") else values
        else:
            raise ValueError(f"unsupported transform for {spec.name}: {spec.transform}")
        frame[spec.name] = pred.astype(np.float32)
        names.append(spec.name)
    return names


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values.astype(np.float32)
    return (
        pd.Series(values)
        .rolling(window=window, min_periods=1, center=True)
        .mean()
        .to_numpy(np.float32)
    )


def robust_gradient(values: np.ndarray, md: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    md = md.astype(np.float64)
    out = np.zeros(len(values), dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(md)
    if finite.sum() < 2:
        return out.astype(np.float32)
    idx = np.flatnonzero(finite)
    vals = values[idx]
    xs = md[idx]
    if np.any(np.diff(xs) <= 0):
        order = np.argsort(xs)
        vals = vals[order]
        xs = xs[order]
        idx = idx[order]
    grad = np.gradient(vals, xs, edge_order=1)
    out[idx] = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


def augment_geometry(
    frame: pd.DataFrame,
    paths: ExperimentPaths,
    config: dict[str, Any],
) -> dict[str, Any]:
    md = np.full(len(frame), np.nan, dtype=np.float32)
    z = np.full(len(frame), np.nan, dtype=np.float32)
    last_md = np.full(len(frame), np.nan, dtype=np.float32)
    last_z = np.full(len(frame), np.nan, dtype=np.float32)
    dz_dmd = np.full(len(frame), np.nan, dtype=np.float32)
    missing_wells: list[str] = []
    md_since_diffs: list[float] = []
    smooth_window = int(get_nested(config, "model.z_slope.smooth_window") or 11)
    clip_abs = float(get_nested(config, "model.z_slope.clip_abs") or 2.0)

    for well, group in frame.groupby("well", sort=False):
        path = paths.train_data_dir / f"{well}__horizontal_well.csv"
        if not path.exists():
            missing_wells.append(str(well))
            continue
        horizontal = pd.read_csv(path, usecols=["MD", "Z", "TVT_input"], low_memory=False)
        row_idx = group["row_idx"].to_numpy(np.int64)
        if row_idx.max(initial=-1) >= len(horizontal):
            raise IndexError(f"row_idx exceeds horizontal rows for {well}: {row_idx.max()}")
        md_all = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float32)
        z_all = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(np.float32)
        tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float32)
        known = np.flatnonzero(np.isfinite(tvt_input))
        if len(known) == 0:
            missing_wells.append(str(well))
            continue
        anchor_idx = int(known[-1])
        slope_all = robust_gradient(z_all, md_all)
        slope_all = rolling_mean(np.clip(slope_all, -clip_abs, clip_abs), smooth_window)
        positions = group.index.to_numpy(np.int64)
        md[positions] = md_all[row_idx]
        z[positions] = z_all[row_idx]
        last_md[positions] = md_all[anchor_idx]
        last_z[positions] = z_all[anchor_idx]
        dz_dmd[positions] = slope_all[row_idx]
        cache_md_since = numeric_array(group, "md_since")
        geom_md_since = md_all[row_idx] - md_all[anchor_idx]
        finite = np.isfinite(cache_md_since) & np.isfinite(geom_md_since)
        if finite.any():
            diff = np.abs(cache_md_since[finite] - geom_md_since[finite])
            md_since_diffs.append(float(np.nanmax(diff)))

    frame["md"] = md
    frame["z"] = z
    frame["last_md"] = last_md
    frame["last_z"] = last_z
    frame["md_delta"] = frame["md"] - frame["last_md"]
    frame["z_delta"] = frame["z"] - frame["last_z"]
    frame["dz_dmd"] = dz_dmd
    frame["minus_dz_dmd"] = -dz_dmd
    return {
        "missing_wells": missing_wells[:50],
        "missing_well_count": len(missing_wells),
        "max_abs_md_since_diff_vs_cache": max(md_since_diffs) if md_since_diffs else None,
        "md_finite_rate": float(np.isfinite(md).mean()),
        "z_finite_rate": float(np.isfinite(z).mean()),
        "dz_dmd_finite_rate": float(np.isfinite(dz_dmd).mean()),
    }


def add_base_path_features(
    frame: pd.DataFrame,
    candidate_names: list[str],
    config: dict[str, Any],
) -> None:
    smooth_window = int(get_nested(config, "model.z_slope.base_slope_smooth_window") or 15)
    disagreement_cols = [
        c
        for c in ["pf_ancc", "beam_mean", "likpf_mean", "sc_ens", "hyb"]
        if c in frame.columns
    ]
    if len(disagreement_cols) >= 2:
        frame["candidate_disagreement_std"] = (
            frame[disagreement_cols].std(axis=1).astype(np.float32)
        )
        frame["candidate_disagreement_range"] = (
            frame[disagreement_cols].max(axis=1) - frame[disagreement_cols].min(axis=1)
        ).astype(np.float32)
    else:
        frame["candidate_disagreement_std"] = 0.0
        frame["candidate_disagreement_range"] = 0.0

    for candidate in candidate_names:
        slope = np.full(len(frame), np.nan, dtype=np.float32)
        rough = np.full(len(frame), np.nan, dtype=np.float32)
        for _, group in frame.groupby("well", sort=False):
            pos = group.index.to_numpy(np.int64)
            order = np.argsort(group["row_idx"].to_numpy(np.int64))
            ordered = pos[order]
            base = numeric_array(frame.loc[ordered], candidate)
            md = numeric_array(frame.loc[ordered], "md")
            local_slope = robust_gradient(base, md)
            local_slope = rolling_mean(local_slope, smooth_window)
            local_rough = np.abs(robust_gradient(local_slope, md))
            slope[ordered] = local_slope
            rough[ordered] = local_rough
        frame[f"{candidate}_dmd"] = slope
        frame[f"{candidate}_roughness"] = rough
        frame[f"{candidate}_z_slope_gap"] = frame["minus_dz_dmd"].to_numpy(np.float32) - slope
        if "pf_z" in frame.columns:
            frame[f"{candidate}_minus_pf_z"] = (
                numeric_array(frame, candidate) - numeric_array(frame, "pf_z")
            ).astype(np.float32)


def parse_variants(config: dict[str, Any]) -> list[VariantSpec]:
    cfg = get_nested(config, "model.z_slope.correction_grid") or {}
    base_candidates = [str(v) for v in cfg.get("base_candidates", ["likpf_mean"])]
    alphas = [float(v) for v in cfg.get("alphas", [0.1])]
    clip_ft = [float(v) for v in cfg.get("clip_ft", [10.0])]
    z_abs_min = [float(v) for v in cfg.get("z_abs_min", [0.05])]
    aux_modes = [str(v) for v in cfg.get("aux_modes", ["none"])]
    disagreement_min = [float(v) for v in cfg.get("disagreement_min", [0.0])]
    near_guard_ft = [float(v) for v in cfg.get("near_guard_ft", [50.0])]
    fade_end_ft = [float(v) for v in cfg.get("fade_end_ft", [250.0])]
    variants: list[VariantSpec] = []
    for base in base_candidates:
        for alpha in alphas:
            for clip in clip_ft:
                for z_min in z_abs_min:
                    for aux in aux_modes:
                        for dis_min in disagreement_min:
                            for guard in near_guard_ft:
                                for fade in fade_end_ft:
                                    name = (
                                        f"zsl_{base}_a{float_tag(alpha)}"
                                        f"_c{float_tag(clip)}_z{float_tag(z_min)}"
                                        f"_d{float_tag(dis_min)}_{aux}"
                                    )
                                    variants.append(
                                        VariantSpec(
                                            name=name,
                                            base_candidate=base,
                                            alpha=alpha,
                                            clip_ft=clip,
                                            z_abs_min=z_min,
                                            aux_mode=aux,
                                            disagreement_min=dis_min,
                                            near_guard_ft=guard,
                                            fade_end_ft=fade,
                                        )
                                    )
    return variants


def md_fade_gate(md_since: np.ndarray, near_guard_ft: float, fade_end_ft: float) -> np.ndarray:
    denom = max(fade_end_ft - near_guard_ft, 1.0)
    return np.clip((md_since - near_guard_ft) / denom, 0.0, 1.0).astype(np.float32)


def cumulative_correction_for_base(
    frame: pd.DataFrame,
    base_candidate: str,
    config: dict[str, Any],
) -> np.ndarray:
    gap_col = f"{base_candidate}_z_slope_gap"
    if gap_col not in frame.columns:
        raise ValueError(f"missing gap column for base candidate: {base_candidate}")
    smooth_window = int(get_nested(config, "model.z_slope.gap_smooth_window") or 21)
    correction = np.full(len(frame), np.nan, dtype=np.float32)
    for _, group in frame.groupby("well", sort=False):
        pos = group.index.to_numpy(np.int64)
        order = np.argsort(group["row_idx"].to_numpy(np.int64))
        ordered = pos[order]
        gap = numeric_array(frame.loc[ordered], gap_col)
        gap = rolling_mean(gap, smooth_window).astype(np.float64)
        md = numeric_array(frame.loc[ordered], "md").astype(np.float64)
        dmd = np.diff(md, prepend=md[0])
        dmd = np.where(np.isfinite(dmd) & (dmd > 0), dmd, 0.0)
        inc = np.nan_to_num(gap, nan=0.0, posinf=0.0, neginf=0.0) * dmd
        correction[ordered] = np.cumsum(inc).astype(np.float32)
    return correction


def apply_variant(
    frame: pd.DataFrame,
    variant: VariantSpec,
    base_correction: np.ndarray,
) -> np.ndarray:
    base = numeric_array(frame, variant.base_candidate)
    md_since = numeric_array(frame, "md_since")
    z_abs = np.abs(numeric_array(frame, "dz_dmd"))
    gate = md_fade_gate(md_since, variant.near_guard_ft, variant.fade_end_ft)
    gate *= (z_abs >= variant.z_abs_min).astype(np.float32)
    if variant.disagreement_min > 0:
        disagreement = numeric_array(frame, "candidate_disagreement_std")
        gate *= (disagreement >= variant.disagreement_min).astype(np.float32)
    correction = base_correction.copy()
    if variant.aux_mode == "pfz_agree":
        if "pf_z" not in frame.columns:
            gate *= 0.0
        else:
            aux_delta = numeric_array(frame, "pf_z") - base
            agree = np.sign(aux_delta) == np.sign(correction)
            gate *= agree.astype(np.float32)
    elif variant.aux_mode == "pfz_pull":
        if "pf_z" not in frame.columns:
            gate *= 0.0
        else:
            aux_delta = numeric_array(frame, "pf_z") - base
            correction = 0.5 * correction + 0.5 * aux_delta
    elif variant.aux_mode != "none":
        raise ValueError(f"unsupported aux_mode: {variant.aux_mode}")
    clipped = np.clip(correction, -variant.clip_ft, variant.clip_ft)
    pred = base + variant.alpha * gate * clipped
    return pred.astype(np.float32)


def score_prediction(pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(pred) & np.isfinite(true)
    if not finite.any():
        return {
            "rows": 0,
            "coverage": 0.0,
            "rmse": None,
            "mae": None,
            "within10": None,
            "bias": None,
        }
    err = pred[finite].astype(np.float64) - true[finite].astype(np.float64)
    return {
        "rows": int(finite.sum()),
        "coverage": float(finite.mean()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "within10": float(np.mean(np.abs(err) <= 10.0)),
        "bias": float(np.mean(err)),
    }


def distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(pd.Series(values), errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def score_columns(frame: pd.DataFrame, columns: list[str], primary_baseline: str) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt")
    baseline_score = None
    if primary_baseline in frame.columns:
        baseline_score = score_prediction(numeric_array(frame, primary_baseline), true)
    rows: list[dict[str, Any]] = []
    for column in columns:
        score = score_prediction(numeric_array(frame, column), true)
        delta = None
        if baseline_score and score["rmse"] is not None and baseline_score["rmse"] is not None:
            delta = float(score["rmse"] - baseline_score["rmse"])
        rows.append({"candidate": column, **score, "delta_rmse_vs_primary_baseline": delta})
    return (
        pd.DataFrame(rows)
        .sort_values(["rmse", "candidate"], na_position="last")
        .reset_index(drop=True)
    )


def compute_bucket_metrics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt")
    buckets = distance_bucket(frame["md_since"])
    rows: list[dict[str, Any]] = []
    for column in columns:
        pred = numeric_array(frame, column)
        for bucket, idx in pd.Series(buckets).groupby(buckets, observed=False).groups.items():
            positions = np.asarray(list(idx), dtype=np.int64)
            if len(positions) == 0:
                continue
            score = score_prediction(pred[positions], true[positions])
            rows.append({"candidate": column, "distance_bucket": str(bucket), **score})
    return pd.DataFrame(rows)


def compute_by_well(frame: pd.DataFrame, columns: list[str], primary_baseline: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=False):
        true = numeric_array(group, "true_tvt")
        base_rmse = None
        if primary_baseline in group.columns:
            base_rmse = score_prediction(numeric_array(group, primary_baseline), true)["rmse"]
        for column in columns:
            score = score_prediction(numeric_array(group, column), true)
            delta = None
            if base_rmse is not None and score["rmse"] is not None:
                delta = float(score["rmse"] - base_rmse)
            rows.append(
                {
                    "well": str(well),
                    "candidate": column,
                    **score,
                    "delta_rmse_vs_primary_baseline": delta,
                }
            )
    return pd.DataFrame(rows)


def compute_group_metrics(
    frame: pd.DataFrame,
    columns: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    groups: dict[str, np.ndarray] = {
        "all": np.ones(len(frame), dtype=bool),
        "near_000_050": numeric_array(frame, "md_since") <= 50.0,
        "longtail_1000_plus": numeric_array(frame, "md_since") >= 1000.0,
    }
    dz = np.abs(numeric_array(frame, "dz_dmd"))
    if np.isfinite(dz).any():
        groups["z_abs_top_quartile"] = dz >= np.nanquantile(dz, 0.75)
    disagreement = numeric_array(frame, "candidate_disagreement_std")
    if np.isfinite(disagreement).any():
        groups["candidate_disagreement_top_quartile"] = disagreement >= np.nanquantile(
            disagreement,
            0.75,
        )
    for well in get_nested(config, "audit.representative_wells") or []:
        groups[f"representative_{well}"] = frame["well"].astype(str).to_numpy() == str(well)

    rows: list[dict[str, Any]] = []
    true = numeric_array(frame, "true_tvt")
    for group_name, mask in groups.items():
        positions = np.flatnonzero(mask)
        if len(positions) == 0:
            continue
        for column in columns:
            score = score_prediction(numeric_array(frame, column)[positions], true[positions])
            rows.append({"group": group_name, "candidate": column, **score})
    return pd.DataFrame(rows)


def max_well_regression(by_well: pd.DataFrame) -> pd.Series:
    return by_well.groupby("candidate", observed=True)["delta_rmse_vs_primary_baseline"].max()


def write_feature_schema(path: Path, columns: list[str]) -> None:
    pd.DataFrame(
        {
            "variant": OUTPUT_PREFIX,
            "feature_index": np.arange(len(columns), dtype=int),
            "feature": columns,
        }
    ).to_csv(path, index=False)


def run_audit(
    config: dict[str, Any] | None = None,
    paths: ExperimentPaths | None = None,
) -> dict[str, Any]:
    start = time.time()
    config = load_config() if config is None else config
    paths = ExperimentPaths() if paths is None else paths
    paths.require_kaggle_runtime()
    paths.ensure_output_dirs()

    specs = parse_candidate_specs(config)
    frame, feature_meta, resolved_sources = read_feature_cache(config, specs)
    candidate_names = materialize_candidates(frame, specs, resolved_sources)
    geometry_meta = augment_geometry(frame, paths, config)
    add_base_path_features(frame, candidate_names, config)

    variants = parse_variants(config)
    variant_columns: list[str] = []
    correction_cache: dict[str, np.ndarray] = {}
    for variant in variants:
        if variant.base_candidate not in frame.columns:
            continue
        if variant.base_candidate not in correction_cache:
            correction_cache[variant.base_candidate] = cumulative_correction_for_base(
                frame,
                variant.base_candidate,
                config,
            )
        frame[variant.name] = apply_variant(
            frame,
            variant,
            correction_cache[variant.base_candidate],
        )
        variant_columns.append(variant.name)

    primary_baseline = str(get_nested(config, "audit.primary_baseline") or "likpf_mean")
    score_columns_all = [*candidate_names, *variant_columns]
    candidate_metrics = score_columns(frame, score_columns_all, primary_baseline)
    bucket_metrics = compute_bucket_metrics(frame, score_columns_all)
    by_well = compute_by_well(frame, score_columns_all, primary_baseline)
    group_metrics = compute_group_metrics(frame, score_columns_all, config)
    regressions = max_well_regression(by_well)
    candidate_metrics["max_well_regression_vs_primary"] = candidate_metrics["candidate"].map(
        regressions
    )

    artifacts = paths.artifacts_dir
    metrics_path = artifacts / f"{OUTPUT_PREFIX}_candidate_metrics.csv"
    bucket_path = artifacts / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well.csv"
    group_path = artifacts / f"{OUTPUT_PREFIX}_group_metrics.csv"
    oof_path = artifacts / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz"
    schema_path = artifacts / f"{OUTPUT_PREFIX}_feature_schema.csv"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"

    candidate_metrics.to_csv(metrics_path, index=False)
    bucket_metrics.to_csv(bucket_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    group_metrics.to_csv(group_path, index=False)

    top_k = int(get_nested(config, "audit.save_oof_top_k") or 8)
    top_variants = [
        str(value)
        for value in candidate_metrics[candidate_metrics["candidate"].isin(variant_columns)]
        .head(top_k)["candidate"]
        .tolist()
    ]
    keep_columns = [
        "id",
        "well",
        "row_idx",
        "target",
        "true_tvt",
        "last_known_tvt",
        "md_since",
        "eval_len",
        "md",
        "z",
        "last_md",
        "last_z",
        "md_delta",
        "z_delta",
        "dz_dmd",
        "minus_dz_dmd",
        "candidate_disagreement_std",
        "candidate_disagreement_range",
        *candidate_names,
        *top_variants,
    ]
    keep_columns = list(
        dict.fromkeys([column for column in keep_columns if column in frame.columns])
    )
    frame[keep_columns].to_csv(oof_path, index=False, compression="gzip")
    write_feature_schema(schema_path, keep_columns)

    best = candidate_metrics.iloc[0].to_dict() if len(candidate_metrics) else {}
    primary = (
        candidate_metrics[candidate_metrics["candidate"] == primary_baseline].iloc[0].to_dict()
        if (candidate_metrics["candidate"] == primary_baseline).any()
        else {}
    )
    best_variant = (
        candidate_metrics[candidate_metrics["candidate"].isin(variant_columns)].iloc[0].to_dict()
        if len(variant_columns)
        else {}
    )
    summary = {
        "experiment": OUTPUT_PREFIX,
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.time() - start,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "feature_cache": feature_meta,
        "geometry": geometry_meta,
        "candidate_names": candidate_names,
        "variant_count": len(variant_columns),
        "primary_baseline": to_jsonable(primary),
        "best_candidate": to_jsonable(best),
        "best_z_slope_variant": to_jsonable(best_variant),
        "decision": {
            "recommendation": "diagnostic_only_until_kaggle_result_review",
            "reason": (
                "Z-slope correction is a train-side posthoc audit over existing "
                "PF/Beam candidates. Raw-test parity, near-prefix guard, and worst-well "
                "regression must be reviewed before any inference port."
            ),
        },
        "artifacts": {
            "candidate_metrics": str(metrics_path),
            "bucket_metrics": str(bucket_path),
            "by_well": str(by_well_path),
            "group_metrics": str(group_path),
            "oof_predictions": str(oof_path),
            "feature_schema": str(schema_path),
            "summary": str(summary_path),
        },
        "artifact_sha256": {
            "candidate_metrics": sha256_path(metrics_path),
            "bucket_metrics": sha256_path(bucket_path),
            "by_well": sha256_path(by_well_path),
            "group_metrics": sha256_path(group_path),
            "oof_predictions_raw": sha256_path(oof_path),
            "oof_predictions_decompressed": sha256_path(oof_path, decompressed=True),
            "feature_schema": sha256_path(schema_path),
        },
    }
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")
    best_z_delta = float(best_variant["delta_rmse_vs_primary_baseline"])
    metrics_status = (
        "completed_train_side_audit_candidate"
        if best_z_delta < 0.0
        else "completed_train_side_rejected_no_submit"
    )
    metrics_note = (
        "Train-side Z-slope posthoc correction audit completed. "
        "Best Z-slope variant improved the primary baseline; "
        "review guards before any inference port."
        if best_z_delta < 0.0
        else "Train-side Z-slope posthoc correction audit completed. "
        "Best Z-slope variant did not improve the primary baseline; "
        "no inference port or submission."
    )
    metrics_json = {
        "experiment": OUTPUT_PREFIX,
        "status": metrics_status,
        "metric": "rmse",
        "cv": float(primary["rmse"]),
        "public_lb": None,
        "private_lb": None,
        "primary_baseline": to_jsonable(primary),
        "best_z_slope_variant": to_jsonable(best_variant),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "summary_path": str(summary_path),
        "notes": metrics_note,
    }
    paths.metrics_path.write_text(
        json.dumps(to_jsonable(metrics_json), indent=2, sort_keys=True) + "\n"
    )
    return summary


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(to_jsonable(result["best_z_slope_variant"]), indent=2, sort_keys=True))
