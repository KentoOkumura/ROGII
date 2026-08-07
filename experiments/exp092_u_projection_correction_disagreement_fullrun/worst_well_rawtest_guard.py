from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUTPUT_PREFIX = "exp092_worst_well_rawtest_guard"
EXP092_PREFIX = "exp092_u_projection_correction_disagreement_fullrun"
EXP092_DIR = Path("experiments") / "exp092_u_projection_correction_disagreement_fullrun"
DEFAULT_OUTPUT_DIR = EXP092_DIR / "artifacts" / "worst_well_rawtest_guard"
DEFAULT_OOF_GUARD_DIR = EXP092_DIR / "artifacts" / "oof_delta_guard"
DEFAULT_TEST_DIR = Path("data/raw/test")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_csv(path: str | Path) -> str | None:
    path = Path(path)
    if path.suffix != ".gz":
        return None
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_sha256(ids: pd.Series, values: np.ndarray, *, label: str) -> str:
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    for raw_id in ids.astype(str).to_numpy():
        digest.update(raw_id.encode("utf-8"))
        digest.update(b"\0")
    digest.update(np.asarray(values, dtype=np.float32).tobytes())
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except TypeError:
        pass
    return value


def existing_path(
    candidates: list[Path],
    *,
    label: str,
    required: bool,
    search_kaggle_input: bool = True,
) -> Path | None:
    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    input_root = Path("/kaggle/input")
    if search_kaggle_input and input_root.exists():
        for candidate in sorted(input_root.glob(f"**/{candidates[0].name}")):
            checked.append(str(candidate))
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
    if required:
        raise FileNotFoundError(f"{label} not found or empty. Checked:\n" + "\n".join(checked))
    return None


def default_candidates(filename: str, roots: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for root in roots:
        candidates.extend([root / filename, root / "artifacts" / filename])
    candidates.extend([Path.cwd() / filename, Path.cwd() / "artifacts" / filename])
    return candidates


def resolve_inputs(args: argparse.Namespace) -> dict[str, Path | None]:
    train_roots = [
        EXP092_DIR / "kaggle/output/train",
        EXP092_DIR / "kaggle/output/train_v1",
        Path("/tmp/kaggle-output/exp092_u_projection_correction_disagreement_fullrun/train"),
        Path("/tmp/kaggle-output/exp092_u_projection_correction_disagreement_fullrun/train_v1"),
        Path("/tmp/kaggle-output/exp092-uproj-corr-disagree-train"),
    ]
    infer_roots = [
        EXP092_DIR / "kaggle/output/inference",
        EXP092_DIR / "kaggle/output/inference_v1",
        Path("/tmp/kaggle-output/exp092_u_projection_correction_disagreement_fullrun/inference"),
        Path("/tmp/kaggle-output/exp092_u_projection_correction_disagreement_fullrun/inference_v1"),
        Path("/tmp/kaggle-output/exp092-uproj-corr-disagree-infer"),
    ]
    exp073_roots = [
        Path(
            "/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_v2"
        ),
        Path(
            "/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_v1"
        ),
        Path(
            "/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_cpu_v2"
        ),
    ]
    exp077_roots = [Path("/tmp/kaggle-output/exp077-full-replay-postprocess-guard-infer-v1")]

    def explicit_or_default(
        raw: str | None,
        filename: str,
        roots: list[Path],
        label: str,
        required: bool,
        *,
        search_kaggle_input: bool = True,
    ) -> Path | None:
        if raw:
            return existing_path([Path(raw)], label=label, required=required)
        return existing_path(
            default_candidates(filename, roots),
            label=label,
            required=required,
            search_kaggle_input=search_kaggle_input,
        )

    return {
        "exp092_predictions": explicit_or_default(
            args.exp092_predictions,
            f"{EXP092_PREFIX}_inference_test_predictions.csv.gz",
            infer_roots,
            "exp092 inference predictions",
            True,
        ),
        "exp073_predictions": explicit_or_default(
            args.exp073_predictions,
            "exp063_full_replay_repro_guard_inference_test_predictions.csv.gz",
            exp073_roots,
            "optional exp073 inference predictions",
            False,
        ),
        "exp077_submission": explicit_or_default(
            args.exp077_submission,
            "submission.csv",
            exp077_roots,
            "optional exp077 inference submission",
            False,
            search_kaggle_input=False,
        ),
        "train_feature_schema": explicit_or_default(
            args.train_feature_schema,
            f"{EXP092_PREFIX}_feature_schema.csv",
            train_roots,
            "optional exp092 train feature schema",
            False,
        ),
        "inference_feature_schema": explicit_or_default(
            args.inference_feature_schema,
            f"{EXP092_PREFIX}_inference_feature_schema.csv",
            infer_roots,
            "optional exp092 inference feature schema",
            False,
        ),
        "train_projection_summary": explicit_or_default(
            args.train_projection_summary,
            f"{EXP092_PREFIX}_projection_feature_summary.csv",
            train_roots,
            "optional exp092 train projection summary",
            False,
        ),
        "inference_projection_summary": explicit_or_default(
            args.inference_projection_summary,
            f"{EXP092_PREFIX}_inference_projection_feature_summary.csv",
            infer_roots,
            "optional exp092 inference projection summary",
            False,
        ),
        "oof_by_well": explicit_or_default(
            args.oof_by_well,
            "exp092_oof_delta_guard_by_well.csv",
            [DEFAULT_OOF_GUARD_DIR],
            "exp092 OOF by-well guard",
            True,
        ),
        "oof_path_continuity": explicit_or_default(
            args.oof_path_continuity,
            "exp092_oof_delta_guard_path_continuity.csv",
            [DEFAULT_OOF_GUARD_DIR],
            "optional exp092 OOF path continuity",
            False,
        ),
    }


def path_meta(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "decompressed_sha256": sha256_decompressed_csv(path),
        "bytes": int(path.stat().st_size),
    }


def read_exp092_predictions(path: Path) -> pd.DataFrame:
    usecols = ["id", "well", "variant", "mode", "model", "last_known_tvt", "pred_delta", "pred_tvt"]
    frame = pd.read_csv(path, usecols=usecols, dtype={"id": str, "well": str})
    for col in ["last_known_tvt", "pred_delta", "pred_tvt"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(np.float32)
    if not np.isfinite(frame[["last_known_tvt", "pred_delta", "pred_tvt"]].to_numpy()).all():
        raise ValueError("exp092 inference predictions contain non-finite numeric values")
    return frame


def read_optional_prediction_surface(path: Path | None, *, label: str) -> pd.DataFrame | None:
    if path is None:
        return None
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if "pred_tvt" in columns:
        usecols = [
            col
            for col in ["id", "well", "mode", "model", "last_known_tvt", "pred_tvt"]
            if col in columns
        ]
        frame = pd.read_csv(path, usecols=usecols, dtype={"id": str, "well": str})
        value_col = "pred_tvt"
    else:
        value_col = "tvt" if "tvt" in columns else columns[1]
        frame = pd.read_csv(path, usecols=["id", value_col], dtype={"id": str})
    frame = frame.rename(columns={value_col: f"pred_{label}"})
    frame[f"pred_{label}"] = pd.to_numeric(frame[f"pred_{label}"], errors="raise").astype(
        np.float32
    )
    return frame[["id", f"pred_{label}"]]


def parse_tail_index(ids: pd.Series) -> np.ndarray:
    suffix = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    parsed = pd.to_numeric(suffix, errors="coerce")
    if parsed.isna().any():
        bad = ids[parsed.isna()].head(5).tolist()
        raise ValueError(f"Cannot parse raw row index suffix from ids: {bad}")
    return parsed.to_numpy(np.int64)


def add_raw_test_context(
    predictions: pd.DataFrame, test_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = predictions.reset_index(drop=True).copy()
    row_index = parse_tail_index(frame["id"])
    md = np.full(len(frame), np.nan, dtype=np.float32)
    z = np.full(len(frame), np.nan, dtype=np.float32)
    raw_tvt_input = np.full(len(frame), np.nan, dtype=np.float32)
    anchor_row_index = np.full(len(frame), -1, dtype=np.int32)
    anchor_md = np.full(len(frame), np.nan, dtype=np.float32)
    anchor_t0 = np.full(len(frame), np.nan, dtype=np.float32)
    known_prefix_rows = np.full(len(frame), -1, dtype=np.int32)
    well_rows: list[dict[str, Any]] = []

    for well, idx in frame.groupby("well", sort=False).indices.items():
        idx_array = np.asarray(idx, dtype=np.int64)
        well_name = str(well)
        raw_path = test_dir / f"{well_name}__horizontal_well.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"raw test horizontal well not found: {raw_path}")
        raw = pd.read_csv(raw_path, usecols=["MD", "Z", "TVT_input"])
        take = row_index[idx_array]
        if int(take.min()) < 0 or int(take.max()) >= len(raw):
            raise ValueError(
                f"row index out of range for {raw_path}: min={take.min()} max={take.max()}"
            )
        selected = raw.iloc[take]
        md[idx_array] = pd.to_numeric(selected["MD"], errors="raise").to_numpy(np.float32)
        z[idx_array] = pd.to_numeric(selected["Z"], errors="raise").to_numpy(np.float32)
        raw_tvt_input[idx_array] = pd.to_numeric(selected["TVT_input"], errors="coerce").to_numpy(
            np.float32
        )
        known = raw[pd.to_numeric(raw["TVT_input"], errors="coerce").notna()]
        if known.empty:
            raise ValueError(f"No known TVT_input prefix rows for raw test well {well_name}")
        anchor = known.iloc[-1]
        anchor_idx = int(known.index[-1])
        anchor_row_index[idx_array] = anchor_idx
        anchor_md[idx_array] = float(anchor["MD"])
        anchor_t0[idx_array] = float(anchor["TVT_input"])
        known_prefix_rows[idx_array] = int(len(known))
        well_rows.append(
            {
                "well": well_name,
                "rows": int(len(idx_array)),
                "raw_rows": int(len(raw)),
                "min_row_index": int(take.min()),
                "max_row_index": int(take.max()),
                "anchor_row_index": anchor_idx,
                "known_prefix_rows": int(len(known)),
            }
        )

    frame["raw_row_index"] = row_index.astype(np.int32)
    frame["md"] = md
    frame["z"] = z
    frame["raw_tvt_input"] = raw_tvt_input
    frame["anchor_row_index"] = anchor_row_index
    frame["anchor_md"] = anchor_md
    frame["anchor_t0"] = anchor_t0
    frame["known_prefix_rows"] = known_prefix_rows
    frame["md_since"] = (frame["md"] - frame["anchor_md"]).astype(np.float32)
    frame["tail_rank"] = (frame["raw_row_index"] - frame["anchor_row_index"]).astype(np.int32)
    frame["tail_length"] = frame.groupby("well")["id"].transform("size").astype(np.int32)
    t0_diff = np.abs(
        frame["last_known_tvt"].to_numpy(np.float32) - frame["anchor_t0"].to_numpy(np.float32)
    )
    meta = {
        "test_dir": str(test_dir),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "anchor_t0_vs_last_known_abs_max": float(t0_diff.max()),
        "anchor_t0_vs_last_known_abs_mean": float(t0_diff.mean()),
        "known_prefix_rows_min": int(frame["known_prefix_rows"].min()),
        "known_prefix_rows_max": int(frame["known_prefix_rows"].max()),
        "sample_wells": well_rows[:10],
    }
    return frame, meta


def distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def tail_rank_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 99, 249, 499, 999, np.inf],
        labels=["000_099", "100_249", "250_499", "500_999", "1000_plus"],
        include_lowest=True,
    )


def step_summary(values: pd.Series) -> dict[str, float | int]:
    step = values.astype(float).diff().abs().dropna()
    return {
        "step_abs_p95": float(step.quantile(0.95)) if not step.empty else 0.0,
        "step_abs_max": float(step.max()) if not step.empty else 0.0,
        "step_abs_ge10": int((step >= 10.0).sum()),
        "step_abs_ge25": int((step >= 25.0).sum()),
    }


def build_test_well_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_cols = [col for col in frame.columns if col.startswith("pred_exp0") and col != "pred_tvt"]
    for well, subset in frame.sort_values(["well", "raw_row_index"]).groupby("well", sort=False):
        pred = subset["pred_tvt"].astype(float)
        record: dict[str, Any] = {
            "well": str(well),
            "rows": int(len(subset)),
            "tail_length": int(subset["tail_length"].max()),
            "tail_rank_min": int(subset["tail_rank"].min()),
            "tail_rank_max": int(subset["tail_rank"].max()),
            "md_since_min": float(subset["md_since"].min()),
            "md_since_max": float(subset["md_since"].max()),
            "known_prefix_rows": int(subset["known_prefix_rows"].max()),
            "anchor_t0_vs_last_known_abs_max": float(
                np.abs(subset["anchor_t0"] - subset["last_known_tvt"]).max()
            ),
            "pred_min": float(pred.min()),
            "pred_max": float(pred.max()),
            "pred_mean": float(pred.mean()),
            "pred_std": float(pred.std(ddof=0)),
            "pred_range": float(pred.max() - pred.min()),
            **{f"exp092_{key}": value for key, value in step_summary(pred).items()},
        }
        for base_col in base_cols:
            label = base_col.replace("pred_", "")
            correction = pred - subset[base_col].astype(float)
            correction_step = correction.diff().abs().dropna()
            record[f"corr_vs_{label}_mean"] = float(correction.mean())
            record[f"corr_vs_{label}_abs_mean"] = float(correction.abs().mean())
            record[f"corr_vs_{label}_abs_p95"] = float(correction.abs().quantile(0.95))
            record[f"corr_vs_{label}_abs_max"] = float(correction.abs().max())
            record[f"corr_vs_{label}_step_abs_p95"] = (
                float(correction_step.quantile(0.95)) if not correction_step.empty else 0.0
            )
            record[f"corr_vs_{label}_step_abs_max"] = (
                float(correction_step.max()) if not correction_step.empty else 0.0
            )
            record[f"corr_vs_{label}_step_abs_ge5"] = int((correction_step >= 5.0).sum())
        rows.append(record)
    return pd.DataFrame(rows).sort_values("exp092_step_abs_p95", ascending=False)


def build_bucket_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["distance_bucket"] = distance_bucket(work["md_since"]).astype(str)
    work["tail_rank_bucket"] = tail_rank_bucket(work["tail_rank"]).astype(str)
    rows: list[pd.DataFrame] = []
    for family in ["distance_bucket", "tail_rank_bucket"]:
        grouped = (
            work.groupby(family, observed=True)
            .agg(
                rows=("id", "size"),
                wells=("well", "nunique"),
                pred_mean=("pred_tvt", "mean"),
                pred_std=("pred_tvt", "std"),
                pred_min=("pred_tvt", "min"),
                pred_max=("pred_tvt", "max"),
                md_since_min=("md_since", "min"),
                md_since_max=("md_since", "max"),
            )
            .reset_index()
            .rename(columns={family: "bucket"})
        )
        grouped.insert(0, "bucket_family", family)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def compare_schema(
    train_path: Path | None, inference_path: Path | None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if train_path is None or inference_path is None:
        return pd.DataFrame(), {
            "status": "missing_input",
            "train_schema": str(train_path),
            "inference_schema": str(inference_path),
        }
    train = pd.read_csv(train_path)
    infer = pd.read_csv(inference_path)
    train_features = train["feature"].astype(str).tolist()
    infer_features = infer["feature"].astype(str).tolist()
    rows = []
    for feature in sorted(set(train_features) | set(infer_features)):
        rows.append(
            {
                "feature": feature,
                "train_index": train_features.index(feature) if feature in train_features else -1,
                "inference_index": infer_features.index(feature)
                if feature in infer_features
                else -1,
                "in_train": feature in train_features,
                "in_inference": feature in infer_features,
                "same_index": (
                    feature in train_features
                    and feature in infer_features
                    and train_features.index(feature) == infer_features.index(feature)
                ),
            }
        )
    parity = pd.DataFrame(rows)
    summary = {
        "status": "pass" if train_features == infer_features else "warning_schema_mismatch",
        "train_feature_count": int(len(train_features)),
        "inference_feature_count": int(len(infer_features)),
        "missing_in_inference": int((~parity["in_inference"]).sum()),
        "extra_in_inference": int((~parity["in_train"]).sum()),
        "order_mismatch": int(
            (~parity["same_index"] & parity["in_train"] & parity["in_inference"]).sum()
        ),
    }
    return parity, summary


def compare_projection_summary(
    train_path: Path | None, inference_path: Path | None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if train_path is None or inference_path is None:
        return pd.DataFrame(), {
            "status": "missing_input",
            "train_projection_summary": str(train_path),
            "inference_projection_summary": str(inference_path),
        }
    train = pd.read_csv(train_path).add_prefix("train_")
    infer = pd.read_csv(inference_path).add_prefix("inference_")
    if "train_source" not in train.columns or "inference_source" not in infer.columns:
        return pd.DataFrame(), {"status": "missing_source_column"}
    merged = train.merge(infer, left_on="train_source", right_on="inference_source", how="outer")
    merged["source"] = merged["train_source"].fillna(merged["inference_source"])
    for col in ["abs_resid_mean", "abs_resid_p95", "resid_mad_mean", "u_std"]:
        left = f"train_{col}"
        right = f"inference_{col}"
        if left in merged.columns and right in merged.columns:
            merged[f"{col}_ratio_inference_over_train"] = pd.to_numeric(
                merged[right], errors="coerce"
            ) / pd.to_numeric(merged[left], errors="coerce").replace(0, np.nan)
    ratio_cols = [col for col in merged.columns if col.endswith("_ratio_inference_over_train")]
    max_ratio = (
        float(np.nanmax(np.abs(merged[ratio_cols].to_numpy(dtype=float))))
        if ratio_cols
        else float("nan")
    )
    return merged, {
        "status": "completed",
        "sources": int(merged["source"].nunique()),
        "max_abs_ratio": max_ratio,
    }


def build_oof_worst_profile(
    by_well_path: Path,
    continuity_path: Path | None,
    *,
    regression_threshold: float,
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    by_well = pd.read_csv(by_well_path)
    worst = by_well.sort_values("exp092_lgb1_rmse_delta_vs_exp077", ascending=False).head(top_n)
    warning = by_well[by_well["exp092_lgb1_rmse_delta_vs_exp077"] > float(regression_threshold)]
    if continuity_path is not None:
        continuity = pd.read_csv(continuity_path)
        worst = worst.merge(continuity, on="well", how="left")
        warning = warning.merge(continuity, on="well", how="left")
    profile = {
        "rows": int(len(by_well)),
        "regression_threshold": float(regression_threshold),
        "warning_wells": int(len(warning)),
        "top_n": int(top_n),
        "max_regression_vs_exp077": float(by_well["exp092_lgb1_rmse_delta_vs_exp077"].max()),
        "max_improvement_vs_exp077": float(by_well["exp092_lgb1_rmse_delta_vs_exp077"].min()),
        "oof_exp092_step_p95_q99": float(
            by_well.get("pred_exp092_lgb1_step_abs_p95", pd.Series([np.nan])).quantile(0.99)
        )
        if "pred_exp092_lgb1_step_abs_p95" in by_well
        else None,
    }
    if continuity_path is not None:
        continuity = pd.read_csv(continuity_path)
        profile.update(
            {
                "oof_exp092_step_p95_p99": float(
                    continuity["pred_exp092_lgb1_step_abs_p95"].quantile(0.99)
                ),
                "oof_exp092_step_max_p99": float(
                    continuity["pred_exp092_lgb1_step_abs_max"].quantile(0.99)
                ),
                "oof_worst_exp092_step_p95_max": float(
                    worst["pred_exp092_lgb1_step_abs_p95"].max()
                ),
                "oof_worst_exp092_step_max_max": float(
                    worst["pred_exp092_lgb1_step_abs_max"].max()
                ),
            }
        )
    return worst, profile


def warning_thresholds(oof_profile: dict[str, Any]) -> dict[str, float]:
    step_p95_values = [
        2.0,
        1.25 * float(oof_profile.get("oof_exp092_step_p95_p99") or 0.0),
        1.10 * float(oof_profile.get("oof_worst_exp092_step_p95_max") or 0.0),
    ]
    step_max_values = [
        10.0,
        1.25 * float(oof_profile.get("oof_exp092_step_max_p99") or 0.0),
        1.10 * float(oof_profile.get("oof_worst_exp092_step_max_max") or 0.0),
    ]
    return {
        "anchor_t0_abs_max": 0.05,
        "exp092_step_abs_p95": float(max(step_p95_values)),
        "exp092_step_abs_max": float(max(step_max_values)),
        "correction_abs_p95": 15.0,
        "correction_step_abs_p95": 5.0,
    }


def apply_test_warnings(well_metrics: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    frame = well_metrics.copy()
    warning_cols: list[str] = []
    checks = {
        "warn_anchor_t0_mismatch": frame["anchor_t0_vs_last_known_abs_max"]
        > thresholds["anchor_t0_abs_max"],
        "warn_exp092_step_p95": frame["exp092_step_abs_p95"] > thresholds["exp092_step_abs_p95"],
        "warn_exp092_step_max": frame["exp092_step_abs_max"] > thresholds["exp092_step_abs_max"],
    }
    for col in [
        col for col in frame.columns if col.startswith("corr_vs_") and col.endswith("_abs_p95")
    ]:
        checks[f"warn_{col}"] = frame[col] > thresholds["correction_abs_p95"]
    for col in [
        col for col in frame.columns if col.startswith("corr_vs_") and col.endswith("_step_abs_p95")
    ]:
        checks[f"warn_{col}"] = frame[col] > thresholds["correction_step_abs_p95"]
    for name, mask in checks.items():
        frame[name] = mask.astype(bool)
        warning_cols.append(name)
    frame["warning_count"] = frame[warning_cols].sum(axis=1).astype(int)
    return frame.sort_values(["warning_count", "exp092_step_abs_p95"], ascending=[False, False])


def run_guard(
    *,
    output_dir: Path,
    test_dir: Path,
    inputs: dict[str, Path | None],
    regression_threshold: float,
    top_n_worst: int,
) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    exp092 = read_exp092_predictions(inputs["exp092_predictions"])  # type: ignore[arg-type]
    frame, raw_meta = add_raw_test_context(exp092, test_dir)
    for label, key in [("exp073", "exp073_predictions"), ("exp077_policy", "exp077_submission")]:
        optional = read_optional_prediction_surface(inputs[key], label=label)
        if optional is not None:
            frame = frame.merge(optional, on="id", how="left", validate="one_to_one")
    missing_optional_predictions = (
        int(
            frame[
                [col for col in frame.columns if col.startswith("pred_exp0") and col != "pred_tvt"]
            ]
            .isna()
            .sum()
            .sum()
        )
        if any(col.startswith("pred_exp0") and col != "pred_tvt" for col in frame.columns)
        else 0
    )
    if missing_optional_predictions:
        raise ValueError(
            f"optional prediction merge produced missing rows: {missing_optional_predictions}"
        )

    well_metrics = build_test_well_metrics(frame)
    bucket_metrics = build_bucket_metrics(frame)
    oof_worst, oof_profile = build_oof_worst_profile(
        inputs["oof_by_well"],  # type: ignore[arg-type]
        inputs["oof_path_continuity"],
        regression_threshold=regression_threshold,
        top_n=top_n_worst,
    )
    thresholds = warning_thresholds(oof_profile)
    well_metrics = apply_test_warnings(well_metrics, thresholds)
    schema_parity, schema_summary = compare_schema(
        inputs["train_feature_schema"],
        inputs["inference_feature_schema"],
    )
    projection_parity, projection_summary = compare_projection_summary(
        inputs["train_projection_summary"],
        inputs["inference_projection_summary"],
    )

    well_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_test_well_metrics.csv", index=False)
    bucket_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_test_bucket_metrics.csv", index=False)
    oof_worst.to_csv(output_dir / f"{OUTPUT_PREFIX}_oof_worst_wells.csv", index=False)
    if not schema_parity.empty:
        schema_parity.to_csv(output_dir / f"{OUTPUT_PREFIX}_schema_parity.csv", index=False)
    if not projection_parity.empty:
        projection_parity.to_csv(
            output_dir / f"{OUTPUT_PREFIX}_projection_summary_parity.csv", index=False
        )

    warning_rows = well_metrics[well_metrics["warning_count"] > 0]
    status = "visible_test_completed_pass"
    if len(warning_rows) > 0 or schema_summary.get("status") not in {"pass", "missing_input"}:
        status = "visible_test_completed_with_warnings"
    summary = {
        "experiment": OUTPUT_PREFIX,
        "parent": EXP092_PREFIX,
        "status": status,
        "mode": "visible_test_only_target_free_probe",
        "code_competition_caveat": {
            "visible_test_only": True,
            "hidden_lb_test_observed": False,
            "note": (
                "Normal Kaggle notebook execution reads only the exposed visible test. "
                "In Code Competition scoring, the hidden LB test is injected only during "
                "code submission rerun; hidden-test probes require assertions in the "
                "submitted inference notebook and pass/fail observation."
            ),
        },
        "inputs": {key: path_meta(value) for key, value in inputs.items()},
        "raw_test_context": raw_meta,
        "prediction_sha256": prediction_sha256(
            frame["id"],
            frame["pred_tvt"].to_numpy(np.float32),
            label="exp092/rawtest/lgb1/pred_tvt",
        ),
        "oof_profile": oof_profile,
        "thresholds": thresholds,
        "schema_parity": schema_summary,
        "projection_summary_parity": projection_summary,
        "test_warning": {
            "wells": int(len(well_metrics)),
            "warning_wells": int(len(warning_rows)),
            "max_warning_count": int(well_metrics["warning_count"].max()),
            "top_warning_wells": well_metrics.head(10).to_dict("records"),
        },
        "artifacts": {
            "test_well_metrics": f"{OUTPUT_PREFIX}_test_well_metrics.csv",
            "test_bucket_metrics": f"{OUTPUT_PREFIX}_test_bucket_metrics.csv",
            "oof_worst_wells": f"{OUTPUT_PREFIX}_oof_worst_wells.csv",
            "schema_parity": f"{OUTPUT_PREFIX}_schema_parity.csv"
            if not schema_parity.empty
            else None,
            "projection_summary_parity": f"{OUTPUT_PREFIX}_projection_summary_parity.csv"
            if not projection_parity.empty
            else None,
            "summary": f"{OUTPUT_PREFIX}_summary.json",
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (output_dir / f"{OUTPUT_PREFIX}_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def create_self_test_fixture(root: Path) -> tuple[Path, Path, dict[str, Path | None]]:
    test_dir = root / "test"
    artifacts = root / "artifacts"
    test_dir.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for well_i, well in enumerate(["aaa00000", "bbb00000", "ccc00000"]):
        raw_rows = []
        for row in range(8):
            tvt_input = 12000.0 + well_i * 20.0 + row * 0.5 if row < 3 else np.nan
            raw_rows.append({"MD": 1000.0 + row * 10.0, "Z": -100.0 - row, "TVT_input": tvt_input})
        pd.DataFrame(raw_rows).to_csv(test_dir / f"{well}__horizontal_well.csv", index=False)
        anchor_t0 = raw_rows[2]["TVT_input"]
        for row in range(3, 8):
            raw_id = f"{well}_{row}"
            pred = anchor_t0 + (row - 2) * (1.0 + 0.2 * well_i)
            pred_rows.append(
                {
                    "id": raw_id,
                    "well": well,
                    "variant": "u_projection_correction_plus_disagreement",
                    "mode": "gpu_repro_guard_dp_threads8",
                    "model": "lgb1",
                    "last_known_tvt": anchor_t0,
                    "pred_delta": pred - anchor_t0,
                    "pred_tvt": pred,
                }
            )
            rows.append({"id": raw_id, "well": well})
    exp092_path = artifacts / f"{EXP092_PREFIX}_inference_test_predictions.csv.gz"
    pd.DataFrame(pred_rows).to_csv(exp092_path, index=False, compression="gzip")
    exp073_path = artifacts / "exp063_full_replay_repro_guard_inference_test_predictions.csv.gz"
    base = pd.DataFrame(pred_rows).assign(pred_tvt=lambda df: df["pred_tvt"] - 0.2)
    base.to_csv(exp073_path, index=False, compression="gzip")
    by_well = pd.DataFrame(
        {
            "well": ["train_a", "train_b", "train_c"],
            "rows": [100, 100, 100],
            "wells": [1, 1, 1],
            "exp092_lgb1_rmse_delta_vs_exp077": [4.0, 0.1, -1.0],
            "exp092_lgb1_rmse_delta_vs_exp073": [5.0, 0.2, -1.2],
        }
    )
    by_well_path = artifacts / "exp092_oof_delta_guard_by_well.csv"
    by_well.to_csv(by_well_path, index=False)
    continuity = pd.DataFrame(
        {
            "well": ["train_a", "train_b", "train_c"],
            "pred_exp092_lgb1_step_abs_p95": [0.8, 0.5, 0.4],
            "pred_exp092_lgb1_step_abs_max": [1.0, 0.8, 0.7],
        }
    )
    continuity_path = artifacts / "exp092_oof_delta_guard_path_continuity.csv"
    continuity.to_csv(continuity_path, index=False)
    schema = pd.DataFrame(
        {"feature_index": [0, 1], "feature": ["a", "b"], "is_projection_feature": [False, True]}
    )
    train_schema = artifacts / f"{EXP092_PREFIX}_feature_schema.csv"
    infer_schema = artifacts / f"{EXP092_PREFIX}_inference_feature_schema.csv"
    schema.to_csv(train_schema, index=False)
    schema.to_csv(infer_schema, index=False)
    proj = pd.DataFrame(
        {
            "source": ["pf_ancc"],
            "abs_resid_mean": [1.0],
            "abs_resid_p95": [2.0],
            "resid_mad_mean": [1.1],
            "u_std": [3.0],
        }
    )
    train_proj = artifacts / f"{EXP092_PREFIX}_projection_feature_summary.csv"
    infer_proj = artifacts / f"{EXP092_PREFIX}_inference_projection_feature_summary.csv"
    proj.to_csv(train_proj, index=False)
    proj.to_csv(infer_proj, index=False)
    return (
        test_dir,
        root / "out",
        {
            "exp092_predictions": exp092_path,
            "exp073_predictions": exp073_path,
            "exp077_submission": None,
            "train_feature_schema": train_schema,
            "inference_feature_schema": infer_schema,
            "train_projection_summary": train_proj,
            "inference_projection_summary": infer_proj,
            "oof_by_well": by_well_path,
            "oof_path_continuity": continuity_path,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="exp092 visible-test worst-well regression guard")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--exp092-predictions")
    parser.add_argument("--exp073-predictions")
    parser.add_argument("--exp077-submission")
    parser.add_argument("--train-feature-schema")
    parser.add_argument("--inference-feature-schema")
    parser.add_argument("--train-projection-summary")
    parser.add_argument("--inference-projection-summary")
    parser.add_argument("--oof-by-well")
    parser.add_argument("--oof-path-continuity")
    parser.add_argument("--regression-threshold", type=float, default=0.25)
    parser.add_argument("--top-n-worst", type=int, default=30)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        with tempfile.TemporaryDirectory(prefix="exp092_guard_selftest_") as temp_dir:
            test_dir, output_dir, inputs = create_self_test_fixture(Path(temp_dir))
            run_guard(
                output_dir=output_dir,
                test_dir=test_dir,
                inputs=inputs,
                regression_threshold=args.regression_threshold,
                top_n_worst=args.top_n_worst,
            )
        return
    inputs = resolve_inputs(args)
    run_guard(
        output_dir=Path(args.output_dir),
        test_dir=Path(args.test_dir),
        inputs=inputs,
        regression_threshold=args.regression_threshold,
        top_n_worst=args.top_n_worst,
    )


if __name__ == "__main__":
    main()
