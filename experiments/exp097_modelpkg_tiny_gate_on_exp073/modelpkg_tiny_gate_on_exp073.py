from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUTPUT_PREFIX = "exp097_modelpkg_tiny_gate_on_exp073"
EXP073_INFERENCE_FILENAME = "exp063_full_replay_repro_guard_inference_test_predictions.csv.gz"
MODEL_PACKAGE_FILENAME = "submission_model_package_only.csv"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if value is pd.NA:
        return None
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except TypeError:
        pass
    return value


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _as_path_list(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, str | Path):
        return [Path(value)]
    if isinstance(value, list | tuple):
        return [Path(item) for item in value if item]
    return []


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_decompressed_csv(path: str | Path) -> str:
    path = Path(path)
    hasher = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def frame_sha256(frame: pd.DataFrame, *, columns: list[str]) -> str:
    return hashlib.sha256(frame[columns].to_csv(index=False).encode("utf-8")).hexdigest()


def prediction_sha256(ids: pd.Series, values: np.ndarray, *, label: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(label.encode("utf-8"))
    for raw_id in ids.astype(str).to_numpy():
        hasher.update(raw_id.encode("utf-8"))
        hasher.update(b"\0")
    hasher.update(np.asarray(values, dtype=np.float32).tobytes())
    return hasher.hexdigest()


def find_input_file(
    filename: str,
    configured: Any = None,
    *,
    local_roots: list[Path] | None = None,
) -> Path:
    candidates: list[Path] = []
    candidates.extend(_as_path_list(configured))
    for root in local_roots or []:
        candidates.append(root / filename)
        candidates.append(root / "artifacts" / filename)
    candidates.extend([Path.cwd() / filename, Path.cwd() / "artifacts" / filename])
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    input_root = Path("/kaggle/input")
    if input_root.exists():
        for candidate in sorted(input_root.glob(f"**/{filename}")):
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate

    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(f"input file not found or empty: {filename}. Checked:\n{checked}")


def read_submission_contract(path: str | Path, *, target_col: str = "tvt") -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns)[:2] != ["id", target_col]:
        if "id" not in frame.columns or target_col not in frame.columns:
            raise ValueError(
                f"{path} must contain id and {target_col} columns, got {list(frame.columns)}"
            )
        frame = frame[["id", target_col]].copy()
    else:
        frame = frame[["id", target_col]].copy()
    frame["id"] = frame["id"].astype(str)
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce").astype(np.float64)
    if frame["id"].duplicated().any():
        duplicated = frame.loc[frame["id"].duplicated(), "id"].head(5).tolist()
        raise ValueError(f"{path} contains duplicated ids: {duplicated}")
    if frame[target_col].isna().any() or not np.isfinite(frame[target_col].to_numpy()).all():
        raise ValueError(f"{path} contains missing or non-finite {target_col} values")
    return frame


def load_exp073_inference_predictions(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    local_roots = [
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
    source = find_input_file(
        EXP073_INFERENCE_FILENAME,
        get_nested(config, "data.exp073_inference_predictions"),
        local_roots=local_roots,
    )
    mode = str(get_nested(config, "audit.selected_mode", "gpu_repro_guard_dp_threads8"))
    model = str(get_nested(config, "audit.selected_model", "lgb_mean"))
    usecols = ["id", "well", "mode", "model", "last_known_tvt", "pred_tvt"]
    dtypes = {
        "id": "string",
        "well": "string",
        "mode": "string",
        "model": "string",
        "last_known_tvt": "float32",
        "pred_tvt": "float32",
    }
    chunks: list[pd.DataFrame] = []
    chunksize = int(get_nested(config, "audit.prediction_read_chunksize", 500_000))
    for chunk in pd.read_csv(source, usecols=usecols, dtype=dtypes, chunksize=chunksize):
        filtered = chunk[(chunk["mode"] == mode) & (chunk["model"] == model)].copy()
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        raise ValueError(f"No exp073 rows found in {source} for mode={mode} model={model}")
    frame = pd.concat(chunks, ignore_index=True)
    for col in ["id", "well", "mode", "model"]:
        frame[col] = frame[col].astype(str)
    frame["pred_tvt"] = pd.to_numeric(frame["pred_tvt"], errors="coerce").astype(np.float64)
    frame["last_known_tvt"] = pd.to_numeric(frame["last_known_tvt"], errors="coerce").astype(
        np.float64
    )
    if frame["id"].duplicated().any():
        raise ValueError("exp073 inference prediction contains duplicated ids")
    if not np.isfinite(frame[["pred_tvt", "last_known_tvt"]].to_numpy()).all():
        raise ValueError("exp073 inference prediction contains non-finite numeric values")
    metadata = {
        "path": str(source),
        "raw_file_sha256": sha256_file(source),
        "decompressed_content_sha256": sha256_decompressed_csv(source),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "mode": mode,
        "model": model,
        "prediction_sha256": prediction_sha256(
            frame["id"], frame["pred_tvt"].to_numpy(np.float32), label=f"exp073/{mode}/{model}"
        ),
    }
    return frame, metadata


def generate_exp073_base_predictions_for_current_test(
    config: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, dict[str, Any]]:
    from exp063_full_replay_reproducibility_guard import run_saved_model_inference

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generation = get_nested(config, "inference.exp073_base_generation", {}) or {}
    feature_generation = generation.get("feature_generation", {}) or {}
    sample_path = Path(
        get_nested(config, "data.sample_submission", "data/raw/sample_submission.csv")
    )
    raw_dir = Path(get_nested(config, "data.raw_dir", "data/raw"))
    temp_submission_path = output_dir / "_exp073_base_submission.csv"
    summary = run_saved_model_inference(
        output_dir=output_dir,
        submission_path=temp_submission_path,
        sample_submission_path=sample_path,
        data_dir=raw_dir,
        tracker_test_path=generation.get("tracker_test_path"),
        model_manifest_path=generation.get("model_manifest_path"),
        mode_name=str(
            generation.get(
                "selected_mode",
                get_nested(config, "audit.selected_mode", "gpu_repro_guard_dp_threads8"),
            )
        ),
        model_name=str(
            generation.get("selected_model", get_nested(config, "audit.selected_model", "lgb_mean"))
        ),
        submission_target_column=str(get_nested(config, "data.submission_target_column", "tvt")),
        regenerate_test_features=bool(generation.get("regenerate_test_features", True)),
        n_jobs=int(feature_generation.get("n_jobs", 8)),
        pf_seeds=int(feature_generation.get("pf_seeds", 128)),
        pf_particles=int(feature_generation.get("pf_particles", 500)),
        fast=bool(feature_generation.get("fast", False)),
        use_gpu=str(feature_generation.get("use_gpu", "auto")),
    )
    prediction_path = output_dir / EXP073_INFERENCE_FILENAME
    if not prediction_path.exists() or prediction_path.stat().st_size <= 0:
        raise FileNotFoundError(f"generated exp073 base prediction missing: {prediction_path}")
    return prediction_path, summary


def load_model_package_prediction(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_input_file(
        MODEL_PACKAGE_FILENAME,
        get_nested(config, "data.model_package_predictions"),
        local_roots=[
            Path("/tmp/kaggle-output/source-check/pilkwang-rogii-target-free-tvt-geosteering"),
            Path("/tmp/kaggle-output/pilkwang-rogii-target-free-tvt-geosteering"),
        ],
    )
    frame = read_submission_contract(
        source, target_col=str(get_nested(config, "data.submission_target_column", "tvt"))
    )
    metadata = {
        "path": str(source),
        "raw_file_sha256": sha256_file(source),
        "rows": int(len(frame)),
        "prediction_sha256": prediction_sha256(
            frame["id"], frame["tvt"].to_numpy(np.float32), label="model_package_only"
        ),
        "source_mode": str(
            get_nested(config, "model_package.prediction_mode", "precomputed_submission")
        ),
    }
    return frame.rename(columns={"tvt": "modelpkg_tvt"}), metadata


def align_inputs(
    exp073: pd.DataFrame,
    modelpkg: pd.DataFrame,
    sample_path: str | Path,
) -> pd.DataFrame:
    sample = pd.read_csv(sample_path)[["id"]].copy()
    sample["id"] = sample["id"].astype(str)
    if sample["id"].duplicated().any():
        raise ValueError(f"sample submission contains duplicated ids: {sample_path}")
    base = sample.merge(
        exp073[["id", "well", "last_known_tvt", "pred_tvt"]],
        on="id",
        how="left",
        validate="one_to_one",
    ).merge(modelpkg[["id", "modelpkg_tvt"]], on="id", how="left", validate="one_to_one")
    missing_cols = [col for col in ["pred_tvt", "modelpkg_tvt"] if base[col].isna().any()]
    if missing_cols:
        counts = {col: int(base[col].isna().sum()) for col in missing_cols}
        raise ValueError(f"aligned predictions are missing rows: {counts}")
    numeric = ["last_known_tvt", "pred_tvt", "modelpkg_tvt"]
    if not np.isfinite(base[numeric].to_numpy(np.float64)).all():
        raise ValueError("aligned predictions contain non-finite values")
    base = base.rename(columns={"pred_tvt": "base_tvt"})
    base["diff_modelpkg_minus_base"] = base["modelpkg_tvt"] - base["base_tvt"]
    base["abs_modelpkg_diff"] = np.abs(base["diff_modelpkg_minus_base"])
    return base


def apply_gate(frame: pd.DataFrame, *, gmax: float, scale: float) -> pd.DataFrame:
    if not (0.0 <= float(gmax) <= 1.0):
        raise ValueError(f"gmax must be in [0, 1], got {gmax}")
    if float(scale) <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    result = frame.copy()
    abs_diff = result["abs_modelpkg_diff"].to_numpy(np.float64)
    gate = float(gmax) / (1.0 + np.square(abs_diff / float(scale)))
    correction = gate * result["diff_modelpkg_minus_base"].to_numpy(np.float64)
    result["gate_weight"] = gate
    result["modelpkg_correction"] = correction
    result["pred_tvt"] = result["base_tvt"].to_numpy(np.float64) + correction
    result["gmax"] = float(gmax)
    result["scale"] = float(scale)
    if not np.isfinite(result["pred_tvt"].to_numpy(np.float64)).all():
        raise ValueError("gated prediction contains non-finite values")
    return result


def summarize_variant(
    frame: pd.DataFrame, *, variant: str, gmax: float, scale: float
) -> dict[str, Any]:
    correction = frame["modelpkg_correction"].to_numpy(np.float64)
    abs_correction = np.abs(correction)
    raw_diff = frame["abs_modelpkg_diff"].to_numpy(np.float64)
    pred = frame["pred_tvt"].to_numpy(np.float64)
    gate = frame["gate_weight"].to_numpy(np.float64)
    submission = frame[["id", "pred_tvt"]].rename(columns={"pred_tvt": "tvt"})
    return {
        "variant": variant,
        "gmax": float(gmax),
        "scale": float(scale),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()) if "well" in frame.columns else None,
        "raw_modelpkg_diff_abs_mean": float(np.mean(raw_diff)),
        "raw_modelpkg_diff_abs_p50": float(np.quantile(raw_diff, 0.50)),
        "raw_modelpkg_diff_abs_p95": float(np.quantile(raw_diff, 0.95)),
        "raw_modelpkg_diff_abs_p99": float(np.quantile(raw_diff, 0.99)),
        "raw_modelpkg_diff_abs_max": float(np.max(raw_diff)),
        "gate_weight_mean": float(np.mean(gate)),
        "gate_weight_p95": float(np.quantile(gate, 0.95)),
        "gate_weight_max": float(np.max(gate)),
        "correction_abs_mean": float(np.mean(abs_correction)),
        "correction_abs_p95": float(np.quantile(abs_correction, 0.95)),
        "correction_abs_p99": float(np.quantile(abs_correction, 0.99)),
        "correction_abs_max": float(np.max(abs_correction)),
        "prediction_min": float(np.min(pred)),
        "prediction_max": float(np.max(pred)),
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred)),
        "prediction_sha256": prediction_sha256(frame["id"], pred.astype(np.float32), label=variant),
        "submission_sha256": frame_sha256(submission, columns=["id", "tvt"]),
    }


def evaluate_gate_grid(
    aligned: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gmax_values = [
        float(value) for value in get_nested(config, "audit.grid.gmax", [0.003, 0.005, 0.010])
    ]
    scales = [float(value) for value in get_nested(config, "audit.grid.scale", [4.0, 5.0, 8.0])]
    selected_gmax = float(get_nested(config, "inference.selected_variant.gmax", 0.005))
    selected_scale = float(get_nested(config, "inference.selected_variant.scale", 4.0))
    max_raw_diff_p95 = float(
        get_nested(config, "audit.selection_guard.max_raw_modelpkg_diff_p95", 35.0)
    )
    max_correction_abs_p95 = float(
        get_nested(config, "audit.selection_guard.max_correction_abs_p95", 0.10)
    )
    max_correction_abs_max = float(
        get_nested(config, "audit.selection_guard.max_correction_abs_max", 1.0)
    )

    aligned_path = output_dir / f"{OUTPUT_PREFIX}_aligned_base_modelpkg.csv.gz"
    aligned.to_csv(aligned_path, index=False, compression="gzip")

    rows: list[dict[str, Any]] = []
    selected_frame: pd.DataFrame | None = None
    selected_summary: dict[str, Any] | None = None
    for gmax in gmax_values:
        for scale in scales:
            variant = (
                f"modelpkg_gate_g{int(round(gmax * 1000)):03d}_s{str(scale).replace('.', 'p')}"
            )
            gated = apply_gate(aligned, gmax=gmax, scale=scale)
            summary = summarize_variant(gated, variant=variant, gmax=gmax, scale=scale)
            summary["passes_raw_diff_guard"] = bool(
                summary["raw_modelpkg_diff_abs_p95"] <= max_raw_diff_p95
            )
            summary["passes_correction_p95_guard"] = bool(
                summary["correction_abs_p95"] <= max_correction_abs_p95
            )
            summary["passes_correction_max_guard"] = bool(
                summary["correction_abs_max"] <= max_correction_abs_max
            )
            summary["passes_all_guards"] = bool(
                summary["passes_raw_diff_guard"]
                and summary["passes_correction_p95_guard"]
                and summary["passes_correction_max_guard"]
            )
            rows.append(summary)
            if abs(gmax - selected_gmax) < 1e-12 and abs(scale - selected_scale) < 1e-12:
                selected_frame = gated
                selected_summary = summary

    if selected_frame is None or selected_summary is None:
        raise ValueError(
            f"selected gate gmax={selected_gmax} scale={selected_scale} is not in audit grid"
        )

    metrics = pd.DataFrame(rows).sort_values(["gmax", "scale"]).reset_index(drop=True)
    metrics_path = output_dir / f"{OUTPUT_PREFIX}_variant_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    selected_variant = str(selected_summary["variant"])
    pred_path = output_dir / f"{OUTPUT_PREFIX}_{selected_variant}_predictions.csv.gz"
    selected_frame.to_csv(pred_path, index=False, compression="gzip")

    selected_submission = selected_frame[["id", "pred_tvt"]].rename(columns={"pred_tvt": "tvt"})
    selected_submission_path = output_dir / f"{OUTPUT_PREFIX}_{selected_variant}_submission.csv"
    selected_submission.to_csv(selected_submission_path, index=False)

    summary = {
        "status": "audit_completed",
        "output_prefix": OUTPUT_PREFIX,
        "selected_variant": selected_variant,
        "selected_passes_all_guards": bool(selected_summary["passes_all_guards"]),
        "selected_summary": selected_summary,
        "guard_limits": {
            "max_raw_modelpkg_diff_p95": max_raw_diff_p95,
            "max_correction_abs_p95": max_correction_abs_p95,
            "max_correction_abs_max": max_correction_abs_max,
        },
        "artifacts": {
            "aligned": str(aligned_path),
            "variant_metrics": str(metrics_path),
            "selected_predictions": str(pred_path),
            "selected_submission": str(selected_submission_path),
        },
        "elapsed_seconds": float(time.time() - started),
    }
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"
    with summary_path.open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
        fp.write("\n")
    summary["artifacts"]["summary"] = str(summary_path)
    return summary


def load_and_align(
    config: dict[str, Any], *, sample_path: str | Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    exp073, exp073_meta = load_exp073_inference_predictions(config)
    modelpkg, modelpkg_meta = load_model_package_prediction(config)
    aligned = align_inputs(exp073, modelpkg, sample_path)
    meta = {
        "exp073_inference_predictions": exp073_meta,
        "model_package_prediction": modelpkg_meta,
        "aligned": {
            "rows": int(len(aligned)),
            "wells": int(aligned["well"].nunique()),
            "base_prediction_sha256": prediction_sha256(
                aligned["id"], aligned["base_tvt"].to_numpy(np.float32), label="aligned_exp073_base"
            ),
            "modelpkg_prediction_sha256": prediction_sha256(
                aligned["id"],
                aligned["modelpkg_tvt"].to_numpy(np.float32),
                label="aligned_model_package",
            ),
        },
    }
    return aligned, meta


def run_audit_from_config(
    config: dict[str, Any],
    *,
    sample_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned, input_meta = load_and_align(config, sample_path=sample_path)
    summary = evaluate_gate_grid(aligned, config, output_dir)
    summary["input_meta"] = input_meta
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"
    with summary_path.open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
        fp.write("\n")
    return summary


def write_base_only_submission(
    config: dict[str, Any],
    *,
    sample_path: str | Path,
    output_dir: str | Path,
    submission_path: str | Path,
    reason: str,
    base_generation_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(sample_path)[["id"]].copy()
    sample["id"] = sample["id"].astype(str)
    exp073, exp073_meta = load_exp073_inference_predictions(config)
    base = sample.merge(
        exp073[["id", "well", "last_known_tvt", "pred_tvt"]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    if base["pred_tvt"].isna().any():
        missing = int(base["pred_tvt"].isna().sum())
        first_ids = base.loc[base["pred_tvt"].isna(), "id"].head(10).tolist()
        raise ValueError(
            "exp073 base fallback does not cover sample_submission ids; "
            f"missing_rows={missing} first_missing_ids={first_ids}"
        )
    if not np.isfinite(base[["last_known_tvt", "pred_tvt"]].to_numpy(np.float64)).all():
        raise ValueError("exp073 base fallback contains non-finite numeric values")

    pred_path = output_dir / f"{OUTPUT_PREFIX}_exp073_base_only_predictions.csv.gz"
    base.to_csv(pred_path, index=False, compression="gzip")
    target_col = str(get_nested(config, "data.submission_target_column", "tvt"))
    final = sample.copy()
    final[target_col] = base["pred_tvt"].to_numpy(np.float64)
    submission_path = Path(submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    final[["id", target_col]].to_csv(submission_path, index=False)

    selected_summary = {
        "variant": "exp073_base_only_hidden_modelpkg_disabled",
        "rows": int(len(final)),
        "wells": int(base["well"].nunique()),
        "prediction_min": float(final[target_col].min()),
        "prediction_max": float(final[target_col].max()),
        "prediction_mean": float(final[target_col].mean()),
        "prediction_std": float(final[target_col].std()),
        "prediction_sha256": prediction_sha256(
            final["id"], final[target_col].to_numpy(np.float32), label="exp073_base_only"
        ),
    }
    summary = {
        "status": "modelpkg_disabled_base_submission_written",
        "output_prefix": OUTPUT_PREFIX,
        "reason": reason,
        "selected_variant": selected_summary["variant"],
        "selected_passes_all_guards": True,
        "selected_summary": selected_summary,
        "input_meta": {
            "exp073_inference_predictions": exp073_meta,
            "exp073_base_generation": base_generation_meta,
            "model_package_prediction": {
                "status": "disabled",
                "reason": reason,
            },
        },
        "submission": {
            "path": str(submission_path),
            "rows": int(len(final)),
            "columns": f"id,{target_col}",
            "submission_sha256": sha256_file(submission_path),
            "prediction_min": selected_summary["prediction_min"],
            "prediction_max": selected_summary["prediction_max"],
            "prediction_mean": selected_summary["prediction_mean"],
            "prediction_std": selected_summary["prediction_std"],
        },
        "artifacts": {
            "base_predictions": str(pred_path),
        },
    }
    summary_path = output_dir / f"{OUTPUT_PREFIX}_inference_summary.json"
    with summary_path.open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
        fp.write("\n")
    summary["artifacts"]["summary"] = str(summary_path)
    return summary


def prefer_runtime_competition_paths(config: dict[str, Any], sample_path: str | Path) -> None:
    sample_path = Path(sample_path)
    if sample_path.exists():
        set_nested(config, "data.sample_submission", str(sample_path))
    raw_dir = sample_path.parent
    if (raw_dir / "train").is_dir() and (raw_dir / "test").is_dir():
        set_nested(config, "data.raw_dir", str(raw_dir))
        set_nested(config, "data.train_dir", str(raw_dir / "train"))
        set_nested(config, "data.test_dir", str(raw_dir / "test"))


def run_inference_from_config(
    config: dict[str, Any],
    *,
    sample_path: str | Path,
    output_dir: str | Path,
    submission_path: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefer_runtime_competition_paths(config, sample_path)
    base_generation_meta: dict[str, Any] | None = None
    if bool(get_nested(config, "inference.generate_exp073_base_on_current_test", False)):
        generated_prediction_path, base_generation_meta = (
            generate_exp073_base_predictions_for_current_test(
                config,
                output_dir,
            )
        )
        configured_sources = _as_path_list(get_nested(config, "data.exp073_inference_predictions"))
        set_nested(
            config,
            "data.exp073_inference_predictions",
            [str(generated_prediction_path), *[str(path) for path in configured_sources]],
        )

    try:
        summary = run_audit_from_config(config, sample_path=sample_path, output_dir=output_dir)
    except Exception as exc:
        message = str(exc)
        can_disable_modelpkg = bool(
            get_nested(config, "inference.disable_modelpkg_on_row_mismatch", False)
        )
        is_modelpkg_row_mismatch = (
            "aligned predictions are missing rows" in message and "modelpkg_tvt" in message
        )
        if not (can_disable_modelpkg and is_modelpkg_row_mismatch):
            raise
        summary = write_base_only_submission(
            config,
            sample_path=sample_path,
            output_dir=output_dir,
            submission_path=submission_path,
            reason=(
                "model-package precomputed public prediction does not cover the current "
                f"sample_submission rows: {message}"
            ),
            base_generation_meta=base_generation_meta,
        )
        return summary

    if base_generation_meta is not None:
        summary.setdefault("input_meta", {})["exp073_base_generation"] = base_generation_meta
    if not bool(summary["selected_passes_all_guards"]):
        summary["status"] = "guard_failed_no_submission"
        summary_path = output_dir / f"{OUTPUT_PREFIX}_inference_summary.json"
        with summary_path.open("w") as fp:
            json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
            fp.write("\n")
        return summary

    selected_submission = Path(summary["artifacts"]["selected_submission"])
    sample = pd.read_csv(sample_path)[["id"]].copy()
    candidate = read_submission_contract(
        selected_submission,
        target_col=str(get_nested(config, "data.submission_target_column", "tvt")),
    )
    final = sample.merge(candidate, on="id", how="left", validate="one_to_one")
    if final["tvt"].isna().any() or not np.isfinite(final["tvt"].to_numpy(np.float64)).all():
        raise ValueError("selected submission does not cover sample_submission ids")
    submission_path = Path(submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    final[["id", "tvt"]].to_csv(submission_path, index=False)
    summary["status"] = "inference_submission_written"
    summary["submission"] = {
        "path": str(submission_path),
        "rows": int(len(final)),
        "columns": "id,tvt",
        "submission_sha256": sha256_file(submission_path),
        "prediction_min": float(final["tvt"].min()),
        "prediction_max": float(final["tvt"].max()),
        "prediction_mean": float(final["tvt"].mean()),
        "prediction_std": float(final["tvt"].std()),
    }
    summary_path = output_dir / f"{OUTPUT_PREFIX}_inference_summary.json"
    with summary_path.open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
        fp.write("\n")
    return summary
