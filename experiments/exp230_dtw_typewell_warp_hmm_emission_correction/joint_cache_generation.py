from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from direct_hmm_comparison import exp072_column, run_direct_comparison
from exact_hmm_smoother import to_jsonable
from exp072_feature_cache import run_train_feature_cache as run_exp072_feature_cache
from feature_cache import main as run_hmm_feature_cache
from settings import ExperimentPaths, get_nested, load_config


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_optional(paths: ExperimentPaths, candidates: list[str]) -> Path | None:
    checked: list[Path] = []
    for raw in candidates:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = paths.root / candidate
        checked.append(candidate)
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for raw in candidates:
            basename = Path(raw).name
            if not basename:
                continue
            for candidate in sorted(kaggle_input.rglob(basename)):
                checked.append(candidate)
                if candidate.exists() and candidate.stat().st_size > 0:
                    return candidate
    return None


def gzip_file_report(
    *,
    paths: ExperimentPaths,
    label: str,
    generated_path: Path | None,
    reference_candidates: list[str],
    expected_raw_gzip_sha256: str | None = None,
    expected_decompressed_sha256: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "label": label,
        "generated_path": str(generated_path) if generated_path is not None else None,
        "generated_exists": bool(generated_path and generated_path.exists()),
        "reference_path": None,
        "reference_exists": False,
    }
    if generated_path is not None and generated_path.exists():
        generated_raw = sha256_path(generated_path)
        generated_decompressed = sha256_gzip_decompressed(generated_path)
        report["generated_raw_gzip_sha256"] = generated_raw
        report["generated_decompressed_sha256"] = generated_decompressed
        if expected_raw_gzip_sha256:
            report["expected_raw_gzip_sha256"] = expected_raw_gzip_sha256
            report["matches_expected_raw_gzip_sha256"] = generated_raw == expected_raw_gzip_sha256
        if expected_decompressed_sha256:
            report["expected_decompressed_sha256"] = expected_decompressed_sha256
            report["matches_expected_decompressed_sha256"] = generated_decompressed == expected_decompressed_sha256

    reference_path = resolve_optional(paths, reference_candidates)
    if reference_path is not None:
        report["reference_path"] = str(reference_path)
        report["reference_exists"] = True
        reference_raw = sha256_path(reference_path)
        reference_decompressed = sha256_gzip_decompressed(reference_path)
        report["reference_raw_gzip_sha256"] = reference_raw
        report["reference_decompressed_sha256"] = reference_decompressed
        if generated_path is not None and generated_path.exists():
            report["matches_reference_raw_gzip_sha256"] = report.get("generated_raw_gzip_sha256") == reference_raw
            report["matches_reference_decompressed_sha256"] = (
                report.get("generated_decompressed_sha256") == reference_decompressed
            )
    return report


def maybe_run_reference_parity_checks(
    paths: ExperimentPaths,
    config: dict[str, Any],
    exp072_summary: dict[str, Any] | None,
    hmm_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    references = get_nested(config, "reference_artifacts") or {}
    reports: dict[str, Any] = {}
    if exp072_summary is not None:
        outputs = exp072_summary.get("outputs") or {}
        generated = paths.artifacts_dir / str(outputs.get("train_features"))
        ref = references.get("exp072_train_features") or {}
        reports["exp072_train_features"] = gzip_file_report(
            paths=paths,
            label="exp072_train_features",
            generated_path=generated,
            reference_candidates=list(ref.get("candidates") or []),
            expected_raw_gzip_sha256=ref.get("expected_raw_gzip_sha256"),
            expected_decompressed_sha256=ref.get("expected_decompressed_sha256"),
        )
    if hmm_summary is not None:
        outputs = hmm_summary.get("outputs") or {}
        generated = paths.artifacts_dir / str(outputs.get("train_features"))
        ref = references.get("exp205_hmm_train_features") or {}
        reports["exp205_hmm_train_features"] = gzip_file_report(
            paths=paths,
            label="exp205_hmm_train_features",
            generated_path=generated,
            reference_candidates=list(ref.get("candidates") or []),
            expected_raw_gzip_sha256=ref.get("expected_raw_gzip_sha256"),
            expected_decompressed_sha256=ref.get("expected_decompressed_sha256"),
        )
    return reports


def metric_parity_report(config: dict[str, Any], comparison_summary: dict[str, Any] | None) -> dict[str, Any]:
    if comparison_summary is None:
        return {"checked": False, "reason": "comparison not run"}
    comparison = get_nested(config, "comparison") or {}
    expected_candidate = comparison.get("expected_best_candidate")
    expected_rmse = comparison.get("expected_best_rmse_tvt")
    tolerance = float(comparison.get("metric_abs_tolerance", 0.0) or 0.0)
    best = comparison_summary.get("best_candidate") or {}
    actual_candidate = best.get("candidate")
    actual_rmse = best.get("rmse")
    report: dict[str, Any] = {
        "checked": True,
        "expected_best_candidate": expected_candidate,
        "actual_best_candidate": actual_candidate,
        "best_candidate_matches": actual_candidate == expected_candidate if expected_candidate else None,
        "expected_best_rmse_tvt": expected_rmse,
        "actual_best_rmse_tvt": actual_rmse,
        "metric_abs_tolerance": tolerance,
    }
    if expected_rmse is not None and actual_rmse is not None:
        diff = abs(float(actual_rmse) - float(expected_rmse))
        report["best_rmse_abs_diff"] = diff
        report["best_rmse_matches"] = diff <= tolerance
    return report


def run_joint_generation() -> dict[str, Any]:
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    config = load_config()
    execution = get_nested(config, "execution") or {}
    started = time.time()

    exp072_summary: dict[str, Any] | None = None
    exp072_frame: pd.DataFrame | None = None
    hmm_summary: dict[str, Any] | None = None
    comparison_summary: dict[str, Any] | None = None

    if bool(execution.get("run_exp072_full_cache", True)):
        exp072_config = get_nested(config, "feature_cache.exp072") or {}
        comparison = get_nested(config, "comparison") or {}
        baseline_candidates = list(comparison.get("baseline_candidate_columns") or ["likpf_mean"])
        return_columns = ["id", "well", "target", "last_known_tvt", "md_since"]
        for candidate in baseline_candidates:
            column = exp072_column(candidate)
            if column not in return_columns:
                return_columns.append(column)
        max_wells = exp072_config.get("max_wells")
        if max_wells is not None:
            max_wells = int(max_wells)
        keep_frame = bool(exp072_config.get("keep_frame_for_comparison", True))
        exp072_result = run_exp072_feature_cache(
            data_dir=paths.raw_data_dir,
            output_dir=paths.artifacts_dir,
            n_jobs=exp072_config.get("n_jobs"),
            pf_seeds=exp072_config.get("pf_seeds"),
            pf_particles=exp072_config.get("pf_particles"),
            fast=bool(exp072_config.get("fast", False)),
            max_wells=max_wells,
            output_prefix=str(exp072_config.get("output_prefix") or "exp063_full_replay_feature_cache"),
            return_frame=keep_frame,
            return_columns=return_columns if keep_frame else None,
        )
        if keep_frame:
            exp072_summary, exp072_frame = exp072_result  # type: ignore[assignment]
        else:
            exp072_summary = exp072_result  # type: ignore[assignment]

    if bool(execution.get("run_hmm_cache", True)):
        hmm_summary = run_hmm_feature_cache()

    if bool(execution.get("run_direct_comparison", True)):
        use_in_memory_exp072 = bool(execution.get("direct_comparison_use_in_memory_exp072", True))
        if use_in_memory_exp072 and exp072_frame is None:
            raise RuntimeError("direct_comparison_use_in_memory_exp072=True requires exp072 frame generation")
        comparison_summary = run_direct_comparison(
            baseline_frame=exp072_frame if use_in_memory_exp072 else None,
            baseline_source="in_memory_exp072_full_cache_generated_by_exp230" if use_in_memory_exp072 else None,
        )

    parity = (
        maybe_run_reference_parity_checks(paths, config, exp072_summary, hmm_summary)
        if bool(execution.get("run_reference_parity_checks", True))
        else {}
    )
    metric_parity = metric_parity_report(config, comparison_summary)

    output_path = paths.artifacts_dir / "exp230_dtw_hmm_generation_summary.json"
    summary = {
        "experiment": paths.experiment_name,
        "status": "joint_generation_completed_pending_review",
        "rows": exp072_summary.get("rows") if exp072_summary else hmm_summary.get("rows") if hmm_summary else None,
        "wells": exp072_summary.get("wells") if exp072_summary else hmm_summary.get("wells") if hmm_summary else None,
        "exp072": exp072_summary,
        "hmm": hmm_summary,
        "comparison": comparison_summary,
        "parity": parity,
        "metric_parity": metric_parity,
        "outputs": {"joint_summary": str(output_path)},
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(output_path, summary)
    summary["sha256"] = {"joint_summary": sha256_path(output_path)}
    write_json(output_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run_joint_generation()
