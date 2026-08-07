from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from settings import ExperimentPaths, get_nested, load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate an existing experiment for code submit.")
    parser.add_argument(
        "--output-prefix",
        default="cv_submit_gate",
        help="Prefix for decision artifacts in the experiment artifacts directory.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        value = json.load(fp)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def resolve_path(root: Path, value: Any) -> Path | None:
    if value in (None, "", "TODO"):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Rule:
    name: str
    passed: bool
    required: bool
    observed: Any
    expected: Any
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "required": self.required,
            "observed": self.observed,
            "expected": self.expected,
            "note": self.note,
        }


def summarize_submission(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}

    rows = 0
    duplicate_ids = 0
    missing_values = 0
    ids_seen: set[str] = set()
    values: list[float] = []
    columns: list[str] = []

    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        columns = list(reader.fieldnames or [])
        for row in reader:
            rows += 1
            row_id = str(row.get("id", ""))
            if row_id in ids_seen:
                duplicate_ids += 1
            ids_seen.add(row_id)
            raw_value = row.get("tvt")
            value = as_float(raw_value)
            if value is None:
                missing_values += 1
            else:
                values.append(value)

    mean_value = sum(values) / len(values) if values else None
    std_value = None
    if values and mean_value is not None:
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        std_value = math.sqrt(variance)

    return {
        "exists": True,
        "path": str(path),
        "columns": columns,
        "rows": rows,
        "duplicate_ids": duplicate_ids,
        "missing_values": missing_values,
        "prediction_min": min(values) if values else None,
        "prediction_max": max(values) if values else None,
        "prediction_mean": mean_value,
        "prediction_std": std_value,
        "sha256": sha256_file(path),
    }


def source_ravaghi_mean(metrics: dict[str, Any]) -> float | None:
    for row in metrics.get("pooled_metrics", []):
        if not isinstance(row, dict):
            continue
        if row.get("variant") == "ravaghi_public_lgbm_replay" and row.get("model") == "lgb_mean":
            return as_float(row.get("rmse_tvt"))
    return None


def build_rules(
    config: dict[str, Any],
    source_metrics: dict[str, Any],
    probe_metrics: dict[str, Any],
    submission_summary: dict[str, Any],
) -> tuple[list[Rule], dict[str, Any]]:
    gate = config.get("gate", {})
    inference = source_metrics.get("inference", {})
    source_submission = inference.get("submission", {})
    source_submit_check = inference.get("submit_check", {})
    hidden_probe = probe_metrics.get("hidden_code_submission", {})

    source_status = source_metrics.get("status")
    inference_status = inference.get("status")
    source_cv = as_float(get_nested(source_metrics, "inference_candidate.rmse_tvt"))
    selected_cv = as_float(get_nested(source_metrics, "selected_candidate.rmse_tvt"))
    ravaghi_mean = source_ravaghi_mean(source_metrics)
    delta_vs_ravaghi = None
    if source_cv is not None and ravaghi_mean is not None:
        delta_vs_ravaghi = ravaghi_mean - source_cv

    expected_rows = as_int(gate.get("expected_submission_rows"))
    local_rows = as_int(submission_summary.get("rows"))
    metrics_rows = as_int(source_submission.get("rows"))
    observed_rows = local_rows if local_rows is not None else metrics_rows

    local_sha = submission_summary.get("sha256")
    metrics_sha = source_submission.get("sha256")
    observed_min = as_float(submission_summary.get("prediction_min"))
    observed_max = as_float(submission_summary.get("prediction_max"))
    if observed_min is None:
        observed_min = as_float(source_submission.get("prediction_min"))
    if observed_max is None:
        observed_max = as_float(source_submission.get("prediction_max"))

    probe_interpretation = str(hidden_probe.get("interpretation", ""))

    evidence = {
        "source_status": source_status,
        "source_cv": source_cv,
        "selected_cv": selected_cv,
        "ravaghi_mean_cv": ravaghi_mean,
        "delta_vs_ravaghi_mean": delta_vs_ravaghi,
        "inference_status": inference_status,
        "submit_check_status": source_submit_check.get("status"),
        "fallback_rows": source_submission.get("fallback_rows"),
        "submission_rows": observed_rows,
        "submission_sha256": local_sha or metrics_sha,
        "metrics_submission_sha256": metrics_sha,
        "local_submission_exists": submission_summary.get("exists"),
        "prediction_min": observed_min,
        "prediction_max": observed_max,
        "probe_status": hidden_probe.get("status"),
        "probe_public_lb": hidden_probe.get("public_lb"),
        "probe_interpretation": probe_interpretation,
    }

    max_cv = as_float(gate.get("max_cv_rmse"))
    min_delta = as_float(gate.get("min_delta_vs_ravaghi_mean"))
    max_fallback = as_int(gate.get("max_fallback_rows"))
    pred_min_lower = as_float(gate.get("prediction_min_lower_bound"))
    pred_max_upper = as_float(gate.get("prediction_max_upper_bound"))

    rules = [
        Rule(
            "source_experiment_completed",
            source_status == gate.get("required_source_status"),
            True,
            source_status,
            gate.get("required_source_status"),
            "exp063 train-side audit completed.",
        ),
        Rule(
            "cv_below_submit_gate",
            source_cv is not None and max_cv is not None and source_cv <= max_cv,
            True,
            source_cv,
            f"<= {max_cv}",
            "Inference candidate pooled RMSE must clear the submit gate.",
        ),
        Rule(
            "improves_ravaghi_replay",
            delta_vs_ravaghi is not None
            and min_delta is not None
            and delta_vs_ravaghi >= min_delta,
            True,
            delta_vs_ravaghi,
            f">= {min_delta}",
            "Pixiux replay should beat Ravaghi replay by a material margin on the same surface.",
        ),
        Rule(
            "inference_completed",
            inference_status == gate.get("required_inference_status"),
            True,
            inference_status,
            gate.get("required_inference_status"),
            "exp063 saved-booster inference v2 completed.",
        ),
        Rule(
            "submit_check_passed",
            source_submit_check.get("status") == gate.get("required_submit_check_status"),
            True,
            source_submit_check.get("status"),
            gate.get("required_submit_check_status"),
            "exp063 inference output passed sample submission checks.",
        ),
        Rule(
            "fallback_rows_zero",
            as_int(source_submission.get("fallback_rows")) == max_fallback,
            True,
            source_submission.get("fallback_rows"),
            max_fallback,
            "No fallback predictions should be used.",
        ),
        Rule(
            "submission_rows_match",
            observed_rows == expected_rows,
            True,
            observed_rows,
            expected_rows,
            "Submission row count should match the sample submission.",
        ),
        Rule(
            "prediction_range_sanity",
            observed_min is not None
            and observed_max is not None
            and pred_min_lower is not None
            and pred_max_upper is not None
            and observed_min >= pred_min_lower
            and observed_max <= pred_max_upper,
            True,
            {"min": observed_min, "max": observed_max},
            {"min_lower": pred_min_lower, "max_upper": pred_max_upper},
            "Prediction range should stay in the historical TVT range.",
        ),
        Rule(
            "submission_file_present",
            bool(submission_summary.get("exists")),
            bool(gate.get("require_submission_file_present", True)),
            submission_summary.get("path"),
            "existing local source submission file",
            "Local file is required when this gate is run before a manual submit.",
        ),
        Rule(
            "submission_sha_matches_metrics",
            local_sha is not None and metrics_sha is not None and local_sha == metrics_sha,
            bool(gate.get("require_sha_match", True)),
            {"local": local_sha, "metrics": metrics_sha},
            "equal sha256",
            "The local file to submit should match exp063 recorded metadata.",
        ),
        Rule(
            "hidden_overlap_probe_completed",
            hidden_probe.get("status") == gate.get("required_probe_status")
            and "no train/test well_id overlap" in probe_interpretation,
            True,
            {
                "status": hidden_probe.get("status"),
                "interpretation": probe_interpretation,
            },
            {
                "status": gate.get("required_probe_status"),
                "interpretation_contains": "no train/test well_id overlap",
            },
            "exp064 code submission probe should complete without the overlap assertion firing.",
        ),
    ]
    return rules, evidence


def write_rule_csv(path: Path, rules: list[Rule]) -> None:
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["name", "passed", "required", "observed", "expected", "note"],
        )
        writer.writeheader()
        for rule in rules:
            row = rule.as_dict()
            row["observed"] = json.dumps(row["observed"], ensure_ascii=False, sort_keys=True)
            row["expected"] = json.dumps(row["expected"], ensure_ascii=False, sort_keys=True)
            writer.writerow(row)


def write_report(path: Path, decision: dict[str, Any], rules: list[Rule]) -> None:
    lines = [
        "# CV Submit Gate Report",
        "",
        f"- experiment: `{decision['experiment']}`",
        f"- decision: `{decision['decision']}`",
        f"- approved_for_code_submit: `{decision['approved_for_code_submit']}`",
        f"- source_experiment: `{decision['source_experiment']}`",
        f"- source_cv: `{decision['evidence']['source_cv']}`",
        f"- delta_vs_ravaghi_mean: `{decision['evidence']['delta_vs_ravaghi_mean']}`",
        f"- submission_sha256: `{decision['evidence']['submission_sha256']}`",
        "",
        "## Rules",
        "",
        "| rule | required | passed | observed | expected |",
        "| --- | --- | --- | --- | --- |",
    ]
    for rule in rules:
        observed = json.dumps(rule.observed, ensure_ascii=False, sort_keys=True)
        expected = json.dumps(rule.expected, ensure_ascii=False, sort_keys=True)
        lines.append(
            f"| `{rule.name}` | {rule.required} | {rule.passed} | `{observed}` | `{expected}` |"
        )

    submit = decision["submit_target"]
    lines.extend(
        [
            "",
            "## Submit Target",
            "",
            f"- kernel: `{submit['kernel']}`",
            f"- version: `{submit['version']}`",
            f"- output_file: `{submit['output_file']}`",
            f"- source_submission_path: `{submit['source_submission_path']}`",
            "",
            "```bash",
            submit["command"],
            "```",
            "",
            "This gate does not submit automatically and does not update any LB anchor.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    config = load_config()

    source_metrics_path = resolve_path(paths.root, get_nested(config, "data.source_metrics"))
    probe_metrics_path = resolve_path(paths.root, get_nested(config, "data.overlap_probe_metrics"))
    source_submission_path = resolve_path(paths.root, get_nested(config, "data.source_submission"))

    if source_metrics_path is None or not source_metrics_path.exists():
        raise FileNotFoundError(f"source metrics not found: {source_metrics_path}")
    if probe_metrics_path is None or not probe_metrics_path.exists():
        raise FileNotFoundError(f"probe metrics not found: {probe_metrics_path}")
    if source_submission_path is None:
        raise ValueError("data.source_submission is required")

    source_metrics = load_json(source_metrics_path)
    probe_metrics = load_json(probe_metrics_path)
    submission_summary = summarize_submission(source_submission_path)

    rules, evidence = build_rules(config, source_metrics, probe_metrics, submission_summary)
    required_rules = [rule for rule in rules if rule.required]
    approved = all(rule.passed for rule in required_rules)

    candidate = get_nested(config, "model.candidate") or {}
    message = (
        "exp063 strict Pixiux replay lgb_mean CV "
        f"{evidence['source_cv']:.6f}; exp066 gate approved"
        if evidence["source_cv"] is not None
        else "exp063 strict Pixiux replay; exp066 gate approved"
    )
    command = (
        "kaggle competitions submit rogii-wellbore-geology-prediction "
        f"-k {candidate.get('inference_kernel')} "
        f"-v {candidate.get('inference_kernel_version')} "
        f"-f {candidate.get('output_file')} "
        f'-m "{message}"'
    )

    decision = {
        "experiment": "exp066_cv_submit_gate",
        "status": "completed",
        "decision": "approved_for_code_submit" if approved else "blocked_do_not_submit",
        "approved_for_code_submit": approved,
        "created_at": datetime.now(UTC).isoformat(),
        "source_experiment": candidate.get("source_experiment"),
        "source_metrics_path": str(source_metrics_path),
        "probe_metrics_path": str(probe_metrics_path),
        "evidence": evidence,
        "rules": [rule.as_dict() for rule in rules],
        "submit_target": {
            "kernel": candidate.get("inference_kernel"),
            "version": candidate.get("inference_kernel_version"),
            "output_file": candidate.get("output_file"),
            "source_submission_path": str(source_submission_path),
            "command": command,
        },
        "notes": [
            "Gate approval only means the exp063 inference v2 kernel is ready for a code submit.",
            "No LB anchor is updated until the code submission completes and is recorded.",
        ],
    }

    prefix = args.output_prefix
    decision_json = paths.artifacts_dir / f"{prefix}_decision.json"
    decision_csv = paths.artifacts_dir / f"{prefix}_decision.csv"
    report_md = paths.artifacts_dir / f"{prefix}_report.md"
    decision_json.write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    write_rule_csv(decision_csv, rules)
    write_report(report_md, decision, rules)

    metrics = {
        "experiment": "exp066_cv_submit_gate",
        "status": "completed",
        "cv": evidence["source_cv"],
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "route": "ml_model",
        "approved_for_code_submit": approved,
        "decision": decision["decision"],
        "source_experiment": candidate.get("source_experiment"),
        "source_kernel": candidate.get("inference_kernel"),
        "source_kernel_version": candidate.get("inference_kernel_version"),
        "source_submission_sha256": evidence["submission_sha256"],
        "delta_vs_ravaghi_mean": evidence["delta_vs_ravaghi_mean"],
        "rule_pass_count": sum(rule.passed for rule in rules),
        "required_rule_pass_count": sum(rule.passed for rule in required_rules),
        "required_rule_count": len(required_rules),
        "artifacts": {
            "decision": decision_json.name,
            "rules": decision_csv.name,
            "report": report_md.name,
        },
        "notes": (
            "Gate passed; exp063 inference v2 is approved for code submit, "
            "but no submission was executed by exp066."
        ),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    paths.metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
