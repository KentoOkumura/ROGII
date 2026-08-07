#!/usr/bin/env python3
"""Census existing exact-HMM position and momentum interventions.

This repository-wide diagnostic scans experiment configs and Python sources
for exp209-style ``sig_p`` contracts.  It answers whether any completed
experiment actually changed position sigma/grid resolution or momentum, rather
than merely proposing a design or changing another transition component.

The census does not execute an HMM and does not evaluate predictions.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_OUTPUT = Path(
    "studies/hmm_exp209_existing_transition_intervention_census_20260726"
)
DEFAULT_SIG_P = 0.02
DEFAULT_STEP = 0.35
DEFAULT_MOMENTUM = 0.998


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def flatten(
    value: Any,
    path: str = "",
) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            rows.append((child_path, child))
            rows.extend(flatten(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(flatten(child, f"{path}[{index}]"))
    return rows


def local_sig_p_contracts(
    value: Any,
    path: str = "",
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "sig_p" in value:
            contracts.append(
                {
                    "path": path or "<root>",
                    "sig_p": value.get("sig_p"),
                    "step": value.get("step"),
                    "mom": value.get("mom", value.get("momentum")),
                }
            )
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            contracts.extend(
                local_sig_p_contracts(child, child_path)
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            contracts.extend(
                local_sig_p_contracts(
                    child,
                    f"{path}[{index}]",
                )
            )
    return contracts


def normalized_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def metric_state(experiment_dir: Path) -> tuple[str | None, Any]:
    path = experiment_dir / "metrics.json"
    if not path.exists():
        return None, None
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "unreadable", None
    status = metrics.get("status")
    cv = metrics.get("cv")
    if cv is None:
        cv = metrics.get("rmse")
    return str(status) if status is not None else None, cv


def source_sig_p_literals(
    experiment_root: Path,
) -> pd.DataFrame:
    pattern = re.compile(
        r"\bsig_p\b"
        r"(?:\s*:\s*float)?"
        r"\s*=\s*"
        r"(?P<value>[0-9]+(?:\.[0-9]+)?)"
    )
    rows: list[dict[str, Any]] = []
    for path in sorted(experiment_root.glob("exp*/**/*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in pattern.finditer(line):
                rows.append(
                    {
                        "experiment": path.relative_to(
                            experiment_root
                        ).parts[0],
                        "source": str(path),
                        "line": line_number,
                        "value": float(match.group("value")),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    experiment_root = root / "experiments"
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)

    config_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    config_count = 0
    for config_path in sorted(
        experiment_root.glob("exp*/config.yaml")
    ):
        config_count += 1
        try:
            config = yaml.safe_load(
                config_path.read_text(encoding="utf-8")
            ) or {}
        except (yaml.YAMLError, OSError) as error:
            config_rows.append(
                {
                    "experiment": config_path.parent.name,
                    "config_path": str(config_path),
                    "parse_error": str(error),
                    "has_sig_p": False,
                }
            )
            continue

        flattened = flatten(config)
        sig_p_entries = [
            (path, value)
            for path, value in flattened
            if path.rsplit(".", 1)[-1] == "sig_p"
        ]
        position_schedule_paths = [
            path
            for path, _value in flattened
            if "position_sigma_schedule" in path
        ]
        momentum_schedule_paths = [
            path
            for path, _value in flattened
            if "momentum_schedule" in path
        ]
        if not (
            sig_p_entries
            or position_schedule_paths
            or momentum_schedule_paths
        ):
            continue
        experiment = config_path.parent.name
        experiment_meta = config.get("experiment") or {}
        metrics_status, metrics_cv = metric_state(config_path.parent)
        contracts = local_sig_p_contracts(config)
        numeric_sig_p = [
            float(contract["sig_p"])
            for contract in contracts
            if is_number(contract["sig_p"])
        ]
        numeric_steps = [
            float(contract["step"])
            for contract in contracts
            if is_number(contract["step"])
        ]
        numeric_momenta = [
            float(contract["mom"])
            for contract in contracts
            if is_number(contract["mom"])
        ]
        nondefault_sig_p = [
            value
            for value in numeric_sig_p
            if abs(value - DEFAULT_SIG_P) > 1e-12
        ]
        nondefault_step = [
            value
            for value in numeric_steps
            if abs(value - DEFAULT_STEP) > 1e-12
        ]
        nondefault_momentum = [
            value
            for value in numeric_momenta
            if abs(value - DEFAULT_MOMENTUM) > 1e-12
        ]
        has_position_intervention_design = bool(
            nondefault_sig_p
            or nondefault_step
            or position_schedule_paths
        )
        has_momentum_intervention_design = bool(
            nondefault_momentum or momentum_schedule_paths
        )
        status = str(experiment_meta.get("status", ""))
        terminal_without_run = any(
            token in status.lower()
            for token in (
                "closed_without_run",
                "design",
                "not_run",
                "unimplemented",
            )
        )
        has_recorded_metric = metrics_cv is not None
        config_rows.append(
            {
                "experiment": experiment,
                "config_path": str(config_path),
                "parse_error": None,
                "status": status,
                "route": experiment_meta.get("route"),
                "metrics_status": metrics_status,
                "metrics_cv": normalized_json(metrics_cv),
                "has_sig_p": bool(sig_p_entries),
                "sig_p_entries": normalized_json(sig_p_entries),
                "local_contract_count": len(contracts),
                "numeric_sig_p_values": normalized_json(
                    numeric_sig_p
                ),
                "numeric_step_values": normalized_json(
                    numeric_steps
                ),
                "numeric_momentum_values": normalized_json(
                    numeric_momenta
                ),
                "nondefault_sig_p_values": normalized_json(
                    nondefault_sig_p
                ),
                "nondefault_step_values": normalized_json(
                    nondefault_step
                ),
                "nondefault_momentum_values": normalized_json(
                    nondefault_momentum
                ),
                "position_sigma_schedule_paths": normalized_json(
                    position_schedule_paths
                ),
                "momentum_schedule_paths": normalized_json(
                    momentum_schedule_paths
                ),
                "has_position_intervention_design": (
                    has_position_intervention_design
                ),
                "has_momentum_intervention_design": (
                    has_momentum_intervention_design
                ),
                "terminal_without_run": terminal_without_run,
                "has_recorded_metric": has_recorded_metric,
                "completed_position_intervention": bool(
                    has_position_intervention_design
                    and has_recorded_metric
                    and not terminal_without_run
                ),
                "completed_momentum_intervention": bool(
                    has_momentum_intervention_design
                    and has_recorded_metric
                    and not terminal_without_run
                ),
            }
        )
        for contract in contracts:
            local_rows.append(
                {
                    "experiment": experiment,
                    "status": status,
                    **contract,
                }
            )

    config_frame = pd.DataFrame(config_rows)
    local_frame = pd.DataFrame(local_rows)
    source_frame = source_sig_p_literals(experiment_root)

    numeric_sig_p_counter: Counter[str] = Counter()
    numeric_step_counter: Counter[str] = Counter()
    numeric_momentum_counter: Counter[str] = Counter()
    for row in local_rows:
        for key, counter in (
            ("sig_p", numeric_sig_p_counter),
            ("step", numeric_step_counter),
            ("mom", numeric_momentum_counter),
        ):
            value = row[key]
            if is_number(value):
                counter[f"{float(value):.12g}"] += 1

    position_designs = config_frame.loc[
        config_frame.get(
            "has_position_intervention_design",
            pd.Series(False, index=config_frame.index),
        ).fillna(False)
    ]
    momentum_designs = config_frame.loc[
        config_frame.get(
            "has_momentum_intervention_design",
            pd.Series(False, index=config_frame.index),
        ).fillna(False)
    ]
    completed_position = config_frame.loc[
        config_frame.get(
            "completed_position_intervention",
            pd.Series(False, index=config_frame.index),
        ).fillna(False)
    ]
    completed_momentum = config_frame.loc[
        config_frame.get(
            "completed_momentum_intervention",
            pd.Series(False, index=config_frame.index),
        ).fillna(False)
    ]
    unusual_source_literals = (
        source_frame.loc[
            (source_frame["value"] - DEFAULT_SIG_P).abs() > 1e-12
        ]
        if not source_frame.empty
        else source_frame
    )

    summary = {
        "scope": {
            "experiment_configs": config_count,
            "hmm_transition_configs": int(len(config_frame)),
            "configs_with_sig_p": int(
                config_frame["has_sig_p"].fillna(False).sum()
            ),
            "local_sig_p_contracts": int(len(local_frame)),
            "python_sig_p_numeric_literals": int(len(source_frame)),
        },
        "local_numeric_values": {
            "sig_p": dict(sorted(numeric_sig_p_counter.items())),
            "step_in_same_mapping_as_sig_p": dict(
                sorted(numeric_step_counter.items())
            ),
            "momentum_in_same_mapping_as_sig_p": dict(
                sorted(numeric_momentum_counter.items())
            ),
        },
        "position_intervention_designs": position_designs[
            ["experiment", "status"]
        ].to_dict(orient="records"),
        "momentum_intervention_designs": momentum_designs[
            ["experiment", "status"]
        ].to_dict(orient="records"),
        "completed_actual_position_interventions": completed_position[
            ["experiment", "status", "metrics_status"]
        ].to_dict(orient="records"),
        "completed_actual_momentum_interventions": completed_momentum[
            ["experiment", "status", "metrics_status"]
        ].to_dict(orient="records"),
        "nondefault_python_sig_p_literals": unusual_source_literals.to_dict(
            orient="records"
        ),
        "interpretation": (
            "A design-only or closed-without-run schedule is not evidence "
            "from an actual HMM intervention. The census treats a completed "
            "intervention as requiring a non-default position/momentum "
            "contract, a recorded metric, and a non-design terminal status."
        ),
    }

    config_frame.to_csv(
        output / "experiment_config_census.csv",
        index=False,
    )
    local_frame.to_csv(
        output / "local_sig_p_contracts.csv",
        index=False,
    )
    source_frame.to_csv(
        output / "python_sig_p_literals.csv",
        index=False,
    )
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
