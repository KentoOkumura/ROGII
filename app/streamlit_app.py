from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from scripts.config_utils import ROOT, load_project_config, project_path

PROJECT_CONFIG = load_project_config()
EXPERIMENTS_DIR = project_path(PROJECT_CONFIG, "paths.experiments_dir")
SUMMARY_PATH = ROOT / "experiment_summary.md"
SUBMISSIONS_PATH = project_path(PROJECT_CONFIG, "paths.submissions_file")


def list_experiments() -> list[Path]:
    return sorted(path for path in EXPERIMENTS_DIR.glob("exp*") if path.is_dir())


def read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as fp:
        value = yaml.safe_load(fp) or {}
    return value if isinstance(value, dict) else {}


def load_score_table() -> pd.DataFrame:
    rows = []
    for experiment_dir in list_experiments():
        config = read_yaml(experiment_dir / "config.yaml")
        metrics_path = experiment_dir / "metrics.json"
        if metrics_path.exists():
            metrics = pd.read_json(metrics_path, typ="series").to_dict()
        else:
            metrics = {}
        experiment = config.get("experiment", {})
        lineage = config.get("lineage", {})
        rows.append(
            {
                "experiment": experiment.get("name", experiment_dir.name),
                "route": experiment.get("route"),
                "parent": lineage.get("parent"),
                "status": metrics.get("status", experiment.get("status")),
                "cv": metrics.get("cv"),
                "public_lb": metrics.get("public_lb"),
                "private_lb": metrics.get("private_lb"),
                "summary": experiment.get("description") or lineage.get("diff_summary"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Kaggle Experiment Dashboard", layout="wide")
    st.title("Kaggle Experiment Dashboard")

    score_table = load_score_table()
    st.subheader("Experiments")
    if score_table.empty:
        st.info("No experiments found.")
    else:
        st.dataframe(score_table, use_container_width=True, hide_index=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Experiment Summary")
        st.markdown(read_text(SUMMARY_PATH) or "No summary yet.")
    with col_right:
        st.subheader("Submission History")
        st.markdown(read_text(SUBMISSIONS_PATH) or "No submissions yet.")


if __name__ == "__main__":
    main()
