from __future__ import annotations

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_update_metrics_preserves_existing_scores_and_nested_evidence(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(ROOT / "templates" / "experiment" / "settings.py"))
    update_metrics = namespace["update_metrics"]
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "experiment": "exp123_test",
                "public_lb": 0.12,
                "evidence": {
                    "artifacts": {"submission_sha": "abc"},
                    "reruns": [{"run": 1}],
                },
            }
        )
    )

    updated = update_metrics(
        metrics_path,
        {
            "status": "debug_completed",
            "evidence": {"artifacts": {"model_count": 5}},
        },
    )

    assert updated["public_lb"] == 0.12
    assert updated["evidence"]["artifacts"] == {
        "submission_sha": "abc",
        "model_count": 5,
    }
    assert updated["evidence"]["reruns"] == [{"run": 1}]
    assert json.loads(metrics_path.read_text()) == updated
    assert not (tmp_path / ".metrics.json.tmp").exists()


def test_automated_run_status_does_not_overwrite_user_decision(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(ROOT / "templates" / "experiment" / "settings.py"))
    update_metrics = namespace["update_metrics"]
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"experiment": "exp123_test", "status": "completed"})
    )

    updated = update_metrics(
        metrics_path,
        {"status": "debug_completed", "debug": True},
    )

    assert updated["status"] == "completed"
    assert updated["debug"] is True


def test_automated_run_status_does_not_erase_leak_risk(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(ROOT / "templates" / "experiment" / "settings.py"))
    update_metrics = namespace["update_metrics"]
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"experiment": "exp123_test", "status": "leak-risk"})
    )

    updated = update_metrics(metrics_path, {"status": "scaffold_completed"})

    assert updated["status"] == "leak-risk"


def test_kaggle_runtime_paths_follow_project_configuration(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(ROOT / "templates" / "experiment" / "settings.py"))
    input_root = tmp_path / "input"
    working_root = tmp_path / "working"
    competition_root = input_root / "custom-competition"
    (competition_root / "training_files").mkdir(parents=True)
    (competition_root / "evaluation_files").mkdir()
    (competition_root / "submission_layout.csv").write_text("id,target\n")
    working_root.mkdir()
    project_config = {
        "competition": {"slug": "custom-competition"},
        "data": {
            "raw_dir": "competition_data",
            "train_dir": "competition_data/training_files",
            "test_dir": "competition_data/evaluation_files",
        },
        "submission": {"sample_file": "competition_data/submission_layout.csv"},
    }

    settings_globals = namespace["kaggle_competition_input_dir"].__globals__
    settings_globals["KAGGLE_INPUT_ROOT"] = input_root
    settings_globals["KAGGLE_WORKING_ROOT"] = working_root
    settings_globals["load_project_config"] = lambda: project_config

    paths = namespace["ExperimentPaths"]()

    assert paths.train_data_dir == competition_root / "training_files"
    assert paths.test_data_dir == competition_root / "evaluation_files"
    assert paths.sample_submission_path == competition_root / "submission_layout.csv"


def test_train_notebook_uses_shared_metrics_update_without_copying_schema() -> None:
    notebook_path = (
        ROOT
        / "templates"
        / "experiment"
        / "{{EXPERIMENT_NAME}}_train.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    source = "".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )

    assert "update_metrics(" in source
    assert "metrics_path.write_text" not in source
    assert '"submission_validation"' not in source
