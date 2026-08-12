from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts import new_experiment


def write_record_templates(root: Path) -> None:
    template = root / "templates" / "experiment"
    template.mkdir(parents=True)
    (template / "README.md").write_text("# {{ EXPERIMENT_NAME }}\n\n## 概要\n")
    (template / "SESSION_NOTES.md").write_text(
        "# {{ EXPERIMENT_NAME }} セッションノート\n\n## 現在の作業\n\n- 次: TODO\n"
    )
    (template / "result.md").write_text(
        "# {{ EXPERIMENT_NAME }} 結果\n\n## 記録の参照先\n\n- 設定: config.yaml\n"
    )
    (template / "metrics.json").write_text(
        '{"experiment": "{{ EXPERIMENT_NAME }}", "status": "planned"}\n'
    )


def test_source_experiment_tests_are_excluded_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(new_experiment, "ROOT", tmp_path)
    source = tmp_path / "experiments" / "exp001_parent"
    (source / "tests").mkdir(parents=True)
    (source / "tests" / "test_parent.py").write_text("def test_parent(): pass\n")
    (source / "config.yaml").write_text("experiment: exp001_parent\n")

    destination = tmp_path / "experiments" / "exp002_child"
    new_experiment.copy_tree(source, destination, force=False, copy_tests=False)

    assert (destination / "config.yaml").exists()
    assert not (destination / "tests").exists()
    for dirname in new_experiment.GENERATED_DIRS:
        generated_dir = destination / dirname
        assert generated_dir.is_dir()
        assert (generated_dir / ".gitkeep").is_file()


def test_source_experiment_tests_can_be_copied_explicitly(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(new_experiment, "ROOT", tmp_path)
    source = tmp_path / "experiments" / "exp001_parent"
    (source / "tests").mkdir(parents=True)
    (source / "tests" / "test_parent.py").write_text("def test_parent(): pass\n")

    destination = tmp_path / "experiments" / "exp002_child"
    new_experiment.copy_tree(source, destination, force=False, copy_tests=True)

    assert (destination / "tests" / "test_parent.py").exists()


def test_parent_copy_replaces_identity_and_resets_execution_records(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(new_experiment, "ROOT", tmp_path)
    write_record_templates(tmp_path)
    source = tmp_path / "experiments" / "exp001_parent"
    source.mkdir(parents=True)
    (source / "README.md").write_text("# exp001_parent\n\nPublic LB: 1.23\n")
    (source / "SESSION_NOTES.md").write_text("# exp001_parent\ncompleted\n")
    (source / "result.md").write_text("# exp001_parent result\ncompleted\n")
    (source / "metrics.json").write_text(
        '{"experiment": "exp001_parent", "status": "completed", "public_lb": 1.23}\n'
    )
    (source / "config.yaml").write_text(
        "experiment:\n"
        "  name: exp001_parent\n"
        "  description: completed parent\n"
        "  route: pf_beam\n"
        "  status: completed\n"
        "lineage:\n"
        "  parent: exp000_base\n"
        "  hypothesis_id: HYP-19000101-93\n"
        "  backlog_candidate: parent_candidate\n"
        "  diff_summary: parent diff\n"
        "model:\n"
        "  name: retained_model\n"
    )
    (source / "settings.py").write_text('EXPERIMENT_NAME = "exp001_parent"\n')
    (source / "exp001_parent_train.ipynb").write_text(
        '{"cells": [{"source": ["# exp001_parent train"]}]}\n'
    )

    destination = tmp_path / "experiments" / "exp002_child"
    new_experiment.copy_tree(source, destination, force=False, copy_tests=False)
    new_experiment.replace_parent_experiment_identity(destination, "exp001_parent", "exp002_child")
    new_experiment.reset_parent_records(destination, "exp002_child", "exp001_parent")

    assert not (destination / "exp001_parent_train.ipynb").exists()
    assert (destination / "exp002_child_train.ipynb").exists()
    assert "exp002_child" in (destination / "settings.py").read_text()
    assert "Public LB" not in (destination / "README.md").read_text()
    assert "exp001_parent" not in (destination / "README.md").read_text()
    assert "exp001_parent" not in (destination / "result.md").read_text()
    metrics = json.loads((destination / "metrics.json").read_text())
    assert metrics == {"experiment": "exp002_child", "status": "planned"}
    config = yaml.safe_load((destination / "config.yaml").read_text())
    assert config["experiment"]["name"] == "exp002_child"
    assert config["experiment"]["description"] == "TODO"
    assert config["experiment"]["route"] == "pf_beam"
    assert "status" not in config["experiment"]
    assert config["lineage"]["parent"] == "exp001_parent"
    assert config["lineage"]["hypothesis_id"] == "TODO"
    assert config["lineage"]["backlog_candidate"] == "TODO"
    assert config["lineage"]["diff_summary"] == "TODO"
    assert config["model"]["name"] == "retained_model"
    assert "ルート:" not in (destination / "README.md").read_text()
    assert "状態:" not in (destination / "README.md").read_text()
    assert "Route:" not in (destination / "SESSION_NOTES.md").read_text()
    assert "状態:" not in (destination / "SESSION_NOTES.md").read_text()
