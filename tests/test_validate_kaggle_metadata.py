from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import scripts.validate_kaggle_metadata as metadata_validator
from scripts.prepare_kaggle_notebooks import ensure_notebook_kernel_metadata, make_support_cell
from scripts.validate_kaggle_metadata import validate_package

PROJECT_RUNTIME = "runtime:\n  kaggle:\n    enable_gpu: false\n    enable_internet: false\n"


def _write_metadata(
    package: Path,
    *,
    notebook_name: str = "notebook.ipynb",
    **overrides: object,
) -> None:
    metadata: dict[str, object] = {
        "id": "owner/canonical-notebook",
        "title": "canonical notebook",
        "code_file": notebook_name,
        "competition_sources": ["competition"],
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
    }
    metadata.update(overrides)
    (package / "kernel-metadata.json").write_text(json.dumps(metadata))


def _write_generated_notebook(package: Path, *, notebook_name: str = "notebook.ipynb") -> None:
    project_path = package / "project.yml"
    if not project_path.is_file():
        project_path.write_text(PROJECT_RUNTIME)
    support_files = {"project.yml": project_path.read_bytes()}
    config_path = package / "config.yaml"
    if config_path.is_file():
        support_files["config.yaml"] = config_path.read_bytes()
    notebook = {
        "cells": [make_support_cell(support_files)],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (package / notebook_name).write_text(json.dumps(notebook))


def test_validate_package_accepts_push_ready_metadata(tmp_path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    _write_generated_notebook(package)
    _write_metadata(package)

    metadata = validate_package(package)

    assert metadata["id"] == "owner/canonical-notebook"


def test_validate_package_rejects_missing_notebook(tmp_path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "project.yml").write_text(PROJECT_RUNTIME)
    _write_metadata(package, notebook_name="missing.ipynb")

    with pytest.raises(ValueError, match="code_file does not exist"):
        validate_package(package)


def test_validate_package_rejects_missing_bootstrap(tmp_path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "project.yml").write_text(PROJECT_RUNTIME)
    (package / "notebook.ipynb").write_text(
        json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5})
    )
    _write_metadata(package)

    with pytest.raises(ValueError, match="generated notebook bootstrap is missing"):
        validate_package(package)


def test_validate_package_rejects_internet_setting_that_differs_from_project(
    tmp_path,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    _write_generated_notebook(package)
    _write_metadata(package, enable_internet=True)

    with pytest.raises(ValueError, match="enable_internet does not match effective runtime"):
        validate_package(package)


def test_validate_package_uses_packaged_notebook_runtime_override(tmp_path) -> None:
    package = tmp_path / "audit"
    package.mkdir()
    (package / "project.yml").write_text(
        "runtime:\n  kaggle:\n    enable_gpu: false\n    enable_internet: false\n"
    )
    (package / "config.yaml").write_text(
        "runtime:\n"
        "  kaggle:\n"
        "    audit:\n"
        "      enable_gpu: true\n"
        "      machine_shape: NvidiaTeslaT4\n"
    )
    _write_generated_notebook(package)
    _write_metadata(
        package,
        id="owner/canonical-audit",
        title="canonical audit",
        enable_gpu=True,
        machine_shape="NvidiaTeslaT4",
    )

    metadata = validate_package(package)

    assert metadata["machine_shape"] == "NvidiaTeslaT4"


@pytest.mark.parametrize(
    ("metadata_override", "expected_error"),
    [
        ({"enable_gpu": False}, "enable_gpu does not match effective runtime"),
        ({"machine_shape": "NvidiaTeslaP100"}, "machine_shape does not match effective runtime"),
    ],
)
def test_validate_package_rejects_metadata_that_differs_from_notebook_runtime(
    tmp_path,
    metadata_override: dict[str, object],
    expected_error: str,
) -> None:
    package = tmp_path / "audit"
    package.mkdir()
    (package / "project.yml").write_text(
        "runtime:\n  kaggle:\n    enable_gpu: false\n    enable_internet: false\n"
    )
    (package / "config.yaml").write_text(
        "runtime:\n"
        "  kaggle:\n"
        "    audit:\n"
        "      enable_gpu: true\n"
        "      machine_shape: NvidiaTeslaT4\n"
    )
    _write_generated_notebook(package)
    metadata = {
        "id": "owner/canonical-audit",
        "title": "canonical audit",
        "code_file": "notebook.ipynb",
        "competition_sources": ["competition"],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4",
    }
    metadata.update(metadata_override)
    (package / "kernel-metadata.json").write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match=expected_error):
        validate_package(package)


def test_validate_repository_package_allows_configured_bootstrap_exclusions(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "repo"
    experiment_dir = root / "experiments" / "exp123_test"
    package = experiment_dir / "kaggle" / "audit"
    package.mkdir(parents=True)
    (root / "project.yml").write_text(PROJECT_RUNTIME)
    config = (
        "experiment:\n"
        "  id: exp123_test\n"
        "runtime:\n"
        "  kaggle:\n"
        "    audit:\n"
        "      include_experiment_sources: false\n"
    )
    (experiment_dir / "config.yaml").write_text(config)
    source_notebook = {
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# Audit\n"]}],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    source_notebook_path = experiment_dir / "exp123_test_audit.ipynb"
    source_notebook_path.write_text(json.dumps(source_notebook))
    (package / "project.yml").write_bytes((root / "project.yml").read_bytes())
    (package / "config.yaml").write_text(config)
    generated_notebook = copy.deepcopy(source_notebook)
    ensure_notebook_kernel_metadata(generated_notebook)
    generated_notebook["cells"].insert(
        0,
        make_support_cell({"project.yml": (root / "project.yml").read_bytes()}),
    )
    (package / "exp123_test_audit.ipynb").write_text(json.dumps(generated_notebook))
    _write_metadata(package, notebook_name="exp123_test_audit.ipynb")
    monkeypatch.setattr(metadata_validator, "ROOT", root)

    validate_package(package)


def test_validate_package_rejects_support_file_that_differs_from_bootstrap(tmp_path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    _write_generated_notebook(package)
    _write_metadata(package)
    (package / "project.yml").write_text(PROJECT_RUNTIME + "metadata:\n  notes: changed\n")

    with pytest.raises(ValueError, match="bootstrap support file does not match"):
        validate_package(package)


@pytest.mark.parametrize(
    "stale_source",
    ["config", "notebook", "notebook_metadata", "removed_support", "src"],
)
def test_validate_repository_package_rejects_current_source_changes(
    tmp_path,
    monkeypatch,
    stale_source: str,
) -> None:
    root = tmp_path / "repo"
    experiment_dir = root / "experiments" / "exp123_test"
    package = experiment_dir / "kaggle" / "audit"
    package.mkdir(parents=True)
    (root / "project.yml").write_text(PROJECT_RUNTIME)
    (root / "src").mkdir()
    (root / "src" / "feature.py").write_text("VALUE = 1\n")
    (experiment_dir / "config.yaml").write_text("experiment:\n  id: exp123_test\n")
    (experiment_dir / "helper.py").write_text("VALUE = 2\n")
    source_notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Audit\n"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    source_notebook_path = experiment_dir / "exp123_test_audit.ipynb"
    source_notebook_path.write_text(json.dumps(source_notebook))
    (package / "project.yml").write_bytes((root / "project.yml").read_bytes())
    (package / "config.yaml").write_bytes((experiment_dir / "config.yaml").read_bytes())
    (package / "helper.py").write_bytes((experiment_dir / "helper.py").read_bytes())
    (package / "src").mkdir()
    (package / "src" / "feature.py").write_bytes((root / "src" / "feature.py").read_bytes())
    support_files = {
        "project.yml": (root / "project.yml").read_bytes(),
        "config.yaml": (experiment_dir / "config.yaml").read_bytes(),
        "helper.py": (experiment_dir / "helper.py").read_bytes(),
        "src/feature.py": (root / "src" / "feature.py").read_bytes(),
    }
    generated_notebook = copy.deepcopy(source_notebook)
    ensure_notebook_kernel_metadata(generated_notebook)
    generated_notebook["cells"].insert(0, make_support_cell(support_files))
    (package / "exp123_test_audit.ipynb").write_text(json.dumps(generated_notebook))
    _write_metadata(package, notebook_name="exp123_test_audit.ipynb")
    monkeypatch.setattr(metadata_validator, "ROOT", root)

    validate_package(package)
    if stale_source == "config":
        (experiment_dir / "config.yaml").write_text(
            "experiment:\n  id: exp123_test\nmetadata:\n  notes: changed\n"
        )
        expected_error = "config.yaml differs"
    elif stale_source == "notebook":
        source_notebook["cells"][0]["source"] = ["# Changed audit\n"]
        source_notebook_path.write_text(json.dumps(source_notebook))
        expected_error = "generated notebook differs from source notebook"
    elif stale_source == "notebook_metadata":
        source_notebook["metadata"]["custom"] = {"changed": True}
        source_notebook_path.write_text(json.dumps(source_notebook))
        expected_error = "generated notebook differs from source notebook"
    elif stale_source == "removed_support":
        (experiment_dir / "helper.py").unlink()
        expected_error = "removed experiment source remains: helper.py"
    else:
        (root / "src" / "new_feature.py").write_text("VALUE = 3\n")
        expected_error = "missing src/new_feature.py"

    with pytest.raises(ValueError, match=expected_error):
        validate_package(package)
