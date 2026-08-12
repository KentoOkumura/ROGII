from __future__ import annotations

import json

import pytest

from scripts.validate_kaggle_metadata import validate_package


def test_validate_package_accepts_push_ready_metadata(tmp_path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "notebook.ipynb").write_text("{}\n")
    (package / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "owner/canonical-notebook",
                "title": "canonical notebook",
                "code_file": "notebook.ipynb",
                "competition_sources": ["competition"],
                "enable_internet": False,
            }
        )
    )

    metadata = validate_package(package)

    assert metadata["id"] == "owner/canonical-notebook"


def test_validate_package_rejects_missing_notebook(tmp_path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "owner/canonical-notebook",
                "title": "canonical notebook",
                "code_file": "missing.ipynb",
                "competition_sources": ["competition"],
                "enable_internet": False,
            }
        )
    )

    with pytest.raises(ValueError, match="code_file does not exist"):
        validate_package(package)


def test_validate_package_rejects_internet_setting_that_differs_from_project(
    tmp_path,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "notebook.ipynb").write_text("{}\n")
    (package / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "owner/canonical-notebook",
                "title": "canonical notebook",
                "code_file": "notebook.ipynb",
                "competition_sources": ["competition"],
                "enable_internet": True,
            }
        )
    )

    with pytest.raises(ValueError, match="enable_internet does not match project.yml"):
        validate_package(package)
