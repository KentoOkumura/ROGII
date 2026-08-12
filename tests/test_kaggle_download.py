from __future__ import annotations

import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import kaggle_download  # noqa: E402


def _write_archive(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_safe_extract_archive_skips_identical_existing_files(tmp_path: Path) -> None:
    archive_path = tmp_path / "competition.zip"
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    existing = data_dir / "train" / "existing.csv"
    existing.parent.mkdir()
    existing.write_text("keep")
    _write_archive(
        archive_path,
        {
            "sample_submission.csv": "id,target\n1,0\n",
            "train/existing.csv": "keep",
        },
    )

    extracted, skipped = kaggle_download.safe_extract_archive(archive_path, data_dir)

    assert (data_dir / "sample_submission.csv").is_file()
    assert existing.read_text() == "keep"
    assert (extracted, skipped) == (1, 1)


def test_safe_extract_archive_rejects_changed_existing_files(tmp_path: Path) -> None:
    archive_path = tmp_path / "competition.zip"
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    existing = data_dir / "train.csv"
    existing.write_text("stale")
    _write_archive(
        archive_path,
        {
            "new.csv": "new",
            "train.csv": "updated",
        },
    )

    with pytest.raises(ValueError, match="differs from archive member"):
        kaggle_download.safe_extract_archive(archive_path, data_dir)

    assert existing.read_text() == "stale"
    assert not (data_dir / "new.csv").exists()


@pytest.mark.parametrize("member", ["../escape.txt", "/tmp/escape.txt"])
def test_safe_extract_archive_rejects_paths_outside_data_dir(tmp_path: Path, member: str) -> None:
    archive_path = tmp_path / "competition.zip"
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    _write_archive(archive_path, {member: "unsafe"})

    with pytest.raises(ValueError, match="path escapes"):
        kaggle_download.safe_extract_archive(archive_path, data_dir)


def test_safe_extract_archive_rejects_symbolic_links(tmp_path: Path) -> None:
    archive_path = tmp_path / "competition.zip"
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    link = zipfile.ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(link, "target")

    with pytest.raises(ValueError, match="symbolic link"):
        kaggle_download.safe_extract_archive(archive_path, data_dir)


def test_main_downloads_extracts_and_checks_sample_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {
        "competition": {"slug": "sample-competition"},
        "paths": {"data_dir": "data"},
        "data": {"raw_dir": "data/raw"},
        "submission": {"sample_file": "data/raw/sample_submission.csv"},
    }
    commands: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        assert check is True
        commands.append(command)
        archive_path = tmp_path / "data" / "raw" / "sample-competition.zip"
        _write_archive(archive_path, {"sample_submission.csv": "id,target\n1,0\n"})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(kaggle_download, "ROOT", tmp_path)
    monkeypatch.setattr(kaggle_download, "load_project_config", lambda: config)
    monkeypatch.setattr(kaggle_download.subprocess, "run", fake_run)

    kaggle_download.main()

    assert commands == [
        [
            "uv",
            "run",
            "kaggle",
            "competitions",
            "download",
            "-c",
            "sample-competition",
            "-p",
            str(tmp_path / "data" / "raw"),
        ]
    ]
    assert (tmp_path / "data" / "raw" / "sample_submission.csv").is_file()
