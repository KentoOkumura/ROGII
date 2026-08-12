from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_project  # noqa: E402
from config_utils import load_project_config  # noqa: E402


def test_strict_validation_rejects_todo_sample_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = copy.deepcopy(load_project_config())
    config["submission"]["sample_file"] = "TODO"
    monkeypatch.setattr(validate_project, "load_project_config", lambda: config)
    monkeypatch.setattr(sys, "argv", ["validate_project.py", "--strict"])

    with pytest.raises(SystemExit):
        validate_project.main()

    assert "strict value still TODO: submission.sample_file" in capsys.readouterr().out


@pytest.mark.parametrize(
    "target_columns",
    [[], "tvt", ["TODO"], [""], [1]],
)
def test_strict_validation_rejects_invalid_target_columns(
    target_columns: object,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = copy.deepcopy(load_project_config())
    config["submission"]["target_columns"] = target_columns
    monkeypatch.setattr(validate_project, "load_project_config", lambda: config)
    monkeypatch.setattr(sys, "argv", ["validate_project.py", "--strict"])

    with pytest.raises(SystemExit):
        validate_project.main()

    assert "submission.target_columns" in capsys.readouterr().out
