from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_project  # noqa: E402
from config_utils import load_project_config  # noqa: E402


def set_nested(config: dict[str, object], key_path: str, value: object) -> None:
    current = config
    parts = key_path.split(".")
    for part in parts[:-1]:
        current = current[part]  # type: ignore[assignment]
    current[parts[-1]] = value


def test_all_required_scalar_project_fields_are_strict() -> None:
    assert set(validate_project.STRICT_KEYS) == set(validate_project.REQUIRED_SCHEMA_KEYS) - {
        "submission.target_columns"
    }


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


@pytest.mark.parametrize(
    "key_path",
    [
        "competition.name",
        "competition.is_code_competition",
        "data.target_column",
        "data.group_column",
        "data.score_rows",
        "defaults.seed",
        "defaults.n_folds",
        "submission.output_file",
        "submission.allow_extra_columns",
        "metadata.owner",
        "metadata.notes",
        "runtime.kaggle.enable_gpu",
        "runtime.kaggle.enable_internet",
        "runtime.kaggle.time_limit_hours",
    ],
)
def test_strict_validation_rejects_todo_in_all_competition_specific_fields(
    key_path: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = copy.deepcopy(load_project_config())
    current: dict[str, object] = config
    parts = key_path.split(".")
    for part in parts[:-1]:
        current = current[part]  # type: ignore[assignment]
    current[parts[-1]] = "TODO"
    monkeypatch.setattr(validate_project, "load_project_config", lambda: config)
    monkeypatch.setattr(sys, "argv", ["validate_project.py", "--strict"])

    with pytest.raises(SystemExit):
        validate_project.main()

    assert f"strict value still TODO: {key_path}" in capsys.readouterr().out


def test_validation_rejects_unintended_competition_slug(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = copy.deepcopy(load_project_config())
    monkeypatch.setattr(validate_project, "load_project_config", lambda: config)
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_project.py", "--strict", "--expected-competition", "other-competition"],
    )

    with pytest.raises(SystemExit):
        validate_project.main()

    assert "competition.slug does not match --expected-competition" in capsys.readouterr().out


def test_validation_rejects_url_for_another_competition(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = copy.deepcopy(load_project_config())
    config["competition"]["url"] = "https://www.kaggle.com/competitions/other-competition"
    monkeypatch.setattr(validate_project, "load_project_config", lambda: config)
    monkeypatch.setattr(sys, "argv", ["validate_project.py", "--strict"])

    with pytest.raises(SystemExit):
        validate_project.main()

    assert "competition.url does not match competition.slug" in capsys.readouterr().out


def test_strict_validation_requires_competition_inputs_under_raw_dir(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = copy.deepcopy(load_project_config())
    config["data"]["train_dir"] = "data/external/train"
    monkeypatch.setattr(validate_project, "load_project_config", lambda: config)
    monkeypatch.setattr(sys, "argv", ["validate_project.py", "--strict"])

    with pytest.raises(SystemExit):
        validate_project.main()

    assert "data.train_dir must be inside data.raw_dir" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("key_path", "value", "expected_error"),
    [
        ("competition.name", 123, "competition.name must be a string"),
        (
            "competition.is_code_competition",
            "false",
            "competition.is_code_competition must be true or false",
        ),
        ("submission.allow_extra_columns", "false", "must be true or false"),
        ("runtime.kaggle.enable_gpu", "false", "must be true or false"),
        ("runtime.kaggle.enable_internet", 0, "must be true or false"),
        ("defaults.seed", True, "defaults.seed must be an integer"),
        ("defaults.seed", -1, "defaults.seed must be at least 0"),
        ("defaults.n_folds", "5", "defaults.n_folds must be an integer"),
        ("defaults.n_folds", 1, "defaults.n_folds must be at least 2"),
        (
            "runtime.kaggle.time_limit_hours",
            0,
            "runtime.kaggle.time_limit_hours must be a positive number",
        ),
        (
            "runtime.kaggle.time_limit_hours",
            float("inf"),
            "runtime.kaggle.time_limit_hours must be a positive number",
        ),
    ],
)
def test_strict_validation_rejects_invalid_scalar_types_and_ranges(
    key_path: str,
    value: object,
    expected_error: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = copy.deepcopy(load_project_config())
    set_nested(config, key_path, value)
    monkeypatch.setattr(validate_project, "load_project_config", lambda: config)
    monkeypatch.setattr(sys, "argv", ["validate_project.py", "--strict"])

    with pytest.raises(SystemExit):
        validate_project.main()

    assert expected_error in capsys.readouterr().out


@pytest.mark.parametrize("key_path", ["data.train_dir", "data.test_dir"])
def test_strict_validation_requires_configured_competition_inputs_to_exist(
    key_path: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = copy.deepcopy(load_project_config())
    value = f"data/raw/missing_{key_path.rsplit('.', 1)[-1]}"
    set_nested(config, key_path, value)
    monkeypatch.setattr(validate_project, "load_project_config", lambda: config)
    monkeypatch.setattr(sys, "argv", ["validate_project.py", "--strict"])

    with pytest.raises(SystemExit):
        validate_project.main()

    assert f"configured path does not exist: {key_path}={value}" in capsys.readouterr().out


def test_strict_validation_rejects_tpu_enablement(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = copy.deepcopy(load_project_config())
    config["runtime"]["kaggle"]["enable_tpu"] = True
    monkeypatch.setattr(validate_project, "load_project_config", lambda: config)
    monkeypatch.setattr(sys, "argv", ["validate_project.py", "--strict"])

    with pytest.raises(SystemExit):
        validate_project.main()

    assert "runtime.kaggle.enable_tpu=true is unsupported" in capsys.readouterr().out
