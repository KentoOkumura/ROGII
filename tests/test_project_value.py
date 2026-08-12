from __future__ import annotations

import pytest

from scripts.project_value import configured_scalar


def test_configured_scalar_reads_dotted_project_value() -> None:
    config = {"competition": {"slug": "sample-competition"}}

    assert configured_scalar(config, "competition.slug") == "sample-competition"


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"competition": {"slug": "TODO"}},
        {"competition": {"slug": ["not", "scalar"]}},
    ],
)
def test_configured_scalar_rejects_missing_todo_or_non_scalar(config: dict) -> None:
    with pytest.raises(ValueError):
        configured_scalar(config, "competition.slug")
