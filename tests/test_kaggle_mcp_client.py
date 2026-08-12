import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / ".agents/skills/kaggle-platform/shared/mcp_client.py"
MODULE_SPEC = importlib.util.spec_from_file_location("kaggle_mcp_client", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
mcp_client = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(mcp_client)


class FakeResponse:
    def __init__(self, body: str) -> None:
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self) -> bytes:
        return self.body.encode("utf-8")


def test_mcp_call_sends_token_in_process_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int) -> FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse('data: {"jsonrpc":"2.0","id":1,"result":{"content":[]}}')

    monkeypatch.setattr(mcp_client.urllib.request, "urlopen", fake_urlopen)

    response = mcp_client.mcp_call(
        "search_datasets",
        {"search": "titanic"},
        token="test-secret-token",
        timeout=7,
        endpoint="https://example.invalid/mcp",
    )

    request = captured["request"]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.get_header("Authorization") == "Bearer test-secret-token"
    assert captured["timeout"] == 7
    assert payload["method"] == "tools/call"
    assert payload["params"]["name"] == "search_datasets"
    assert response["result"] == {"content": []}


def test_mcp_list_tools_returns_structured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_timeout(*args: Any, **kwargs: Any) -> None:
        raise TimeoutError

    monkeypatch.setattr(mcp_client.urllib.request, "urlopen", raise_timeout)

    assert mcp_client.mcp_list_tools(token="test-secret-token") == {
        "error": {"message": "timeout"}
    }


def test_resolve_token_accepts_api_token_without_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KAGGLE_MCP_TOKEN", raising=False)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "generated-token-without-assumed-prefix")

    assert mcp_client.resolve_token() == "generated-token-without-assumed-prefix"


def test_resolve_token_accepts_access_token_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setattr(mcp_client, "get_access_token", lambda: "stored-api-token")

    assert mcp_client.resolve_token() == "stored-api-token"


def test_resolve_token_rejects_legacy_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setenv("KAGGLE_MCP_TOKEN", "unsupported-alias")
    monkeypatch.setenv("KAGGLE_KEY", "legacy-key")
    monkeypatch.setattr(mcp_client, "get_access_token", lambda: "")

    assert mcp_client.resolve_token() == ""


MCP_ENTRYPOINTS = (
    ".agents/skills/kaggle-platform/modules/kllm/scripts/list_competition_pages.py",
    ".agents/skills/kaggle-platform/modules/kllm/hackathon/scripts/hackathon_overview.py",
    ".agents/skills/kaggle-platform/modules/kllm/hackathon/scripts/list_writeups.py",
    ".agents/skills/kaggle-platform/modules/kllm/hackathon/scripts/fetch_writeup.py",
)


@pytest.mark.parametrize("relative_path", MCP_ENTRYPOINTS)
def test_mcp_entrypoint_help_starts_without_import_errors(relative_path: str) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / relative_path), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
