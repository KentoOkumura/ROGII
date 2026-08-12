"""Shared MCP JSON-RPC client for the Kaggle MCP server.

Used by both the integration test suite and the hackathon module scripts.
Single source of truth for SSE parsing, auth header construction, response
classification, and credential discovery.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MCP_ENDPOINT = "https://www.kaggle.com/mcp"


def get_api_token() -> str:
    """Return the explicitly configured API token without inferring its type."""
    return os.getenv("KAGGLE_API_TOKEN", "")


def get_username() -> str:
    """Return Kaggle username from env or ~/.kaggle/kaggle.json."""
    u = os.getenv("KAGGLE_USERNAME", "")
    if u:
        return u
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            return json.loads(kaggle_json.read_text()).get("username", "")
        except (json.JSONDecodeError, KeyError):
            pass
    return ""


def get_access_token() -> str:
    """Return token from ~/.kaggle/access_token if present."""
    p = Path.home() / ".kaggle" / "access_token"
    if p.exists():
        return p.read_text().strip()
    return ""


def resolve_token() -> str:
    """Return an API token supported by the Kaggle MCP server."""
    return get_api_token() or get_access_token() or ""


def _parse_mcp_response(raw: str) -> dict[str, Any]:
    """Parse either an SSE-framed or a raw JSON-RPC response."""
    for line in raw.splitlines():
        if line.startswith("data:"):
            try:
                return json.loads(line[5:].strip())
            except json.JSONDecodeError:
                pass
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {"raw": raw[:300]}


def _post_json_rpc(
    payload: dict[str, Any],
    token: str,
    timeout: int,
    endpoint: str,
) -> dict[str, Any]:
    """Post JSON-RPC without exposing the bearer token in process arguments."""
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        parsed = _parse_mcp_response(raw)
        if "raw" not in parsed:
            return parsed
        return {"error": {"message": f"HTTP {exc.code}: {raw[:200]}"}}
    except TimeoutError:
        return {"error": {"message": "timeout"}}
    except urllib.error.URLError as exc:
        return {"error": {"message": f"connection failed: {exc.reason}"}}
    return _parse_mcp_response(raw)


def mcp_call(
    tool: str,
    arguments: dict[str, Any],
    token: str,
    timeout: int = 30,
    endpoint: str = MCP_ENDPOINT,
) -> dict[str, Any]:
    """Call an MCP tool via JSON-RPC over HTTP. Returns parsed response.

    Handles both SSE-framed and raw JSON responses. On timeout/parse failure
    returns a structured error rather than raising.
    """
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
        "id": 1,
    }
    return _post_json_rpc(payload, token, timeout, endpoint)


def mcp_list_tools(token: str, timeout: int = 30, endpoint: str = MCP_ENDPOINT) -> dict[str, Any]:
    """Call tools/list and return the parsed response."""
    payload = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
    return _post_json_rpc(payload, token, timeout, endpoint)


def classify_result(resp: dict[str, Any]) -> str:
    """Classify MCP response: ok | empty | unauthenticated | error:<msg> | parse_fail."""
    if "raw" in resp:
        return "parse_fail"
    error = resp.get("error", {})
    if error:
        return f"error: {error.get('message', 'unknown')[:80]}"
    result = resp.get("result", {})
    content = result.get("content", [])
    if isinstance(content, str):
        if "unauthenticated" in content.lower():
            return "unauthenticated"
        return "ok"
    if isinstance(content, list):
        for c in content:
            if not isinstance(c, dict):
                continue
            text = c.get("text", "")
            tl = text.lower()
            if "unauthenticated" in tl:
                return "unauthenticated"
            if tl.startswith("error") or '"error"' in tl or "server error" in tl:
                return f"error: {text[:100]}"
    if result and result != {}:
        return "ok"
    return "empty"


def extract_text(resp: dict[str, Any]) -> str:
    """Pull the first text block out of an MCP response. Returns '' if none."""
    result = resp.get("result", {})
    content = result.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and "text" in c:
                return c["text"]
    return ""


def extract_json(resp: dict[str, Any]) -> dict[str, Any] | list[Any] | None:
    """Pull the first text block and parse it as JSON. Returns None if not JSON."""
    text = extract_text(resp)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
