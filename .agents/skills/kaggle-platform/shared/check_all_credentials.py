#!/usr/bin/env python3
"""Unified Kaggle credential checker.

Checks all credential sources in priority order:
  1. KAGGLE_API_TOKEN env var
  2. ~/.kaggle/access_token file (recommended for local agents)
  3. ~/.kaggle/credentials.json (OAuth login from `kaggle auth login`)
  4. KAGGLE_USERNAME + KAGGLE_KEY env vars (legacy)
  5. ~/.kaggle/kaggle.json (legacy)

Returns structured JSON output for easy parsing.
Never prints actual credential values — only masked status.

Usage:
    uv run python .agents/skills/kaggle-platform/shared/check_all_credentials.py
    uv run python .agents/skills/kaggle-platform/shared/check_all_credentials.py --json
    uv run python .agents/skills/kaggle-platform/shared/check_all_credentials.py --require api-token
    uv run python \
        .agents/skills/kaggle-platform/shared/check_all_credentials.py --require python-api
    uv run python .agents/skills/kaggle-platform/shared/check_all_credentials.py --require cli

The default ``--require any`` accepts any usable credential source. Use
``api-token`` for MCP and other Bearer-token-only clients, ``python-api`` for
Kaggle Python API or kagglehub operations, and ``cli`` for Kaggle CLI operations.

Exit codes:
    0 — A credential source satisfying the selected requirement was found
    1 — No configured credential source satisfies the selected requirement
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _ensure_mode_600(path: Path) -> None:
    """Auto-tighten file mode to 600 if anything else is set.

    Credential files must never be group- or world-readable. Previously this
    only warned and continued; now it self-heals because credentials in a
    world-readable file are an active leak, not a future risk.
    """
    mode = path.stat().st_mode & 0o777
    if mode != 0o600:
        try:
            path.chmod(0o600)
            print(f"[INFO] Tightened {path} permissions from {oct(mode)[-3:]} to 600")
        except OSError as e:
            print(f"[WARN] {path} permissions are {oct(mode)[-3:]}, could not chmod 600: {e}")


def _read_access_token() -> str:
    """Read ~/.kaggle/access_token if it exists."""
    access_token = Path.home() / ".kaggle" / "access_token"
    if not access_token.exists():
        return ""
    token = access_token.read_text().strip()
    if token:
        _ensure_mode_600(access_token)
    return token


def _read_kaggle_json() -> dict:
    """Read ~/.kaggle/kaggle.json if it exists and is valid."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        return {}
    try:
        creds = json.loads(kaggle_json.read_text())
        _ensure_mode_600(kaggle_json)
        return creds
    except (json.JSONDecodeError, KeyError):
        print(f"[WARN] {kaggle_json} exists but is malformed")
        return {}


def _oauth_credentials_path() -> Path:
    """Return the Kaggle CLI OAuth credentials path."""
    return Path.home() / ".kaggle" / "credentials.json"


def _mask(value: str, prefix_len: int = 0) -> str:
    """Mask a credential value, showing only first prefix_len and last 4 chars."""
    if not value:
        return "****"
    if len(value) <= prefix_len + 4:
        return "****"
    return value[:prefix_len] + "*" * max(0, len(value) - prefix_len - 4) + value[-4:]


def check_all_credentials(output_json: bool = False, requirement: str = "any") -> bool:
    """Check credentials and enforce an optional client-specific requirement."""
    if requirement not in {"any", "api-token", "python-api", "cli"}:
        raise ValueError(f"Unsupported credential requirement: {requirement}")
    results = {}
    found_any = False

    # KAGGLE_TOKEN is not a supported alias: its token type is ambiguous, and
    # treating it as a legacy KAGGLE_KEY can silently select the wrong auth
    # mechanism.
    if os.getenv("KAGGLE_TOKEN") and not os.getenv("KAGGLE_API_TOKEN"):
        print("[WARN] KAGGLE_TOKEN is not used; rename it to KAGGLE_API_TOKEN")

    # --- API Token (primary, recommended) ---
    # An explicit environment value must override a persistent local file so
    # CI, Colab, and managed secret stores can select their own credential.
    access_token_file = _read_access_token()
    api_token_env = os.getenv("KAGGLE_API_TOKEN", "")
    api_token = api_token_env or access_token_file

    if api_token:
        source = "env" if api_token_env else "~/.kaggle/access_token"
        results["KAGGLE_API_TOKEN"] = {
            "status": "OK", "value": _mask(api_token, 5),
            "source": source, "type": "API token",
        }
        print(f"[OK] API Token: {_mask(api_token, 5)} (from {source})")
        found_any = True
    else:
        results["KAGGLE_API_TOKEN"] = {"status": "MISSING", "value": None, "source": None}
        print("[MISSING] API Token")
        print("          Generate at: https://www.kaggle.com/settings")
        print("          → API Tokens (Recommended) → Generate New Token")
        print("          Save as ~/.kaggle/access_token or set KAGGLE_API_TOKEN env var")

    # --- OAuth credentials from `kaggle auth login` ---
    oauth_path = _oauth_credentials_path()
    oauth_found = oauth_path.exists()
    if oauth_found:
        _ensure_mode_600(oauth_path)
        results["KAGGLE_OAUTH_CREDENTIALS"] = {
            "status": "OK", "value": None, "source": str(oauth_path),
        }
        print(f"[OK] OAuth credentials: {oauth_path} (from kaggle auth login)")
        found_any = True
    else:
        results["KAGGLE_OAUTH_CREDENTIALS"] = {
            "status": "MISSING", "value": None, "source": None,
        }
        print(
            "[INFO] OAuth credentials not found "
            "(optional; run `uv run kaggle auth login` for interactive CLI auth)"
        )

    # --- Legacy credentials (optional) ---
    kaggle_json_data = _read_kaggle_json()

    # KAGGLE_USERNAME
    username = os.getenv("KAGGLE_USERNAME") or kaggle_json_data.get("username")
    if username:
        source = "env" if os.getenv("KAGGLE_USERNAME") else "kaggle.json"
        results["KAGGLE_USERNAME"] = {"status": "OK", "value": username, "source": source}
        print(f"[OK] KAGGLE_USERNAME: {username} (from {source})")
    else:
        results["KAGGLE_USERNAME"] = {"status": "MISSING", "value": None, "source": None}
        print("[INFO] KAGGLE_USERNAME not set (optional with API token)")

    # KAGGLE_KEY
    key = os.getenv("KAGGLE_KEY") or kaggle_json_data.get("key")
    if key:
        source = "env" if os.getenv("KAGGLE_KEY") else "kaggle.json"
        results["KAGGLE_KEY"] = {
            "status": "OK", "value": _mask(key),
            "source": source, "type": "Legacy API key",
        }
        print(f"[OK] KAGGLE_KEY: {_mask(key)} (Legacy API key, from {source})")
    else:
        results["KAGGLE_KEY"] = {"status": "MISSING", "value": None, "source": None}
        if not api_token:
            print("[MISSING] KAGGLE_KEY")
            print("          Legacy API key. Generate at: https://www.kaggle.com/settings")
            print("          → Legacy API Credentials → Create Legacy API Key")
        else:
            print("[INFO] KAGGLE_KEY not set (optional when API token is available)")

    if key and not username:
        print("[WARN] Legacy KAGGLE_KEY is unusable without KAGGLE_USERNAME")

    # --- Summary ---
    legacy_pair_found = bool(username and key)
    found_any = bool(api_token) or oauth_found or legacy_pair_found
    print()
    if requirement == "api-token":
        requirement_met = bool(api_token)
    elif requirement == "python-api":
        requirement_met = bool(api_token) or legacy_pair_found
    else:
        requirement_met = found_any

    if found_any:
        if api_token:
            print("API token found — you're ready to go!")
            print("(Supported by kaggle CLI >= 1.8.0, kagglehub >= 0.4.1, MCP Server)")
        elif legacy_pair_found:
            print(
                "Legacy username/key credentials found — "
                "supported by CLI and Python API clients."
            )
            print("MCP operations require an API token from Generate New Token.")
        elif oauth_found:
            print("OAuth credentials found — kaggle CLI can authenticate interactively.")
            print("Python API clients need an API token or a legacy username/key pair.")
    else:
        print("No Kaggle credentials found. To set up:")
        print()
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Under 'API Tokens (Recommended)', click 'Generate New Token'")
        print("  3. Keep the token local; do not paste it into chat or command arguments")
        print()
        print("     # Option 0: Interactive CLI OAuth login")
        print("     uv run kaggle auth login")
        print()
        print("     # Option A: Local hidden-input helper")
        print(
            "     uv run python .agents/skills/kaggle-platform/modules/registration/"
            "scripts/configure_token.py"
        )
        print()
        print(
            "  Full guide: .agents/skills/kaggle-platform/modules/registration/"
            "references/kaggle-setup.md"
        )

    if found_any and not requirement_met:
        print()
        print("Configured credentials do not satisfy this client's requirement.")
        if requirement == "api-token":
            print("This operation requires an API token from Generate New Token.")
            print("OAuth and legacy username/key credentials cannot authenticate MCP.")
        elif requirement == "python-api":
            print("This operation requires an API token or a legacy username/key pair.")
            print("OAuth-only CLI credentials are insufficient for this Python API client.")

    if output_json:
        print()
        print("--- JSON ---")
        print(json.dumps(results, indent=2))

    return requirement_met


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check configured Kaggle credentials")
    parser.add_argument("--json", action="store_true", help="Print structured JSON details")
    parser.add_argument(
        "--require",
        choices=("any", "api-token", "python-api", "cli"),
        default="any",
        help="Require credentials compatible with a specific client type",
    )
    args = parser.parse_args()
    ok = check_all_credentials(output_json=args.json, requirement=args.require)
    sys.exit(0 if ok else 1)
