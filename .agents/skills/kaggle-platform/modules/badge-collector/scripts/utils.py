"""Shared utilities for the badge collector."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Rate limiting: seconds between API calls
API_DELAY = 5

# Prefix for all created resources
RESOURCE_PREFIX = "badge-collector-"

# Skill root: 3 levels up from modules/badge-collector/scripts/utils.py
SKILL_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = SKILL_ROOT.parents[2]
SHARED_SCRIPTS = SKILL_ROOT / "shared"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATE_DIR = REPO_ROOT / ".badge-collector"


def get_username() -> str:
    """Get the explicitly configured Kaggle username."""
    username = os.getenv("KAGGLE_USERNAME")
    if username:
        return username
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            creds = json.loads(kaggle_json.read_text())
            return creds.get("username", "")
        except (json.JSONDecodeError, OSError):
            return ""
    return ""


def get_kaggle_cli() -> tuple[str, ...]:
    """Return the repository-locked Kaggle CLI command."""
    return ("uv", "run", "--project", str(REPO_ROOT), "kaggle")


def run_kaggle_cli(
    args: list[str],
    check: bool = True,
    timeout: int = 120,
    stream_output: bool = False,
) -> subprocess.CompletedProcess:
    """Run a kaggle CLI command with rate limiting."""
    cmd = [*get_kaggle_cli(), *args]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=not stream_output,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "see streamed output above"
        print(f"  [STDERR] {stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    time.sleep(API_DELAY)
    return result


def make_temp_dir(suffix: str = "") -> Path:
    """Create a machine-temporary directory outside the tracked skill tree."""
    return Path(tempfile.mkdtemp(prefix=RESOURCE_PREFIX, suffix=suffix))


def credential_requirement_for_phases(phases: list[int]) -> str | None:
    """Return the credential capability required by selected phases."""
    selected = set(phases)
    if 1 in selected:
        return "python-api"
    if selected & {2, 3, 5}:
        return "cli"
    return None


def check_credentials(phases: list[int]) -> bool:
    """Verify credentials against the clients used by selected phases."""
    requirement = credential_requirement_for_phases(phases)
    if requirement is None:
        return True

    script = SHARED_SCRIPTS / "check_all_credentials.py"
    if script.exists():
        result = subprocess.run(
            [sys.executable, str(script), "--require", requirement],
            capture_output=True,
            text=True,
        )
        print(result.stdout.strip())
        return result.returncode == 0

    # Fallback mirrors the canonical source types if the shared checker is absent.
    access_token = Path.home() / ".kaggle" / "access_token"
    has_api_token = bool(os.getenv("KAGGLE_API_TOKEN")) or access_token.exists()
    if requirement == "api-token":
        return has_api_token

    oauth_credentials = Path.home() / ".kaggle" / "credentials.json"
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_legacy = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    if kaggle_json.exists():
        try:
            legacy_file = json.loads(kaggle_json.read_text())
            has_legacy = has_legacy or bool(legacy_file.get("username") and legacy_file.get("key"))
        except (json.JSONDecodeError, OSError):
            pass
    if requirement == "python-api":
        return has_api_token or has_legacy
    return has_api_token or oauth_credentials.exists() or has_legacy


def resource_name(kind: str, suffix: str = "") -> str:
    """Generate a unique resource name with prefix."""
    ts = int(time.time())
    name = f"{RESOURCE_PREFIX}{kind}-{ts}"
    if suffix:
        name += f"-{suffix}"
    return name


def slug(name: str) -> str:
    """Convert a name to a Kaggle-compatible slug."""
    return name.lower().replace(" ", "-").replace("_", "-")
