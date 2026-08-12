from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PARENT_SOURCE = (
    ROOT
    / "experiments/exp413_scale5_likpf_full_replacement_on_exp335"
    / "exp413_scale5_likpf_full_replacement_on_exp335_current_test_inference.py"
)
OUTPUT = (
    ROOT
    / "experiments/exp510_exp413_exact_public_preoverride_hedge"
    / "exp510_exp413_hidden_safe_runtime.py"
)
EXPECTED_PARENT_SHA256 = (
    "0f6fc81e56556aa6db828584ab2a2e58dde9db9cc4b54d6c12fa60e1c68f1388"
)


def replace_exact(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(
            f"expected exactly one transform anchor, got {text.count(old)}: {old!r}"
        )
    return text.replace(old, new)


def build_runtime_source() -> str:
    parent_bytes = PARENT_SOURCE.read_bytes()
    parent_sha = hashlib.sha256(parent_bytes).hexdigest()
    if parent_sha != EXPECTED_PARENT_SHA256:
        raise RuntimeError(
            f"exp413 parent source SHA mismatch: {parent_sha} != {EXPECTED_PARENT_SHA256}"
        )

    parent = parent_bytes.decode()
    body = parent[parent.index("import gc\n") :]
    body = replace_exact(
        body,
        "from settings import EXPERIMENT_NAME, ExperimentPaths, load_config",
        "from exp413_runtime.settings import EXPERIMENT_NAME, ExperimentPaths, load_config",
    )
    body = replace_exact(
        body,
        "PACKAGE_DIR = Path.cwd()\n"
        "if not (PACKAGE_DIR / \"config.yaml\").exists():\n"
        "    PACKAGE_DIR = Path(\"experiments/exp413_scale5_likpf_full_replacement_on_exp335\")",
        "PACKAGE_DIR = Path(\"exp413_runtime\")\n"
        "if not (PACKAGE_DIR / \"config.yaml\").exists():\n"
        "    raise FileNotFoundError(\"embedded exp413 runtime config is missing\")",
    )
    body = replace_exact(
        body,
        "config = load_config()\nparent_config = yaml.safe_load(",
        "config = load_config()\n"
        "# exp510 execution authorization activates the already-approved exp413 v4\n"
        "# inference path in memory. The vendored parent config remains immutable.\n"
        "config[\"inference\"][\"run_enabled\"] = True\n"
        "parent_config = yaml.safe_load(",
    )

    header = f'''\
from __future__ import annotations

"""Generated hidden-safe exp413 runtime used by exp510.

Source: {PARENT_SOURCE.relative_to(ROOT)}
Source SHA256: {EXPECTED_PARENT_SHA256}

Do not edit this generated file directly. Regenerate it with
``uv run python experiments/exp510_exp413_exact_public_preoverride_hedge/prepare_exp510_hidden_safe_runtime.py``.
"""

PARENT_SOURCE_SHA256 = "{EXPECTED_PARENT_SHA256}"


def generate_dynamic_exp413_prediction():
'''
    footer = '''
    return predictions.copy(), dict(metrics), Path(prediction_path)
'''
    return header + textwrap.indent(body.rstrip() + "\n", "    ") + footer


def main() -> None:
    generated = build_runtime_source()
    OUTPUT.write_text(generated)
    print(
        f"generated {OUTPUT.relative_to(ROOT)} "
        f"({len(generated.splitlines())} lines, "
        f"sha256={hashlib.sha256(generated.encode()).hexdigest()})"
    )


if __name__ == "__main__":
    main()
