from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from collections.abc import Iterable
from pathlib import Path

import pytest


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def require_saved_files(*paths: Path | str) -> None:
    """Skip a test when its explicitly declared untracked execution evidence is absent."""

    missing = [Path(path) for path in paths if not Path(path).is_file()]
    if missing:
        pytest.skip(
            "requires saved execution evidence that is not tracked by Git: "
            + ", ".join(_display_path(path) for path in missing)
        )


def require_one_saved_file(paths: Iterable[Path | str]) -> Path:
    """Return the first available evidence path or skip when none is present."""

    candidates = [Path(path) for path in paths]
    for path in candidates:
        if path.is_file():
            return path
    pytest.skip(
        "requires one of the saved execution evidence files not tracked by Git: "
        + ", ".join(_display_path(path) for path in candidates)
    )


def ensure_numba_test_stub() -> types.ModuleType:
    """Provide the small Numba API used by pure-Python contract tests when absent."""

    module = sys.modules.get("numba")
    if module is None and importlib.util.find_spec("numba") is not None:
        import numba

        return numba
    if module is None:
        module = types.ModuleType("numba")
        module.__spec__ = importlib.machinery.ModuleSpec("numba", loader=None)
        sys.modules["numba"] = module

    def identity_njit(*args, **kwargs):
        del kwargs
        if args and callable(args[0]):
            return args[0]

        def decorator(function):
            return function

        return decorator

    if not hasattr(module, "njit"):
        module.njit = identity_njit
    if not hasattr(module, "prange"):
        module.prange = range
    if not hasattr(module, "get_num_threads"):
        module.get_num_threads = lambda: 1
    if not hasattr(module, "set_num_threads"):
        module.set_num_threads = lambda _threads: None
    if not hasattr(module, "__version__"):
        module.__version__ = "test-stub"
    return module
