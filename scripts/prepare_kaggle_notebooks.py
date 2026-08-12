from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import yaml

try:
    from .config_utils import (
        ROOT,
        effective_kaggle_runtime,
        get_nested,
        is_todo,
        kaggle_runtime_errors,
        load_project_config,
        validate_notebook_kind,
    )
except ImportError:  # Direct execution: `uv run python scripts/prepare_kaggle_notebooks.py`
    from config_utils import (
        ROOT,
        effective_kaggle_runtime,
        get_nested,
        is_todo,
        kaggle_runtime_errors,
        load_project_config,
        validate_notebook_kind,
    )

IGNORE_PATTERNS = shutil.ignore_patterns("__pycache__", ".pytest_cache", ".ruff_cache")
NOTEBOOK_KINDS = ("train", "inference")
DEFAULT_KERNELSPEC = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3",
}
DEFAULT_LANGUAGE_INFO = {
    "name": "python",
    "pygments_lexer": "ipython3",
}
MAX_KERNEL_SLUG_LENGTH = 50
_UNSET = object()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare experiment notebooks for Kaggle kernel push."
    )
    parser.add_argument("--experiment", required=True, help="Experiment name, e.g. expXXX_model")
    parser.add_argument(
        "--notebook",
        default="both",
        metavar="KIND",
        help=(
            "Notebook suffix to prepare, or 'both' for train and inference. "
            "Suffixes use lowercase letters, digits, and underscores."
        ),
    )
    parser.add_argument(
        "--kernel-id",
        default=None,
        help="Explicit Kaggle kernel id. Only valid when --notebook is a single notebook kind.",
    )
    parser.add_argument("--train-kernel-id", default=None, help="Kaggle kernel id for train.")
    parser.add_argument(
        "--inference-kernel-id",
        default=None,
        help="Kaggle kernel id for inference.",
    )
    parser.add_argument(
        "--kernel-id-prefix",
        default=None,
        help="Base Kaggle kernel id, e.g. username/exp002. Adds -train/-inference.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Explicit Kaggle title. Only valid with a single notebook kind.",
    )
    parser.add_argument(
        "--title-prefix",
        default=None,
        help="Prefix the default title and derive a matching default kernel slug.",
    )
    parser.add_argument("--competition-slug", default=None, help="Override competition slug.")
    parser.add_argument(
        "--run-on-push",
        action="store_true",
        help="Ask Kaggle to execute the notebook after push.",
    )
    parser.add_argument(
        "--no-src", action="store_true", help="Do not copy the repository src/ package."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail unless generated Kaggle metadata is ready for kernel push.",
    )
    return parser.parse_args()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def ensure_notebook_kernel_metadata(notebook: dict[str, Any]) -> None:
    metadata = notebook.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("notebook metadata must be a JSON object")
    kernelspec = metadata.get("kernelspec")
    if not isinstance(kernelspec, dict) or not kernelspec.get("name"):
        metadata["kernelspec"] = dict(DEFAULT_KERNELSPEC)
    language_info = metadata.get("language_info")
    if not isinstance(language_info, dict) or not language_info.get("name"):
        metadata["language_info"] = dict(DEFAULT_LANGUAGE_INFO)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path.relative_to(ROOT)} must contain a YAML mapping")
    return value


def default_kernel_id(
    experiment: str,
    kind: str,
    owner: str | None = None,
    kernel_slug: str | None = None,
) -> str | None:
    username = os.environ.get("KAGGLE_USERNAME") or owner
    if not username:
        return None
    resolved_slug = kernel_slug or (
        f"{experiment.lower().replace('_', '-')}-{kind.replace('_', '-')}"
    )
    return f"{username}/{resolved_slug}"


def suffixed_kernel_id(kernel_id_prefix: str | None, kind: str) -> str | None:
    if not kernel_id_prefix:
        return None
    username, _, slug = kernel_id_prefix.partition("/")
    if not username or not slug:
        raise ValueError("--kernel-id-prefix must look like username/kernel-slug")
    return f"{username}/{slug}-{kaggle_slug(kind)}"


def kaggle_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def metadata_validation_errors(
    metadata: dict[str, Any],
    *,
    expected_enable_gpu: bool | object = _UNSET,
    expected_enable_internet: bool | object = _UNSET,
    expected_machine_shape: str | None | object = _UNSET,
) -> list[str]:
    errors: list[str] = []
    if metadata.get("enable_tpu") is not False:
        errors.append(
            "enable_tpu is unsupported by this repository and must be explicitly false"
        )
    if expected_enable_gpu is not _UNSET and metadata.get("enable_gpu") is not expected_enable_gpu:
        errors.append(
            "enable_gpu does not match effective runtime config: "
            f"expected {expected_enable_gpu}, got {metadata.get('enable_gpu')!r}"
        )
    if (
        expected_enable_internet is not _UNSET
        and metadata.get("enable_internet") is not expected_enable_internet
    ):
        errors.append(
            "enable_internet does not match effective runtime config: "
            f"expected {expected_enable_internet}, got "
            f"{metadata.get('enable_internet')!r}"
        )
    if (
        expected_machine_shape is not _UNSET
        and metadata.get("machine_shape") != expected_machine_shape
    ):
        errors.append(
            "machine_shape does not match effective runtime config: "
            f"expected {expected_machine_shape!r}, got {metadata.get('machine_shape')!r}"
        )
    for key in ("id", "title"):
        value = metadata.get(key)
        if is_todo(value) or str(value).startswith("TODO") or str(value).startswith("INSERT_"):
            errors.append(f"{key} contains a TODO value")

    competition_sources = metadata.get("competition_sources")
    if not isinstance(competition_sources, list) or not competition_sources:
        errors.append("competition_sources is empty")
    elif any(is_todo(value) or str(value).startswith("TODO") for value in competition_sources):
        errors.append("competition_sources contains a TODO value")

    kernel_id = str(metadata.get("id") or "")
    owner, separator, kernel_slug = kernel_id.partition("/")
    if separator != "/" or not owner or not kernel_slug or "/" in kernel_slug:
        errors.append("id must use owner/kernel-slug format")
    elif len(kernel_slug) > MAX_KERNEL_SLUG_LENGTH:
        errors.append(f"kernel slug exceeds {MAX_KERNEL_SLUG_LENGTH} characters: {kernel_slug}")

    title = str(metadata.get("title") or "")
    title_slug = kaggle_slug(title)
    if not title_slug:
        errors.append("title does not produce a non-empty Kaggle slug")
    elif kernel_slug and title_slug != kernel_slug:
        errors.append(
            f"id/title slug mismatch: id uses {kernel_slug!r}, title resolves to {title_slug!r}"
        )

    return errors


def copy_src(destination_dir: Path) -> None:
    source = ROOT / "src"
    if not source.exists():
        return

    destination = destination_dir / "src"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=IGNORE_PATTERNS)


def is_experiment_support_file(path: Path) -> bool:
    return path.suffix == ".py" or path.suffix in {".yaml", ".yml"} or path.name == "metrics.json"


def copy_experiment_sources(experiment_dir: Path, destination_dir: Path) -> None:
    for path in sorted(experiment_dir.iterdir()):
        if not path.is_file():
            continue
        if is_experiment_support_file(path):
            shutil.copy2(path, destination_dir / path.name)


def collect_support_files(
    experiment_dir: Path,
    copy_repository_src: bool,
    bootstrap_files: list[str] | None = None,
    bootstrap_dependency_files: list[dict[str, str]] | None = None,
    include_experiment_sources: bool = True,
) -> dict[str, bytes]:
    support_files: dict[str, bytes] = {}
    if include_experiment_sources:
        for path in sorted(experiment_dir.iterdir()):
            if not path.is_file():
                continue
            if is_experiment_support_file(path):
                support_files[path.name] = path.read_bytes()

    support_files["project.yml"] = (ROOT / "project.yml").read_bytes()

    if copy_repository_src:
        src_dir = ROOT / "src"
        if src_dir.exists():
            for path in sorted(src_dir.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    support_files[str(path.relative_to(ROOT))] = path.read_bytes()
    for item in bootstrap_files or []:
        relative_path = Path(str(item))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe bootstrap file path: {item}")
        source = experiment_dir / relative_path
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(
                f"configured bootstrap file missing: {source.relative_to(ROOT)}"
            )
        support_files[str(relative_path)] = source.read_bytes()
    for item in bootstrap_dependency_files or []:
        source_path = Path(str(item.get("source", "")))
        destination_path = Path(str(item.get("destination", "")))
        if (
            source_path.is_absolute()
            or destination_path.is_absolute()
            or ".." in source_path.parts
            or ".." in destination_path.parts
        ):
            raise ValueError(f"unsafe bootstrap dependency mapping: {item}")
        source = ROOT / source_path
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"configured bootstrap dependency missing: {source_path}")
        support_files[str(destination_path)] = source.read_bytes()
    return support_files


def build_support_bundle(support_files: dict[str, bytes]) -> tuple[str, dict[str, dict[str, Any]]]:
    manifest: dict[str, dict[str, Any]] = {}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative_path, contents in sorted(support_files.items()):
            info = zipfile.ZipInfo(relative_path)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, contents)
            manifest[relative_path] = {
                "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return encoded, manifest


def make_support_cell(support_files: dict[str, bytes]) -> dict[str, Any]:
    encoded_zip, manifest = build_support_bundle(support_files)
    chunks = [encoded_zip[index : index + 88] for index in range(0, len(encoded_zip), 88)]
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True)
    source = [
        "# Kaggle push sends only the notebook body, so recreate local support files first.\n",
        "import base64 as _kaggle_base64\n",
        "import hashlib as _kaggle_hashlib\n",
        "import io as _kaggle_io\n",
        "import time as _kaggle_time\n",
        "import zipfile as _kaggle_zipfile\n",
        "from pathlib import Path as _KagglePath\n",
        "\n",
        "_KAGGLE_BOOTSTRAP_STARTED = _kaggle_time.time()\n",
        "_KAGGLE_SUPPORT_ZIP_B64 = (\n",
    ]
    source.extend(f"    {chunk!r}\n" for chunk in chunks)
    source.extend(
        [
            ")\n",
            f"_KAGGLE_SUPPORT_MANIFEST = {manifest_json}\n",
            "_zip_bytes = _kaggle_base64.b64decode(_KAGGLE_SUPPORT_ZIP_B64)\n",
            "with _kaggle_zipfile.ZipFile(_kaggle_io.BytesIO(_zip_bytes)) as _zip:\n",
            "    for _member in _zip.infolist():\n",
            "        _path = _KagglePath(_member.filename)\n",
            "        if _path.is_absolute() or '..' in _path.parts:\n",
            "            raise RuntimeError(f'unsafe bootstrap path: {_member.filename}')\n",
            "    _zip.extractall('.')\n",
            "for _relative_path, _info in _KAGGLE_SUPPORT_MANIFEST.items():\n",
            "    _contents = _KagglePath(_relative_path).read_bytes()\n",
            "    _digest = _kaggle_hashlib.sha256(_contents).hexdigest()\n",
            "    if _digest != _info['sha256']:\n",
            "        raise RuntimeError(f'bootstrap hash mismatch: {_relative_path}')\n",
            "print(\n",
            '    f"Prepared {len(_KAGGLE_SUPPORT_MANIFEST)} Kaggle support files "\n',
            '    "from zip bootstrap."\n',
            ")\n",
        ]
    )
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "kaggle-support-files",
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def write_kaggle_notebook(
    source_notebook: Path,
    destination_notebook: Path,
    support_files: dict[str, bytes],
) -> None:
    notebook = json.loads(source_notebook.read_text())
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"{source_notebook.relative_to(ROOT)} has no notebook cells")
    ensure_notebook_kernel_metadata(notebook)
    cells.insert(0, make_support_cell(support_files))
    write_json(destination_notebook, notebook)


def selected_kinds(value: str) -> tuple[str, ...]:
    if value == "both":
        return NOTEBOOK_KINDS
    return (validate_notebook_kind(value),)


def build_metadata(
    *,
    experiment: str,
    kind: str,
    notebook_name: str,
    kernel_id: str | None,
    title_prefix: str | None,
    title: str | None,
    competition_slug: str | None,
    competition_name: str | None,
    enable_gpu: bool | None,
    enable_internet: bool | None,
    machine_shape: str | None,
    run_on_push: bool = False,
    owner: str | None = None,
    kernel_sources: list[str] | None = None,
    dataset_sources: list[str] | None = None,
    model_sources: list[str] | None = None,
) -> dict[str, Any]:
    del competition_name
    default_title = (
        f"{title_prefix} {experiment} {kind}" if title_prefix else f"{experiment} {kind}"
    )
    title_for_default_id = title or default_title
    resolved_kernel_id = (
        kernel_id
        or default_kernel_id(
            experiment,
            kind,
            owner,
            kernel_slug=kaggle_slug(title_for_default_id),
        )
        or "TODO_KAGGLE_USERNAME/TODO_KERNEL_SLUG"
    )
    _, separator, resolved_kernel_slug = resolved_kernel_id.partition("/")
    if title is not None:
        resolved_title = title
    elif kernel_id is not None and separator and resolved_kernel_slug:
        resolved_title = resolved_kernel_slug.replace("-", " ")
    else:
        resolved_title = default_title
    metadata: dict[str, Any] = {
        "id": resolved_kernel_id,
        "title": resolved_title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False if enable_gpu is None else enable_gpu,
        "enable_tpu": False,
        "enable_internet": False if enable_internet is None else enable_internet,
        "run_on_push": run_on_push,
        "dataset_sources": [] if dataset_sources is None else dataset_sources,
        "competition_sources": [],
        "kernel_sources": [] if kernel_sources is None else kernel_sources,
        "model_sources": [] if model_sources is None else model_sources,
    }

    if competition_slug and not is_todo(competition_slug):
        metadata["competition_sources"] = [competition_slug]
    if machine_shape and not is_todo(machine_shape):
        metadata["machine_shape"] = machine_shape
    return metadata


def prepare_one(
    *,
    experiment_dir: Path,
    experiment: str,
    kind: str,
    kernel_id: str | None,
    title_prefix: str | None,
    title: str | None,
    competition_slug: str | None,
    competition_name: str | None,
    owner: str | None,
    enable_gpu: bool | None,
    enable_internet: bool | None,
    machine_shape: str | None,
    run_on_push: bool,
    copy_repository_src: bool,
    bootstrap_files: list[str] | None,
    bootstrap_dependency_files: list[dict[str, str]] | None,
    include_experiment_sources: bool,
    kernel_sources: list[str] | None,
    dataset_sources: list[str] | None,
    model_sources: list[str] | None,
) -> tuple[Path, list[str]]:
    notebook_name = f"{experiment}_{kind}.ipynb"
    source_notebook = experiment_dir / notebook_name
    if not source_notebook.exists():
        raise FileNotFoundError(f"required notebook missing: {source_notebook.relative_to(ROOT)}")

    destination_dir = experiment_dir / "kaggle" / kind
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.mkdir(parents=True)

    copy_experiment_sources(experiment_dir, destination_dir)
    support_files = collect_support_files(
        experiment_dir,
        copy_repository_src,
        bootstrap_files,
        bootstrap_dependency_files,
        include_experiment_sources,
    )
    write_kaggle_notebook(source_notebook, destination_dir / notebook_name, support_files)
    shutil.copy2(ROOT / "project.yml", destination_dir / "project.yml")
    if copy_repository_src:
        copy_src(destination_dir)

    metadata = build_metadata(
        experiment=experiment,
        kind=kind,
        notebook_name=notebook_name,
        kernel_id=kernel_id,
        title_prefix=title_prefix,
        title=title,
        competition_slug=competition_slug,
        competition_name=competition_name,
        owner=owner,
        enable_gpu=enable_gpu,
        enable_internet=enable_internet,
        machine_shape=machine_shape,
        run_on_push=run_on_push,
        kernel_sources=kernel_sources,
        dataset_sources=dataset_sources,
        model_sources=model_sources,
    )
    write_json(destination_dir / "kernel-metadata.json", metadata)
    return destination_dir, metadata_validation_errors(metadata)


def main() -> None:
    args = parse_args()
    try:
        notebook_kinds = selected_kinds(args.notebook)
    except ValueError as exc:
        raise SystemExit(f"invalid --notebook value: {exc}") from exc
    if args.kernel_id and args.notebook == "both":
        raise SystemExit("--kernel-id is only valid with a single --notebook kind")
    if args.title and args.notebook == "both":
        raise SystemExit("--title is only valid with a single --notebook kind")

    config = load_project_config()
    experiment_dir = ROOT / "experiments" / args.experiment
    if not experiment_dir.exists():
        raise FileNotFoundError(f"experiment does not exist: {experiment_dir.relative_to(ROOT)}")
    experiment_config = read_yaml(experiment_dir / "config.yaml")

    competition_slug = args.competition_slug or get_nested(config, "competition.slug")
    competition_name = get_nested(config, "competition.name")
    owner = get_nested(config, "metadata.owner")
    configured_kernel_sources = get_nested(experiment_config, "runtime.kaggle.kernel_sources")
    configured_dataset_sources = get_nested(experiment_config, "runtime.kaggle.dataset_sources")
    configured_model_sources = get_nested(experiment_config, "runtime.kaggle.model_sources")
    configured_bootstrap_files = get_nested(experiment_config, "runtime.kaggle.bootstrap_files")
    configured_bootstrap_dependency_files = get_nested(
        experiment_config, "runtime.kaggle.bootstrap_dependency_files"
    )
    configured_train_kernel_sources = get_nested(
        experiment_config,
        "runtime.kaggle.train_kernel_sources",
    )
    configured_train_dataset_sources = get_nested(
        experiment_config,
        "runtime.kaggle.train_dataset_sources",
    )
    configured_train_model_sources = get_nested(
        experiment_config,
        "runtime.kaggle.train_model_sources",
    )
    configured_inference_kernel_sources = get_nested(
        experiment_config,
        "runtime.kaggle.inference_kernel_sources",
    )
    configured_inference_dataset_sources = get_nested(
        experiment_config,
        "runtime.kaggle.inference_dataset_sources",
    )
    configured_inference_model_sources = get_nested(
        experiment_config,
        "runtime.kaggle.inference_model_sources",
    )
    configured_guard_kernel_sources = get_nested(
        experiment_config,
        "runtime.kaggle.guard_kernel_sources",
    )
    configured_guard_dataset_sources = get_nested(
        experiment_config,
        "runtime.kaggle.guard_dataset_sources",
    )
    configured_guard_model_sources = get_nested(
        experiment_config,
        "runtime.kaggle.guard_model_sources",
    )
    default_kernel_sources = (
        [str(value) for value in configured_kernel_sources]
        if isinstance(configured_kernel_sources, list)
        else None
    )
    default_dataset_sources = (
        [str(value) for value in configured_dataset_sources]
        if isinstance(configured_dataset_sources, list)
        else None
    )
    default_model_sources = (
        [str(value) for value in configured_model_sources]
        if isinstance(configured_model_sources, list)
        else None
    )
    default_bootstrap_files = (
        [str(value) for value in configured_bootstrap_files]
        if isinstance(configured_bootstrap_files, list)
        else None
    )

    prepared: list[tuple[Path, list[str]]] = []
    for kind in notebook_kinds:
        runtime_settings = effective_kaggle_runtime(config, experiment_config, kind)
        runtime_errors = kaggle_runtime_errors(runtime_settings)
        if runtime_errors:
            raise ValueError("; ".join(runtime_errors))
        kind_bootstrap_files = get_nested(
            experiment_config,
            f"runtime.kaggle.{kind}.bootstrap_files",
        )
        kind_bootstrap_dependency_files = get_nested(
            experiment_config,
            f"runtime.kaggle.{kind}.bootstrap_dependency_files",
        )
        kind_include_experiment_sources = get_nested(
            experiment_config,
            f"runtime.kaggle.{kind}.include_experiment_sources",
        )
        resolved_bootstrap_files = (
            [str(value) for value in kind_bootstrap_files]
            if isinstance(kind_bootstrap_files, list)
            else default_bootstrap_files
        )
        resolved_bootstrap_dependency_files = (
            kind_bootstrap_dependency_files
            if isinstance(kind_bootstrap_dependency_files, list)
            else (
                configured_bootstrap_dependency_files
                if isinstance(configured_bootstrap_dependency_files, list)
                else None
            )
        )
        if kind_include_experiment_sources is not None and not isinstance(
            kind_include_experiment_sources, bool
        ):
            raise ValueError(
                f"runtime.kaggle.{kind}.include_experiment_sources must be true or false"
            )
        resolved_include_experiment_sources = (
            kind_include_experiment_sources if kind_include_experiment_sources is not None else True
        )
        explicit_kernel_id = {
            "train": args.train_kernel_id,
            "inference": args.inference_kernel_id,
        }.get(kind)
        if args.kernel_id:
            explicit_kernel_id = args.kernel_id
        kernel_id = explicit_kernel_id or suffixed_kernel_id(args.kernel_id_prefix, kind)
        kind_configured_kernel_sources = get_nested(
            experiment_config,
            f"runtime.kaggle.{kind}.kernel_sources",
        )
        if kind_configured_kernel_sources is None:
            kind_configured_kernel_sources = get_nested(
                experiment_config,
                f"runtime.kaggle.{kind}_kernel_sources",
            )
        kind_configured_dataset_sources = get_nested(
            experiment_config,
            f"runtime.kaggle.{kind}.dataset_sources",
        )
        if kind_configured_dataset_sources is None:
            kind_configured_dataset_sources = get_nested(
                experiment_config,
                f"runtime.kaggle.{kind}_dataset_sources",
            )
        kind_configured_model_sources = get_nested(
            experiment_config,
            f"runtime.kaggle.{kind}.model_sources",
        )
        if kind_configured_model_sources is None:
            kind_configured_model_sources = get_nested(
                experiment_config,
                f"runtime.kaggle.{kind}_model_sources",
            )
        if kind == "train":
            if kind_configured_kernel_sources is None:
                kind_configured_kernel_sources = configured_train_kernel_sources
            if kind_configured_dataset_sources is None:
                kind_configured_dataset_sources = configured_train_dataset_sources
            if kind_configured_model_sources is None:
                kind_configured_model_sources = configured_train_model_sources
        elif kind == "inference":
            if kind_configured_kernel_sources is None:
                kind_configured_kernel_sources = configured_inference_kernel_sources
            if kind_configured_dataset_sources is None:
                kind_configured_dataset_sources = configured_inference_dataset_sources
            if kind_configured_model_sources is None:
                kind_configured_model_sources = configured_inference_model_sources
        elif kind in {"prefix_crop_features", "gr_matcher_features"}:
            if kind_configured_kernel_sources is None:
                kind_configured_kernel_sources = default_kernel_sources
            if kind_configured_dataset_sources is None:
                kind_configured_dataset_sources = default_dataset_sources
            if kind_configured_model_sources is None:
                kind_configured_model_sources = default_model_sources
        elif kind == "guard":
            if kind_configured_kernel_sources is None:
                kind_configured_kernel_sources = configured_guard_kernel_sources
            if kind_configured_dataset_sources is None:
                kind_configured_dataset_sources = configured_guard_dataset_sources
            if kind_configured_model_sources is None:
                kind_configured_model_sources = configured_guard_model_sources
        elif (
            kind in {"selector_train", "signed_selector_train", "downstream_gpu_train"}
            or kind.startswith("train_lgb")
            or kind.startswith("train_variant")
        ):
            if kind_configured_kernel_sources is None:
                kind_configured_kernel_sources = configured_train_kernel_sources
            if kind_configured_dataset_sources is None:
                kind_configured_dataset_sources = configured_train_dataset_sources
            if kind_configured_model_sources is None:
                kind_configured_model_sources = configured_train_model_sources
        if kind == "pfbeam_features" and all(
            value is None
            for value in (
                kind_configured_kernel_sources,
                kind_configured_dataset_sources,
                kind_configured_model_sources,
            )
        ):
            kernel_sources = None
            dataset_sources = None
            model_sources = None
        else:
            kernel_sources = (
                [str(value) for value in kind_configured_kernel_sources]
                if isinstance(kind_configured_kernel_sources, list)
                else default_kernel_sources
            )
            dataset_sources = (
                [str(value) for value in kind_configured_dataset_sources]
                if isinstance(kind_configured_dataset_sources, list)
                else default_dataset_sources
            )
            model_sources = (
                [str(value) for value in kind_configured_model_sources]
                if isinstance(kind_configured_model_sources, list)
                else default_model_sources
            )
        prepared.append(
            prepare_one(
                experiment_dir=experiment_dir,
                experiment=args.experiment,
                kind=kind,
                kernel_id=kernel_id,
                title_prefix=args.title_prefix,
                title=args.title,
                competition_slug=str(competition_slug) if competition_slug is not None else None,
                competition_name=str(competition_name) if competition_name is not None else None,
                owner=str(owner) if owner is not None and not is_todo(owner) else None,
                enable_gpu=runtime_settings["enable_gpu"],
                enable_internet=runtime_settings["enable_internet"],
                machine_shape=runtime_settings.get("machine_shape"),
                run_on_push=args.run_on_push,
                copy_repository_src=not args.no_src,
                bootstrap_files=resolved_bootstrap_files,
                bootstrap_dependency_files=resolved_bootstrap_dependency_files,
                include_experiment_sources=resolved_include_experiment_sources,
                kernel_sources=kernel_sources,
                dataset_sources=dataset_sources,
                model_sources=model_sources,
            )
        )

    metadata_errors: dict[str, list[str]] = {
        path.name: errors for path, errors in prepared if errors
    }
    if args.strict and metadata_errors:
        raise SystemExit(f"kernel metadata validation failed: {metadata_errors}")

    for path, errors in prepared:
        print(f"Prepared Kaggle notebook directory: {path.relative_to(ROOT)}")
        if errors:
            print(f"Metadata issues remain in {path.name}: {'; '.join(errors)}")


if __name__ == "__main__":
    main()
