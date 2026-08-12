from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

try:
    from .config_utils import (
        ROOT,
        effective_kaggle_runtime,
        get_nested,
        kaggle_runtime_errors,
        load_project_config,
    )
    from .prepare_kaggle_notebooks import (
        ensure_notebook_kernel_metadata,
        metadata_validation_errors,
    )
except ImportError:  # Direct execution: `uv run python scripts/validate_kaggle_metadata.py ...`
    from config_utils import (
        ROOT,
        effective_kaggle_runtime,
        get_nested,
        kaggle_runtime_errors,
        load_project_config,
    )
    from prepare_kaggle_notebooks import (
        ensure_notebook_kernel_metadata,
        metadata_validation_errors,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one prepared Kaggle notebook package before push."
    )
    parser.add_argument("--package-dir", required=True, type=Path)
    return parser.parse_args()


def _cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list) and all(isinstance(line, str) for line in source):
        return "".join(source)
    if isinstance(source, str):
        return source
    raise ValueError("generated notebook contains a cell with invalid source")


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            try:
                return ast.literal_eval(node.value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"bootstrap assignment is not literal: {name}") from exc
    raise ValueError(f"generated notebook bootstrap is missing {name}")


def _bootstrap_files(notebook_path: Path) -> dict[str, bytes]:
    try:
        notebook = json.loads(notebook_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {notebook_path}: {exc}") from exc
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError(f"generated notebook has no cells: {notebook_path}")

    if not notebook["cells"] or not isinstance(notebook["cells"][0], dict):
        raise ValueError("generated notebook bootstrap is missing from the first cell")
    bootstrap_source = _cell_source(notebook["cells"][0])
    if "_KAGGLE_SUPPORT_MANIFEST" not in bootstrap_source:
        raise ValueError("generated notebook bootstrap is missing from the first cell")
    try:
        tree = ast.parse(bootstrap_source)
    except SyntaxError as exc:
        raise ValueError(f"generated notebook bootstrap is invalid Python: {exc}") from exc

    manifest = _literal_assignment(tree, "_KAGGLE_SUPPORT_MANIFEST")
    encoded_zip = _literal_assignment(tree, "_KAGGLE_SUPPORT_ZIP_B64")
    if not isinstance(manifest, dict) or not all(
        isinstance(path, str) and isinstance(info, dict) for path, info in manifest.items()
    ):
        raise ValueError("generated notebook bootstrap manifest is invalid")
    if not isinstance(encoded_zip, str):
        raise ValueError("generated notebook bootstrap ZIP is invalid")
    try:
        archive_bytes = base64.b64decode(encoded_zip, validate=True)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("generated notebook bootstrap ZIP contains duplicate paths")
            files: dict[str, bytes] = {}
            for name in names:
                relative_path = Path(name)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(f"unsafe bootstrap path: {name}")
                files[name] = archive.read(name)
    except (ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(
            ("generated notebook", "unsafe bootstrap path")
        ):
            raise
        raise ValueError(f"generated notebook bootstrap ZIP is invalid: {exc}") from exc

    if set(manifest) != set(files):
        raise ValueError("bootstrap manifest paths do not match embedded ZIP paths")
    for relative_path, contents in files.items():
        info = manifest[relative_path]
        expected_bytes = info.get("bytes")
        expected_sha = info.get("sha256")
        actual_sha = hashlib.sha256(contents).hexdigest()
        if expected_bytes != len(contents) or expected_sha != actual_sha:
            raise ValueError(f"bootstrap manifest does not match embedded file: {relative_path}")
    return files


def _repository_package_context(package_dir: Path) -> tuple[Path, str] | None:
    try:
        relative = package_dir.resolve().relative_to((ROOT / "experiments").resolve())
    except ValueError:
        return None
    if len(relative.parts) != 3 or relative.parts[1] != "kaggle":
        return None
    return ROOT / "experiments" / relative.parts[0], relative.parts[2]


def _compare_bytes(
    errors: list[str],
    *,
    current_path: Path,
    packaged_path: Path,
    label: str,
) -> None:
    if not packaged_path.is_file():
        errors.append(f"prepared package is stale; missing {label}: {packaged_path}")
    elif current_path.read_bytes() != packaged_path.read_bytes():
        errors.append(f"prepared package is stale; {label} differs from {current_path}")


def _repository_consistency_errors(
    package_dir: Path,
    notebook_path: Path,
    bootstrap_files: dict[str, bytes],
    experiment_config: dict[str, Any],
) -> list[str]:
    context = _repository_package_context(package_dir)
    if context is None:
        return []
    experiment_dir, notebook_kind = context
    errors: list[str] = []

    _compare_bytes(
        errors,
        current_path=ROOT / "project.yml",
        packaged_path=package_dir / "project.yml",
        label="project.yml",
    )
    support_sources = sorted(
        path
        for path in experiment_dir.iterdir()
        if path.is_file()
        and (path.suffix in {".py", ".yaml", ".yml"} or path.name == "metrics.json")
    )
    for source_path in support_sources:
        _compare_bytes(
            errors,
            current_path=source_path,
            packaged_path=package_dir / source_path.name,
            label=source_path.name,
        )
    current_support_names = {path.name for path in support_sources}
    packaged_support_names = {
        path.name
        for path in package_dir.iterdir()
        if path.is_file()
        and path.name != "project.yml"
        and (path.suffix in {".py", ".yaml", ".yml"} or path.name == "metrics.json")
    }
    for extra_name in sorted(packaged_support_names - current_support_names):
        errors.append(f"prepared package is stale; removed experiment source remains: {extra_name}")

    packaged_src = package_dir / "src"
    if packaged_src.is_dir():
        current_src = ROOT / "src"
        current_src_files = {
            path.relative_to(current_src): path
            for path in current_src.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        }
        packaged_src_files = {
            path.relative_to(packaged_src): path
            for path in packaged_src.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        }
        for missing_path in sorted(current_src_files.keys() - packaged_src_files.keys()):
            errors.append(f"prepared package is stale; missing src/{missing_path}")
        for extra_path in sorted(packaged_src_files.keys() - current_src_files.keys()):
            errors.append(f"prepared package is stale; removed src/{extra_path} remains")
        for relative_path in sorted(current_src_files.keys() & packaged_src_files.keys()):
            if (
                current_src_files[relative_path].read_bytes()
                != packaged_src_files[relative_path].read_bytes()
            ):
                errors.append(f"prepared package is stale; src/{relative_path} differs")

    source_notebook = experiment_dir / notebook_path.name
    if not source_notebook.is_file():
        errors.append(f"prepared package is stale; source notebook is missing: {source_notebook}")
    else:
        try:
            source = json.loads(source_notebook.read_text())
            generated = json.loads(notebook_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"notebook JSON is invalid: {exc}")
        else:
            source_cells = source.get("cells") if isinstance(source, dict) else None
            generated_cells = generated.get("cells") if isinstance(generated, dict) else None
            if not isinstance(source_cells, list) or not isinstance(generated_cells, list):
                errors.append("source or generated notebook has no cells")
            else:
                ensure_notebook_kernel_metadata(source)
                generated_without_bootstrap = dict(generated)
                generated_without_bootstrap["cells"] = generated_cells[1:]
                if generated_without_bootstrap != source:
                    errors.append(
                        "prepared package is stale; generated notebook differs from source notebook"
                    )

    include_experiment_sources = get_nested(
        experiment_config,
        f"runtime.kaggle.{notebook_kind}.include_experiment_sources",
    )
    if include_experiment_sources is not False:
        for source_path in support_sources:
            if source_path.name not in bootstrap_files:
                errors.append(
                    f"generated notebook bootstrap is missing experiment source: {source_path.name}"
                )

    dependency_files = get_nested(
        experiment_config,
        f"runtime.kaggle.{notebook_kind}.bootstrap_dependency_files",
    )
    if dependency_files is None:
        dependency_files = get_nested(
            experiment_config,
            "runtime.kaggle.bootstrap_dependency_files",
        )
    dependency_sources: dict[str, Path] = {}
    if isinstance(dependency_files, list):
        for item in dependency_files:
            if not isinstance(item, dict):
                continue
            source_value = item.get("source")
            destination_value = item.get("destination")
            if isinstance(source_value, str) and isinstance(destination_value, str):
                dependency_sources[destination_value] = ROOT / source_value

    for relative_path, contents in bootstrap_files.items():
        relative = Path(relative_path)
        dependency_source = dependency_sources.get(relative_path)
        source_candidates = (
            [dependency_source]
            if dependency_source is not None
            else [experiment_dir / relative, ROOT / relative]
        )
        current_source = next((path for path in source_candidates if path.is_file()), None)
        if current_source is not None and current_source.read_bytes() != contents:
            errors.append(
                "prepared package is stale; bootstrap file differs from "
                f"{current_source}: {relative_path}"
            )

    return errors


def validate_package(package_dir: Path) -> dict[str, Any]:
    metadata_path = package_dir / "kernel-metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"missing Kaggle metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {metadata_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Kaggle metadata must be a JSON object: {metadata_path}")

    packaged_project_path = package_dir / "project.yml"
    if not packaged_project_path.is_file():
        raise ValueError(f"missing prepared project config: {packaged_project_path}")
    project_config = load_project_config(packaged_project_path)
    packaged_experiment_path = package_dir / "config.yaml"
    experiment_config = (
        load_project_config(packaged_experiment_path) if packaged_experiment_path.is_file() else {}
    )
    runtime_settings = effective_kaggle_runtime(
        project_config,
        experiment_config,
        package_dir.name,
    )
    runtime_errors = kaggle_runtime_errors(runtime_settings)
    if runtime_errors:
        raise ValueError("; ".join(runtime_errors))
    errors = metadata_validation_errors(
        metadata,
        expected_enable_gpu=runtime_settings["enable_gpu"],
        expected_enable_internet=runtime_settings["enable_internet"],
        expected_machine_shape=runtime_settings.get("machine_shape"),
    )
    code_file = metadata.get("code_file")
    notebook_path: Path | None = None
    if not isinstance(code_file, str) or not code_file:
        errors.append("code_file is empty")
    elif not (package_dir / code_file).is_file():
        errors.append(f"code_file does not exist in package: {code_file}")
    else:
        notebook_path = package_dir / code_file
    if notebook_path is not None:
        try:
            embedded_files = _bootstrap_files(notebook_path)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if "project.yml" not in embedded_files:
                errors.append("generated notebook bootstrap is missing project.yml")
            for relative_path, contents in embedded_files.items():
                packaged_support = package_dir / relative_path
                if packaged_support.is_file() and packaged_support.read_bytes() != contents:
                    errors.append(
                        "bootstrap support file does not match prepared package: "
                        f"{relative_path}"
                    )
            errors.extend(
                _repository_consistency_errors(
                    package_dir,
                    notebook_path,
                    embedded_files,
                    experiment_config,
                )
            )
    if errors:
        raise ValueError("; ".join(errors))
    return metadata


def main() -> None:
    args = parse_args()
    try:
        metadata = validate_package(args.package_dir)
    except ValueError as exc:
        raise SystemExit(f"Kaggle metadata validation failed: {exc}") from exc
    print(f"Kaggle metadata is push-ready: {metadata['id']}")


if __name__ == "__main__":
    main()
