import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_kaggle_notebooks as notebook_prep  # noqa: E402
from prepare_kaggle_notebooks import (  # noqa: E402
    build_metadata,
    build_support_bundle,
    make_support_cell,
    metadata_validation_errors,
)


def test_build_notebook_metadata_uses_kaggle_notebook_fields() -> None:
    metadata = build_metadata(
        experiment="exp123_test",
        kind="inference",
        notebook_name="exp123_test_inference.ipynb",
        kernel_id="user/exp123-test-inference",
        title_prefix="ROGII",
        title="exp123 test inference",
        competition_slug="rogii-wellbore-geology-prediction",
        competition_name="ROGII - Wellbore Geology Prediction",
        enable_gpu=True,
        enable_internet=False,
        run_on_push=True,
        machine_shape=None,
        kernel_sources=["owner/kernel-output"],
        dataset_sources=["owner/external-dataset"],
        model_sources=["owner/model/variant/1"],
    )

    assert metadata["id"] == "user/exp123-test-inference"
    assert metadata["code_file"] == "exp123_test_inference.ipynb"
    assert metadata["kernel_type"] == "notebook"
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["run_on_push"] is True
    assert metadata["competition_sources"] == ["rogii-wellbore-geology-prediction"]
    assert metadata["kernel_sources"] == ["owner/kernel-output"]
    assert metadata["dataset_sources"] == ["owner/external-dataset"]
    assert metadata["model_sources"] == ["owner/model/variant/1"]
    assert metadata_validation_errors(metadata) == []


def test_build_notebook_metadata_can_default_matching_id_and_title(monkeypatch) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)

    metadata = build_metadata(
        experiment="exp123_test",
        kind="train",
        notebook_name="exp123_test_train.ipynb",
        kernel_id=None,
        title_prefix="ROGII",
        title=None,
        competition_slug="rogii-wellbore-geology-prediction",
        competition_name="ROGII - Wellbore Geology Prediction",
        enable_gpu=False,
        enable_internet=False,
        run_on_push=False,
        machine_shape=None,
        owner="projectowner",
    )

    assert metadata["id"] == "projectowner/rogii-exp123-test-train"
    assert metadata["title"] == "ROGII exp123_test train"
    assert metadata_validation_errors(metadata) == []


def test_build_notebook_metadata_defaults_title_from_explicit_kernel_id() -> None:
    metadata = build_metadata(
        experiment="exp123_test",
        kind="train",
        notebook_name="exp123_test_train.ipynb",
        kernel_id="projectowner/short-canonical-train",
        title_prefix=None,
        title=None,
        competition_slug="competition",
        competition_name="Competition Name",
        enable_gpu=False,
        enable_internet=False,
        run_on_push=False,
        machine_shape=None,
    )

    assert metadata["title"] == "short canonical train"
    assert metadata_validation_errors(metadata) == []


def test_metadata_validation_rejects_long_kernel_slug() -> None:
    slug = "x" * 51
    metadata = {
        "id": f"owner/{slug}",
        "title": slug,
        "competition_sources": ["competition"],
    }

    assert any("exceeds 50 characters" in error for error in metadata_validation_errors(metadata))


def test_metadata_validation_rejects_tpu() -> None:
    metadata = {
        "id": "owner/exp123-test-train",
        "title": "exp123 test train",
        "competition_sources": ["competition"],
        "enable_tpu": True,
    }

    assert any(
        "enable_tpu is unsupported" in error
        for error in metadata_validation_errors(metadata)
    )


def test_support_cell_uses_zip_bootstrap_not_inline_file_json() -> None:
    encoded_zip, manifest = build_support_bundle({"baseline.py": b"VALUE = 1\n"})
    cell = make_support_cell({"baseline.py": b"VALUE = 1\n"})
    source = "".join(cell["source"])

    assert encoded_zip
    assert manifest["baseline.py"]["bytes"] == 10
    assert "VALUE = 1" not in source
    assert "_KAGGLE_SUPPORT_ZIP_B64" in source
    assert "_KAGGLE_BOOTSTRAP_STARTED" in source
    assert "_zip.extractall" in source


def test_support_bundle_excludes_repository_bytecode(tmp_path, monkeypatch) -> None:
    experiment_dir = tmp_path / "experiments" / "exp123_test"
    src_dir = tmp_path / "src"
    cache_dir = src_dir / "__pycache__"
    experiment_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    (experiment_dir / "config.yaml").write_text("experiment: {}\n")
    (tmp_path / "project.yml").write_text("competition: {}\n")
    (src_dir / "feature.py").write_text("VALUE = 1\n")
    (cache_dir / "feature.cpython-311.pyc").write_bytes(b"bytecode")

    monkeypatch.setattr(notebook_prep, "ROOT", tmp_path)
    support = notebook_prep.collect_support_files(
        experiment_dir,
        copy_repository_src=True,
    )

    assert "src/feature.py" in support
    assert not any("__pycache__" in path or path.endswith(".pyc") for path in support)


def test_support_bundle_includes_experiment_metrics_schema(tmp_path, monkeypatch) -> None:
    experiment_dir = tmp_path / "experiments" / "exp123_test"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "metrics.json").write_text('{"evidence": {"reruns": []}}\n')
    (tmp_path / "project.yml").write_text("competition: {}\n")

    monkeypatch.setattr(notebook_prep, "ROOT", tmp_path)
    support = notebook_prep.collect_support_files(
        experiment_dir,
        copy_repository_src=False,
    )

    assert support["metrics.json"] == b'{"evidence": {"reruns": []}}\n'


def test_support_bundle_can_exclude_experiment_sources(tmp_path, monkeypatch) -> None:
    experiment_dir = tmp_path / "experiments" / "exp123_test"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "large_candidate.py").write_text("VALUE = 1\n")
    (experiment_dir / "config.yaml").write_text("experiment: {}\n")
    (tmp_path / "project.yml").write_text("competition: {}\n")

    monkeypatch.setattr(notebook_prep, "ROOT", tmp_path)
    support = notebook_prep.collect_support_files(
        experiment_dir,
        copy_repository_src=False,
        include_experiment_sources=False,
    )

    assert set(support) == {"project.yml"}
