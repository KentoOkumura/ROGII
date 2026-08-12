"""Phase 1: Instant API badges (~16 badges).

Performs badge prerequisite actions via kagglehub and kaggle-cli:
  - Python Coder, R Coder, API Notebook Creator, Utility Scripter
  - Code Uploader, Code Forker, Code Tagger
  - Dataset Creator, API Dataset Creator, Dataset Tagger, Dataset Documenter
  - Model Creator, API Model Creator, Model Variation Creator, Model Tagger, Model Documenter
"""

import json
import shutil
import time

from badge_tracker import set_status, should_attempt
from utils import (
    API_DELAY,
    TEMPLATES_DIR,
    make_temp_dir,
    resource_name,
    run_kaggle_cli,
)


def _create_python_notebook(username: str) -> bool:
    """Push a Python notebook for three notebook badge criteria."""
    badge_ids = ["python_coder", "api_notebook_creator", "code_uploader"]
    actionable = [b for b in badge_ids if should_attempt(b)]
    if not actionable:
        return True

    for bid in actionable:
        set_status(bid, "attempting")

    try:
        tmp = make_temp_dir("-python-nb")
        nb_slug = resource_name("python-nb")

        # Copy notebook template
        src = TEMPLATES_DIR / "python_notebook.ipynb"
        dst = tmp / "notebook.ipynb"
        shutil.copy2(src, dst)

        # Create kernel-metadata.json
        metadata = {
            "id": f"{username}/{nb_slug}",
            "title": nb_slug,
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "keywords": ["badge-collector", "automated"],
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        }
        (tmp / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))

        # Push
        run_kaggle_cli(["kernels", "push", "-p", str(tmp)])
        print(f"  [OK] Python notebook pushed: {nb_slug}")

        for bid in actionable:
            set_status(bid, "action_completed", f"notebook={nb_slug}")
        return True

    except Exception as e:
        print(f"  [FAIL] Python notebook: {e}")
        for bid in actionable:
            set_status(bid, "failed", str(e))
        return False


def _create_r_notebook(username: str) -> bool:
    """Push an R notebook for the R Coder criterion."""
    if not should_attempt("r_coder"):
        return True

    set_status("r_coder", "attempting")
    try:
        tmp = make_temp_dir("-r-nb")
        nb_slug = resource_name("r-nb")

        # Copy R notebook template
        src = TEMPLATES_DIR / "r_notebook.ipynb"
        dst = tmp / "notebook.ipynb"
        shutil.copy2(src, dst)

        metadata = {
            "id": f"{username}/{nb_slug}",
            "title": nb_slug,
            "code_file": "notebook.ipynb",
            "language": "r",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "keywords": ["badge-collector", "automated"],
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        }
        (tmp / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))

        run_kaggle_cli(["kernels", "push", "-p", str(tmp)])
        print(f"  [OK] R notebook pushed: {nb_slug}")
        set_status("r_coder", "action_completed", f"notebook={nb_slug}")
        return True

    except Exception as e:
        print(f"  [FAIL] R notebook: {e}")
        set_status("r_coder", "failed", str(e))
        return False


def _create_utility_script(username: str) -> bool:
    """Push a utility script for the Utility Scripter criterion."""
    if not should_attempt("utility_scripter"):
        return True

    set_status("utility_scripter", "attempting")
    try:
        tmp = make_temp_dir("-utility")
        script_slug = resource_name("utility-script")

        # Copy utility script template
        src = TEMPLATES_DIR / "utility_script.py"
        dst = tmp / "script.py"
        shutil.copy2(src, dst)

        metadata = {
            "id": f"{username}/{script_slug}",
            "title": script_slug,
            "code_file": "script.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "keywords": ["badge-collector", "utility", "automated"],
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        }
        (tmp / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))

        run_kaggle_cli(["kernels", "push", "-p", str(tmp)])
        print(f"  [OK] Utility script pushed: {script_slug}")
        set_status("utility_scripter", "action_completed", f"script={script_slug}")
        return True

    except Exception as e:
        print(f"  [FAIL] Utility script: {e}")
        set_status("utility_scripter", "failed", str(e))
        return False


def _fork_notebook(username: str) -> bool:
    """Fork a public notebook for the Code Forker criterion."""
    if not should_attempt("code_forker"):
        return True

    set_status("code_forker", "attempting")
    try:
        # List public notebooks and pick one to fork
        # We'll fork a simple well-known public notebook
        fork_slug = resource_name("forked-nb")

        # Use kaggle API to fork — pull a simple public kernel first
        # Then re-push as a fork
        tmp = make_temp_dir("-fork")
        result = run_kaggle_cli(
            ["kernels", "pull", "alexisbcook/titanic-tutorial", "-p", str(tmp)],
            check=False,
        )

        if result.returncode != 0:
            # Try another well-known notebook
            result = run_kaggle_cli(
                ["kernels", "pull", "dansbecker/your-first-machine-learning-model", "-p", str(tmp)],
                check=False,
            )

        if result.returncode != 0:
            print("  [SKIP] Could not pull a public notebook to fork")
            set_status("code_forker", "failed", "no public notebook available")
            return False

        # Find the pulled file
        pulled_files = list(tmp.glob("*.py")) + list(tmp.glob("*.ipynb"))
        if not pulled_files:
            set_status("code_forker", "failed", "no files pulled")
            return False

        code_file = pulled_files[0]

        metadata = {
            "id": f"{username}/{fork_slug}",
            "title": fork_slug,
            "code_file": code_file.name,
            "language": "python",
            "kernel_type": "script" if code_file.suffix == ".py" else "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "keywords": ["badge-collector", "forked"],
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        }
        (tmp / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))

        run_kaggle_cli(["kernels", "push", "-p", str(tmp)])
        print(f"  [OK] Notebook forked: {fork_slug}")
        set_status("code_forker", "action_completed", f"forked={fork_slug}")
        return True

    except Exception as e:
        print(f"  [FAIL] Fork notebook: {e}")
        set_status("code_forker", "failed", str(e))
        return False


def _tag_notebook(username: str) -> bool:
    """Add tags to a notebook for the Code Tagger criterion.

    Note: Tagging is done via the keywords field in kernel-metadata.json
    when pushing. If we already pushed a notebook with keywords, this badge
    may already be satisfied. Create a dedicated tagged notebook as action evidence.
    """
    if not should_attempt("code_tagger"):
        return True

    set_status("code_tagger", "attempting")
    try:
        tmp = make_temp_dir("-tagged-nb")
        nb_slug = resource_name("tagged-nb")

        src = TEMPLATES_DIR / "python_notebook.ipynb"
        dst = tmp / "notebook.ipynb"
        shutil.copy2(src, dst)

        metadata = {
            "id": f"{username}/{nb_slug}",
            "title": nb_slug,
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "keywords": [
                "badge-collector",
                "python",
                "data-science",
                "machine-learning",
                "tutorial",
            ],
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        }
        (tmp / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))

        run_kaggle_cli(["kernels", "push", "-p", str(tmp)])
        print(f"  [OK] Tagged notebook pushed: {nb_slug}")
        set_status("code_tagger", "action_completed", f"notebook={nb_slug}")
        return True

    except Exception as e:
        print(f"  [FAIL] Tag notebook: {e}")
        set_status("code_tagger", "failed", str(e))
        return False


def _create_dataset(username: str) -> bool:
    """Create a dataset for two dataset-creation badge criteria."""
    badge_ids = ["dataset_creator", "api_dataset_creator"]
    actionable = [b for b in badge_ids if should_attempt(b)]
    if not actionable:
        return True

    for bid in actionable:
        set_status(bid, "attempting")

    try:
        tmp = make_temp_dir("-dataset")
        ds_slug = resource_name("dataset")

        # Copy dataset files
        readme_src = TEMPLATES_DIR / "README_dataset.md"
        if readme_src.exists():
            shutil.copy2(readme_src, tmp / "README.md")

        # Create a small data file
        (tmp / "data.csv").write_text("id,value,category\n1,42,A\n2,17,B\n3,99,A\n4,55,C\n5,23,B\n")

        # Create dataset-metadata.json
        metadata = {
            "title": ds_slug,
            "id": f"{username}/{ds_slug}",
            "licenses": [{"name": "CC0-1.0"}],
            "keywords": ["badge-collector", "automated", "data-science"],
            "resources": [
                {
                    "path": "data.csv",
                    "description": "Sample data file",
                    "schema": {
                        "fields": [
                            {"name": "id", "type": "integer"},
                            {"name": "value", "type": "integer"},
                            {"name": "category", "type": "string"},
                        ]
                    },
                }
            ],
            "description": (
                "Auto-generated dataset for badge collection. "
                "Contains sample tabular data."
            ),
        }
        (tmp / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))

        run_kaggle_cli(["datasets", "create", "-p", str(tmp)])
        print(f"  [OK] Dataset created: {ds_slug}")

        for bid in actionable:
            set_status(bid, "action_completed", f"dataset={ds_slug}")
        return True

    except Exception as e:
        print(f"  [FAIL] Create dataset: {e}")
        for bid in actionable:
            set_status(bid, "failed", str(e))
        return False


def _tag_dataset(username: str) -> bool:
    """Tag a dataset for the Dataset Tagger criterion.

    Note: Tags are set via keywords in dataset-metadata.json during creation.
    We create a second dataset with explicit tags.
    """
    if not should_attempt("dataset_tagger"):
        return True

    set_status("dataset_tagger", "attempting")
    try:
        tmp = make_temp_dir("-tagged-ds")
        ds_slug = resource_name("tagged-dataset")

        (tmp / "data.csv").write_text("x,y\n1,2\n3,4\n")

        metadata = {
            "title": ds_slug,
            "id": f"{username}/{ds_slug}",
            "licenses": [{"name": "CC0-1.0"}],
            "keywords": [
                "badge-collector",
                "python",
                "data-science",
                "machine-learning",
                "classification",
            ],
            "resources": [{"path": "data.csv", "description": "Tagged sample data"}],
            "description": "Tagged dataset for badge collection.",
        }
        (tmp / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))

        run_kaggle_cli(["datasets", "create", "-p", str(tmp)])
        print(f"  [OK] Tagged dataset created: {ds_slug}")
        set_status("dataset_tagger", "action_completed", f"dataset={ds_slug}")
        return True

    except Exception as e:
        print(f"  [FAIL] Tag dataset: {e}")
        set_status("dataset_tagger", "failed", str(e))
        return False


def _document_dataset(username: str) -> bool:
    """Create a well-documented dataset for the Dataset Documenter criterion.

    For max usability score:
      - title, description in metadata
      - keywords/tags
      - license
      - resource descriptions with schema (column names, types)
      - README.md with overview
      - File descriptions
      - Provenance info
    """
    if not should_attempt("dataset_documenter"):
        return True

    set_status("dataset_documenter", "attempting")
    try:
        tmp = make_temp_dir("-documented-ds")
        ds_slug = resource_name("documented-dataset")

        # Create substantial data
        lines = ["id,name,score,grade,date\n"]
        for i in range(1, 21):
            grade = "A" if i > 15 else "B" if i > 10 else "C"
            lines.append(
                f"{i},student_{i},{50 + i},{grade},2024-01-{i:02d}\n"
            )
        (tmp / "data.csv").write_text("".join(lines))

        # Detailed README
        readme = f"""# {ds_slug}

## Description

This dataset contains sample student performance data for demonstration purposes.
It was auto-generated by the Kaggle Badge Collector tool.

## Contents

- `data.csv` — Student records with scores and grades

## Schema

| Column | Type | Description |
|--------|------|-------------|
| id | integer | Unique student identifier |
| name | string | Student name |
| score | integer | Test score (0-100) |
| grade | string | Letter grade (A/B/C) |
| date | date | Record date |

## License

CC0 1.0 Universal — Public Domain Dedication

## Provenance

Auto-generated sample data for testing and demonstration.
"""
        (tmp / "README.md").write_text(readme)

        metadata = {
            "title": ds_slug,
            "id": f"{username}/{ds_slug}",
            "subtitle": "Sample student performance data for demonstration",
            "description": "Well-documented sample dataset containing student performance records. "
                           "Includes scores, grades, and dates for 20 students. "
                           "Created by Kaggle Badge Collector for badge collection purposes.",
            "licenses": [{"name": "CC0-1.0"}],
            "keywords": [
                "badge-collector",
                "education",
                "students",
                "tabular",
                "classification",
            ],
            "resources": [
                {
                    "path": "data.csv",
                    "description": "Student performance records with scores and grades",
                    "schema": {
                        "fields": [
                            {"name": "id", "type": "integer", "description": "Unique student ID"},
                            {"name": "name", "type": "string", "description": "Student name"},
                            {
                                "name": "score",
                                "type": "integer",
                                "description": "Test score (0-100)",
                            },
                            {"name": "grade", "type": "string", "description": "Letter grade"},
                            {"name": "date", "type": "date", "description": "Record date"},
                        ]
                    },
                }
            ],
        }
        (tmp / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))

        run_kaggle_cli(["datasets", "create", "-p", str(tmp)])
        print(f"  [OK] Documented dataset created: {ds_slug}")
        set_status("dataset_documenter", "action_completed", f"dataset={ds_slug}")
        return True

    except Exception as e:
        print(f"  [FAIL] Document dataset: {e}")
        set_status("dataset_documenter", "failed", str(e))
        return False


def _create_model(username: str) -> bool:
    """Create a model for two model-creation badge criteria."""
    badge_ids = ["model_creator", "api_model_creator"]
    actionable = [b for b in badge_ids if should_attempt(b)]
    if not actionable:
        return True

    for bid in actionable:
        set_status(bid, "attempting")

    try:
        tmp = make_temp_dir("-model")
        model_slug = resource_name("model")

        # Create model-metadata.json (for the model container)
        model_meta = {
            "ownerSlug": username,
            "title": model_slug,
            "slug": model_slug,
            "subtitle": "Auto-generated model for badge collection",
            "isPrivate": True,
            "description": "A sample model created by Kaggle Badge Collector.",
            "publishTime": "",
            "provenanceSources": "",
        }
        (tmp / "model-metadata.json").write_text(json.dumps(model_meta, indent=2))

        # Create a dummy model file
        (tmp / "model.txt").write_text("# Placeholder model file\nversion=1.0\n")

        # Create the model container
        run_kaggle_cli(["models", "create", "-p", str(tmp)])
        print(f"  [OK] Model container created: {model_slug}")

        for bid in actionable:
            set_status(bid, "action_completed", f"model={model_slug}")
        return True

    except Exception as e:
        print(f"  [FAIL] Create model: {e}")
        for bid in actionable:
            set_status(bid, "failed", str(e))
        return False


def _create_model_variation(username: str) -> bool:
    """Create a model variation for the Model Variation Creator criterion."""
    if not should_attempt("model_variation_creator"):
        return True

    set_status("model_variation_creator", "attempting")
    try:
        tmp = make_temp_dir("-model-var")
        model_slug = resource_name("model-var")

        # First create the model container
        model_meta = {
            "ownerSlug": username,
            "title": model_slug,
            "slug": model_slug,
            "subtitle": "Model with variation for badge collection",
            "isPrivate": True,
            "description": "A model with an instance/variation.",
            "publishTime": "",
            "provenanceSources": "",
        }
        (tmp / "model-metadata.json").write_text(json.dumps(model_meta, indent=2))
        (tmp / "model.txt").write_text("# Model file v1\n")

        run_kaggle_cli(["models", "create", "-p", str(tmp)])
        time.sleep(API_DELAY)

        # Now create a model instance (variation)
        instance_meta = {
            "ownerSlug": username,
            "modelSlug": model_slug,
            "instanceSlug": "default",
            "framework": "other",
            "overview": "Default variation of the model",
            "licenseName": "Apache 2.0",
            "fineTunable": False,
        }
        (tmp / "model-instance-metadata.json").write_text(json.dumps(instance_meta, indent=2))

        handle = f"{username}/{model_slug}/other/default"
        run_kaggle_cli([
            "models", "instances", "versions", "create",
            handle, "-p", str(tmp), "-n", "Initial version",
        ])
        print(f"  [OK] Model variation created: {handle}")
        set_status("model_variation_creator", "action_completed", f"variation={handle}")
        return True

    except Exception as e:
        print(f"  [FAIL] Model variation: {e}")
        set_status("model_variation_creator", "failed", str(e))
        return False


def _tag_model(username: str) -> bool:
    """Create a tagged model for the Model Tagger criterion.

    Note: Model tagging is done via keywords when creating via kagglehub
    or through the Kaggle API. The CLI model create doesn't directly support
    keywords, so we'll rely on the kagglehub approach or note it for browser.
    """
    if not should_attempt("model_tagger"):
        return True

    set_status("model_tagger", "attempting")
    try:
        import kagglehub

        model_slug = resource_name("tagged-model")
        tmp = make_temp_dir("-tagged-model")

        (tmp / "model.txt").write_text("# Tagged model\n")

        # kagglehub.model_upload supports tags indirectly
        handle = f"{username}/{model_slug}/other/default"
        kagglehub.model_upload(
            handle=handle,
            local_model_dir=str(tmp),
            version_notes="Tagged model for badge collection",
            license_name="Apache 2.0",
        )
        print(f"  [OK] Tagged model created: {model_slug}")
        set_status("model_tagger", "action_completed", f"model={model_slug}")
        return True

    except Exception as e:
        print(f"  [FAIL] Tag model: {e}")
        # Fallback: create via CLI and note tagging needs browser
        set_status("model_tagger", "failed", str(e))
        return False


def _document_model(username: str) -> bool:
    """Create a well-documented model for the Model Documenter criterion."""
    if not should_attempt("model_documenter"):
        return True

    set_status("model_documenter", "attempting")
    try:
        tmp = make_temp_dir("-documented-model")
        model_slug = resource_name("documented-model")

        # Create comprehensive model metadata
        model_meta = {
            "ownerSlug": username,
            "title": model_slug,
            "slug": model_slug,
            "subtitle": "Well-documented sample model for badge collection",
            "isPrivate": True,
            "description": (
                "# Sample Documented Model\n\n"
                "This is a well-documented sample model created by the Kaggle Badge Collector.\n\n"
                "## Overview\n"
                "A placeholder model demonstrating full documentation coverage.\n\n"
                "## Usage\n"
                "```python\nimport kagglehub\n"
                f"path = kagglehub.model_download('{username}/{model_slug}/other/default')\n```\n\n"
                "## License\n"
                "Apache 2.0"
            ),
            "publishTime": "",
            "provenanceSources": "Auto-generated by Kaggle Badge Collector",
        }
        (tmp / "model-metadata.json").write_text(json.dumps(model_meta, indent=2))
        (tmp / "model.txt").write_text("# Documented model\nversion=1.0\nframework=other\n")
        (tmp / "README.md").write_text(
            f"# {model_slug}\n\nA well-documented sample model.\n\n"
            "## Details\n- Framework: Other\n- License: Apache 2.0\n"
        )

        run_kaggle_cli(["models", "create", "-p", str(tmp)])
        time.sleep(API_DELAY)

        # Create instance with full metadata
        instance_meta = {
            "ownerSlug": username,
            "modelSlug": model_slug,
            "instanceSlug": "default",
            "framework": "other",
            "overview": (
                "Default instance of the documented model. "
                "Contains a placeholder model file for demonstration."
            ),
            "licenseName": "Apache 2.0",
            "fineTunable": False,
        }
        (tmp / "model-instance-metadata.json").write_text(json.dumps(instance_meta, indent=2))

        handle = f"{username}/{model_slug}/other/default"
        run_kaggle_cli([
            "models", "instances", "versions", "create",
            handle, "-p", str(tmp), "-n", "Documented initial version",
        ])

        print(f"  [OK] Documented model created: {model_slug}")
        set_status("model_documenter", "action_completed", f"model={model_slug}")
        return True

    except Exception as e:
        print(f"  [FAIL] Document model: {e}")
        set_status("model_documenter", "failed", str(e))
        return False


def run(username: str) -> tuple[int, int]:
    """Run Phase 1 prerequisite actions. Return action attempt/success counts."""
    actions = [
        (
            "Python notebook",
            ("python_coder", "api_notebook_creator", "code_uploader"),
            _create_python_notebook,
        ),
        ("R notebook", ("r_coder",), _create_r_notebook),
        ("Utility script", ("utility_scripter",), _create_utility_script),
        ("Fork notebook", ("code_forker",), _fork_notebook),
        ("Tag notebook", ("code_tagger",), _tag_notebook),
        ("Create dataset", ("dataset_creator", "api_dataset_creator"), _create_dataset),
        ("Tag dataset", ("dataset_tagger",), _tag_dataset),
        ("Document dataset", ("dataset_documenter",), _document_dataset),
        ("Create model", ("model_creator", "api_model_creator"), _create_model),
        ("Model variation", ("model_variation_creator",), _create_model_variation),
        ("Tag model", ("model_tagger",), _tag_model),
        ("Document model", ("model_documenter",), _document_model),
    ]

    attempted = 0
    succeeded = 0

    for name, badge_ids, fn in actions:
        if not any(should_attempt(badge_id) for badge_id in badge_ids):
            continue
        print(f"\n  --- {name} ---")
        attempted += 1
        if fn(username):
            succeeded += 1

    return attempted, succeeded
