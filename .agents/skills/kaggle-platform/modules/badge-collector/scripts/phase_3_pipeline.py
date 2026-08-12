"""Phase 3: Pipeline badges (~3 badges).

Performs prerequisite actions that require Kaggle Notebook execution and output:
  - Dataset Pipeline Creator (create dataset from notebook output)
  - Model Pipeline Creator (create model from notebook output)
  - R Markdown Coder (push and execute R Markdown as a Kaggle Notebook)

These badges require execution in the Kaggle Notebook environment,
so the workflow follows live logs until the execution stream closes.
"""

import json
import subprocess

from badge_tracker import set_status, should_attempt
from utils import (
    make_temp_dir,
    resource_name,
    run_kaggle_cli,
)


def _follow_kernel_logs(username: str, kernel_slug: str, timeout: int = 600) -> bool:
    """Follow live logs until the execution stream closes or times out."""
    full_slug = f"{username}/{kernel_slug}"
    try:
        result = run_kaggle_cli(
            ["kernels", "logs", "-f", full_slug],
            check=False,
            timeout=timeout,
            stream_output=True,
        )
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] Live logs did not close within {timeout}s")
        return False
    if result.returncode != 0:
        print("    [FAIL] Live log stream ended with an error")
        return False
    return True


def _dataset_pipeline(username: str) -> bool:
    """Create a notebook that outputs a dataset, then create a dataset from that output.

    Targets: Dataset Pipeline Creator.
    """
    if not should_attempt("dataset_pipeline_creator"):
        return True

    set_status("dataset_pipeline_creator", "attempting")
    try:
        tmp = make_temp_dir("-ds-pipeline")
        nb_slug = resource_name("ds-pipeline-nb")

        # Create a notebook that generates a CSV output
        notebook_content = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import pandas as pd\n",
                        "\n",
                        "# Generate output data\n",
                        "data = pd.DataFrame({\n",
                        "    'x': range(1, 11),\n",
                        "    'y': [i**2 for i in range(1, 11)],\n",
                        "    'label': ['even' if i%2==0 else 'odd' for i in range(1, 11)]\n",
                        "})\n",
                        "\n",
                        "# Save to output\n",
                        "data.to_csv('output_data.csv', index=False)\n",
                        "print(f'Output shape: {data.shape}')\n",
                        "print(data.head())\n",
                    ],
                }
            ],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.10.0"},
            },
            "nbformat": 4,
            "nbformat_minor": 4,
        }
        (tmp / "notebook.ipynb").write_text(json.dumps(notebook_content, indent=2))

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
            "keywords": ["badge-collector", "pipeline", "dataset"],
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        }
        (tmp / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))

        # Push the Kaggle Notebook
        run_kaggle_cli(["kernels", "push", "-p", str(tmp)])
        print(f"  [OK] Pipeline notebook pushed: {nb_slug}")
        print("  Waiting for Kaggle Notebook execution to complete...")

        # Follow the primary live SSE path until the execution stream closes.
        if not _follow_kernel_logs(username, nb_slug):
            print("  [FAIL] Notebook execution failed or timed out")
            set_status("dataset_pipeline_creator", "failed", "Kaggle Notebook execution failed")
            return False

        # Download notebook output
        output_dir = make_temp_dir("-ds-pipeline-output")
        run_kaggle_cli(
            [
                "kernels",
                "output",
                f"{username}/{nb_slug}",
                "--path",
                str(output_dir),
            ]
        )

        # Create a dataset from the output
        ds_slug = resource_name("pipeline-dataset")
        ds_metadata = {
            "title": ds_slug,
            "id": f"{username}/{ds_slug}",
            "licenses": [{"name": "CC0-1.0"}],
            "keywords": ["badge-collector", "pipeline"],
            "resources": [{"path": "output_data.csv", "description": "Pipeline output"}],
            "description": f"Dataset created from notebook output ({nb_slug})",
        }
        (output_dir / "dataset-metadata.json").write_text(json.dumps(ds_metadata, indent=2))

        run_kaggle_cli(["datasets", "create", "-p", str(output_dir)])
        print(f"  [OK] Pipeline dataset created: {ds_slug}")
        set_status(
            "dataset_pipeline_creator",
            "action_completed",
            f"dataset={ds_slug} from notebook={nb_slug}",
        )
        return True

    except Exception as e:
        print(f"  [FAIL] Dataset pipeline: {e}")
        set_status("dataset_pipeline_creator", "failed", str(e))
        return False


def _model_pipeline(username: str) -> bool:
    """Create a notebook that outputs a model, then create a model from that output.

    Targets: Model Pipeline Creator.
    """
    if not should_attempt("model_pipeline_creator"):
        return True

    set_status("model_pipeline_creator", "attempting")
    try:
        tmp = make_temp_dir("-model-pipeline")
        nb_slug = resource_name("model-pipeline-nb")

        # Create a notebook that trains and saves a simple model
        notebook_content = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import json\n",
                        "\n",
                        "# Create a simple 'model' (just config/weights as JSON)\n",
                        "model = {\n",
                        "    'type': 'linear',\n",
                        "    'weights': [0.5, -0.3, 0.8],\n",
                        "    'bias': 0.1,\n",
                        "    'features': ['x1', 'x2', 'x3']\n",
                        "}\n",
                        "\n",
                        "with open('model.json', 'w') as f:\n",
                        "    json.dump(model, f, indent=2)\n",
                        "\n",
                        "print('Model saved to model.json')\n",
                        "print(json.dumps(model, indent=2))\n",
                    ],
                }
            ],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.10.0"},
            },
            "nbformat": 4,
            "nbformat_minor": 4,
        }
        (tmp / "notebook.ipynb").write_text(json.dumps(notebook_content, indent=2))

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
            "keywords": ["badge-collector", "pipeline", "model"],
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        }
        (tmp / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))

        # Push notebook
        run_kaggle_cli(["kernels", "push", "-p", str(tmp)])
        print(f"  [OK] Model pipeline notebook pushed: {nb_slug}")
        print("  Waiting for Kaggle Notebook execution to complete...")

        if not _follow_kernel_logs(username, nb_slug):
            print("  [FAIL] Notebook execution failed or timed out")
            set_status("model_pipeline_creator", "failed", "Kaggle Notebook execution failed")
            return False

        # Download output
        output_dir = make_temp_dir("-model-pipeline-output")
        run_kaggle_cli(
            [
                "kernels",
                "output",
                f"{username}/{nb_slug}",
                "--path",
                str(output_dir),
            ]
        )

        # Create model from output
        model_slug = resource_name("pipeline-model")
        model_meta = {
            "ownerSlug": username,
            "title": model_slug,
            "slug": model_slug,
            "subtitle": f"Model from notebook pipeline ({nb_slug})",
            "isPrivate": True,
            "description": "Model created from notebook output via pipeline.",
            "publishTime": "",
            "provenanceSources": "",
        }
        (output_dir / "model-metadata.json").write_text(json.dumps(model_meta, indent=2))

        run_kaggle_cli(["models", "create", "-p", str(output_dir)])
        print(f"  [OK] Pipeline model created: {model_slug}")
        set_status(
            "model_pipeline_creator",
            "action_completed",
            f"model={model_slug} from notebook={nb_slug}",
        )
        return True

    except Exception as e:
        print(f"  [FAIL] Model pipeline: {e}")
        set_status("model_pipeline_creator", "failed", str(e))
        return False


def _r_markdown(username: str) -> bool:
    """Push an R notebook/script for the R Markdown Coder criterion.

    Note: kaggle CLI v1.8 doesn't support .Rmd files or 'rmarkdown' language
    (returns JSON parse error). We push an R script (language=r, kernel_type=script)
    which is the closest automatable equivalent.
    """
    if not should_attempt("r_markdown_coder"):
        return True

    set_status("r_markdown_coder", "attempting")
    try:
        tmp = make_temp_dir("-rmd")
        nb_slug = resource_name("r-markdown")

        # Create an R script (kaggle CLI v1.8 doesn't support .Rmd directly)
        r_content = """# Badge Collector R Markdown equivalent
# R script for Kaggle badge collection

library(datasets)

# Load and analyze iris data
data(iris)
cat("=== Iris Dataset Summary ===\\n")
print(summary(iris))

# Group means by species
cat("\\n=== Species Means ===\\n")
print(aggregate(. ~ Species, data = iris, FUN = mean))

# Generate output
cat("\\nR script completed successfully\\n")
"""
        (tmp / "script.R").write_text(r_content)

        metadata = {
            "id": f"{username}/{nb_slug}",
            "title": nb_slug,
            "code_file": "script.R",
            "language": "r",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "keywords": ["r", "rmarkdown", "data-analysis"],
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        }
        (tmp / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))

        run_kaggle_cli(["kernels", "push", "-p", str(tmp)])
        print(f"  [OK] R script pushed: {nb_slug}")
        set_status("r_markdown_coder", "action_completed", f"notebook={nb_slug}")
        return True

    except Exception as e:
        print(f"  [FAIL] R Markdown: {e}")
        set_status("r_markdown_coder", "failed", str(e))
        return False


def run(username: str) -> tuple[int, int]:
    """Run Phase 3 prerequisite actions. Return action attempt/success counts."""
    actions = [
        ("Dataset pipeline", "dataset_pipeline_creator", _dataset_pipeline),
        ("Model pipeline", "model_pipeline_creator", _model_pipeline),
        ("R Markdown", "r_markdown_coder", _r_markdown),
    ]

    attempted = 0
    succeeded = 0

    for name, badge_id, fn in actions:
        if not should_attempt(badge_id):
            continue
        print(f"\n  --- {name} ---")
        attempted += 1
        if fn(username):
            succeeded += 1

    return attempted, succeeded
