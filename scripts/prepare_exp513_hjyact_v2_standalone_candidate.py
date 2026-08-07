from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments/exp513_hjyact_v2_final_standalone_public_lb_audit"
OUTPUT = EXP_DIR / "exp513_hjyact_v2_final_standalone_public_lb_audit_compact_selfcontained_inference.py"
PARENT_GENERATOR = ROOT / "scripts/prepare_exp512_hjyact_v2_candidate.py"
EXPECTED_PARENT_GENERATOR_SHA256 = (
    "de18a4ecaa76c7c5be483d1c303239fb4afcb9a6dd7414aed62db5ee2110a57e"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_parent_generator():
    raw = PARENT_GENERATOR.read_bytes()
    observed = sha256_bytes(raw)
    if observed != EXPECTED_PARENT_GENERATOR_SHA256:
        raise RuntimeError(
            "exp512 generator SHA mismatch: "
            f"{observed} != {EXPECTED_PARENT_GENERATOR_SHA256}"
        )
    spec = importlib.util.spec_from_file_location("exp512_hjyact_v2_generator", PARENT_GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load parent generator: {PARENT_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEADER = '''\
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp513 hjyact-v2 final standalone Public LB audit
#
# This candidate dynamically regenerates the complete version-2 public final path
# and writes that prediction unchanged. It intentionally contains no downstream
# blend, cross-consumer cache, static visible prediction, or external submit call.

# %% [markdown]
# ## Contents
# 1. Imports, source identity, and mount-safe input audit
# 2. SP45 PF / Beam selector helpers
# 3. Ridge/PF anchor and deterministic candidate surface
# 4. Saved ridge artifact inference and runtime Ridge
# 5. Projection and learned trajectory replay
# 6. Guarded overlap and final hjyact-v2 layers
# 7. Standalone submission and reproducibility outputs

# %%
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from IPython.display import display

EXPERIMENT_NAME = "exp513_hjyact_v2_final_standalone_public_lb_audit"
SOURCE_KERNEL = "hjyact/ultimate-pf-config-strategy-a-reproducible-score"
SOURCE_KERNEL_ID = 128161011
SOURCE_VERSION = 2
SOURCE_RUN_ID = 337064157
SOURCE_PROFILE = "vp_balanced_modelpkg_005"
SOURCE_PULL_NOTEBOOK_SHA256 = "4b4879a6d427422c127a300e09dc763b71ea5e7878eb3639941c75753a23933c"
SOURCE_CODE_CELL_SHA256 = "ee93ce4c80c6490cbf2f9cfe518e8e3b54516c212aa813c4a045a64b4c126088"
SOURCE_VISIBLE_FINAL_SHA256 = "b192d3f348ae00680dc4df942b95cef5fd708c636a741f77dfb6b6e89b9ded4a"
VISIBLE_SAMPLE_ID_ORDER_SHA256 = "e6a2a380b8751443333064563fe94289055b95a739a3c8ac42d672df28a7e269"
GENERATOR_SHA256 = "__GENERATOR_SHA256__"
EXPERIMENT_STARTED_AT = time.time()

SOURCE_TRAINER_FILES = {
    "models/lightgbm-1": (
        "lgbmregressor_trainer_20260526182612.pkl",
        "5b8c34db51508138826c682d9dc0787557f6bccdb9fbb858d3fad27dc0a84a39",
    ),
    "models/lightgbm-2": (
        "lgbmregressor_trainer_20260526190415.pkl",
        "4384298a78736530f7b9b00a2908f9c8be478e5856b64cbc8f1f34a3d5889944",
    ),
    "models/lightgbm-3": (
        "lgbmregressor_trainer_20260526192806.pkl",
        "053d458f382aa1bdc2f22525e13d7ecb3c6b20a7637144eb8b708ef005cef5be",
    ),
    "models/catboost-1": (
        "catboostregressor_trainer_20260526193740.pkl",
        "5ea8b0705fae314bbd4194ee92c3eb5d2292ba761bd177b91ed898ac72ddf867",
    ),
    "models/catboost-2": (
        "catboostregressor_trainer_20260526194838.pkl",
        "3abcc44faebd1f374320e5939b19f6a5e911f9b2ac75f55054dd860e37aca4f4",
    ),
}

HJYACT_REQUIRED_INPUTS = {
    "koolbox": {
        "koolbox-0.1.3-py3-none-any.whl": "f654008252fe17463f27548b6a327926aa5451fdba52fe60fbb7639f5afc4bdc",
    },
    "ridge": {
        "data/train.csv": "68689ad02338581669f5392dc38741e6920a00537c3a5131820e723c2c9bcbcf",
        **{
            f"{directory}/{filename}": sha
            for directory, (filename, sha) in SOURCE_TRAINER_FILES.items()
        },
    },
    "learned": {
        "features.json": "ea9042f88cb3d8716b83e40c5c5ecb39f8bc8fcfeb52edb40d1871cd99496308",
        "lgb0.pkl": "a6451b3c42aeace6778e952b088287654946dca5412b818990d3f6b397e501e1",
        "lgb1.pkl": "4d61ab162af864bd3cfe37bde4421299746f28147faa3239e1ad14f15453f547",
        "lgb2.pkl": "1ee24121ecf455d904f3433bba49857d076fc33ca0b6b7a71ff9d538b3b8acf5",
    },
    "model_package": {
        "feature_builders/build_features.py": "0da2b08d900e1707b9136df1cd2b72b6defe43ec1dfba4bf67f87ba9a8f01e7c",
        "feature_builders/feature_columns.json": "95bf750725c3e8ccdb0363d7cb4bd51560b9cefb60662954ca9ef7d8776e8559",
        "feature_builders/rogii_feature_core.py": "ab877aad4537af30c1e1e8b1bace1c3d513c2a2228323bb57295c45bff636f0d",
        "metadata/model_package_manifest.json": "a19d5742e7ef2f43ca3661f6bcde059d13754299411577366cd278dee7025a1a",
        "stacking/blend_config.json": "1a67b9ecc509154d7413eff8ca5c98ab84caa6563efecd74bfa7fc1eab2e3814",
        "models/drift_ncc_xgb_alltrain.json": "4ece08e78da10579640428265bc060028efa0eb7aa27a4336da69933945bf607",
        "models/drift_ncc_catboost_alltrain.cbm": "954682b1775d773aa85d2697bfe19cf9196bea3d9758173359ab9dbeb60282b4",
        "models/drift_ncc_hgb_alltrain.joblib": "7b497ed9d2f09b4336eee819bfa6c42ed04d7c8ef44393d84e7e0f740fa45e40",
        "models/drift_ncc_lgb_alltrain.txt": "c7e5af52038188ce683e05f377a13114a61a874976fdd24d58b39f77f8532ed7",
        "models/sequence_tcn_tcn_residual.pt": "ad001b7e898fd50059a5a6b5ce13ba24aaf0b2a57b18a8bccb0163aeee880efd",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_content_sha(frame, columns):
    canonical = frame.loc[:, columns].copy()
    payload = canonical.to_csv(index=False, lineterminator="\\n").encode()
    return hashlib.sha256(payload).hexdigest()


def id_order_sha(ids) -> str:
    return hashlib.sha256("\\n".join(str(value) for value in ids).encode()).hexdigest()


def resolve_input_root(candidates, required):
    matches = []
    for candidate in candidates:
        root = Path(candidate)
        if root.is_dir() and all((root / relative).is_file() for relative in required):
            matches.append(root.resolve())
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise RuntimeError(f"expected one exact input root, got {unique}")
    return unique[0]


def resolve_competition_data_root() -> Path:
    candidates = (
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    )
    matches = [
        root.resolve()
        for root in candidates
        if (root / "train").is_dir()
        and (root / "test").is_dir()
        and (root / "sample_submission.csv").is_file()
    ]
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise RuntimeError(f"expected one competition data root, got {unique}")
    return unique[0]


def verify_required_files(root: Path, required):
    records = []
    for relative, expected in required.items():
        path = root / relative
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"input SHA mismatch: {path}: {observed} != {expected}")
        records.append(
            {
                "path": str(path),
                "relative_path": relative,
                "sha256": observed,
                "bytes": path.stat().st_size,
            }
        )
    return records


def verify_hjyact_inputs():
    roots = {
        "koolbox": resolve_input_root(
            [
                "/kaggle/input/datasets/phongnguyn23021656/koolbox-offline",
                "/kaggle/input/koolbox-offline",
            ],
            HJYACT_REQUIRED_INPUTS["koolbox"],
        ),
        "ridge": resolve_input_root(
            [
                "/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts",
                "/kaggle/input/wellbore-geology-prediction-artifacts",
            ],
            HJYACT_REQUIRED_INPUTS["ridge"],
        ),
        "learned": resolve_input_root(
            [
                "/kaggle/input/datasets/fleongg/rogii-claude-models-pub",
                "/kaggle/input/rogii-claude-models-pub",
            ],
            HJYACT_REQUIRED_INPUTS["learned"],
        ),
        "model_package": resolve_input_root(
            [
                "/kaggle/input/datasets/pilkwang/rogii-model-package",
                "/kaggle/input/rogii-model-package",
            ],
            HJYACT_REQUIRED_INPUTS["model_package"],
        ),
    }
    files = {
        name: verify_required_files(root, HJYACT_REQUIRED_INPUTS[name])
        for name, root in roots.items()
    }
    return {
        "roots": {name: str(root) for name, root in roots.items()},
        "files": files,
    }


COMPETITION_DATA_ROOT = str(resolve_competition_data_root())
HJYACT_INPUT_AUDIT = verify_hjyact_inputs()
RIDGE_ARTIFACT_ROOT = HJYACT_INPUT_AUDIT["roots"]["ridge"]
print("experiment:", EXPERIMENT_NAME)
print("source:", SOURCE_KERNEL, "version", SOURCE_VERSION, "run", SOURCE_RUN_ID)
print("profile:", SOURCE_PROFILE)
print("competition data root:", COMPETITION_DATA_ROOT)
print("ridge artifact root:", RIDGE_ARTIFACT_ROOT)
print("input roots:", HJYACT_INPUT_AUDIT["roots"])

'''


FINAL_ORCHESTRATION = '''\
WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
FINAL_SUBMISSION_PATH = WORKING_DIR / "submission.csv"
COMPONENT_COPY_PATH = WORKING_DIR / "hjyact_v2_final_component.csv"
INPUT_MANIFEST_PATH = WORKING_DIR / "exp513_input_manifest.json"
MODEL_MANIFEST_PATH = WORKING_DIR / "exp513_model_manifest.json"
REPRODUCIBILITY_MANIFEST_PATH = WORKING_DIR / "exp513_reproducibility_manifest.json"

if not FINAL_SUBMISSION_PATH.is_file():
    raise RuntimeError("complete hjyact-v2 final path did not produce submission.csv")
submission = pd.read_csv(FINAL_SUBMISSION_PATH, dtype={"id": str})
sample = pd.read_csv(Path(COMPETITION_DATA_ROOT) / "sample_submission.csv", dtype={"id": str})
if list(sample.columns) != ["id", "tvt"]:
    raise RuntimeError(f"unexpected dynamic sample schema: {list(sample.columns)}")
if list(submission.columns) != list(sample.columns):
    raise RuntimeError("standalone submission schema differs from dynamic sample")
if len(submission) != len(sample):
    raise RuntimeError("standalone submission row count differs from dynamic sample")
if not submission["id"].equals(sample["id"]):
    raise RuntimeError("standalone submission ID order differs from dynamic sample")
if submission["id"].duplicated().any():
    raise RuntimeError("standalone submission contains duplicate IDs")
if not np.isfinite(submission["tvt"].to_numpy(np.float64)).all():
    raise RuntimeError("standalone submission contains non-finite predictions")

sample_id_sha = id_order_sha(sample["id"])
submission_sha = sha256_file(FINAL_SUBMISSION_PATH)
submission_content_sha = dataframe_content_sha(submission, ["id", "tvt"])
visible_reference_checks = {
    "sample_id_order_sha256": sample_id_sha,
    "sample_id_order_match": sample_id_sha == VISIBLE_SAMPLE_ID_ORDER_SHA256,
    "submission_sha256": submission_sha,
}
if visible_reference_checks["sample_id_order_match"]:
    visible_reference_checks["submission_match"] = submission_sha == SOURCE_VISIBLE_FINAL_SHA256
    if not visible_reference_checks["submission_match"]:
        raise RuntimeError(
            "visible hjyact-v2 standalone parity failed: "
            f"{submission_sha} != {SOURCE_VISIBLE_FINAL_SHA256}"
        )
else:
    visible_reference_checks["submission_match"] = None

shutil.copyfile(FINAL_SUBMISSION_PATH, COMPONENT_COPY_PATH)
if sha256_file(COMPONENT_COPY_PATH) != submission_sha:
    raise RuntimeError("standalone component copy changed submission bytes")

input_manifest = {
    "experiment": EXPERIMENT_NAME,
    "competition_data_root": COMPETITION_DATA_ROOT,
    "competition_root_contract": ["train", "test", "sample_submission.csv"],
    "hjyact_input_audit": HJYACT_INPUT_AUDIT,
    "rows": int(len(sample)),
    "wells": int(sample["id"].str.rsplit("_", n=1).str[0].nunique()),
    "sample_id_order_sha256": sample_id_sha,
    "legacy_mount_fallback_used": str(COMPETITION_DATA_ROOT).startswith("/kaggle/input/competitions/"),
}
INPUT_MANIFEST_PATH.write_text(
    json.dumps(input_manifest, indent=2, sort_keys=True, default=str) + "\\n",
    encoding="utf-8",
)

model_manifest = {
    "experiment": EXPERIMENT_NAME,
    "route": "ensemble",
    "source_kernel": SOURCE_KERNEL,
    "source_version": SOURCE_VERSION,
    "source_run_id": SOURCE_RUN_ID,
    "source_profile": SOURCE_PROFILE,
    "new_booster_training_count": 0,
    "parent_control_retraining_count": 0,
    "runtime_ridge_fit_count": 5,
    "saved_model_file_count": 13,
    "contained_estimator_count": 33,
    "component_model_inventory": {
        "trainer_wrapper_files": 5,
        "trainer_fold_estimators": 25,
        "learned_trajectory_models": 3,
        "model_package_models": 5,
    },
    "learned_zero_filled_columns": list(globals().get("HJYACT_LEARNED_ZERO_FILLED_COLUMNS", [])),
    "exp413_model_files": 0,
    "downstream_blend": False,
    "cross_consumer_candidate_reuse": False,
}
MODEL_MANIFEST_PATH.write_text(
    json.dumps(model_manifest, indent=2, sort_keys=True, default=str) + "\\n",
    encoding="utf-8",
)

metrics = {
    "experiment": EXPERIMENT_NAME,
    "status": "kaggle_inference_completed_with_submission_output",
    "route": "ensemble",
    "standalone_component": "hjyact_v2_final",
    "source_kernel": SOURCE_KERNEL,
    "source_version": SOURCE_VERSION,
    "source_run_id": SOURCE_RUN_ID,
    "source_profile": SOURCE_PROFILE,
    "rows": int(len(submission)),
    "wells": int(input_manifest["wells"]),
    "new_booster_training_count": 0,
    "runtime_ridge_fit_count": 5,
    "external_submission_performed": False,
    "mount_safe_resolution": {
        "competition_root": COMPETITION_DATA_ROOT,
        "ridge_root": HJYACT_INPUT_AUDIT["roots"]["ridge"],
        "all_required_input_sha_verified": True,
    },
    "visible_reference_checks": visible_reference_checks,
    "prediction_stats": {
        "min": float(submission["tvt"].min()),
        "max": float(submission["tvt"].max()),
        "mean": float(submission["tvt"].mean()),
        "std": float(submission["tvt"].std()),
    },
    "sha256": {
        "generator": GENERATOR_SHA256,
        "source_pull_notebook": SOURCE_PULL_NOTEBOOK_SHA256,
        "source_code_cells": SOURCE_CODE_CELL_SHA256,
        "input_manifest": sha256_file(INPUT_MANIFEST_PATH),
        "model_manifest": sha256_file(MODEL_MANIFEST_PATH),
        "prediction_content": submission_content_sha,
        "submission": submission_sha,
    },
    "runtime_seconds": float(time.time() - EXPERIMENT_STARTED_AT),
}
(WORKING_DIR / "metrics.json").write_text(
    json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\\n",
    encoding="utf-8",
)
reproducibility_manifest = {
    **metrics,
    "input_manifest": input_manifest,
    "model_manifest": model_manifest,
    "hidden_test_policy": "dynamic raw/sample inputs only; visible hashes are post-hoc assertions",
    "exp512_failure_guards": {
        "competition_root_resolved_by_required_content": True,
        "ridge_root_taken_from_sha_audited_input_root": True,
        "legacy_direct_assignment_forbidden": True,
    },
}
REPRODUCIBILITY_MANIFEST_PATH.write_text(
    json.dumps(reproducibility_manifest, indent=2, sort_keys=True, default=str) + "\\n",
    encoding="utf-8",
)
print("exp513 standalone submission generated:", FINAL_SUBMISSION_PATH, submission.shape)
print("visible source parity:", visible_reference_checks)
print("external submission performed: False")
display(submission.head(20))
display(metrics)
'''


SECTION_BEFORE_CELL = {
    6: ("1. Imports, source identity, and mount-safe input audit",),
    11: ("2. SP45 PF / Beam selector helpers",),
    14: ("3. Ridge/PF anchor and deterministic candidate surface",),
    15: ("4. Saved ridge artifact inference and runtime Ridge",),
    30: ("5. Projection and learned trajectory replay",),
    48: ("6. Guarded overlap and final hjyact-v2 layers",),
}


def standalone_transform(parent, index: int, source: str) -> str:
    transformed = parent.transform_source_cell(index, source)
    if index == 43:
        replacements = {
            "HJYACT_SHARED_FEATURE_RUNTIME_SECONDS": "HJYACT_FEATURE_RUNTIME_SECONDS",
            "HJYACT_SHARED_FEATURE_FRAME": "_hjyact_feature_frame",
        }
        for old, new in replacements.items():
            transformed = transformed.replace(old, new)
        original = "sub, cv_final, _hjyact_feature_frame = main()\nsub.head()"
        replacement = '''\
sub, cv_final, _hjyact_feature_frame = main()
HJYACT_LEARNED_FEATURE_ROWS = int(len(_hjyact_feature_frame))
del _hjyact_feature_frame
sub.head()'''
        if transformed.count(original) != 1:
            raise RuntimeError("standalone learned-frame release transform failed")
        transformed = transformed.replace(original, replacement, 1)
    return transformed


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: prepare_exp513_hjyact_v2_standalone_candidate.py SOURCE.ipynb"
        )
    parent = load_parent_generator()
    source_path = Path(sys.argv[1]).resolve()
    raw = source_path.read_bytes()
    observed_notebook_sha = sha256_bytes(raw)
    if observed_notebook_sha != parent.EXPECTED_SOURCE_NOTEBOOK_SHA256:
        raise RuntimeError(
            "hjyact source notebook SHA mismatch: "
            f"{observed_notebook_sha} != {parent.EXPECTED_SOURCE_NOTEBOOK_SHA256}"
        )
    notebook = json.loads(raw)
    observed_code_sha = parent.source_code_sha(notebook)
    if observed_code_sha != parent.EXPECTED_SOURCE_CODE_SHA256:
        raise RuntimeError(
            f"hjyact code-cell SHA mismatch: {observed_code_sha} != "
            f"{parent.EXPECTED_SOURCE_CODE_SHA256}"
        )

    generator_sha = sha256_bytes(Path(__file__).read_bytes())
    pieces = [HEADER.replace("__GENERATOR_SHA256__", generator_sha)]
    cells = notebook["cells"]
    for index in parent.ACTIVE_SOURCE_CELLS:
        for section in SECTION_BEFORE_CELL.get(index, ()):
            pieces.append(parent.markdown_cell(section))
        source = standalone_transform(
            parent,
            index,
            "".join(cells[index].get("source", [])),
        )
        pieces.append(parent.code_cell(source))

    pieces.append(parent.markdown_cell("7. Standalone submission and reproducibility outputs"))
    pieces.append(parent.code_cell(FINAL_ORCHESTRATION))
    generated = "".join(pieces)
    forbidden = (
        "generate_dynamic_exp413_prediction",
        "CANDIDATE_REUSE_TRACKER",
        "0.50 * exp413",
        "exp413_component_submission",
    )
    found = [token for token in forbidden if token in generated]
    if found:
        raise RuntimeError(f"standalone output contains forbidden downstream tokens: {found}")
    OUTPUT.write_text(generated)
    print(
        f"generated {OUTPUT.relative_to(ROOT)} "
        f"({len(generated.splitlines())} lines, {len(generated.encode())} bytes, "
        f"sha256={sha256_bytes(generated.encode())})"
    )


if __name__ == "__main__":
    main()
