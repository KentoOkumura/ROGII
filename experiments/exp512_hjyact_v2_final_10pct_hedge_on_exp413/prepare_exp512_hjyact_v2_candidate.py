from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments/exp512_hjyact_v2_final_10pct_hedge_on_exp413"
OUTPUT = EXP_DIR / "exp512_hjyact_v2_final_10pct_hedge_on_exp413_compact_selfcontained_inference.py"
EXP413_RUNTIME = (
    ROOT
    / "experiments/exp510_exp413_exact_public_preoverride_hedge"
    / "exp510_exp413_hidden_safe_runtime.py"
)
EXPECTED_SOURCE_NOTEBOOK_SHA256 = (
    "4b4879a6d427422c127a300e09dc763b71ea5e7878eb3639941c75753a23933c"
)
EXPECTED_SOURCE_CODE_SHA256 = (
    "ee93ce4c80c6490cbf2f9cfe518e8e3b54516c212aa813c4a045a64b4c126088"
)
EXPECTED_EXP413_RUNTIME_SHA256 = (
    "0eea5b11d6852d0c2170914e993d6aba1204c02f2de00c3a809b299c028ef1dd"
)

ACTIVE_SOURCE_CELLS = (
    6,
    7,
    8,
    9,
    10,
    11,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    30,
    32,
    33,
    36,
    37,
    39,
    40,
    41,
    42,
    43,
    45,
    48,
    50,
    51,
    52,
    53,
)

SECTION_BEFORE_CELL = {
    6: ("1. Imports, source identity, and frozen profile",),
    11: ("2. SP45 PF / Beam selector helpers",),
    14: ("3. Ridge/PF anchor and shared deterministic candidate surface",),
    15: ("4. Saved ridge artifact inference and runtime Ridge",),
    30: ("5. Projection and learned trajectory replay",),
    48: ("6. Guarded overlap and final hjyact-v2 layers",),
}

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
# # exp512 hjyact-v2 final fixed 50/50 blend on exp413
#
# This candidate extracts only the active version-2 source path, regenerates both
# components on the dynamic sample, reuses the deterministic learned-replay surface,
# and writes `0.50 * exp413 + 0.50 * hjyact_v2_final` in float64.

# %% [markdown]
# ## Contents
# 1. Imports, source identity, and frozen profile
# 2. SP45 PF / Beam selector helpers
# 3. Ridge/PF anchor and shared deterministic candidate surface
# 4. Saved ridge artifact inference and runtime Ridge
# 5. Projection and learned trajectory replay
# 6. Guarded overlap and final hjyact-v2 layers
# 7. Embedded hidden-safe exp413 runtime
# 8. Shared-DAG manifest, fixed blend, and reproducibility outputs

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from IPython.display import display

SOURCE_KERNEL = "hjyact/ultimate-pf-config-strategy-a-reproducible-score"
SOURCE_KERNEL_ID = 128161011
SOURCE_VERSION = 2
SOURCE_RUN_ID = 337064157
SOURCE_PROFILE = "vp_balanced_modelpkg_005"
SOURCE_PULL_NOTEBOOK_SHA256 = "4b4879a6d427422c127a300e09dc763b71ea5e7878eb3639941c75753a23933c"
SOURCE_CODE_CELL_SHA256 = "ee93ce4c80c6490cbf2f9cfe518e8e3b54516c212aa813c4a045a64b4c126088"
SOURCE_VISIBLE_FINAL_SHA256 = "b192d3f348ae00680dc4df942b95cef5fd708c636a741f77dfb6b6e89b9ded4a"
EXP413_VISIBLE_REFERENCE_CONTENT_SHA256 = "875a1334ae3c90f841414f8f98d8877fb06234e17e0fd0b8d46385170a584dc4"
EXP413_VISIBLE_NUMERICAL_WITNESS_CONTENT_SHA256 = "3a9bbd1f7e6ab93189c90b4c9c0da9d6a2858746028e93b25fe2a10c7be68d87"
EXP413_VISIBLE_NUMERICAL_MAX_ABS_TOLERANCE_FT = 0.02
EXP413_VISIBLE_NUMERICAL_RMSE_TOLERANCE_FT = 0.001
EXP413_VISIBLE_NUMERICAL_WITNESS_MAX_ABS_FT = 0.0165
EXP413_VISIBLE_NUMERICAL_WITNESS_RMSE_FT = 0.0007530119954096194
EXP413_K16_PINNED_KAPPA = (
    1.271301613102484,
    0.7640828298998221,
    0.24211584390788987,
    0.009167473766386033,
    0.20431363370309996,
    -0.1534724474917627,
    0.29178856553015436,
    0.7166407555416966,
    0.7509050893373884,
    0.2782989780194615,
    -0.6039646412169122,
    0.4528285596106727,
)
EXP413_K16_RUNTIME_KAPPA_AUDIT_MAX_ABS = 1.0e-7
EXP413_WEIGHT = 0.50
HJYACT_WEIGHT = 0.50
FORMULA_TOLERANCE_FT = 1.0e-9
VISIBLE_SAMPLE_ID_ORDER_SHA256 = "e6a2a380b8751443333064563fe94289055b95a739a3c8ac42d672df28a7e269"

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


def fixed_blend(exp413_values, hjyact_values):
    exp413_array = np.asarray(exp413_values, dtype=np.float64)
    hjyact_array = np.asarray(hjyact_values, dtype=np.float64)
    if exp413_array.shape != hjyact_array.shape:
        raise ValueError("50/50 blend component shapes differ")
    return EXP413_WEIGHT * exp413_array + HJYACT_WEIGHT * hjyact_array


def sha256_gzip_content(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        records.append({"path": str(path), "relative_path": relative, "sha256": observed, "bytes": path.stat().st_size})
    return records


def verify_hjyact_inputs():
    roots = {
        "koolbox": resolve_input_root(
            ["/kaggle/input/datasets/phongnguyn23021656/koolbox-offline", "/kaggle/input/koolbox-offline"],
            HJYACT_REQUIRED_INPUTS["koolbox"],
        ),
        "ridge": resolve_input_root(
            ["/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts", "/kaggle/input/wellbore-geology-prediction-artifacts"],
            HJYACT_REQUIRED_INPUTS["ridge"],
        ),
        "learned": resolve_input_root(
            ["/kaggle/input/datasets/fleongg/rogii-claude-models-pub", "/kaggle/input/rogii-claude-models-pub"],
            HJYACT_REQUIRED_INPUTS["learned"],
        ),
        "model_package": resolve_input_root(
            ["/kaggle/input/datasets/pilkwang/rogii-model-package", "/kaggle/input/rogii-model-package"],
            HJYACT_REQUIRED_INPUTS["model_package"],
        ),
    }
    files = {name: verify_required_files(root, HJYACT_REQUIRED_INPUTS[name]) for name, root in roots.items()}
    return {"roots": {name: str(root) for name, root in roots.items()}, "files": files}


EXP413_K16_HASWELL_CHILD_CODE = r"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info

source_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
train_dir = Path(sys.argv[3])
test_dir = Path(sys.argv[4])
output_path = Path(sys.argv[5])
summary_path = Path(sys.argv[6])
pinned_kappa = np.asarray(json.loads(sys.argv[7]), dtype=np.float64)
audit_max_abs = float(sys.argv[8])

spec = importlib.util.spec_from_file_location("exp512_k16_haswell_source", source_path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import exp413 K16 source in Haswell subprocess")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

openblas = [item for item in threadpool_info() if item.get("internal_api") == "openblas"]
architectures = sorted({str(item.get("architecture")) for item in openblas})
if architectures != ["Haswell"]:
    raise RuntimeError(f"exp413 K16 subprocess did not load Haswell OpenBLAS: {architectures}")

source_config = json.loads(config_path.read_text())
params = module.params_from_config(source_config)
max_train = source_config.get("inference", {}).get("max_train_wells")
max_test = source_config.get("inference", {}).get("max_test_wells")
started = time.time()
train_wells = module.load_train_wells(
    train_dir,
    params,
    max_wells=int(max_train) if max_train is not None else None,
)
test_wells = module.load_test_wells(
    test_dir,
    params,
    max_wells=int(max_test) if max_test is not None else None,
)
if not train_wells or not test_wells:
    raise FileNotFoundError("exp413 K16 Haswell subprocess requires train/test wells")
fields = module.build_fields(train_wells, params)
runtime_fit_kappa = np.asarray(module.fit_kappa(train_wells, fields, params), dtype=np.float64)
if runtime_fit_kappa.shape != pinned_kappa.shape:
    raise RuntimeError("exp413 K16 runtime/pinned kappa shapes differ")
runtime_fit_vs_pinned_max_abs = float(np.max(np.abs(runtime_fit_kappa - pinned_kappa)))
if runtime_fit_vs_pinned_max_abs > audit_max_abs:
    raise RuntimeError(
        "exp413 K16 Haswell runtime fit differs from pinned train-only parameter: "
        f"{runtime_fit_vs_pinned_max_abs}"
    )

rows = []
well_summaries = []
for order, well in enumerate(test_wells, start=1):
    inference = module.predict_well(well, fields, pinned_kappa, params)
    row_idx = np.arange(well.s + 1, well.s + well.n + 1, dtype=np.int32)
    if len(row_idx) != len(inference.pred) or len(inference.pred) != len(inference.delta):
        raise ValueError(f"exp413 K16 row contract mismatch for well={well.wid}")
    rows.append(
        pd.DataFrame(
            {
                "id": [f"{well.wid}_{int(index)}" for index in row_idx],
                "well": str(well.wid),
                "well_row_idx": row_idx,
                "candidate_tvt": np.asarray(inference.pred, dtype=np.float32),
                "geometry_gr_delta": np.asarray(inference.delta, dtype=np.float32),
            }
        )
    )
    summary = dict(inference.summary)
    summary["order"] = order
    well_summaries.append(summary)

result = pd.concat(rows, ignore_index=True)
output_path.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(output_path, index=False, compression="zstd")
summary_path.write_text(
    json.dumps(
        {
            "train_wells": len(train_wells),
            "test_wells": len(test_wells),
            "rows": len(result),
            "kappa": [float(value) for value in pinned_kappa],
            "kappa_source": "pinned_train_only_exp413_visible_reference",
            "runtime_fit_kappa": [float(value) for value in runtime_fit_kappa],
            "runtime_fit_vs_pinned_max_abs": runtime_fit_vs_pinned_max_abs,
            "runtime_fit_audit_max_abs": audit_max_abs,
            "blas_architecture": architectures[0],
            "well_summaries": well_summaries,
            "runtime_seconds": time.time() - started,
        },
        indent=2,
        sort_keys=True,
    )
    + "\\n"
)
print(
    "exp413 K16 Haswell subprocess:",
    {"rows": len(result), "runtime_fit_vs_pinned_max_abs": runtime_fit_vs_pinned_max_abs},
    flush=True,
)
"""


def run_exp413_k16_haswell_subprocess(
    module,
    *,
    train_dir,
    test_dir,
    source_config,
    finalize_primitive_confidence,
    frame_content_sha256,
):
    import subprocess
    import sys

    work_dir = Path("/kaggle/working/artifacts/exp512_k16_haswell")
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / "source_config.json"
    output_path = work_dir / "exp226_k16_primitive.parquet"
    summary_path = work_dir / "summary.json"
    config_path.write_text(json.dumps(source_config, sort_keys=True) + "\\n")
    environment = dict(os.environ)
    environment["OPENBLAS_CORETYPE"] = "Haswell"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            EXP413_K16_HASWELL_CHILD_CODE,
            str(Path(module.__file__).resolve()),
            str(config_path),
            str(Path(train_dir).resolve()),
            str(Path(test_dir).resolve()),
            str(output_path),
            str(summary_path),
            json.dumps(EXP413_K16_PINNED_KAPPA),
            str(EXP413_K16_RUNTIME_KAPPA_AUDIT_MAX_ABS),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "exp413 K16 Haswell subprocess failed: "
            f"returncode={completed.returncode}\\n{completed.stderr[-8000:]}"
        )
    result = pd.read_parquet(output_path)
    result = finalize_primitive_confidence(result)
    if result.duplicated("id").any() or not np.isfinite(
        result[["candidate_tvt", "geometry_gr_delta"]].to_numpy()
    ).all():
        raise ValueError("exp413 K16 Haswell output violates duplicate/finite contract")
    summary = json.loads(summary_path.read_text())
    summary["prediction_and_confidence_content_sha256"] = frame_content_sha256(result)
    summary["execution_mode"] = "isolated_haswell_openblas_subprocess"
    return result, summary


HJYACT_INPUT_AUDIT = verify_hjyact_inputs()

'''


RIDGE_INPUT_CELL = '''\
_ridge_train_path = CFG.artifacts_path / "data" / "train.csv"
if not _ridge_train_path.is_file():
    raise FileNotFoundError(f"required source ridge feature table is missing: {_ridge_train_path}")
train_df = pd.read_csv(_ridge_train_path, low_memory=False)

test_paths = sorted((CFG.dataset_path / "test").glob("*__horizontal_well.csv"))
test_df = build_dataset(test_paths, is_train=False, label="test")

features = [c for c in train_df.columns if c not in {"well", "id", "target"}]
missing_test_features = [column for column in features if column not in test_df]
if missing_test_features:
    raise RuntimeError(f"source ridge test features are missing: {missing_test_features[:40]}")

X = train_df[features]
y = train_df["target"]
g = train_df["well"]
X_test = test_df[features]
'''


SOURCE_PARAMETER_CELL = '''\
ridge_params = {
    "random_state": 42,
    "alpha": 1.6602834637650032,
    "tol": 0.0005030247295617308,
    "positive": True,
    "fit_intercept": True,
}

pp_params = {"alpha": 1.0, "tau": 85, "w_pf": 0.09}
'''


LGB_LOAD_CELL = '''\
for i, save_path in enumerate(("models/lightgbm-1", "models/lightgbm-2", "models/lightgbm-3"), start=1):
    filename, expected_sha = SOURCE_TRAINER_FILES[save_path]
    trainer_path = CFG.artifacts_path / save_path / filename
    if sha256_file(trainer_path) != expected_sha:
        raise RuntimeError(f"source trainer SHA mismatch: {trainer_path}")
    print(f"Loading lightgbm-{i} from {trainer_path}...")
    trainer = joblib.load(trainer_path)
    key = f"lightgbm-{i}"
    oof_preds[key] = trainer.oof_preds
    test_preds[key] = trainer.predict(X_test)
    overall_scores[key] = trainer.overall_score
    fold_scores[key] = trainer.fold_scores
'''


CB_LOAD_CELL = '''\
for i, save_path in enumerate(("models/catboost-1", "models/catboost-2"), start=1):
    filename, expected_sha = SOURCE_TRAINER_FILES[save_path]
    trainer_path = CFG.artifacts_path / save_path / filename
    if sha256_file(trainer_path) != expected_sha:
        raise RuntimeError(f"source trainer SHA mismatch: {trainer_path}")
    print(f"Loading catboost-{i} from {trainer_path}...")
    trainer = joblib.load(trainer_path)
    key = f"catboost-{i}"
    oof_preds[key] = trainer.oof_preds
    test_preds[key] = trainer.predict(X_test)
    overall_scores[key] = trainer.overall_score
    fold_scores[key] = trainer.fold_scores
'''


LEARNED_MAIN = '''\
HJYACT_SHARED_FEATURE_RUNTIME_SECONDS = None


def main():
    import json
    import joblib

    t0 = time.time()
    train_wids = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA / "train").glob("*__horizontal_well.csv"))
    test_wids = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA / "test").glob("*__horizontal_well.csv"))
    if CFG.N_TRAIN_WELLS:
        train_wids = train_wids[: CFG.N_TRAIN_WELLS]
    print(f"train wells: {len(train_wids)} | test wells: {len(test_wids)}")
    init_imputers(train_wids)

    print("building lik-PF + shared deterministic features (test)...", flush=True)
    likpf_test = build_likpf(test_wids, "test")
    shared_started = time.time()
    shared = build_features(test_wids, "test", is_train=False)
    globals()["HJYACT_SHARED_FEATURE_RUNTIME_SECONDS"] = time.time() - shared_started
    test_df = add_likpf_features(shared, likpf_test).reset_index(drop=True)

    models_dir = _find_models()
    if models_dir is None:
        raise FileNotFoundError("SHA-pinned learned trajectory models are required; inference-time training is forbidden")
    expected_root = Path(HJYACT_INPUT_AUDIT["roots"]["learned"])
    if models_dir.resolve() != expected_root.resolve():
        raise RuntimeError(f"learned model root differs from the pinned dataset: {models_dir}")
    feats = json.loads((models_dir / "features.json").read_text())
    models = [joblib.load(models_dir / f"lgb{index}.pkl") for index in range(3)]
    zero_filled = [column for column in feats if column not in test_df]
    for column in zero_filled:
        test_df[column] = 0.0
    matrix = test_df[feats].to_numpy(np.float32)
    if not np.isfinite(matrix).all():
        raise RuntimeError("learned trajectory feature matrix contains non-finite values")
    meta_test = np.mean([model.predict(matrix) for model in models], axis=0)
    test_pred = make_prediction(test_df, meta_test, None)
    sample = pd.read_csv(CFG.DATA / "sample_submission.csv", dtype={"id": str})
    mapping = dict(zip(test_df["id"].astype(str), test_pred, strict=True))
    sample["tvt"] = sample["id"].map(mapping)
    if sample["tvt"].isna().any() or not np.isfinite(sample["tvt"]).all():
        raise RuntimeError("learned trajectory does not cover the dynamic sample exactly")
    sample.to_csv(CFG.OUT / "submission.csv", index=False)
    globals()["HJYACT_LEARNED_ZERO_FILLED_COLUMNS"] = zero_filled
    print(f"submission.csv written ({len(sample)} rows) in {time.time() - t0:.0f}s")
    return sample, None, test_df


sub, cv_final, HJYACT_SHARED_FEATURE_FRAME = main()
sub.head()
'''


CANDIDATE_REUSE_RUNTIME = '''\
SHARED_NODE_COLUMNS = {
    "raw_well_and_typewell_alignment": ["id", "well", "last_known_tvt"],
    "learned_replay_beam7_bank": [
        "beam_cons_d", "beam_loose_d", "beam_vcons_d", "beam_sm5_d",
        "beam_vloose_d", "beam_mid_d", "beam_stiff_d", "beam_mean_d",
        "beam_std_d", "beam_med_d",
    ],
    "multiscale_ncc_bank": [
        "sc8_d", "sc8_sc", "sc15_d", "sc15_sc", "sc25_d", "sc25_sc",
        "sc_cons_d", "sc_ens_d", "sc_trust", "hyb_d",
    ],
    "formation_dense_geometry_bank": [
        column
        for column in HJYACT_SHARED_FEATURE_FRAME.columns
        if column.startswith(("tvtF", "bw_", "bww_", "bw50_", "frm_rmse_", "form_", "dense_", "spatial_"))
        or column in {"tvt_dense_d", "tvt_densew_d", "tvt_dense50_d"}
    ],
    "deterministic_gr_geometry_feature_block": [
        column
        for column in HJYACT_SHARED_FEATURE_FRAME.columns
        if column not in {"target"}
        and not column.startswith(("pf_", "likpf_", "tdpf"))
    ],
}


class CandidateReuseTracker:
    def __init__(self, frame):
        self.records = []
        self.consumer_hits = []
        for well, group in frame.groupby("well", sort=False):
            ordered = group.reset_index(drop=True)
            row_sha = id_order_sha(ordered["id"].astype(str))
            raw_paths = [
                CFG.DATA / "test" / f"{well}__horizontal_well.csv",
                CFG.DATA / "test" / f"{well}__typewell.csv",
            ]
            raw_sha = hashlib.sha256("".join(sha256_file(path) for path in raw_paths).encode()).hexdigest()
            for node, configured_columns in SHARED_NODE_COLUMNS.items():
                columns = [column for column in configured_columns if column in ordered]
                if not columns:
                    raise RuntimeError(f"shared node {node} has no columns for well {well}")
                definition_sha = hashlib.sha256(f"{SOURCE_CODE_CELL_SHA256}:{node}".encode()).hexdigest()
                parameter_sha = hashlib.sha256(json.dumps({"node": node, "source_profile": SOURCE_PROFILE}, sort_keys=True).encode()).hexdigest()
                content_sha = dataframe_content_sha(ordered, columns)
                fingerprint = hashlib.sha256(
                    f"{definition_sha}:{raw_sha}:{parameter_sha}:none:float32:{row_sha}".encode()
                ).hexdigest()
                self.records.append(
                    {
                        "well": str(well),
                        "node": node,
                        "fingerprint": fingerprint,
                        "definition_sha256": definition_sha,
                        "input_content_sha256": raw_sha,
                        "parameter_sha256": parameter_sha,
                        "seed_policy": "none",
                        "dtype": "float32",
                        "row_order_sha256": row_sha,
                        "content_sha256": content_sha,
                        "columns": columns,
                        "consumers": ["hjyact_learned_trajectory", "exp413_candidate_regeneration"],
                        "generation_count": 1,
                        "cache_hit_count": 0,
                    }
                )

    def mark_exp413_hit(self, well):
        matched = [record for record in self.records if record["well"] == str(well)]
        if len(matched) != len(SHARED_NODE_COLUMNS):
            raise RuntimeError(f"shared-node tracker coverage mismatch for well {well}")
        for record in matched:
            record["cache_hit_count"] += 1
        self.consumer_hits.append({"well": str(well), "consumer": "exp413_candidate_regeneration"})

    def manifest(self):
        if any(record["generation_count"] != 1 or record["cache_hit_count"] != 1 for record in self.records):
            raise RuntimeError("shared candidate DAG generation/hit count contract failed")
        return {
            "source_code_cell_sha256": SOURCE_CODE_CELL_SHA256,
            "cache_scope": "process_local_per_dynamic_sample",
            "storage": "immutable_in_memory",
            "fallback_to_duplicate_generation": False,
            "shared_frame_generation_seconds": float(HJYACT_SHARED_FEATURE_RUNTIME_SECONDS),
            "records": self.records,
            "consumer_hits": self.consumer_hits,
        }


CANDIDATE_REUSE_TRACKER = CandidateReuseTracker(HJYACT_SHARED_FEATURE_FRAME)
'''


EXP413_SHARED_FRAME_BLOCK = '''\
    if shared_deterministic_frame is None or reuse_tracker is None:
        raise RuntimeError("exp413 requires the in-memory hjyact deterministic candidate frame")
    pf_frame = shared_deterministic_frame.copy(deep=True).reset_index(drop=True)
    pf_frame["id"] = pf_frame["id"].astype(str)
    pf_frame["well"] = pf_frame["well"].astype(str)
    route_started = time.time()
    test_wells = replay_list_test_wells()
    if set(pf_frame["well"]) != set(test_wells):
        raise ValueError("shared deterministic frame well set differs from dynamic raw test")

    likpf_columns = [column for column in pf_frame if column.startswith("likpf_")]
    pf_frame = pf_frame.drop(columns=likpf_columns)
    id_to_position = {value: index for index, value in enumerate(pf_frame["id"].astype(str))}
    route_pf_records = []
    for well in test_wells:
        horizontal, typewell = replay_source.load_well(well, "test")
        typewell = typewell.sort_values("TVT")
        known = horizontal["TVT_input"].notna()
        evaluation = horizontal[~known]
        expected_ids = [f"{well}_{int(index)}" for index in evaluation.index]
        if not expected_ids:
            continue
        if any(value not in id_to_position for value in expected_ids):
            raise ValueError(f"shared deterministic frame is missing exp413 IDs for well {well}")
        positions = np.asarray([id_to_position[value] for value in expected_ids], dtype=np.int64)
        ordered = pf_frame.iloc[positions].copy().reset_index(drop=True)
        if ordered["id"].astype(str).tolist() != expected_ids:
            raise ValueError(f"shared deterministic row order differs for well {well}")

        tw_tvt = typewell["TVT"].to_numpy(np.float32)
        tw_gr = typewell["GR"].to_numpy(np.float32)
        pf_ancc, pf_ancc_std = replay_source.run_pf_ancc(
            horizontal,
            tw_tvt,
            tw_gr,
            seed=replay_stable_seed("pf_ancc", well),
        )
        pf_z, _ = replay_source.run_pf_z(
            horizontal,
            tw_tvt,
            tw_gr,
            seed=replay_stable_seed("pf_z", well),
        )
        if len(pf_ancc) != len(expected_ids):
            raise ValueError(f"exp413 pf_ancc row mismatch for well {well}")
        last_tvt = ordered["last_known_tvt"].to_numpy(np.float32)
        if not np.all(last_tvt == last_tvt[0]):
            raise ValueError(f"last_known_tvt must be constant within well {well}")
        has_z = len(pf_z) == len(pf_ancc) and np.isfinite(pf_z).all()
        pf_z_use = pf_z.astype(np.float32) if has_z else last_tvt.copy()
        updates = {
            "pf_ancc": pf_ancc.astype(np.float32),
            "pf_ancc_std": pf_ancc_std.astype(np.float32),
            "pf_ancc_delta": (pf_ancc - last_tvt).astype(np.float32),
            "pf_z": pf_z_use,
            "pf_z_delta": (pf_z_use - last_tvt).astype(np.float32),
            "pf_vs_z": (pf_ancc - pf_z_use).astype(np.float32),
            "pf_vs_spatial": (pf_ancc - ordered["tvtF_ANCC"].to_numpy(np.float32)).astype(np.float32),
            "pf_vs_dense": (pf_ancc - (last_tvt + ordered["tvt_dense_d"].to_numpy(np.float32))).astype(np.float32),
        }
        beam_abs = [
            last_tvt + ordered[f"beam_{tag}_d"].to_numpy(np.float32)
            for _, _, _, _, tag in replay_source.BEAMS
        ]
        shared_abs = [
            last_tvt + ordered[column].to_numpy(np.float32)
            for column in ("sc8_d", "sc15_d", "sc25_d", "sc_ens_d")
        ]
        shared_abs.extend(
            [
                ordered["tvtF_ANCC"].to_numpy(np.float32),
                last_tvt + ordered["tvt_dense_d"].to_numpy(np.float32),
            ]
        )
        signal_matrix = np.stack([pf_ancc.astype(np.float32), *beam_abs, *shared_abs], axis=1)
        updates["sig_std"] = signal_matrix.std(axis=1).astype(np.float32)
        updates["sig_mean_d"] = (signal_matrix.mean(axis=1) - last_tvt).astype(np.float32)
        gr_full = horizontal["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
        hidden_gr = gr_full.loc[evaluation.index].to_numpy(np.float32)
        for offset in replay_source.PF_OFFS:
            updates[f"tdpf{int(offset)}"] = (
                hidden_gr - np.interp(pf_ancc + offset, tw_tvt, tw_gr).astype(np.float32)
            ).astype(np.float32)
        for column, values in updates.items():
            pf_frame.loc[positions, column] = np.asarray(values, dtype=np.float32)
        reuse_tracker.mark_exp413_hit(well)
        route_pf_records.append(
            {
                "well": str(well),
                "rows": len(expected_ids),
                "pf_ancc_seed": int(replay_stable_seed("pf_ancc", well)),
                "pf_z_seed": int(replay_stable_seed("pf_z", well)),
            }
        )

    likpf_test = replay_source.build_likpf(test_wells, "test")
    pf_frame = replay_source.add_likpf_features(pf_frame, likpf_test).reset_index(drop=True)
    pf_meta = {
        "shared_deterministic_dag_reused": True,
        "test_wells": len(test_wells),
        "test_rows": len(pf_frame),
        "test_likpf_rows": len(likpf_test),
        "route_specific_pf_records": route_pf_records,
        "elapsed_feature_seconds": round(time.time() - route_started, 3),
    }
'''


FINAL_ORCHESTRATION = '''\
WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
FINAL_SUBMISSION_PATH = WORKING_DIR / "submission.csv"
HJYACT_COMPONENT_PATH = WORKING_DIR / "hjyact_v2_final_submission.csv"
EXP413_COMPONENT_PATH = WORKING_DIR / "exp413_component_submission.csv"

source_submission_path = Path(CFG.OUT) / "submission.csv"
if not source_submission_path.is_file():
    raise RuntimeError("hjyact-v2 final source path did not produce submission.csv")
hjyact_component = pd.read_csv(source_submission_path, dtype={"id": str})
sample = pd.read_csv(CFG.DATA / "sample_submission.csv", dtype={"id": str})
if list(sample.columns) != ["id", "tvt"]:
    raise RuntimeError(f"unexpected dynamic sample schema: {list(sample.columns)}")
if list(hjyact_component.columns) != list(sample.columns):
    raise RuntimeError("hjyact component schema differs from dynamic sample")
if not hjyact_component["id"].equals(sample["id"]):
    raise RuntimeError("hjyact component ID order differs from dynamic sample")
if hjyact_component["id"].duplicated().any() or not np.isfinite(hjyact_component["tvt"]).all():
    raise RuntimeError("hjyact component duplicate/finite contract failed")

sample_id_sha = id_order_sha(sample["id"])
visible_reference_checks = {"sample_id_order_match": sample_id_sha == VISIBLE_SAMPLE_ID_ORDER_SHA256}
if visible_reference_checks["sample_id_order_match"]:
    observed_hjyact_sha = sha256_file(source_submission_path)
    visible_reference_checks["hjyact_submission_sha256"] = observed_hjyact_sha
    visible_reference_checks["hjyact_submission_match"] = observed_hjyact_sha == SOURCE_VISIBLE_FINAL_SHA256
    if not visible_reference_checks["hjyact_submission_match"]:
        raise RuntimeError(
            f"visible hjyact-v2 parity failed: {observed_hjyact_sha} != {SOURCE_VISIBLE_FINAL_SHA256}"
        )
else:
    visible_reference_checks["hjyact_submission_match"] = None

if HJYACT_COMPONENT_PATH.exists():
    HJYACT_COMPONENT_PATH.unlink()
shutil.move(str(source_submission_path), HJYACT_COMPONENT_PATH)
if FINAL_SUBMISSION_PATH.exists():
    raise RuntimeError("submission.csv must be absent before exp413 component regeneration")

exp413_predictions_memory, exp413_metrics, exp413_prediction_path = generate_dynamic_exp413_prediction(
    shared_deterministic_frame=HJYACT_SHARED_FEATURE_FRAME,
    reuse_tracker=CANDIDATE_REUSE_TRACKER,
)
if not exp413_prediction_path.is_file():
    raise RuntimeError("exp413 did not produce its declared prediction artifact")
exp413_predictions = pd.read_csv(exp413_prediction_path, compression="gzip", dtype={"id": str})
if "pred_tvt" not in exp413_predictions:
    raise RuntimeError("exp413 prediction artifact is missing pred_tvt")
if len(exp413_predictions_memory) != len(exp413_predictions):
    raise RuntimeError("exp413 returned frame and CSV boundary row counts differ")

if not FINAL_SUBMISSION_PATH.is_file():
    raise RuntimeError("exp413 did not produce its component submission.csv")
exp413_component_written = pd.read_csv(FINAL_SUBMISSION_PATH, dtype={"id": str})
if not exp413_component_written["id"].equals(sample["id"]):
    raise RuntimeError("exp413 component submission ID order differs from dynamic sample")
if EXP413_COMPONENT_PATH.exists():
    EXP413_COMPONENT_PATH.unlink()
shutil.move(str(FINAL_SUBMISSION_PATH), EXP413_COMPONENT_PATH)

component_frame = sample[["id"]].merge(
    hjyact_component.rename(columns={"tvt": "hjyact_tvt"}),
    on="id",
    how="left",
    validate="one_to_one",
).merge(
    exp413_predictions[["id", "pred_tvt"]].rename(columns={"pred_tvt": "exp413_tvt"}),
    on="id",
    how="left",
    validate="one_to_one",
)
if not component_frame["id"].equals(sample["id"]):
    raise RuntimeError("component merge changed dynamic sample ID order")
if component_frame[["exp413_tvt", "hjyact_tvt"]].isna().any().any():
    raise RuntimeError("component merge did not cover the dynamic sample exactly")
component_values = component_frame[["exp413_tvt", "hjyact_tvt"]].to_numpy(np.float64)
if not np.isfinite(component_values).all():
    raise RuntimeError("component predictions contain non-finite values")

exp413_written_aligned = sample[["id"]].merge(
    exp413_component_written.rename(columns={"tvt": "written_exp413_tvt"}),
    on="id",
    how="left",
    validate="one_to_one",
)
exp413_boundary_max_abs = float(
    np.max(np.abs(exp413_written_aligned["written_exp413_tvt"].to_numpy(np.float64) - component_frame["exp413_tvt"].to_numpy(np.float64)))
)
if exp413_boundary_max_abs > FORMULA_TOLERANCE_FT:
    raise RuntimeError(f"exp413 CSV boundary parity failed: {exp413_boundary_max_abs}")

blend_values = fixed_blend(
    component_frame["exp413_tvt"],
    component_frame["hjyact_tvt"],
)
submission = sample[["id"]].copy()
submission["tvt"] = blend_values
formula_expected = 0.50 * component_values[:, 0] + 0.50 * component_values[:, 1]
formula_max_abs = float(np.max(np.abs(submission["tvt"].to_numpy(np.float64) - formula_expected)))
if formula_max_abs > FORMULA_TOLERANCE_FT:
    raise RuntimeError(f"fixed 50/50 formula parity failed: {formula_max_abs}")
if submission["id"].duplicated().any() or not np.isfinite(submission["tvt"]).all():
    raise RuntimeError("final submission duplicate/finite contract failed")
if not submission["id"].equals(sample["id"]):
    raise RuntimeError("final submission ID order differs from dynamic sample")
submission.to_csv(FINAL_SUBMISSION_PATH, index=False)

if visible_reference_checks["sample_id_order_match"]:
    observed_exp413_content_sha = sha256_gzip_content(exp413_prediction_path)
    visible_reference_checks["exp413_prediction_content_sha256"] = observed_exp413_content_sha
    visible_reference_checks["exp413_prediction_exact_match"] = (
        observed_exp413_content_sha == EXP413_VISIBLE_REFERENCE_CONTENT_SHA256
    )
    visible_reference_checks["exp413_prediction_numerical_witness_match"] = (
        observed_exp413_content_sha == EXP413_VISIBLE_NUMERICAL_WITNESS_CONTENT_SHA256
    )
    if visible_reference_checks["exp413_prediction_exact_match"]:
        exp413_parity_mode = "exact_reference_content_sha"
        exp413_reference_max_abs_ft = 0.0
        exp413_reference_rmse_ft = 0.0
    elif visible_reference_checks["exp413_prediction_numerical_witness_match"]:
        exp413_parity_mode = "preaudited_platform_numerical_tolerance_witness"
        exp413_reference_max_abs_ft = EXP413_VISIBLE_NUMERICAL_WITNESS_MAX_ABS_FT
        exp413_reference_rmse_ft = EXP413_VISIBLE_NUMERICAL_WITNESS_RMSE_FT
    else:
        raise RuntimeError(
            "visible exp413 regenerated prediction is neither the exact reference nor the "
            "pre-audited numerical tolerance witness: "
            f"{observed_exp413_content_sha}"
        )
    visible_reference_checks["exp413_prediction_reference_max_abs_ft"] = (
        exp413_reference_max_abs_ft
    )
    visible_reference_checks["exp413_prediction_reference_rmse_ft"] = exp413_reference_rmse_ft
    visible_reference_checks["exp413_prediction_numerical_max_abs_tolerance_ft"] = (
        EXP413_VISIBLE_NUMERICAL_MAX_ABS_TOLERANCE_FT
    )
    visible_reference_checks["exp413_prediction_numerical_rmse_tolerance_ft"] = (
        EXP413_VISIBLE_NUMERICAL_RMSE_TOLERANCE_FT
    )
    visible_reference_checks["exp413_prediction_parity_mode"] = exp413_parity_mode
    visible_reference_checks["exp413_prediction_match"] = bool(
        exp413_reference_max_abs_ft <= EXP413_VISIBLE_NUMERICAL_MAX_ABS_TOLERANCE_FT
        and exp413_reference_rmse_ft <= EXP413_VISIBLE_NUMERICAL_RMSE_TOLERANCE_FT
    )
    if not visible_reference_checks["exp413_prediction_match"]:
        raise RuntimeError(
            "visible exp413 numerical tolerance failed: "
            f"max_abs={exp413_reference_max_abs_ft} > "
            f"{EXP413_VISIBLE_NUMERICAL_MAX_ABS_TOLERANCE_FT} or "
            f"rmse={exp413_reference_rmse_ft} > "
            f"{EXP413_VISIBLE_NUMERICAL_RMSE_TOLERANCE_FT}"
        )
else:
    visible_reference_checks["exp413_prediction_exact_match"] = None
    visible_reference_checks["exp413_prediction_numerical_witness_match"] = None
    visible_reference_checks["exp413_prediction_parity_mode"] = "hidden_dynamic_no_visible_gate"
    visible_reference_checks["exp413_prediction_match"] = None

component_frame["blend_tvt"] = submission["tvt"].to_numpy(np.float64)
component_frame.to_csv(WORKING_DIR / "exp512_component_readout.csv", index=False)
reuse_manifest = CANDIDATE_REUSE_TRACKER.manifest()
(WORKING_DIR / "candidate_reuse_manifest.json").write_text(
    json.dumps(reuse_manifest, indent=2, sort_keys=True, default=str) + "\\n",
    encoding="utf-8",
)

model_manifest = {
    "experiment": "exp512_hjyact_v2_final_10pct_hedge_on_exp413",
    "legacy_directory_suffix": "10pct_hedge",
    "actual_formula": "0.50 * exp413 + 0.50 * hjyact_v2_final",
    "route": "ensemble",
    "new_booster_training_count": 0,
    "runtime_ridge_fit_count": 5,
    "saved_model_file_count": 88,
    "contained_estimator_count": 108,
    "component_model_inventory": {
        "exp413": {"parent_selectors": 40, "signed_selectors": 20, "tvt_models": 15},
        "hjyact": {
            "trainer_wrapper_files": 5,
            "trainer_fold_estimators": 25,
            "learned_trajectory_models": 3,
            "model_package_models": 5,
        },
    },
    "hjyact_input_audit": HJYACT_INPUT_AUDIT,
    "exp413_metrics": exp413_metrics,
}
(WORKING_DIR / "exp512_model_manifest.json").write_text(
    json.dumps(model_manifest, indent=2, sort_keys=True, default=str) + "\\n",
    encoding="utf-8",
)

metrics = {
    "experiment": "exp512_hjyact_v2_final_10pct_hedge_on_exp413",
    "status": "kaggle_inference_completed_with_submission_output",
    "route": "ensemble",
    "weights": {"exp413": EXP413_WEIGHT, "hjyact_v2_final": HJYACT_WEIGHT},
    "formula": "0.50 * exp413 + 0.50 * hjyact_v2_final",
    "formula_tolerance_ft": FORMULA_TOLERANCE_FT,
    "formula_max_abs_error_ft": formula_max_abs,
    "exp413_csv_boundary_max_abs_error_ft": exp413_boundary_max_abs,
    "exp413_visible_numerical_tolerance_ft": {
        "max_abs": EXP413_VISIBLE_NUMERICAL_MAX_ABS_TOLERANCE_FT,
        "rmse": EXP413_VISIBLE_NUMERICAL_RMSE_TOLERANCE_FT,
    },
    "rows": int(len(submission)),
    "wells": int(HJYACT_SHARED_FEATURE_FRAME["well"].nunique()),
    "new_booster_training_count": 0,
    "runtime_ridge_fit_count": 5,
    "submission_file_generated": True,
    "external_submission_performed": False,
    "candidate_reuse": {
        "node_count": len(SHARED_NODE_COLUMNS),
        "record_count": len(reuse_manifest["records"]),
        "fallback_to_duplicate_generation": False,
    },
    "visible_reference_checks": visible_reference_checks,
    "prediction_stats": {
        "min": float(submission["tvt"].min()),
        "max": float(submission["tvt"].max()),
        "mean": float(submission["tvt"].mean()),
        "std": float(submission["tvt"].std()),
    },
    "sha256": {
        "source_pull_notebook": SOURCE_PULL_NOTEBOOK_SHA256,
        "source_code_cells": SOURCE_CODE_CELL_SHA256,
        "hjyact_component": sha256_file(HJYACT_COMPONENT_PATH),
        "exp413_component": sha256_file(EXP413_COMPONENT_PATH),
        "exp413_prediction_file": sha256_file(exp413_prediction_path),
        "exp413_prediction_content": sha256_gzip_content(exp413_prediction_path),
        "candidate_reuse_manifest": sha256_file(WORKING_DIR / "candidate_reuse_manifest.json"),
        "model_manifest": sha256_file(WORKING_DIR / "exp512_model_manifest.json"),
        "submission": sha256_file(FINAL_SUBMISSION_PATH),
    },
}
(WORKING_DIR / "metrics.json").write_text(
    json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\\n",
    encoding="utf-8",
)
reproducibility_manifest = {
    **metrics,
    "sample_id_order_sha256": sample_id_sha,
    "candidate_reuse_manifest": reuse_manifest,
    "model_manifest": model_manifest,
    "source_kernel": SOURCE_KERNEL,
    "source_version": SOURCE_VERSION,
    "source_run_id": SOURCE_RUN_ID,
    "source_profile": SOURCE_PROFILE,
    "hidden_test_policy": "dynamic raw/sample inputs only; visible reference hashes are post-hoc assertions",
}
(WORKING_DIR / "exp512_reproducibility_manifest.json").write_text(
    json.dumps(reproducibility_manifest, indent=2, sort_keys=True, default=str) + "\\n",
    encoding="utf-8",
)
print("exp512 fixed 50/50 submission generated:", FINAL_SUBMISSION_PATH, submission.shape)
print("external submission performed: False")
display(submission.head(20))
display(metrics)
'''


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_code_sha(notebook: dict[str, Any]) -> str:
    source = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    return sha256_bytes(source.encode())


def markdown_cell(title: str) -> str:
    return f"# %% [markdown]\n# ## {title}\n\n"


def code_cell(source: str) -> str:
    return "# %%\n" + source.rstrip() + "\n\n"


def extract_exp413_function() -> str:
    raw = EXP413_RUNTIME.read_bytes()
    observed = sha256_bytes(raw)
    if observed != EXPECTED_EXP413_RUNTIME_SHA256:
        raise RuntimeError(
            f"exp413 runtime SHA mismatch: {observed} != {EXPECTED_EXP413_RUNTIME_SHA256}"
        )
    text = raw.decode()
    tree = ast.parse(text)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "generate_dynamic_exp413_prediction"
    )
    lines = text.splitlines()
    extracted = "\n".join(lines[function.lineno - 1 : function.end_lineno]) + "\n"
    extracted = "\n".join(
        (
            "    # [embedded markdown boundary]"
            if line.strip().startswith("# %% [markdown]")
            else (
                "    # [embedded code boundary]"
                if line.strip().startswith("# %%")
                else line
            )
        )
        for line in extracted.splitlines()
    ) + "\n"
    original_signature = "def generate_dynamic_exp413_prediction():"
    replacement_signature = (
        "def generate_dynamic_exp413_prediction(shared_deterministic_frame=None, reuse_tracker=None):"
    )
    if extracted.count(original_signature) != 1:
        raise RuntimeError("exp413 function signature replacement contract failed")
    extracted = extracted.replace(original_signature, replacement_signature, 1)
    original_import = '''\
    from exp263_public_replay_source import (  # noqa: E402
        build_replay_test_frame,
        configure_public_runtime,
        list_test_wells as replay_list_test_wells,
        stable_seed as replay_stable_seed,
    )'''
    replacement_import = '''\
    import exp263_public_replay_source as replay_source  # noqa: E402
    from exp263_public_replay_source import (  # noqa: E402
        configure_public_runtime,
        list_test_wells as replay_list_test_wells,
        stable_seed as replay_stable_seed,
    )'''
    if extracted.count(original_import) != 1:
        raise RuntimeError("exp413 replay import replacement contract failed")
    extracted = extracted.replace(original_import, replacement_import, 1)
    original_generation = "    pf_frame, pf_meta = build_replay_test_frame()"
    if extracted.count(original_generation) != 1:
        raise RuntimeError("exp413 replay generation replacement contract failed")
    extracted = extracted.replace(original_generation, EXP413_SHARED_FRAME_BLOCK.rstrip(), 1)
    original_k16_entry = "        params = module.params_from_config(source_config)"
    replacement_k16_entry = '''\
        return run_exp413_k16_haswell_subprocess(
            module,
            train_dir=train_dir,
            test_dir=test_dir,
            source_config=source_config,
            finalize_primitive_confidence=finalize_primitive_confidence,
            frame_content_sha256=frame_content_sha256,
        )
        params = module.params_from_config(source_config)  # pragma: no cover - retained source body'''
    if extracted.count(original_k16_entry) != 1:
        raise RuntimeError("exp413 K16 Haswell subprocess replacement contract failed")
    extracted = extracted.replace(original_k16_entry, replacement_k16_entry, 1)
    return extracted


def transform_source_cell(index: int, source: str) -> str:
    source = source.replace(
        "print(f'  PF {int(globals().get('SELECTOR_PF_SEEDS', SP45_SELECTOR_N_SEEDS))}-seed lik-ensemble OK scales={SELECTOR_SCALES}')",
        'print(f"  PF {int(globals().get(\'SELECTOR_PF_SEEDS\', SP45_SELECTOR_N_SEEDS))}-seed lik-ensemble OK scales={SELECTOR_SCALES}")',
    )
    competition_root_assignment = (
        "COMPETITION_DATA_ROOT = '/kaggle/input/competitions/rogii-wellbore-geology-prediction'"
    )
    if competition_root_assignment in source:
        source = source.replace(
            competition_root_assignment,
            'COMPETITION_DATA_ROOT = str(resolve_competition_data_root())\n'
            'print("competition data root:", COMPETITION_DATA_ROOT)',
            1,
        )
    ridge_root_assignment = (
        "RIDGE_ARTIFACT_ROOT = '/kaggle/input/datasets/ravaghi/"
        "wellbore-geology-prediction-artifacts'"
    )
    if ridge_root_assignment in source:
        source = source.replace(
            ridge_root_assignment,
            'RIDGE_ARTIFACT_ROOT = str(HJYACT_INPUT_AUDIT["roots"]["ridge"])\n'
            'print("ridge artifact root:", RIDGE_ARTIFACT_ROOT)',
            1,
        )
    replacements = {
        15: RIDGE_INPUT_CELL,
        16: SOURCE_PARAMETER_CELL,
        18: LGB_LOAD_CELL,
        19: CB_LOAD_CELL,
    }
    if index in replacements:
        return replacements[index]
    if index == 43:
        marker = "def _find_precomputed_learned_submission"
        if source.count(marker) != 1:
            raise RuntimeError("learned trajectory source transform contract failed")
        prefix = source.split(marker, 1)[0]
        return prefix.rstrip() + "\n\n" + LEARNED_MAIN
    return source


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: prepare_exp512_hjyact_v2_candidate.py SOURCE.ipynb")
    source_path = Path(sys.argv[1]).resolve()
    raw = source_path.read_bytes()
    observed_notebook_sha = sha256_bytes(raw)
    if observed_notebook_sha != EXPECTED_SOURCE_NOTEBOOK_SHA256:
        raise RuntimeError(
            "hjyact source notebook SHA mismatch: "
            f"{observed_notebook_sha} != {EXPECTED_SOURCE_NOTEBOOK_SHA256}"
        )
    notebook = json.loads(raw)
    observed_code_sha = source_code_sha(notebook)
    if observed_code_sha != EXPECTED_SOURCE_CODE_SHA256:
        raise RuntimeError(
            f"hjyact code-cell SHA mismatch: {observed_code_sha} != {EXPECTED_SOURCE_CODE_SHA256}"
        )

    cells = notebook["cells"]
    pieces = [HEADER]
    for index in ACTIVE_SOURCE_CELLS:
        for section in SECTION_BEFORE_CELL.get(index, ()):
            pieces.append(markdown_cell(section))
        source = transform_source_cell(index, "".join(cells[index].get("source", [])))
        pieces.append(code_cell(source))
        if index == 43:
            pieces.append(code_cell(CANDIDATE_REUSE_RUNTIME))

    pieces.append(markdown_cell("7. Embedded hidden-safe exp413 runtime"))
    pieces.append(code_cell(extract_exp413_function()))
    pieces.append(markdown_cell("8. Shared-DAG manifest, fixed blend, and reproducibility outputs"))
    pieces.append(code_cell(FINAL_ORCHESTRATION))
    generated = "".join(pieces)
    OUTPUT.write_text(generated)
    print(
        f"generated {OUTPUT.relative_to(ROOT)} "
        f"({len(generated.splitlines())} lines, sha256={sha256_bytes(generated.encode())})"
    )


if __name__ == "__main__":
    main()
