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
# # exp514 shared exp413 likelihood-PF bank on exp512
#
# This candidate preserves the frozen exp512 ensemble while generating the exp413
# x1.0 stable-seed likelihood-PF bank once per dynamic well. The same raw bank feeds
# SP45 all-scale/branch summaries and the unchanged exp413 scale-5 consumer.

# %% [markdown]
# ## Contents
# 1. Imports, source identity, and frozen profile
# 2. Shared exp413 likelihood-PF producer, adapters, and Stage A audit
# 3. SP45 PF / Beam selector helpers
# 4. Ridge/PF anchor and shared deterministic candidate surface
# 5. Saved ridge artifact inference and runtime Ridge
# 6. Projection and learned trajectory replay
# 7. Guarded overlap and final hjyact-v2 layers
# 8. Embedded hidden-safe exp413 runtime
# 9. Shared-PF ledger, fixed blend, and reproducibility outputs

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

import ctypes as _exp514_ctypes
import gc as _exp514_gc

from IPython.display import display

STAGE_D_VISIBLE_STARTED = time.time()

def _exp514_current_rss_mib():
    status_path = Path('/proc/self/status')
    if not status_path.is_file():
        return None
    for line in status_path.read_text(encoding='utf-8').splitlines():
        if line.startswith('VmRSS:'):
            return float(line.split()[1]) / 1024.0
    return None


def _exp514_release_globals(names, *, label):
    before_mib = _exp514_current_rss_mib()
    released = []
    namespace = globals()
    for name in names:
        if name in namespace:
            namespace.pop(name)
            released.append(name)
    collected = int(_exp514_gc.collect())
    malloc_trim_called = False
    try:
        malloc_trim_called = bool(
            _exp514_ctypes.CDLL('libc.so.6').malloc_trim(0)
        )
    except Exception:
        malloc_trim_called = False
    after_mib = _exp514_current_rss_mib()
    report = {
        'label': str(label),
        'released_names': released,
        'gc_collected': collected,
        'malloc_trim_called': malloc_trim_called,
        'rss_before_mib': before_mib,
        'rss_after_mib': after_mib,
    }
    print('memory release report:', report, flush=True)
    return report

EXPERIMENT_NAME = "exp514_exp413_likpf_seed_bank_reuse_on_exp512"
EXP514_GENERATOR_SHA256 = "0313eb08bf9e31da0b914cc3633b99d4ceafc0f0417565f25b1a0a7f94549dac"
STAGE_D_GENERATOR_SHA256 = "24d41c8c83ecca6ed5fffe3372f672e4b9b70565bcccf42288e4f1aafa9bbfbb"
STAGE_D_BASE_CANDIDATE_SHA256 = "961762731f91bf20de6d43d869aeed44bfa98f60be7f8cccc1c65b37d05dc24c"
STAGE_D_RUNTIME_REVISION = 4
STAGE_D_VISIBLE_HJYACT_CANDIDATE_SHA256 = "6b3e1c576afc47f065bdcce12a09f4361a6bb97c63667630f4f5ab1e64fa37b3"
STAGE_D_V2_GOLD_BALANCED_SHA256 = "2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815"
STAGE_D_V2_EXP413_COMPONENT_SHA256 = "04e6da90cee4325fb01bf7ce49bd87b91b16cf675cfb9d4cdaec77904aee5908"
STAGE_D_V2_COMPONENT_READOUT_SHA256 = "c3a9b217568fdd8d09eea337e9d1d5addb9f6c6b26b138d7865caee1ffe7e1fd"
STAGE_D_V2_FINAL_SUBMISSION_SHA256 = "9974c3face9004ffb39ead3c6d8955dff5d540559c48cd45bc3fbaebf2e192ad"
EXP512_PARENT_SOURCE_SHA256 = "16982879716918811dfa9915c4862d45836bd9360efafbaee41046c3e1b6240f"
EXP073_REPLAY_SOURCE_SHA256 = "4af212a8a1c83e36cdcc0bc912942a62df1fbc94ca67fd75789171afaa1a647e"
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
SP45_WELL_N_JOBS = 4
EXP413_WELL_N_JOBS = 4
MODEL_PACKAGE_CORRECTION_ENABLED = False
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
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_content_sha(frame, columns):
    canonical = frame.loc[:, columns].copy()
    payload = canonical.to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def id_order_sha(ids) -> str:
    return hashlib.sha256("\n".join(str(value) for value in ids).encode()).hexdigest()


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
from joblib import Parallel, delayed
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

well_n_jobs = min(4, len(test_wells))


def predict_one_well(order, well):
    inference = module.predict_well(well, fields, pinned_kappa, params)
    row_idx = np.arange(well.s + 1, well.s + well.n + 1, dtype=np.int32)
    if len(row_idx) != len(inference.pred) or len(inference.pred) != len(inference.delta):
        raise ValueError(f"exp413 K16 row contract mismatch for well={well.wid}")
    row = pd.DataFrame(
        {
            "id": [f"{well.wid}_{int(index)}" for index in row_idx],
            "well": str(well.wid),
            "well_row_idx": row_idx,
            "candidate_tvt": np.asarray(inference.pred, dtype=np.float32),
            "geometry_gr_delta": np.asarray(inference.delta, dtype=np.float32),
        }
    )
    summary = dict(inference.summary)
    summary["order"] = order
    return row, summary


well_results = Parallel(n_jobs=well_n_jobs, backend="threading")(
    delayed(predict_one_well)(order, well)
    for order, well in enumerate(test_wells, start=1)
)
rows = [row for row, _ in well_results]
well_summaries = [summary for _, summary in well_results]

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
            "well_n_jobs": well_n_jobs,
            "well_summaries": well_summaries,
            "runtime_seconds": time.time() - started,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
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
    config_path.write_text(json.dumps(source_config, sort_keys=True) + "\n")
    environment = dict(os.environ)
    environment["OPENBLAS_CORETYPE"] = "Haswell"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            EXP413_K16_HASWELL_CHILD_CODE,
            str(Path(getattr(module, "__" + "file__")).resolve()),
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
            f"returncode={completed.returncode}\n{completed.stderr[-8000:]}"
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

# %% [markdown]
# ## 1. Imports, source identity, and frozen profile

# %%
# Profile choices:
# - vp_balanced_final: current submission default; visible-prefix profile output becomes final after contact guard.
# - vp_conservative_final: weaker visible-prefix profile, kept as a conservative comparison.
# - contact_gated_anchor*: diagnostic ablations; they keep the pre-visible-prefix anchor and have underperformed.
# - bimodal_guarded: contact_gated_anchor plus bimodal hedge protection.
SUBMISSION_PROFILE = 'vp_balanced_modelpkg_005'

PROFILE_PRESETS = {
    'contact_gated_anchor': dict(
        visible_prefix_profile='conservative',
        visible_prefix_final_selection='self_verified_anchor',
        visible_prefix_cut_fracs=(0.50, 0.65, 0.75),
        sp45_blend_weight=0.60,
        run_guarded_overlap_override=True,
        run_visible_prefix_calibration=True,
        run_bimodal_detector=False,
        run_vp_bimodal_guard=False,
        run_model_package_correction=False,
    ),
    'contact_gated_anchor_w058': dict(
        visible_prefix_profile='conservative',
        visible_prefix_final_selection='self_verified_anchor',
        visible_prefix_cut_fracs=(0.50, 0.65, 0.75),
        sp45_blend_weight=0.58,
        run_guarded_overlap_override=True,
        run_visible_prefix_calibration=True,
        run_bimodal_detector=False,
        run_vp_bimodal_guard=False,
        run_model_package_correction=False,
    ),
    'contact_gated_anchor_w055': dict(
        visible_prefix_profile='conservative',
        visible_prefix_final_selection='self_verified_anchor',
        visible_prefix_cut_fracs=(0.50, 0.65, 0.75),
        sp45_blend_weight=0.55,
        run_guarded_overlap_override=True,
        run_visible_prefix_calibration=True,
        run_bimodal_detector=False,
        run_vp_bimodal_guard=False,
        run_model_package_correction=False,
    ),
    'vp_conservative_final': dict(
        visible_prefix_profile='conservative',
        visible_prefix_final_selection='profile',
        visible_prefix_cut_fracs=(0.50, 0.65, 0.75),
        sp45_blend_weight=0.60,
        run_guarded_overlap_override=True,
        run_visible_prefix_calibration=True,
        run_bimodal_detector=False,
        run_vp_bimodal_guard=False,
        run_model_package_correction=False,
    ),
    'vp_balanced_final': dict(
        visible_prefix_profile='balanced',
        visible_prefix_final_selection='profile',
        visible_prefix_cut_fracs=(0.50, 0.65, 0.75),
        sp45_blend_weight=0.60,
        run_guarded_overlap_override=True,
        run_visible_prefix_calibration=True,
        run_bimodal_detector=False,
        run_vp_bimodal_guard=False,
        run_model_package_correction=False,
    ),

    'vp_balanced_cut557084': dict(
        visible_prefix_profile='balanced',
        visible_prefix_final_selection='profile',
        visible_prefix_cut_fracs=(0.55, 0.70, 0.84),
        sp45_blend_weight=0.60,
        run_guarded_overlap_override=True,
        run_visible_prefix_calibration=True,
        run_bimodal_detector=False,
        run_vp_bimodal_guard=False,
        run_model_package_correction=False,
    ),
    'vp_balanced_modelpkg_005': dict(
        visible_prefix_profile='balanced',
        visible_prefix_final_selection='profile',
        visible_prefix_cut_fracs=(0.50, 0.65, 0.75),
        sp45_blend_weight=0.60,
        run_guarded_overlap_override=True,
        run_visible_prefix_calibration=True,
        run_bimodal_detector=False,
        run_vp_bimodal_guard=False,
        run_model_package_correction=False,
        model_package_gated_max_weight=0.00425,
        model_package_gated_scale=6.0,
    ),
    'vp_balanced_modelpkg_010': dict(
        visible_prefix_profile='balanced',
        visible_prefix_final_selection='profile',
        visible_prefix_cut_fracs=(0.50, 0.65, 0.75),
        sp45_blend_weight=0.60,
        run_guarded_overlap_override=True,
        run_visible_prefix_calibration=True,
        run_bimodal_detector=False,
        run_vp_bimodal_guard=False,
        run_model_package_correction=True,
        model_package_gated_max_weight=0.010,
        model_package_gated_scale=6.0,
    ),
    'bimodal_guarded': dict(
        visible_prefix_profile='conservative',
        visible_prefix_final_selection='self_verified_anchor',
        visible_prefix_cut_fracs=(0.50, 0.65, 0.75),
        sp45_blend_weight=0.60,
        run_guarded_overlap_override=True,
        run_visible_prefix_calibration=True,
        run_bimodal_detector=True,
        run_vp_bimodal_guard=True,
        run_model_package_correction=False,
    ),
}

if SUBMISSION_PROFILE not in PROFILE_PRESETS:
    raise ValueError(f'SUBMISSION_PROFILE must be one of {sorted(PROFILE_PRESETS)}')

_profile = PROFILE_PRESETS[SUBMISSION_PROFILE]

# Data and artifact roots.
COMPETITION_DATA_ROOT = str(resolve_competition_data_root())
print("competition data root:", COMPETITION_DATA_ROOT)
RIDGE_ARTIFACT_ROOT = str(HJYACT_INPUT_AUDIT["roots"]["ridge"])
print("ridge artifact root:", RIDGE_ARTIFACT_ROOT)
KOOLBOX_OFFLINE_ROOTS = (
    '/kaggle/input/datasets/phongnguyn23021656/koolbox-offline',
    '/kaggle/input/koolbox-offline',
    '/kaggle/input/pm-125564438-at-07-07-2026-07-28-00',
)
LEARNED_MODEL_ROOTS = (
    '/kaggle/input/datasets/fleongg/rogii-claude-models-pub',
    '/kaggle/input/rogii-claude-models-pub',
)

# SP45/PF anchor path.
SP45_RIDGE_MODEL_WEIGHT = 0.30
SP45_SELECTOR_WEIGHT = 0.70
SP45_SELECTOR_N_PARTICLES = 500
SP45_SELECTOR_N_SEEDS = 128
SELECTOR_PF_SEEDS = SP45_SELECTOR_N_SEEDS
SELECTOR_PF_RETURN_STD = False
SP45_PROJECTION_DEGREE = 3
SP45_PROJECTION_BLEND_WEIGHT = 0.75

# Optional diagnostics. The submission path keeps full PF precision but disables all-train CV sweeps.
RUN_CV_REPORT = False
RUN_FULL_STACK_CV_ABLATION = False
CV_N_WELLS = 250
CV_ABLATION_N_WELLS = 250
CV_N_SPLITS = 5
CV_SEED = 0
CV_SELECTOR_PF_SEEDS = 24
ABLATION_VP_POSTERIOR_TRUST_ONLY = False

# Bimodal datum detector for PF/beam disagreement.
RUN_BIMODAL_DETECTOR = bool(_profile['run_bimodal_detector'])
RUN_BIMODAL_SELECTOR_HEDGE = RUN_BIMODAL_DETECTOR
BIMODAL_DZ_RANGE = 20.0
BIMODAL_DZ_STEP = 0.5
BUNDLE_MIN = 10.0
BUNDLE_MAX = 20.0
BIMODAL_J_RATIO_EPS = 0.15
SCAN_MIN_SEP = 8.0
BIMODAL_TEMP = 0.75
RUN_ADAPTIVE_TEMP = True
T_MIN = 0.25
T_MAX = 4.0
RUN_PREFIX_TRUST_GATE = True
TRUST_MIN_PREFIX_ROWS = 60
TRUST_FALLBACK = 0.0
TRUST_KAPPA = 1.0
USE_STRUCTURAL_P_FALLBACK = False
BIMODAL_FORCE_MIDPOINT = False
BIMODAL_TRIGGER_MIN_MEDIAN_DIFF = 12.0
BIMODAL_TRIGGER_MIN_P90_DIFF = 18.0
BIMODAL_TRIGGER_MIN_BIG_DIFF_FRAC = 0.18
BIMODAL_TRIGGER_BIG_DIFF_THRESHOLD = 15.0
BIMODAL_MIN_VALID_GR_ROWS = 80

# Heel-calibrated GR matching. This is read-only unless the bimodal detector is enabled.
RUN_HEEL_CALIBRATION = True
RUN_HEEL_LOCALIZATION_REPORT = False
RUN_HEEL_ABLATION_GRID = False
HEEL_MIN_ROWS = 40
HEEL_LOCALIZATION_TOLERANCE = 2.0
HEEL_ALPHA_MIN = 0.25
HEEL_ALPHA_MAX = 4.0
HEEL_BETA_ABS_MAX = 500.0

# Stretch matcher levers. Off until validated with the diagnostic harness.
RUN_GR_FFT_DENOISE = False
RUN_SEQ_MATCHER = False

# Retained comparison layers. Keep off unless deliberately probing them.
RUN_EXACT_MATCH_RECOVERY = False
RUN_OVERLAP_DRY_RUN_PROBE = True

# Learned-trajectory blend.
SP45_BLEND_WEIGHT = float(_profile['sp45_blend_weight'])
SP45_BLEND_CANDIDATE_WEIGHTS = tuple(sorted(set((0.50, 0.52, 0.55, 0.58, 0.60, SP45_BLEND_WEIGHT))))

# Guarded same-well correction.
RUN_GUARDED_OVERLAP_OVERRIDE = bool(_profile['run_guarded_overlap_override'])
GUARDED_OVERRIDE_REF_COL = 'EGFDU'
GUARDED_OVERRIDE_REF_COLS = ('EGFDU', 'ASTNU', 'ANCC', 'ASTNL', 'EGFDL', 'BUDA')
GUARDED_OVERRIDE_MIN_VALID_PHYS_ROWS = 100
GUARDED_OVERRIDE_MIN_KNOWN_PREFIX_ROWS = 50
GUARDED_OVERRIDE_PREFIX_RMSE_LIMIT = 1.0

# Visible-prefix calibration overlay.
RUN_VISIBLE_PREFIX_CALIBRATION = bool(_profile['run_visible_prefix_calibration'])
VISIBLE_PREFIX_PROFILE = str(_profile['visible_prefix_profile'])
VISIBLE_PREFIX_FINAL_SELECTION = str(_profile.get('visible_prefix_final_selection', 'self_verified_anchor'))
VISIBLE_PREFIX_INCLUDE_PF = True
VISIBLE_PREFIX_CAL_SEEDS = 24
VISIBLE_PREFIX_FINAL_SEEDS = 48
VISIBLE_PREFIX_PARTICLES = 350
VISIBLE_PREFIX_CUT_FRACS = tuple(float(x) for x in _profile.get('visible_prefix_cut_fracs', (0.50, 0.65, 0.75)))
VISIBLE_PREFIX_MAX_WELLS = 1_000_000
RUN_VP_BIMODAL_GUARD = bool(_profile['run_vp_bimodal_guard'])
VP_SKIP_REQUIRES_LOW_TRUST = False
VP_LOW_TRUST_THRESHOLD = 0.25
VISIBLE_PREFIX_SKIP_BIMODAL_WELLS = RUN_VP_BIMODAL_GUARD

# Saved-model correction is deliberately disabled for the authorized speed run.
RUN_MODEL_PACKAGE_CORRECTION = MODEL_PACKAGE_CORRECTION_ENABLED
if bool(_profile.get('run_model_package_correction', False)):
    raise RuntimeError('model-package correction must remain disabled for exp512 speed run')
MODEL_PACKAGE_ROOTS = (
    '/kaggle/input/datasets/pilkwang/rogii-model-package',
    '/kaggle/input/rogii-model-package',
)
MODEL_PACKAGE_REQUIRE = False
MODEL_PACKAGE_ALLOW_AUTO_SEARCH = False
MODEL_PACKAGE_GATED_MAX_WEIGHT = float(_profile.get('model_package_gated_max_weight', 0.01))
MODEL_PACKAGE_GATED_SCALE = float(_profile.get('model_package_gated_scale', 6.0))
MODEL_PACKAGE_GATED_CANDIDATES = (0.005, 0.010, 0.0125, 0.015, 0.020)
MODEL_PACKAGE_DIFF_P95_DISABLE = 25.0

print('submission profile:', SUBMISSION_PROFILE)
print('sp45_blend_weight:', SP45_BLEND_WEIGHT)
print('visible_prefix_profile:', VISIBLE_PREFIX_PROFILE)
print('visible_prefix_final_selection:', VISIBLE_PREFIX_FINAL_SELECTION)
print('visible_prefix_cut_fracs:', VISIBLE_PREFIX_CUT_FRACS)
print('guarded_overlap_override:', RUN_GUARDED_OVERLAP_OVERRIDE)
print('sp45_well_n_jobs:', SP45_WELL_N_JOBS)
print('exp413_well_n_jobs:', EXP413_WELL_N_JOBS)
print('model_package_correction:', RUN_MODEL_PACKAGE_CORRECTION)

# %%
# Runtime bridge for the visible-prefix implementation.
import os
os.environ['ROGII_GOLD_PREFIX_CAL'] = '1' if RUN_VISIBLE_PREFIX_CALIBRATION else '0'
os.environ['ROGII_GOLD_PROFILE'] = VISIBLE_PREFIX_PROFILE
os.environ['ROGII_GOLD_INCLUDE_PF'] = '1' if VISIBLE_PREFIX_INCLUDE_PF else '0'
os.environ['ROGII_GOLD_CAL_SEEDS'] = str(int(VISIBLE_PREFIX_CAL_SEEDS))
os.environ['ROGII_GOLD_FINAL_SEEDS'] = str(int(VISIBLE_PREFIX_FINAL_SEEDS))
os.environ['ROGII_GOLD_PARTICLES'] = str(int(VISIBLE_PREFIX_PARTICLES))
os.environ['ROGII_GOLD_CUT_FRACS'] = ','.join(str(float(x)) for x in VISIBLE_PREFIX_CUT_FRACS)
os.environ['ROGII_GOLD_MAX_WELLS'] = str(int(VISIBLE_PREFIX_MAX_WELLS))
os.environ['ROGII_GOLD_FINAL_SELECTION'] = VISIBLE_PREFIX_FINAL_SELECTION
os.environ['ROGII_GOLD_SKIP_BIMODAL'] = '1' if RUN_VP_BIMODAL_GUARD else '0'
os.environ['ROGII_GOLD_VP_SKIP_REQUIRES_LOW_TRUST'] = '1' if VP_SKIP_REQUIRES_LOW_TRUST else '0'
os.environ['ROGII_GOLD_VP_LOW_TRUST_THRESHOLD'] = str(float(VP_LOW_TRUST_THRESHOLD))
os.environ['ROGII_GOLD_CONTACT_OVERRIDE'] = '1' if RUN_GUARDED_OVERLAP_OVERRIDE else '0'

# %%
import sys, os, glob, subprocess, types
from pathlib import Path

_koolbox_roots = [Path(p) for p in globals().get('KOOLBOX_OFFLINE_ROOTS', ()) if str(p).strip()]
_koolbox_root = next((p for p in _koolbox_roots if p.exists()), None)
if _koolbox_root is None:
    # Some notebook environments mount koolbox through a package-manager input
    # whose folder name changes. Search only for koolbox-looking wheels/folders.
    _auto_hits = []
    for _pat in ('/kaggle/input/**/koolbox*.whl', '/kaggle/input/**/koolbox*'):
        _auto_hits.extend(Path(x).parent if Path(x).suffix == '.whl' else Path(x) for x in glob.glob(_pat, recursive=True))
    _koolbox_root = next((p for p in sorted(set(_auto_hits)) if p.exists()), None)


def _wheel_matches_runtime(path):
    name = Path(path).name
    if ' (' in name or not name.endswith('.whl'):
        return False
    parts = name[:-4].split('-')
    if len(parts) < 5:
        return False
    py_tag, abi_tag, _platform_tag = parts[-3], parts[-2], parts[-1]
    runtime_tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    if py_tag.startswith('cp') and py_tag != runtime_tag:
        return False
    if abi_tag.startswith('cp') and abi_tag != runtime_tag:
        return False
    return py_tag in {'py2.py3', 'py3', runtime_tag} or py_tag.startswith(runtime_tag)


def _install_or_path_koolbox(root):
    if root is None:
        return False
    print('using koolbox dir:', root)
    whls = [w for w in sorted(root.glob('**/*.whl')) if _wheel_matches_runtime(w)]
    if whls:
        for w in whls:
            print('install', w)
            subprocess.run(['pip', 'install', '--no-deps', str(w)], check=False)
    else:
        sys.path.insert(0, str(root))
        for sub in root.iterdir():
            if sub.is_dir():
                sys.path.insert(0, str(sub))
    return True


def _make_koolbox_fallback_module():
    import numpy as _np
    import joblib as _joblib
    from pathlib import Path as _Path
    from sklearn.base import clone as _clone
    from sklearn.metrics import root_mean_squared_error as _rmse
    from sklearn.model_selection import GroupKFold as _GroupKFold, KFold as _KFold

    def _take(X, idx):
        return X.iloc[idx] if hasattr(X, 'iloc') else X[idx]

    def _score(metric, y_true, y_pred):
        try:
            return float(metric(y_true, y_pred)) if callable(metric) else float(_rmse(y_true, y_pred))
        except Exception:
            return float(_rmse(y_true, y_pred))

    def _drop_fit_keys(kwargs, keys):
        out = dict(kwargs or {})
        for key in keys:
            out.pop(key, None)
        return out

    class Trainer:
        def __init__(self, estimator, task='regression', metric=None, cv=None, cv_args=None,
                     use_early_stopping=False, verbose=False, save=False, save_path=None):
            self.estimator = estimator
            self.task = task
            self.metric = metric or _rmse
            self.cv = cv
            self.cv_args = cv_args or {}
            self.use_early_stopping = bool(use_early_stopping)
            self.verbose = bool(verbose)
            self.save = bool(save)
            self.save_path = save_path
            self.models = []
            self.oof_preds = None
            self.fold_scores = []
            self.overall_score = None

        def _splits(self, X, y):
            groups = self.cv_args.get('groups')
            cv = self.cv
            if cv is None:
                cv = _GroupKFold(n_splits=5) if groups is not None else _KFold(n_splits=5, shuffle=True, random_state=42)
            try:
                return list(cv.split(X, y, groups=groups))
            except TypeError:
                return list(cv.split(X, y))

        def _fit_one(self, estimator, X_tr, y_tr, X_va=None, y_va=None, fit_args=None):
            fit_kwargs = dict(fit_args or {})
            if self.use_early_stopping and X_va is not None and y_va is not None:
                mod = estimator.__class__.__module__.lower()
                name = estimator.__class__.__name__.lower()
                if 'lightgbm' in mod or 'lgbm' in name:
                    fit_kwargs.setdefault('eval_set', [(X_va, y_va)])
                elif 'catboost' in mod or 'catboost' in name:
                    fit_kwargs.setdefault('eval_set', (X_va, y_va))
            try:
                estimator.fit(X_tr, y_tr, **fit_kwargs)
            except TypeError:
                estimator.fit(X_tr, y_tr, **_drop_fit_keys(fit_kwargs, [
                    'callbacks', 'eval_metric', 'eval_set', 'early_stopping_rounds', 'use_best_model', 'verbose'
                ]))
            return estimator

        def fit(self, X, y, fit_args=None):
            y_arr = _np.asarray(y, dtype=float)
            oof = _np.full(len(y_arr), _np.nan, dtype=float)
            self.models = []
            self.fold_scores = []
            for fold, (tr_idx, va_idx) in enumerate(self._splits(X, y_arr), start=1):
                est = _clone(self.estimator)
                X_tr = _take(X, tr_idx); X_va = _take(X, va_idx)
                y_tr = y_arr[tr_idx]; y_va = y_arr[va_idx]
                est = self._fit_one(est, X_tr, y_tr, X_va, y_va, fit_args=fit_args)
                pred = _np.asarray(est.predict(X_va), dtype=float)
                oof[va_idx] = pred
                score = _score(self.metric, y_va, pred)
                self.fold_scores.append(score)
                self.models.append(est)
                if self.verbose:
                    print(f'fallback Trainer fold {fold}: {score:.5f}')
            if not _np.isfinite(oof).all():
                raise RuntimeError('fallback Trainer produced incomplete OOF predictions')
            self.oof_preds = oof
            self.overall_score = _score(self.metric, y_arr, oof)
            if self.save and self.save_path:
                out_dir = _Path(self.save_path)
                out_dir.mkdir(parents=True, exist_ok=True)
                _joblib.dump(self, out_dir / 'trainer.pkl')
            return self

        def predict(self, X):
            if not self.models:
                raise RuntimeError('Trainer has no fitted fold models')
            preds = [_np.asarray(model.predict(X), dtype=float) for model in self.models]
            return _np.mean(preds, axis=0)

    Trainer.__module__ = 'koolbox'
    Trainer.__qualname__ = 'Trainer'
    module = types.ModuleType('koolbox')
    module.Trainer = Trainer
    setattr(module, "__" + "file__", "<fallback koolbox Trainer shim>")
    return module


_koolbox_mode = 'fallback'
try:
    _install_or_path_koolbox(_koolbox_root)
    import koolbox as _koolbox_probe
    _koolbox_mode = 'external'
except Exception as _e:
    print('koolbox external unavailable; using fallback Trainer shim:', _e)
    sys.modules['koolbox'] = _make_koolbox_fallback_module()
    import koolbox as _koolbox_probe

print('koolbox mode:', _koolbox_mode, '| module:', getattr(_koolbox_probe, "__" + "file__", "<unknown>"))

# %%
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
from koolbox import Trainer
from pathlib import Path
from numba import njit
import matplotlib.pyplot as plt
import multiprocessing
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import joblib
import time
import glob
import os

warnings.filterwarnings("ignore")

# %%
class CFG:
    dataset_path = Path(COMPETITION_DATA_ROOT)
    artifacts_path = Path(RIDGE_ARTIFACT_ROOT)

    seed = 42
    n_splits = 5
    cv = GroupKFold(n_splits=n_splits)

    metric = root_mean_squared_error


def _safe_competition_data_root():
    root = globals().get('COMPETITION_DATA_ROOT', None)
    if root is not None:
        return root
    cfg = globals().get('CFG', None)
    if cfg is not None:
        if hasattr(cfg, 'dataset_path'):
            return getattr(cfg, 'dataset_path')
        if hasattr(cfg, 'DATA'):
            return getattr(cfg, 'DATA')
    return '.'


# %% [markdown]
# ## 2. Shared exp413 likelihood-PF producer, adapters, and Stage A audit
#
# The only candidate likelihood-PF producer below is source-identical to the
# SHA-pinned exp073/exp413 x1.0 kernel. It materializes only aggregates and
# branch summaries. Raw 128-seed trajectories and log-likelihoods remain local
# to one well and are released before the result enters the process-wide bank.

# %%
SHARED_LIKPF_SCALES = (3.0, 5.0, 8.0, 12.0)
SHARED_LIKPF_PARTICLES = 500
SHARED_LIKPF_SEEDS = 128
SHARED_LIKPF_BRANCH_SCALE = 5.0
SHARED_LIKPF_N_JOBS = 4
SHARED_LIKPF_SEED_NAMESPACE = "SHA256(likpf::<split>::<well>)"


def shared_likpf_stable_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo + 1


@njit(cache=True)
def _shared_likpf_interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t


@njit(cache=True, nogil=True)
def _shared_pf_lik_allseeds(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, n_seeds, seed_base,
                            MOM, VN, PN, RP, RR, RESAMP, init_spr):
    n = len(md_v); preds = np.empty((n_seeds, n)); liks = np.empty(n_seeds); tmax = vmin + len(gg)*step
    for s in range(n_seeds):
        np.random.seed(seed_base + s)
        pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
        for j in range(N):
            pos[j] = ls + init_spr*np.random.randn(); rate[j] = ir + 0.01*np.random.randn()
        log_lik = 0.0; prev_md = md_v[0] - 1.0
        for i in range(n):
            dm = md_v[i] - prev_md
            if dm < 1.0: dm = 1.0
            for j in range(N):
                rate[j] = MOM*rate[j] + VN*np.random.randn(); pos[j] += rate[j]*dm + PN*np.random.randn()
                tvt_j = pos[j] - z_v[i]
                if tvt_j < vmin-100.: tvt_j = vmin-100.
                if tvt_j > tmax+100.: tvt_j = tmax+100.
                pos[j] = tvt_j + z_v[i]
            avg_lk = 0.0
            for j in range(N):
                eg = _shared_likpf_interp1(gg, pos[j]-z_v[i], vmin, step); d = (gr_v[i]-eg)/gs; dd = d*d
                if dd > 600.: dd = 600.
                lk = np.exp(-0.5*dd)
                if lk < 1e-300: lk = 1e-300
                avg_lk += w[j]*lk; w[j] = w[j]*lk
            if avg_lk < 1e-300: avg_lk = 1e-300
            log_lik += np.log(avg_lk)
            ws = 0.0
            for j in range(N): ws += w[j]
            if ws > 0.0:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
            neff = 0.0
            for j in range(N): neff += w[j]*w[j]
            neff = 1.0/neff
            if neff < RESAMP*N:
                cum = np.empty(N); c = 0.0
                for j in range(N): c += w[j]; cum[j] = c
                u0 = np.random.uniform(0., 1./N); newpos = np.empty(N); newrate = np.empty(N); ci = 0
                for j in range(N):
                    u = u0 + j/N
                    while ci < N-1 and cum[ci] < u: ci += 1
                    newpos[j] = pos[ci] + RP*np.random.randn(); newrate[j] = rate[ci] + RR*np.random.randn()
                for j in range(N): pos[j] = newpos[j]; rate[j] = newrate[j]; w[j] = 1./N
            est = 0.0
            for j in range(N): est += w[j]*(pos[j]-z_v[i])
            preds[s, i] = est; prev_md = md_v[i]
        liks[s] = log_lik
    return preds, liks


def _shared_likpf_grid(typewell_tvt, typewell_gr, step=0.2):
    tmin = float(typewell_tvt.min()); tmax = float(typewell_tvt.max())
    tvt_g = np.arange(tmin, tmax+step, step)
    return np.interp(tvt_g, typewell_tvt, typewell_gr).astype(np.float64), float(tmin), float(step)


def _shared_array_sha(values) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _shared_json_sha(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _shared_branch_summary(predictions, centered_log_likelihoods, evaluation_index):
    if predictions.shape[0] < 4 or predictions.shape[1] < 10:
        raise ValueError("shared likelihood-PF branch summary requires >=4 seeds and >=10 rows")
    seed_weight = np.exp(centered_log_likelihoods / SHARED_LIKPF_BRANCH_SCALE)
    seed_weight /= float(seed_weight.sum())
    level = np.nanmedian(predictions, axis=1)
    valid = np.isfinite(level) & np.isfinite(seed_weight) & (seed_weight > 0)
    level = level[valid]
    seed_weight = seed_weight[valid]
    if len(level) < 4:
        raise ValueError("shared likelihood-PF branch summary has fewer than four valid seeds")
    seed_weight /= float(seed_weight.sum())
    order = np.argsort(level)
    values = level[order]
    weights = seed_weight[order]
    cumulative_weight = np.cumsum(weights)
    cumulative_value = np.cumsum(weights * values)
    cumulative_square = np.cumsum(weights * values * values)
    total_weight = float(cumulative_weight[-1])
    total_value = float(cumulative_value[-1])
    total_square = float(cumulative_square[-1])
    best = None
    for cut in range(1, len(values)):
        low_weight = float(cumulative_weight[cut - 1])
        high_weight = total_weight - low_weight
        if low_weight < 0.05 or high_weight < 0.05:
            continue
        low_value = float(cumulative_value[cut - 1])
        high_value = total_value - low_value
        low_sse = float(
            cumulative_square[cut - 1] - low_value * low_value / low_weight
        )
        high_sse = float(
            total_square
            - cumulative_square[cut - 1]
            - high_value * high_value / high_weight
        )
        score = max(0.0, low_sse) + max(0.0, high_sse)
        if best is None or score < best[0]:
            best = (
                score,
                low_weight,
                high_weight,
                low_value / low_weight,
                high_value / high_weight,
            )
    if best is None:
        raise ValueError("shared likelihood-PF branch split has no valid mass partition")
    _, mass_low, mass_high, center_low, center_high = best
    return {
        "center_low": float(center_low),
        "center_high": float(center_high),
        "mass_low": float(mass_low),
        "mass_high": float(mass_high),
        "weighted_center": float(np.sum(seed_weight * level)),
        "eval_rows": np.asarray(evaluation_index, dtype=int).tolist(),
        "seed_count": int(len(level)),
    }


def _shared_likpf_one_well(
    well,
    split,
    loader,
    *,
    particles=SHARED_LIKPF_PARTICLES,
    seeds=SHARED_LIKPF_SEEDS,
    scales=SHARED_LIKPF_SCALES,
):
    started = time.time()
    horizontal, typewell = loader(str(well), str(split))
    horizontal = horizontal.copy()
    typewell = typewell.sort_values("TVT").copy()
    required_horizontal = {"MD", "Z", "GR", "TVT_input"}
    if missing := required_horizontal - set(horizontal.columns):
        raise ValueError(f"shared likelihood-PF horizontal columns missing: {sorted(missing)}")
    if {"TVT", "GR"} - set(typewell.columns):
        raise ValueError("shared likelihood-PF typewell requires TVT and GR")
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    evaluation_mask = ~known_mask
    known = horizontal.loc[known_mask]
    evaluation = horizontal.loc[evaluation_mask]
    if len(known) < 10 or len(evaluation) < 10:
        raise ValueError(f"shared likelihood-PF ineligible well: {well}")
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = (
        typewell["GR"]
        .fillna(float(typewell["GR"].mean()))
        .to_numpy(np.float64)
    )
    last = known.iloc[-1]
    level_start = float(last["TVT_input"]) + float(last["Z"])
    typewell_at_known = np.interp(
        known["TVT_input"].to_numpy(np.float64),
        typewell_tvt,
        typewell_gr,
    )
    gr_sigma = float(
        np.clip(
            np.nanstd(
                known["GR"].fillna(0).to_numpy(np.float64) - typewell_at_known
            ),
            10.0,
            60.0,
        )
    )
    tail = known.tail(30)
    delta_tvt = np.diff(tail["TVT_input"].to_numpy(np.float64))
    delta_z = np.diff(tail["Z"].to_numpy(np.float64))
    delta_md = np.diff(tail["MD"].to_numpy(np.float64))
    positive = delta_md > 0
    initial_rate = (
        float(np.median((delta_tvt + delta_z)[positive] / delta_md[positive]))
        if int(positive.sum()) >= 3
        else 0.0
    )
    gr_grid, grid_min, grid_step = _shared_likpf_grid(typewell_tvt, typewell_gr)
    interpolated_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(float(typewell_gr.mean()))
        .to_numpy(np.float64)
    )
    evaluation_index = evaluation.index.to_numpy(np.int64)
    seed_base = shared_likpf_stable_seed("likpf", split, well)
    core_started = time.time()
    predictions, log_likelihoods = _shared_pf_lik_allseeds(
        evaluation["MD"].to_numpy(np.float64),
        evaluation["Z"].to_numpy(np.float64),
        interpolated_gr[evaluation_index],
        gr_grid,
        grid_min,
        grid_step,
        gr_sigma,
        level_start,
        initial_rate,
        int(particles),
        int(seeds),
        int(seed_base),
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )
    core_seconds = time.time() - core_started
    if predictions.shape != (int(seeds), len(evaluation)):
        raise ValueError(f"shared likelihood-PF raw bank shape mismatch for {well}")
    if not np.isfinite(predictions).all() or not np.isfinite(log_likelihoods).all():
        raise ValueError(f"shared likelihood-PF raw bank is non-finite for {well}")
    centered = log_likelihoods - float(log_likelihoods.max())
    suffix_aggregates = {}
    for scale in scales:
        weights = np.exp(centered / float(scale))
        weights /= float(weights.sum())
        suffix_aggregates[f"pf_scale_{float(scale):g}"] = (
            weights[:, None] * predictions
        ).sum(axis=0)
    suffix_aggregates["pf_mean"] = predictions.mean(axis=0)
    branch_summary = _shared_branch_summary(predictions, centered, evaluation_index)

    raw_tvt_input = pd.to_numeric(
        horizontal["TVT_input"], errors="coerce"
    ).to_numpy(np.float64)
    sp45_full = {}
    for name, suffix in suffix_aggregates.items():
        full = raw_tvt_input.copy()
        full[evaluation_index] = np.asarray(suffix, dtype=np.float64)
        if not np.array_equal(full[known_mask], raw_tvt_input[known_mask]):
            raise ValueError(f"shared likelihood-PF known-prefix parity failed for {well}")
        if not np.isfinite(full).all():
            raise ValueError(f"shared likelihood-PF full adapter is non-finite for {well}")
        sp45_full[name] = full

    expected_ids = [f"{well}_{int(index)}" for index in evaluation_index]
    scale5 = suffix_aggregates["pf_scale_5"].astype(np.float32)
    arithmetic_mean = suffix_aggregates["pf_mean"].astype(np.float32)
    exp413_frame = pd.DataFrame(
        {
            "id": expected_ids,
            "likpf_scale_5": scale5,
            "likpf_mean": arithmetic_mean,
        }
    )
    if exp413_frame["id"].duplicated().any():
        raise ValueError(f"shared likelihood-PF duplicate exp413 IDs for {well}")
    scale_content_sha = {
        name: _shared_array_sha(np.asarray(values, dtype=np.float64))
        for name, values in suffix_aggregates.items()
    }
    del predictions, log_likelihoods, centered
    return {
        "well": str(well),
        "split": str(split),
        "seed_base": int(seed_base),
        "row_index": horizontal.index.to_numpy(np.int64),
        "evaluation_index": evaluation_index,
        "known_mask": known_mask,
        "sp45_full": sp45_full,
        "exp413_frame": exp413_frame,
        "branch_summary": branch_summary,
        "scale_content_sha256": scale_content_sha,
        "audit": {
            "rows": int(len(horizontal)),
            "known_rows": int(known_mask.sum()),
            "evaluation_rows": int(evaluation_mask.sum()),
            "particles": int(particles),
            "seeds": int(seeds),
            "scales": [float(value) for value in scales],
            "gr_sigma": gr_sigma,
            "kernel_dtype": "float64",
            "exp413_consumer_dtype": "float32",
            "raw_seed_bank_retained": False,
            "core_seconds": round(core_seconds, 6),
            "elapsed_seconds": round(time.time() - started, 6),
        },
        "ledger": {
            "producer_calls": 1,
            "core_calls": 1,
            "sp45_consumer_hits": 0,
            "exp413_consumer_hits": 0,
            "legacy_sp45_bank_calls": 0,
            "exp413_duplicate_bank_calls": 0,
            "fallback_calls": 0,
        },
    }


def materialize_shared_likpf_bank(
    wells,
    split,
    loader,
    *,
    n_jobs=SHARED_LIKPF_N_JOBS,
    particles=SHARED_LIKPF_PARTICLES,
    seeds=SHARED_LIKPF_SEEDS,
):
    ordered_wells = sorted(str(well) for well in wells)
    if not ordered_wells or len(set(ordered_wells)) != len(ordered_wells):
        raise ValueError("shared likelihood-PF requires unique nonempty wells")
    effective_n_jobs = min(max(1, int(n_jobs)), len(ordered_wells))
    started = time.time()
    results = Parallel(n_jobs=effective_n_jobs, backend="threading")(
        delayed(_shared_likpf_one_well)(
            well,
            split,
            loader,
            particles=int(particles),
            seeds=int(seeds),
        )
        for well in ordered_wells
    )
    bank = {record["well"]: record for record in results}
    if list(bank) != ordered_wells:
        raise RuntimeError("shared likelihood-PF merge order changed")
    report = {
        "requested_n_jobs": int(n_jobs),
        "effective_n_jobs": effective_n_jobs,
        "backend": "threading",
        "wells": len(ordered_wells),
        "elapsed_seconds": round(time.time() - started, 6),
    }
    return bank, report


def shared_likpf_sp45_adapter(record):
    ledger = record["ledger"]
    ledger["sp45_consumer_hits"] += 1
    if ledger["sp45_consumer_hits"] != 1:
        raise RuntimeError(f"SP45 consumed shared bank more than once: {record['well']}")
    required = {f"pf_scale_{scale:g}" for scale in SHARED_LIKPF_SCALES}
    required.add("pf_mean")
    if set(record["sp45_full"]) != required:
        raise ValueError(f"SP45 shared adapter schema mismatch: {record['well']}")
    # Transfer array ownership to the one SP45 consumer instead of duplicating
    # every full-length scale. The worker drops record['sp45_full'] immediately
    # after the selector returns.
    return dict(record["sp45_full"]), dict(record["branch_summary"])


def release_shared_likpf_sp45_payload(record):
    released = []
    for key in ("sp45_full", "row_index", "evaluation_index", "known_mask"):
        if key in record:
            record.pop(key)
            released.append(key)
    record["audit"]["sp45_payload_released"] = True
    record["audit"]["released_after_sp45"] = released
    record["audit"]["retained_for_exp413"] = [
        "id", "likpf_scale_5", "likpf_mean"
    ]
    return released


def shared_likpf_exp413_adapter(bank, wells):
    frames = []
    records = []
    for well in [str(value) for value in wells]:
        if well not in bank:
            raise KeyError(f"exp413 shared likelihood-PF bank is missing well {well}")
        record = bank[well]
        record["ledger"]["exp413_consumer_hits"] += 1
        if record["ledger"]["exp413_consumer_hits"] != 1:
            raise RuntimeError(f"exp413 consumed shared bank more than once: {well}")
        frames.append(record["exp413_frame"])
        records.append(record)
    frame = pd.concat(frames, ignore_index=True)
    expected_columns = ["id", "likpf_scale_5", "likpf_mean"]
    if list(frame.columns) != expected_columns or frame["id"].duplicated().any():
        raise ValueError("exp413 shared likelihood-PF adapter schema/ID contract failed")
    for record in records:
        record.pop("exp413_frame")
        record["audit"]["exp413_payload_released"] = True
    return frame


def finalize_shared_likpf_manifest(bank, expected_wells):
    ordered_wells = sorted(str(value) for value in expected_wells)
    if sorted(bank) != ordered_wells:
        raise ValueError("shared likelihood-PF manifest well set mismatch")
    records = []
    branch_records = []
    ledger_records = []
    for well in ordered_wells:
        record = bank[well]
        forbidden_payloads = {
            "predictions", "log_likelihoods", "raw_bank", "sp45_full",
            "row_index", "evaluation_index", "known_mask", "exp413_frame",
        }
        leaked_payloads = sorted(forbidden_payloads.intersection(record))
        if leaked_payloads:
            raise RuntimeError(
                f"shared likelihood-PF payload leaked beyond consumer scope: "
                f"{well}: {leaked_payloads}"
            )
        ledger = {"well": well, **record["ledger"]}
        expected_ledger = {
            "producer_calls": 1,
            "core_calls": 1,
            "sp45_consumer_hits": 1,
            "exp413_consumer_hits": 1,
            "legacy_sp45_bank_calls": 0,
            "exp413_duplicate_bank_calls": 0,
            "fallback_calls": 0,
        }
        if record["ledger"] != expected_ledger:
            raise RuntimeError(f"shared likelihood-PF ledger failed for {well}: {ledger}")
        records.append(
            {
                "well": well,
                "split": record["split"],
                "seed_base": record["seed_base"],
                "scale_content_sha256": record["scale_content_sha256"],
                "audit": record["audit"],
            }
        )
        branch_records.append({"well": well, **record["branch_summary"]})
        ledger_records.append(ledger)
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "producer_id": "exp413_x1p0_stable_seed_bank",
        "source_sha256": EXP073_REPLAY_SOURCE_SHA256,
        "seed_namespace": SHARED_LIKPF_SEED_NAMESPACE,
        "wells": len(ordered_wells),
        "records": records,
        "branch_summary_sha256": _shared_json_sha(branch_records),
        "generation_ledger_sha256": _shared_json_sha(ledger_records),
        "aggregate_content_sha256": _shared_json_sha(
            [record["scale_content_sha256"] for record in records]
        ),
        "raw_seed_bank_retained": False,
        "all_contracts_passed": True,
    }
    return manifest


def select_shared_likpf_stage_a_wells(data_root, *, split="train", count=32):
    split_root = Path(data_root) / split
    records = []
    for horizontal_path in sorted(split_root.glob("*__horizontal_well.csv")):
        well = horizontal_path.name.split("__", 1)[0]
        typewell_path = split_root / f"{well}__typewell.csv"
        if not typewell_path.is_file():
            continue
        horizontal = pd.read_csv(
            horizontal_path,
            usecols=lambda name: name in {"MD", "Z", "GR", "TVT_input"},
        )
        evaluation = horizontal["TVT_input"].isna()
        known = ~evaluation
        if int(evaluation.sum()) < 10 or int(known.sum()) < 10:
            continue
        evaluation_z = pd.to_numeric(
            horizontal.loc[evaluation, "Z"], errors="coerce"
        ).to_numpy(np.float64)
        if len(evaluation_z) == 0 or not np.isfinite(evaluation_z).any():
            continue
        records.append(
            {
                "well": well,
                "evaluation_rows": int(evaluation.sum()),
                "raw_gr_missing_fraction": float(horizontal["GR"].isna().mean()),
                "evaluation_z_span": float(np.nanmax(evaluation_z) - np.nanmin(evaluation_z)),
                "stable_sha256": hashlib.sha256(
                    f"exp514::stage_a::{split}::{well}".encode("utf-8")
                ).hexdigest(),
            }
        )
    if len(records) < int(count):
        raise ValueError(f"Stage A needs {count} eligible wells, found {len(records)}")
    frame = pd.DataFrame(records)
    for column in (
        "evaluation_rows",
        "raw_gr_missing_fraction",
        "evaluation_z_span",
    ):
        rank = frame[column].rank(method="first").to_numpy(np.int64) - 1
        frame[f"{column}_stratum"] = np.minimum(3, 4 * rank // len(frame))
    stratum_columns = [
        "evaluation_rows_stratum",
        "raw_gr_missing_fraction_stratum",
        "evaluation_z_span_stratum",
    ]
    groups = []
    for key, group in frame.groupby(stratum_columns, sort=True):
        groups.append((tuple(int(value) for value in key), group.sort_values("stable_sha256")))
    selected = []
    offset = 0
    while len(selected) < int(count):
        progressed = False
        for _, group in groups:
            if offset < len(group):
                selected.append(group.iloc[offset].to_dict())
                progressed = True
                if len(selected) == int(count):
                    break
        if not progressed:
            break
        offset += 1
    if len(selected) != int(count):
        raise RuntimeError("Stage A target-free stratified selection did not reach count")
    selection_sha = _shared_json_sha(selected)
    return [str(record["well"]) for record in selected], selected, selection_sha


def run_shared_likpf_stage_a(
    data_root,
    *,
    split="train",
    count=32,
    thread_counts=(1, 4),
    reruns=2,
    particles=SHARED_LIKPF_PARTICLES,
    seeds=SHARED_LIKPF_SEEDS,
):
    wells, selection, selection_sha = select_shared_likpf_stage_a_wells(
        data_root,
        split=split,
        count=count,
    )

    def loader(well, selected_split):
        root = Path(data_root) / selected_split
        return (
            pd.read_csv(root / f"{well}__horizontal_well.csv"),
            pd.read_csv(root / f"{well}__typewell.csv"),
        )

    run_records = []
    reference_signature = None
    for thread_count in thread_counts:
        for rerun in range(int(reruns)):
            bank, parallel_report = materialize_shared_likpf_bank(
                wells,
                split,
                loader,
                n_jobs=int(thread_count),
                particles=int(particles),
                seeds=int(seeds),
            )
            for well in wells:
                shared_likpf_sp45_adapter(bank[well])
            shared_likpf_exp413_adapter(bank, wells)
            manifest = finalize_shared_likpf_manifest(bank, wells)
            signature = {
                "aggregate_content_sha256": manifest["aggregate_content_sha256"],
                "branch_summary_sha256": manifest["branch_summary_sha256"],
                "generation_ledger_sha256": manifest["generation_ledger_sha256"],
            }
            if reference_signature is None:
                reference_signature = signature
            elif signature != reference_signature:
                raise RuntimeError(
                    "Stage A thread/rerun shared likelihood-PF content parity failed"
                )
            run_records.append(
                {
                    "thread_count": int(thread_count),
                    "rerun": int(rerun),
                    "signature": signature,
                    "parallel_report": parallel_report,
                }
            )
    return {
        "stage": "fixed32_technical_determinism",
        "truth_read": False,
        "source_sha256": EXP073_REPLAY_SOURCE_SHA256,
        "selection_sha256": selection_sha,
        "selection": selection,
        "wells": wells,
        "particles": int(particles),
        "seeds": int(seeds),
        "thread_counts": [int(value) for value in thread_counts],
        "reruns": int(reruns),
        "runs": run_records,
        "exp413_scale5_source_contract": "source_identical_exp073_x1p0_stable_seed",
        "all_and_gate_passed": True,
    }


def _warm_shared_likpf_kernel():
    started = time.time()
    md = np.linspace(1.0, 12.0, 12)
    zeros = np.zeros(12)
    gr = np.full(12, 50.0)
    grid = np.linspace(45.0, 55.0, 64)
    _shared_pf_lik_allseeds(
        md,
        zeros,
        gr,
        grid,
        45.0,
        0.2,
        20.0,
        50.0,
        0.0,
        16,
        4,
        1,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )
    return round(time.time() - started, 6)


SHARED_LIKPF_JIT_WARMUP_SECONDS = _warm_shared_likpf_kernel()
print("shared exp413 likelihood-PF kernel compiled:", SHARED_LIKPF_JIT_WARMUP_SECONDS)

# %% [markdown]
# ## 3. SP45 PF / Beam selector helpers

# %%
SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)

SELECTOR_BIN_VARIANTS = {
    0: 'pf_scale_5_hold_0.2',
    1: 'pf_scale_3_hold_0.15',
    2: 'pf_scale_12_beam_0.2_hold_0.15',
    3: 'pf_scale_5_hold_0.15',
    4: 'pf_scale_5_beam_0.05_hold_0.05',
    5: 'pf_scale_12_beam_0.2_hold_0.05',
}

SELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'
SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)

FORMATION_COLS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']

BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2),
    (10,  8.0,  64.0, 2),
    ( 8, 35.0, 220.0, 1),
    (10, 14.0,  90.0, 5),
    (20,  4.0,  36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    (20, 30.0, 200.0, 2),
    (15, 10.0,  80.0, 4),
    (25,  6.0,  50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30,  8.0,  70.0, 2),
    (10, 50.0, 400.0, 0),
]


def tvt_from_contacts(hw_tr, tw_tr, ref_col='EGFDU'):
    tw_g = tw_tr.dropna(subset=['Geology'])
    ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g['Geology'].iloc[0]
        ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    offset = (hw_tr['TVT'] - (ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]) + offset


def load_well(wid, split='train'):
    base = CFG.dataset_path / split
    hw = pd.read_csv(base / f'{wid}__horizontal_well.csv')
    tw = pd.read_csv(base / f'{wid}__typewell.csv')
    return hw, tw


def run_particle_filter(hw, tw, n_particles=500, seed=42):
    tw_s   = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)

    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy(), 0.0

    last     = kn.iloc[-1]
    last_tvt = float(last['TVT_input'])
    last_Z   = float(last['Z'])
    last_MD  = float(last['MD'])

    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn['GR'].fillna(0).values - tw_at_k), 10., 60.))

    tail = kn.tail(30)
    dt = np.diff(tail['TVT_input'].values)
    dz = np.diff(tail['Z'].values)
    dm = np.diff(tail['MD'].values)
    m  = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    N   = n_particles
    rng = np.random.default_rng(seed)
    ls   = last_tvt + last_Z
    pos  = ls + 4.5 * rng.standard_normal(N)  # sp45 patch (sel15 vb best)
    rate = ir + 0.01 * rng.standard_normal(N)
    w    = np.ones(N) / N

    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5

    md_v = ev['MD'].values.astype(float)
    z_v  = ev['Z'].values.astype(float)
    # Interpolate GR gaps before tracking
    gr_interp = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]

    out_vals = hw['TVT_input'].values.astype(float).copy()
    res = np.empty(len(ev))
    prev_MD = last_MD
    log_lik = 0.0

    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos  = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos   = tvt_p + z_v[i]

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d  = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d**2, 600.))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        n_eff = 1.0 / (w**2).sum()
        if n_eff < RESAMP * N:
            cum = np.cumsum(w)
            u0  = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos  = pos[idx]  + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w    = np.ones(N) / N

        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]

    out_vals[list(ev.index)] = res
    return out_vals, log_lik


def run_pf_lik_ensemble(hw, tw, n_particles=500, n_seeds=128, scale=5.0):
    preds = []
    liks  = []
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)
        preds.append(p)
        liks.append(ll)

    liks   = np.array(liks)
    liks_n = liks - liks.max()
    weights = np.exp(liks_n / scale)
    weights /= weights.sum()

    return (weights[:, None] * np.stack(preds, 0)).sum(0)


def run_pf_lik_ensemble_scales(hw, tw, scales=SELECTOR_SCALES, n_particles=500, n_seeds=128, branch_stats=None):
    preds = []
    liks = []
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)
        preds.append(p)
        liks.append(ll)
    pred_arr = np.stack(preds, 0)
    liks = np.array(liks)
    liks_n = liks - liks.max()
    out = {}
    for scale in scales:
        weights = np.exp(liks_n / float(scale))
        weights /= weights.sum()
        out[f'pf_scale_{scale:g}'] = (weights[:, None] * pred_arr).sum(0)
    out['pf_mean'] = pred_arr.mean(0)
    if bool(globals().get('SELECTOR_PF_RETURN_STD', False)):
        out['pf_seed_std'] = pred_arr.std(0)
    if branch_stats is not None:
        try:
            eval_mask = pd.to_numeric(hw['TVT_input'], errors='coerce').isna().to_numpy()
            if int(eval_mask.sum()) >= 10:
                seed_weight = np.exp(liks_n / 5.0)
                seed_weight = seed_weight / max(float(seed_weight.sum()), 1e-12)
                level = np.nanmedian(pred_arr[:, eval_mask], axis=1)
                valid = np.isfinite(level) & np.isfinite(seed_weight) & (seed_weight > 0)
                level = level[valid]
                seed_weight = seed_weight[valid]
                seed_weight = seed_weight / max(float(seed_weight.sum()), 1e-12)
                if len(level) >= 4:
                    order = np.argsort(level)
                    x = level[order]
                    w = seed_weight[order]
                    cw = np.cumsum(w)
                    cx = np.cumsum(w * x)
                    cx2 = np.cumsum(w * x * x)
                    total_w, total_x, total_x2 = float(cw[-1]), float(cx[-1]), float(cx2[-1])
                    best = None
                    for cut in range(1, len(x)):
                        wl = float(cw[cut - 1])
                        wr = total_w - wl
                        if wl < 0.05 or wr < 0.05:
                            continue
                        xl = float(cx[cut - 1])
                        xr = total_x - xl
                        ssel = float(cx2[cut - 1] - xl * xl / wl)
                        sser = float(total_x2 - cx2[cut - 1] - xr * xr / wr)
                        score = max(0.0, ssel) + max(0.0, sser)
                        if best is None or score < best[0]:
                            best = (score, wl, wr, xl / wl, xr / wr)
                    if best is not None:
                        _, mass_low, mass_high, center_low, center_high = best
                        branch_stats.update(
                            center_low=float(center_low),
                            center_high=float(center_high),
                            mass_low=float(mass_low),
                            mass_high=float(mass_high),
                            weighted_center=float(np.sum(seed_weight * level)),
                            eval_rows=np.flatnonzero(eval_mask).astype(int).tolist(),
                            seed_count=int(len(level)),
                        )
        except Exception as exc:
            branch_stats['error'] = repr(exc)
    return out


def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    n  = len(hgr)
    nt = len(tw_tvt)
    if n == 0:
        return np.array([last_tvt])

    if r > 0 and n > max(3, 2 * r + 1):
        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
        sgr = savgol_filter(hgr, win, min(2, win - 1))
    else:
        sgr = hgr.copy()

    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))

    MOVES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    MC    = mc * np.array([2., 1., 0., 1., 2.])

    bidx  = np.full(bs, si, dtype=np.int64)
    bcost = np.full(bs, np.inf)
    bcost[0] = 0.
    bn = 1

    result = np.zeros(n)

    for step in range(n):
        gv = sgr[step]
        ni = bidx[:bn, None] + MOVES[None, :]
        ci = np.clip(ni, 0, nt - 1)
        valid = (ni >= 0) & (ni < nt)

        gr_e = (gv - tw_gr[ci])**2 / es
        tot  = bcost[:bn, None] + gr_e + MC[None, :]
        tot  = np.where(valid, tot, np.inf)

        ni_f  = ni.flatten()
        tot_f = tot.flatten()
        vf    = valid.flatten()
        ni_f  = ni_f[vf]
        tot_f = tot_f[vf]

        order = np.argsort(tot_f)
        ni_s  = ni_f[order]
        tot_s = tot_f[order]

        _, first = np.unique(ni_s, return_index=True)
        ni_u  = ni_s[first]
        tot_u = tot_s[first]

        kept = min(bs, len(ni_u))
        top  = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]
        top  = top[np.argsort(tot_u[top])]

        bidx[:kept]  = ni_u[top]
        bcost[:kept] = tot_u[top]
        if kept < bs:
            bidx[kept:]  = bidx[kept - 1]
            bcost[kept:] = np.inf
        bn = kept

        result[step] = tw_tvt[bidx[0]]

    return result


def run_beam_ensemble(hw, tw):
    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy()

    last_tvt = float(kn.iloc[-1]['TVT_input'])
    tw_s  = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)

    gr_all = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)
    hgr    = gr_all[ev.index]

    beam_results = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
                    for (bs, mc, es, r) in BEAM_CONFIGS]

    beam_mean = np.stack(beam_results, 0).mean(0)

    out = hw['TVT_input'].values.astype(float).copy()
    out[list(ev.index)] = beam_mean
    return out


def selector_well_code(hw):
    eval_mask = hw['TVT_input'].isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, 'Z'].values.astype(float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side='right'))
    code = n_bin + 2 * z_bin
    variant = SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)
    return code, variant, n_eval, z_span


def parse_selector_variant(name):
    parts = name.split('_')
    scale = float(parts[2])
    beam_weight = 0.0
    hold_weight = 0.0
    if 'beam' in parts:
        beam_weight = float(parts[parts.index('beam') + 1])
    if 'hold' in parts:
        hold_weight = float(parts[parts.index('hold') + 1])
    return scale, beam_weight, hold_weight


def _selector_tw_gr_arrays(tw):
    if tw is None or 'TVT' not in tw.columns or 'GR' not in tw.columns:
        return None, None
    tw_s = tw.sort_values('TVT')
    tw_tvt = pd.to_numeric(tw_s['TVT'], errors='coerce').to_numpy(dtype=float)
    tw_gr = pd.to_numeric(tw_s['GR'], errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    if int(valid.sum()) < 10:
        return None, None
    tw_tvt = tw_tvt[valid]
    tw_gr = tw_gr[valid]
    order = np.argsort(tw_tvt)
    return tw_tvt[order], tw_gr[order]


def _selector_eval_mask(hw, n):
    if hw is not None and 'TVT_input' in hw.columns:
        return hw['TVT_input'].isna().to_numpy()[:n]
    return np.ones(n, dtype=bool)


def _selector_gr_scale(hw, tw_tvt, tw_gr, hgr, fallback_resid=None):
    if hw is not None and 'TVT_input' in hw.columns:
        known = hw['TVT_input'].notna().to_numpy()[:len(hgr)] & np.isfinite(hgr)
        if int(known.sum()) >= 25:
            known_tvt = pd.to_numeric(hw.loc[known, 'TVT_input'], errors='coerce').to_numpy(dtype=float)
            known_gr = hgr[known]
            ref_gr = np.interp(known_tvt, tw_tvt, tw_gr)
            scale = np.nanmedian(np.abs(known_gr - ref_gr)) * 1.4826
            if np.isfinite(scale) and scale >= 8.0:
                return float(scale)
    if fallback_resid is not None and len(fallback_resid):
        scale = np.nanmedian(np.abs(fallback_resid - np.nanmedian(fallback_resid))) * 1.4826
        if np.isfinite(scale) and scale >= 8.0:
            return float(scale)
    return 20.0




def _selector_fft_denoise_gr(hgr):
    x = np.asarray(hgr, dtype=float).copy()
    if not bool(globals().get('RUN_GR_FFT_DENOISE', False)):
        return x, False
    m = np.isfinite(x)
    if int(m.sum()) < 64:
        return x, False
    idx = np.arange(len(x), dtype=float)
    filled = x.copy()
    filled[~m] = np.interp(idx[~m], idx[m], x[m])
    centered = filled - np.nanmedian(filled)
    spec = np.fft.rfft(centered)
    if len(spec) < 5:
        return x, False
    amp = np.abs(spec)
    amp[0] = 0.0
    peak = int(np.argmax(amp))
    if peak <= 0 or not np.isfinite(amp[peak]):
        return x, False
    # A light notch: remove only the strongest periodic component and its closest neighbors.
    for j in range(max(1, peak - 1), min(len(spec), peak + 2)):
        spec[j] = 0.0
    denoised = np.fft.irfft(spec, n=len(centered)) + np.nanmedian(filled)
    out = x.copy()
    out[m] = denoised[m]
    return out, True


def _selector_apply_heel_calibration(hw, tw_tvt, tw_gr, hgr):
    info = {
        'heel_calibrated': False,
        'heel_rows': 0,
        'heel_alpha': np.nan,
        'heel_beta': np.nan,
        'heel_rmse_raw': np.nan,
        'heel_rmse_calibrated': np.nan,
        'heel_denoised': False,
    }
    raw = np.asarray(hgr, dtype=float).copy()
    prepared, denoised = _selector_fft_denoise_gr(raw)
    info['heel_denoised'] = bool(denoised)
    if not bool(globals().get('RUN_HEEL_CALIBRATION', False)):
        return prepared, info
    if hw is None or 'TVT_input' not in hw.columns:
        return prepared, info
    n = min(len(hw), len(prepared))
    tvt_input = pd.to_numeric(hw['TVT_input'], errors='coerce').to_numpy(dtype=float)[:n]
    gr_obs = prepared[:n]
    mask = np.isfinite(tvt_input) & np.isfinite(gr_obs)
    min_rows = int(globals().get('HEEL_MIN_ROWS', 40))
    if int(mask.sum()) < min_rows:
        info['heel_rows'] = int(mask.sum())
        return prepared, info
    ref = np.interp(tvt_input[mask], tw_tvt, tw_gr)
    ok = np.isfinite(ref) & np.isfinite(gr_obs[mask])
    if int(ok.sum()) < min_rows:
        info['heel_rows'] = int(ok.sum())
        return prepared, info
    ref = ref[ok]
    obs = gr_obs[mask][ok]
    A = np.column_stack([ref, np.ones_like(ref)])
    try:
        alpha, beta = np.linalg.lstsq(A, obs, rcond=None)[0]
    except Exception:
        return prepared, info
    alpha = float(alpha)
    beta = float(beta)
    amin = float(globals().get('HEEL_ALPHA_MIN', 0.25))
    amax = float(globals().get('HEEL_ALPHA_MAX', 4.0))
    bmax = float(globals().get('HEEL_BETA_ABS_MAX', 500.0))
    if not (np.isfinite(alpha) and np.isfinite(beta)):
        return prepared, info
    if alpha < amin or alpha > amax or abs(beta) > bmax:
        info.update({'heel_rows': int(ok.sum()), 'heel_alpha': alpha, 'heel_beta': beta})
        return prepared, info
    calibrated = (prepared - beta) / max(alpha, 1e-12)
    raw_resid = obs - ref
    cal_resid = ((obs - beta) / max(alpha, 1e-12)) - ref
    info.update({
        'heel_calibrated': True,
        'heel_rows': int(ok.sum()),
        'heel_alpha': alpha,
        'heel_beta': beta,
        'heel_rmse_raw': float(np.sqrt(np.nanmean(raw_resid * raw_resid))),
        'heel_rmse_calibrated': float(np.sqrt(np.nanmean(cal_resid * cal_resid))),
    })
    return calibrated, info


def _selector_heel_report_fields(info):
    info = info or {}
    return {
        'heel_calibrated': bool(info.get('heel_calibrated', False)),
        'heel_rows': int(info.get('heel_rows', 0) or 0),
        'heel_alpha': float(info.get('heel_alpha', np.nan)) if np.isfinite(info.get('heel_alpha', np.nan)) else np.nan,
        'heel_beta': float(info.get('heel_beta', np.nan)) if np.isfinite(info.get('heel_beta', np.nan)) else np.nan,
        'heel_rmse_raw': float(info.get('heel_rmse_raw', np.nan)) if np.isfinite(info.get('heel_rmse_raw', np.nan)) else np.nan,
        'heel_rmse_calibrated': float(info.get('heel_rmse_calibrated', np.nan)) if np.isfinite(info.get('heel_rmse_calibrated', np.nan)) else np.nan,
        'heel_denoised': bool(info.get('heel_denoised', False)),
    }

def _selector_gr_misfit(hw, tw, tvt_path, eval_mask=None):
    tw_tvt, tw_gr = _selector_tw_gr_arrays(tw)
    if tw_tvt is None:
        return np.nan, 0
    if hw is None or 'GR' not in hw.columns or tvt_path is None:
        return np.nan, 0
    hgr = pd.to_numeric(hw['GR'], errors='coerce').interpolate(limit_direction='both').to_numpy(dtype=float)
    hgr, _heel_info = _selector_apply_heel_calibration(hw, tw_tvt, tw_gr, hgr)
    path = np.asarray(tvt_path, dtype=float)
    n = min(len(hgr), len(path))
    mask = _selector_eval_mask(hw, n) if eval_mask is None else np.asarray(eval_mask, dtype=bool)[:n]
    mask &= np.isfinite(hgr[:n]) & np.isfinite(path[:n])
    if int(mask.sum()) < int(globals().get('BIMODAL_MIN_VALID_GR_ROWS', 80)):
        return np.nan, int(mask.sum())
    pred_gr = np.interp(path[:n][mask], tw_tvt, tw_gr)
    resid = hgr[:n][mask] - pred_gr
    scale = _selector_gr_scale(hw, tw_tvt, tw_gr, hgr[:n], fallback_resid=resid)
    z = np.clip(resid / scale, -6.0, 6.0)
    return float(np.nanmean(z * z)), int(mask.sum())


def _selector_lag1_autocorr(resid):
    x = np.asarray(resid, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    x0 = x[:-1] - np.nanmean(x[:-1])
    x1 = x[1:] - np.nanmean(x[1:])
    denom = float(np.sqrt(np.sum(x0 * x0) * np.sum(x1 * x1)))
    if denom <= 1e-12 or not np.isfinite(denom):
        return 0.0
    return float(np.clip(np.sum(x0 * x1) / denom, 0.0, 0.999))


def _selector_temp_from_resid(resid, n_valid):
    legacy = max(float(globals().get('BIMODAL_TEMP', 0.75)), 1e-6)
    if not bool(globals().get('RUN_ADAPTIVE_TEMP', False)):
        return legacy, np.nan, np.nan
    rho1 = _selector_lag1_autocorr(resid)
    n_eff = float(max(n_valid, 1)) * (1.0 - rho1) / max(1.0 + rho1, 1e-6)
    temp = 2.0 / max(n_eff, 1.0)
    temp = float(np.clip(temp, float(globals().get('T_MIN', 0.25)), float(globals().get('T_MAX', 4.0))))
    return temp, float(rho1), float(n_eff)


def _selector_prefix_trust(hw, tw, base, delta_b):
    fallback = float(globals().get('TRUST_FALLBACK', 0.0))
    if not bool(globals().get('RUN_PREFIX_TRUST_GATE', False)):
        return 1.0, np.nan, np.nan, np.nan, 0
    if hw is None or 'TVT_input' not in hw.columns:
        return fallback, np.nan, np.nan, np.nan, 0
    n = min(len(hw), len(base))
    prefix_mask = hw['TVT_input'].notna().to_numpy()[:n]
    try:
        j_pre0, prefix_rows = _selector_gr_misfit(hw, tw, base, eval_mask=prefix_mask)
        j_predb, _ = _selector_gr_misfit(hw, tw, np.asarray(base, dtype=float) + float(delta_b), eval_mask=prefix_mask)
    except Exception:
        return fallback, np.nan, np.nan, np.nan, 0
    if int(prefix_rows) < int(globals().get('TRUST_MIN_PREFIX_ROWS', 60)):
        return fallback, j_pre0, j_predb, np.nan, int(prefix_rows)
    if not (np.isfinite(j_pre0) and np.isfinite(j_predb)):
        return fallback, j_pre0, j_predb, np.nan, int(prefix_rows)
    margin = float(j_predb - j_pre0)
    denom = max(float(globals().get('TRUST_KAPPA', 1.0)) * max(float(j_pre0), 1e-12), 1e-12)
    trust = float(np.clip(margin / denom, 0.0, 1.0))
    return trust, float(j_pre0), float(j_predb), margin, int(prefix_rows)


def selector_bimodal_scan(hw, tw, base_path, eval_mask=None):
    tw_tvt, tw_gr = _selector_tw_gr_arrays(tw)
    if tw_tvt is None or hw is None or 'GR' not in hw.columns or base_path is None:
        return None
    hgr = pd.to_numeric(hw['GR'], errors='coerce').interpolate(limit_direction='both').to_numpy(dtype=float)
    hgr, heel_info = _selector_apply_heel_calibration(hw, tw_tvt, tw_gr, hgr)
    heel_fields = _selector_heel_report_fields(heel_info)
    base = np.asarray(base_path, dtype=float)
    n = min(len(hgr), len(base))
    mask = _selector_eval_mask(hw, n) if eval_mask is None else np.asarray(eval_mask, dtype=bool)[:n]
    mask &= np.isfinite(hgr[:n]) & np.isfinite(base[:n])
    if int(mask.sum()) < int(globals().get('BIMODAL_MIN_VALID_GR_ROWS', 80)):
        return None

    h = hgr[:n][mask]
    b = base[:n][mask]
    pred0 = np.interp(b, tw_tvt, tw_gr)
    resid0 = h - pred0
    scale = _selector_gr_scale(hw, tw_tvt, tw_gr, hgr[:n], fallback_resid=resid0)

    dz_range = float(globals().get('BIMODAL_DZ_RANGE', 20.0))
    dz_step = float(globals().get('BIMODAL_DZ_STEP', 0.5))
    if dz_step <= 0:
        dz_step = 0.5
    deltas = np.arange(-dz_range, dz_range + 0.5 * dz_step, dz_step, dtype=float)
    scores = []
    for dz in deltas:
        pred_gr = np.interp(b + dz, tw_tvt, tw_gr)
        z = np.clip((h - pred_gr) / scale, -6.0, 6.0)
        scores.append(float(np.nanmean(z * z)))
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    if int(finite.sum()) < 3:
        return None

    best_idx = int(np.nanargmin(scores))
    delta_a = float(deltas[best_idx])
    score_a = float(scores[best_idx])
    sep_min = float(globals().get('SCAN_MIN_SEP', 8.0))
    bundle_min = float(globals().get('BUNDLE_MIN', 10.0))
    bundle_max = float(globals().get('BUNDLE_MAX', 20.0))
    sep = np.abs(deltas - delta_a)
    candidate = finite & (sep >= max(sep_min, bundle_min)) & (sep <= bundle_max)
    pred_a = np.interp(b + delta_a, tw_tvt, tw_gr)
    resid_a = h - pred_a
    temp, rho1, n_eff = _selector_temp_from_resid(resid_a, int(mask.sum()))
    if not bool(candidate.any()):
        return {
            'is_bimodal': False,
            'delta_a': delta_a,
            'delta_b': np.nan,
            'delta_star': delta_a,
            'p_base': 1.0,
            'p_eff': 1.0,
            'prefix_trust': 1.0,
            'j_prefix_base': np.nan,
            'j_prefix_decoy': np.nan,
            'prefix_trust_margin': np.nan,
            'prefix_rows': 0,
            'temperature': float(temp),
            'rho1': float(rho1) if np.isfinite(rho1) else np.nan,
            'n_eff': float(n_eff) if np.isfinite(n_eff) else np.nan,
            'score_a': score_a,
            'score_b': np.nan,
            'j_ratio': np.nan,
            'dz_gap': np.nan,
            'valid_gr_rows': int(mask.sum()),
            'forced_midpoint': False,
            **heel_fields,
        }
    idxs = np.flatnonzero(candidate)
    second_idx = int(idxs[np.nanargmin(scores[idxs])])
    delta_b = float(deltas[second_idx])
    score_b = float(scores[second_idx])
    j_ratio = float(score_b / max(score_a, 1e-12))
    dz_gap = float(abs(delta_b - delta_a))
    eps = float(globals().get('BIMODAL_J_RATIO_EPS', 0.15))
    is_bimodal = bool(score_b <= (1.0 + eps) * max(score_a, 1e-12))
    p_base = 1.0 / (1.0 + np.exp(np.clip((score_a - score_b) / max(temp, 1e-6), -50.0, 50.0)))
    trust, j_prefix_base, j_prefix_decoy, trust_margin, prefix_rows = _selector_prefix_trust(hw, tw, base, delta_b)
    if bool(globals().get('BIMODAL_FORCE_MIDPOINT', False)):
        p_eff = 0.5
        forced_midpoint = True
    else:
        p_eff = 0.5 + float(trust) * (float(p_base) - 0.5)
        forced_midpoint = False
    delta_star = float(p_eff * delta_a + (1.0 - p_eff) * delta_b)
    return {
        'is_bimodal': is_bimodal,
        'delta_a': delta_a,
        'delta_b': delta_b,
        'delta_star': delta_star,
        'p_base': float(p_base),
        'p_eff': float(p_eff),
        'prefix_trust': float(trust),
        'j_prefix_base': float(j_prefix_base) if np.isfinite(j_prefix_base) else np.nan,
        'j_prefix_decoy': float(j_prefix_decoy) if np.isfinite(j_prefix_decoy) else np.nan,
        'prefix_trust_margin': float(trust_margin) if np.isfinite(trust_margin) else np.nan,
        'prefix_rows': int(prefix_rows),
        'temperature': float(temp),
        'rho1': float(rho1) if np.isfinite(rho1) else np.nan,
        'n_eff': float(n_eff) if np.isfinite(n_eff) else np.nan,
        'score_a': score_a,
        'score_b': score_b,
        'j_ratio': j_ratio,
        'dz_gap': dz_gap,
        'valid_gr_rows': int(mask.sum()),
        'forced_midpoint': bool(forced_midpoint),
        **heel_fields,
    }

def _bimodal_selector_weight(base, beam, hw=None, tw=None):
    base = np.asarray(base, dtype=float)
    beam = np.asarray(beam, dtype=float)
    n = min(len(base), len(beam))
    if n == 0:
        return None
    eval_mask = _selector_eval_mask(hw, n)
    diff = np.abs(base[:n] - beam[:n])
    valid = eval_mask & np.isfinite(diff)
    if int(valid.sum()) < int(globals().get('BIMODAL_MIN_VALID_GR_ROWS', 80)):
        return None
    med_diff = float(np.nanmedian(diff[valid]))
    p90_diff = float(np.nanquantile(diff[valid], 0.90))
    big_frac = float(np.nanmean(diff[valid] >= float(globals().get('BIMODAL_TRIGGER_BIG_DIFF_THRESHOLD', 15.0))))
    if med_diff < float(globals().get('BIMODAL_TRIGGER_MIN_MEDIAN_DIFF', 12.0)) and p90_diff < float(globals().get('BIMODAL_TRIGGER_MIN_P90_DIFF', 18.0)):
        return None
    if big_frac < float(globals().get('BIMODAL_TRIGGER_MIN_BIG_DIFF_FRAC', 0.18)):
        return None

    scan = selector_bimodal_scan(hw, tw, base, eval_mask=eval_mask)
    if not scan or not scan.get('is_bimodal'):
        return None
    return {
        'delta_star': float(scan['delta_star']),
        'p_base': float(scan['p_base']),
        'p_eff': float(scan.get('p_eff', scan['p_base'])),
        'prefix_trust': float(scan.get('prefix_trust', np.nan)),
        'j_prefix_base': float(scan.get('j_prefix_base', np.nan)),
        'j_prefix_decoy': float(scan.get('j_prefix_decoy', np.nan)),
        'prefix_trust_margin': float(scan.get('prefix_trust_margin', np.nan)),
        'prefix_rows': int(scan.get('prefix_rows', 0)),
        'temperature': float(scan.get('temperature', np.nan)),
        'rho1': float(scan.get('rho1', np.nan)),
        'n_eff': float(scan.get('n_eff', np.nan)),
        'forced_midpoint': bool(scan.get('forced_midpoint', False)),
        'score_base': float(scan['score_a']),
        'score_second': float(scan['score_b']),
        'delta_a': float(scan['delta_a']),
        'delta_b': float(scan['delta_b']),
        'j_ratio': float(scan['j_ratio']),
        'dz_gap': float(scan['dz_gap']),
        'median_abs_diff': med_diff,
        'p90_abs_diff': p90_diff,
        'big_diff_frac': big_frac,
        'valid_gr_rows': int(scan['valid_gr_rows']),
        'heel_calibrated': bool(scan.get('heel_calibrated', False)),
        'heel_rows': int(scan.get('heel_rows', 0)),
        'heel_alpha': float(scan.get('heel_alpha', np.nan)),
        'heel_beta': float(scan.get('heel_beta', np.nan)),
        'heel_rmse_raw': float(scan.get('heel_rmse_raw', np.nan)),
        'heel_rmse_calibrated': float(scan.get('heel_rmse_calibrated', np.nan)),
        'heel_denoised': bool(scan.get('heel_denoised', False)),
    }


def apply_selector_variant(name, pf_by_scale, tvt_beam, last_known_tvt, hw=None, tw=None, return_info=False):
    scale, beam_weight, hold_weight = parse_selector_variant(name)
    base = pf_by_scale.get(f'pf_scale_{scale:g}')
    if base is None:
        base = pf_by_scale[SELECTOR_GLOBAL_VARIANT.split('_beam_')[0].split('_hold_')[0]]
    base = np.asarray(base, dtype=float)
    tvt_beam = np.asarray(tvt_beam, dtype=float)
    info = {
        'bimodal_active': False,
        'base_beam_weight': float(beam_weight),
        'effective_beam_weight': float(beam_weight),
        'delta_star': 0.0,
        'p_base': np.nan,
        'p_eff': np.nan,
        'prefix_trust': np.nan,
        'temperature': np.nan,
        'rho1': np.nan,
        'n_eff': np.nan,
        'heel_calibrated': False,
        'heel_rows': 0,
        'heel_alpha': np.nan,
        'heel_beta': np.nan,
        'heel_rmse_raw': np.nan,
        'heel_rmse_calibrated': np.nan,
        'heel_denoised': False,
    }
    if bool(globals().get('RUN_BIMODAL_DETECTOR', globals().get('RUN_BIMODAL_SELECTOR_HEDGE', False))):
        hedge = _bimodal_selector_weight(base, tvt_beam, hw=hw, tw=tw)
        if hedge is not None:
            info.update(hedge)
            info['bimodal_active'] = True
            info['effective_beam_weight'] = float(beam_weight)
            pred = base + float(hedge['delta_star'])
        else:
            pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam
    else:
        pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    if return_info:
        return pred, info
    return pred

# %% [markdown]
# ## 4. Ridge/PF anchor and shared deterministic candidate surface

# %%
SEED=42
NCPU=min(4,multiprocessing.cpu_count())

FORMATIONS=["ANCC","ASTNU","ASTNL","EGFDU","EGFDL","BUDA"]
PLANE_K=10; DENSE_SPW=60; DENSE_K=20; N_SPLITS=5

BEAMS=[
    (10,20.0,144.0,2,"cons"),
    (10, 8.0, 64.0,2,"loose"),
    ( 8,35.0,220.0,1,"vcons"),
    (10,14.0, 90.0,5,"sm5"),
    (20, 4.0, 36.0,3,"vloose"),
    (12,12.0,100.0,3,"mid"),
    (15,25.0,180.0,2,"stiff"),
]

PF_N=600; ANCC_N=600
PF_MOM=0.993; PF_VN=0.005; PF_PN=0.01
PF_GR_SIG_MIN=10.; PF_GR_SIG_MAX=60.; PF_GR_SIG_DEF=30.
PF_INIT_V_STD=0.02; PF_INIT_SPR=0.5; PF_RESAMP=0.5
PF_ROUGH_P=0.2; PF_ROUGH_V=0.003; PF_GR_WIN=5; PF_GR_WT=0.3
ANCC_ALPHA=0.998; ANCC_RN=0.002; ANCC_PN=0.005
ANCC_IR=0.01; ANCC_IS=0.3; ANCC_RP=0.1; ANCC_RR=0.001

@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t

@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1]=cum[j]+w[j]
    u0=np.random.uniform(0.,1./N)
    np2=np.empty(N); na=np.empty(N); ci=0
    for j in range(N):
        u=u0+j/N
        while ci<N-1 and cum[ci+1]<u: ci+=1
        np2[j]=pos[ci]+rp*np.random.randn()
        na[j] =aux[ci]+rv*np.random.randn()
    return np2,na

@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    """Beam search Ã‚Â±2 delta, Numba JIT."""
    n=len(sgr); nt=len(tw_gr); MAX=BS*6
    bidx=np.zeros(BS,np.int64); bidx[0]=si
    bcost=np.full(BS,1e30);     bcost[0]=0.; bn=np.int64(1)
    hI=np.zeros((n,BS),np.int64); hP=np.zeros((n,BS),np.int64)
    cI=np.zeros(MAX,np.int64); cC=np.full(MAX,1e30); cP=np.zeros(MAX,np.int64)
    for step in range(n):
        gv=sgr[step]; nc=np.int64(0)
        for bi in range(bn):
            idx=bidx[bi]; cost=bcost[bi]
            for d in range(-2,3):            # Ã‚Â±2: TVT can go down
                ni=idx+d
                if ni<0 or ni>=nt: continue
                tot=cost+(gv-tw_gr[ni])**2/es+mc*(d if d>=0 else -d)
                fnd=np.int64(-1)
                for ci in range(nc):
                    if cI[ci]==ni: fnd=ci; break
                if fnd>=0:
                    if tot<cC[fnd]: cC[fnd]=tot; cP[fnd]=bi
                else:
                    if nc<MAX: cI[nc]=ni; cC[nc]=tot; cP[nc]=bi; nc+=1
        kept=min(BS,nc)
        for i in range(kept):
            mi=i
            for j in range(i+1,nc):
                if cC[j]<cC[mi]: mi=j
            if mi!=i:
                cI[i],cI[mi]=cI[mi],cI[i]
                cC[i],cC[mi]=cC[mi],cC[i]
                cP[i],cP[mi]=cP[mi],cP[i]
        hI[step,:kept]=cI[:kept]; hP[step,:kept]=cP[:kept]
        bidx[:kept]=cI[:kept]; bcost[:kept]=cC[:kept]; bn=kept
    best=np.int64(0)
    for b in range(1,bn):
        if bcost[b]<bcost[best]: best=b
    path=np.zeros(n,np.int64); b=best
    for s in range(n-1,-1,-1): path[s]=hI[s,b]; b=hP[s,b]
    return path

@njit(cache=True)
def _pf_ancc(md_v,z_v,gr_v,gg,vmin,step,gs,ls,ir,N,
              ALPHA,RN,PN,IS,RP,RR,RESAMP):
    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ls+IS*np.random.randn()
        rate[j]=ir+0.01*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        for j in range(N):
            rate[j]=ALPHA*rate[j]+RN*np.random.randn()
            pos[j]+=rate[j]*dm+PN*np.random.randn()
            tvt_j=pos[j]-z_v[i]
            tvt_j=max(tvt_j,vmin-50.); tvt_j=min(tvt_j,vmin+len(gg)*step+50.)
            pos[j]=tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                eg=_interp1(gg,pos[j]-z_v[i],vmin,step)
                d=(gr_v[i]-eg)/gs
                lk=max(np.exp(-0.5*d*d) if d*d<600. else 0.,1e-300)
                w[j]*=lk; ws+=w[j]
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,rate=_resamp(pos,rate,w,N,RP,RR)
            for j in range(N): w[j]=1./N
        tv=0.
        for j in range(N): tv+=w[j]*(pos[j]-z_v[i])
        pts[i]=tv; va=0.
        for j in range(N): va+=w[j]*(pos[j]-z_v[i]-tv)**2
        std_[i]=va**0.5; pm=md_v[i]
    return pts,std_

@njit(cache=True)
def _pf_z(md_v,z_v,gr_v,gr_sm_v,gg_p,gg_s,vmin,step,
          gs,ip,iv,beta,icpt,zsig,N,
          MOM,VN,PN,GR_WT,RP,RV,RESAMP):
    pos=np.empty(N); vel=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ip+0.5*np.random.randn()
        vel[j]=iv+0.02*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.; pz=z_v[0]-1.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        dzd=(z_v[i]-pz)/dm; ve=beta*dzd+icpt
        for j in range(N):
            vel[j]=MOM*vel[j]+VN*np.random.randn()
            pos[j]+=vel[j]*dm+PN*np.random.randn()
            pos[j]=max(pos[j],vmin-50.); pos[j]=min(pos[j],vmin+len(gg_p)*step+50.)
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                ep=_interp1(gg_p,pos[j],vmin,step)
                dp=(gr_v[i]-ep)/gs
                lp=max(np.exp(-0.5*dp*dp) if dp*dp<600. else 0.,1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es=_interp1(gg_s,pos[j],vmin,step)
                    ds=(gr_sm_v[i]-es)/(gs*1.5)
                    ls=max(np.exp(-0.5*ds*ds) if ds*ds<600. else 0.,1e-300)
                    lk=(1.-GR_WT)*lp+GR_WT*ls
                else: lk=lp
                lk=max(lk,1e-300); w[j]*=lk; ws+=w[j]
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ws2=0.
        for j in range(N):
            dv=(vel[j]-ve)/max(zsig*2.,0.005)
            lz=max(np.exp(-0.5*dv*dv) if dv*dv<600. else 0.,1e-300)
            w[j]*=lz; ws2+=w[j]
        if ws2>0.:
            for j in range(N): w[j]/=ws2
        else:
            for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,vel=_resamp(pos,vel,w,N,RP,RV)
            for j in range(N): w[j]=1./N
        wm=0.
        for j in range(N): wm+=w[j]*pos[j]
        pts[i]=wm; va=0.
        for j in range(N): va+=w[j]*(pos[j]-wm)**2
        std_[i]=va**0.5; pm=md_v[i]; pz=z_v[i]
    return pts,std_

# Dense grid for O(1) typewell lookup
def _grid(tw_tvt,tw_gr,step=0.2):
    tmin=float(tw_tvt.min()); tmax=float(tw_tvt.max())
    tvt_g=np.arange(tmin,tmax+step,step)
    return np.interp(tvt_g,tw_tvt,tw_gr).astype(np.float64),float(tmin),float(step)

def _gr_sig(hw,tw_tvt,tw_gr):
    kn=hw[hw['TVT_input'].notna()&hw['GR'].notna()]
    if len(kn)<20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn['GR'].values-np.interp(kn['TVT_input'].values,tw_tvt,tw_gr)),
                          PF_GR_SIG_MIN,PF_GR_SIG_MAX))

def _nn(arr,v):
    i=int(np.searchsorted(arr,v,'left'))
    if i>=len(arr): return len(arr)-1
    if i>0 and abs(arr[i-1]-v)<=abs(arr[i]-v): return i-1
    return i

def _smooth(vals,fb,r):
    s=pd.Series(vals,dtype='float32').interpolate(limit_direction='both').fillna(fb)
    return (s.rolling(r*2+1,center=True,min_periods=1).mean() if r>0 else s).to_numpy(np.float32)

def beam_search(gr_h,tw_tvt,tw_gr,start_tvt,bs,mc,es,r):
    si=_nn(tw_tvt,start_tvt)
    sgr=_smooth(gr_h,float(np.nanmean(tw_gr)),r).astype(np.float64)
    path=_beam_jit(sgr,tw_gr.astype(np.float64),si,bs,float(mc),float(es))
    return tw_tvt[path].astype(np.float32)

def run_pf_ancc(hw,tw_tvt,tw_gr,N=ANCC_N):
    gs=_gr_sig(hw,tw_tvt,tw_gr)
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([])
    ls=float(kn['TVT_input'].iloc[-1]+kn['Z'].iloc[-1])
    tail=kn.tail(30); dt=np.diff(tail['TVT_input'].values)
    dz=np.diff(tail['Z'].values); dm=np.diff(tail['MD'].values); m=dm>0
    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr)
    pts,std=_pf_ancc(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                      ev['GR'].values.astype(np.float64),gg,gmin,gst,
                      gs,ls,ir,N,ANCC_ALPHA,ANCC_RN,ANCC_PN,ANCC_IS,ANCC_RP,ANCC_RR,PF_RESAMP)
    return pts.astype(np.float32),std.astype(np.float32)

def run_pf_z(hw,tw_tvt,tw_gr,N=PF_N):
    gs=_gr_sig(hw,tw_tvt,tw_gr)
    tw_s=pd.Series(tw_gr).rolling(PF_GR_WIN,center=True,min_periods=1).mean().values.astype(np.float32)
    kna=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([])
    dz_k=np.diff(kna['Z'].values); dvt=np.diff(kna['TVT_input'].values)
    dmd_k=np.diff(kna['MD'].values); m2=dmd_k>0
    if m2.sum()>=10:
        vz=dz_k[m2]/dmd_k[m2]; vt=dvt[m2]/dmd_k[m2]
        A=np.column_stack([vz,np.ones_like(vz)]); c,_,_,_=np.linalg.lstsq(A,vt,rcond=None)
        beta,icpt,zsig=float(c[0]),float(c[1]),max(float(np.std(vt-(c[0]*vz+c[1]))),0.001)
    else: beta,icpt,zsig=-1.,0.,0.1
    t2=kna.tail(20); dvt2=np.diff(t2['TVT_input'].values); dmd2=np.diff(t2['MD'].values); m3=dmd2>0
    iv=float(np.median(dvt2[m3]/dmd2[m3])) if m3.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr)
    gs2,_,_=_grid(tw_tvt,tw_s)
    gr_sm=hw['GR'].rolling(PF_GR_WIN,center=True,min_periods=1).mean()
    pts,std=_pf_z(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                   ev['GR'].values.astype(np.float64),
                   gr_sm.loc[ev.index].values.astype(np.float64),
                   gg,gs2,gmin,gst,gs,float(kna['TVT_input'].iloc[-1]),iv,
                   beta,icpt,zsig,N,
                   PF_MOM,PF_VN,PF_PN,PF_GR_WT,PF_ROUGH_P,PF_ROUGH_V,PF_RESAMP)
    return pts.astype(np.float32),std.astype(np.float32)


_md=np.linspace(1,50,20,np.float64); _z=np.zeros(20,np.float64); _gr=np.full(20,50.,np.float64)
_gg=np.linspace(45,55,100,np.float64)
_pf_ancc(_md,_z,_gr,_gg,45.,0.1,20.,50.,0.,8,0.998,0.002,0.005,0.3,0.1,0.001,0.5)
_pf_z(_md,_z,_gr,_gr,_gg,_gg,45.,0.1,20.,50.,0.,-1.,0.,0.1,8,0.993,0.005,0.01,0.3,0.2,0.003,0.5)
_beam_jit(np.random.randn(30),np.random.randn(50),25,8,15.,100.)

def robust_slope(x,y,w=None):
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<2 or np.std(x[m])<1e-6: return 0.
    return float(np.polyfit(x[m],y[m],1)[0])

def affine_cal(kgr,tw_at_k,min_pts=20):
    v=np.isfinite(kgr)&np.isfinite(tw_at_k)
    if v.sum()<min_pts or np.std(tw_at_k[v])<1e-6:
        return 1.,float(np.nanmean(kgr)-np.nanmean(tw_at_k)) if v.any() else 0.
    a,b=np.polyfit(tw_at_k[v],kgr[v],1); return float(a),float(b)

def seg_b_well(ktvt,kz,form_col):
    """Segment b_well: early/mid/late thirds + full prefix.
    Returns (b_full, b_early, b_mid, b_late, b_wls) for feature richness."""
    bv=ktvt+kz-form_col; n=len(bv)
    b_full=float(np.median(bv))
    b_late=float(np.median(bv[max(0,n-50):])) if n>=5 else b_full
    t1,t2=n//3, 2*n//3
    b_early=float(np.median(bv[:max(1,t1)])) if t1>0 else b_full
    b_mid  =float(np.median(bv[t1:max(t1+1,t2)])) if t2>t1 else b_full
    # WLS (tail-upweighted)
    w=np.exp(0.02*np.arange(n)); w/=w.sum()
    b_wls=float(np.dot(w,bv))
    return b_full,b_early,b_mid,b_late,b_wls

def multi_scale_ncc(kgr,ktvt,hgr,hws=(8,15,25),stride=3):
    """Multi-scale NCC. Returns score-weighted ensemble + per-scale signals."""
    out=[]
    for hw in hws:
        win=2*hw+1; nk=len(kgr); nh=len(hgr)
        if nk<win+1 or nh==0:
            out.append((np.full(nh,ktvt[-1],np.float32),np.zeros(nh,np.float32))); continue
        kg=pd.Series(kgr).rolling(5,center=True,min_periods=1).mean().values.astype(np.float32)
        hg=pd.Series(hgr).rolling(5,center=True,min_periods=1).mean().values.astype(np.float32)
        sts=np.arange(0,nk-win+1,stride,dtype=np.int32); M=len(sts)
        if M==0:
            out.append((np.full(nh,ktvt[-1],np.float32),np.zeros(nh,np.float32))); continue
        C=kg[sts[:,None]+np.arange(win,dtype=np.int32)[None,:]].astype(np.float32)
        Cn=(C-C.mean(1,keepdims=True))/(C.std(1,keepdims=True)+1e-6)
        hp=np.pad(hg,hw,mode='edge')
        H=hp[np.arange(nh)[:,None]+np.arange(win)[None,:]].astype(np.float32)
        Hn=(H-H.mean(1,keepdims=True))/(H.std(1,keepdims=True)+1e-6)
        ncc=Hn@Cn.T/win; best=ncc.argmax(1); score=ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best]+hw,0,nk-1)].astype(np.float32),score))
    # Score-weighted ensemble (NEW: softmax-weighted combination)
    tvts=np.stack([o[0] for o in out],1); scores=np.stack([o[1] for o in out],1)
    sw=np.exp(3.*scores); sw/=sw.sum(1,keepdims=True)+1e-9
    sc_ens=(tvts*sw).sum(1).astype(np.float32)
    return out, sc_ens   # [(tvt8,sc8),(tvt15,sc15),(tvt25,sc25)], ensemble

class FormationPlaneKNN:
    def __init__(self,well_ids,data_dir):
        rows=[]
        for wid in well_ids:
            p=data_dir/f'{wid}__horizontal_well.csv'
            try: df=pd.read_csv(p,usecols=['X','Y']+FORMATIONS).dropna()
            except: continue
            if len(df)==0: continue
            row={'wid':wid,'x':float(df['X'].median()),'y':float(df['Y'].median())}
            for c in FORMATIONS: row[f'{c}_m']=float(df[c].median())
            rows.append(row)
        self.df=pd.DataFrame(rows); self.wmap={w:i for i,w in enumerate(self.df['wid'])}
        xy=self.df[['x','y']].to_numpy(); self.scale=np.where(xy.std(0)<1e-3,1.,xy.std(0))
        self.tree=cKDTree(xy/self.scale)
        self.xa=self.df['x'].to_numpy(); self.ya=self.df['y'].to_numpy()
        self.fa=self.df[[f'{c}_m' for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self,xy_q,self_wid=None,k=PLANE_K):
        q=xy_q/self.scale; nf=min(k+5,len(self.df))
        dist,idx=self.tree.query(q,k=nf,workers=int(globals().get('EXP514_KDTREE_WORKERS', -1)))
        if self_wid in self.wmap: dist=np.where(idx==self.wmap[self_wid],np.inf,dist)
        ord=np.argpartition(dist,min(k-1,nf-1),1)[:,:k]
        dk=np.take_along_axis(dist,ord,1); ik=np.take_along_axis(idx,ord,1)
        vk=np.isfinite(dk); w=np.where(vk,1./(dk+1e-3),0.).astype(np.float64)
        xn=self.xa[ik]; yn=self.ya[ik]; fn=self.fa[ik]; wx=w*xn; wy=w*yn
        A=np.zeros((len(q),3,3))
        A[:,0,0]=(wx*xn).sum(1); A[:,0,1]=(wx*yn).sum(1); A[:,0,2]=wx.sum(1)
        A[:,1,0]=A[:,0,1]; A[:,1,1]=(wy*yn).sum(1); A[:,1,2]=wy.sum(1)
        A[:,2,0]=A[:,0,2]; A[:,2,1]=A[:,1,2]; A[:,2,2]=w.sum(1)
        A[:,0,0]+=1e-9; A[:,1,1]+=1e-9; A[:,2,2]+=1e-9
        rhs=np.stack([(wx[:,:,None]*fn).sum(1),(wy[:,:,None]*fn).sum(1),(w[:,:,None]*fn).sum(1)],1)
        try: coef=np.linalg.solve(A,rhs)
        except:
            coef=np.zeros((len(q),3,6))
            for r in range(len(q)):
                try: coef[r]=np.linalg.pinv(A[r])@rhs[r]
                except: pass
        Xq=xy_q[:,0]; Yq=xy_q[:,1]
        pred=(Xq[:,None]*coef[:,0,:]+Yq[:,None]*coef[:,1,:]+coef[:,2,:]).astype(np.float32)
        pred[~vk.any(1)]=self.fa.mean(0)
        return pred,np.where(vk,dk,np.inf).min(1).astype(np.float32)

class DenseANCCImputer:
    def __init__(self,well_ids,data_dir,spw=DENSE_SPW):
        xs,ys,anccs,wids=[],[],[],[]
        for wid in well_ids:
            p=data_dir/f'{wid}__horizontal_well.csv'
            try: df=pd.read_csv(p,usecols=['X','Y','ANCC']).dropna()
            except: continue
            if len(df)==0: continue
            ix=np.linspace(0,len(df)-1,min(spw,len(df)),dtype=int); s=df.iloc[ix]
            xs.append(s['X'].values); ys.append(s['Y'].values)
            anccs.append(s['ANCC'].values); wids.extend([wid]*len(s))
        self.xy=np.column_stack([np.concatenate(xs),np.concatenate(ys)])
        self.ancc=np.concatenate(anccs).astype(np.float32); self.wids=np.array(wids)
        self.scale=np.where(self.xy.std(0)<1e-3,1.,self.xy.std(0))
        self.tree=cKDTree(self.xy/self.scale)

    def impute(self,xy_q,self_wid=None,k=DENSE_K,nfetch=5000):
        xy_q=np.atleast_2d(xy_q); q=xy_q/self.scale; nf=min(nfetch,len(self.ancc))
        dist,idx=self.tree.query(q,k=nf,workers=int(globals().get('EXP514_KDTREE_WORKERS', -1)))
        if self_wid: dist=np.where(self.wids[idx]==self_wid,np.inf,dist)
        ord=np.argpartition(dist,min(k-1,nf-1),1)[:,:k]
        dk=np.take_along_axis(dist,ord,1); ik=np.take_along_axis(idx,ord,1)
        vk=np.isfinite(dk); w=np.where(vk,1./(dk+1e-3),0.)
        sw=w.sum(1); safe=np.where(sw<1e-9,1.,sw); an=self.ancc[ik]
        ap=(an*w).sum(1)/safe; ap=np.where(sw<1e-9,float(self.ancc.mean()),ap)
        var=((an-ap[:,None])**2*w).sum(1)/safe
        return ap.astype(np.float32),np.sqrt(np.maximum(var,0.)).astype(np.float32),np.where(vk,dk,np.inf).min(1).astype(np.float32)

hw_paths=sorted((CFG.dataset_path / "train").glob('*__horizontal_well.csv'))
train_wids=[p.stem.replace('__horizontal_well','') for p in hw_paths]
FI=FormationPlaneKNN(train_wids,CFG.dataset_path / "train")
DI=DenseANCCImputer(train_wids,CFG.dataset_path / "train")

_FI=FI; _DI=DI
ANCH_OFFS=np.array([-80,-40,-20,-10,-5,0,5,10,20,40,80],np.float32)
BEAM_OFFS=np.array([-40,-20,-10,-5,-3,0,3,5,10,20,40],np.float32)
SC_OFFS  =np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30],np.float32)
PF_OFFS  =np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30],np.float32)

def build_well(hw_path,tw_path,is_train):
    global _FI,_DI
    wid=Path(hw_path).stem.replace('__horizontal_well','')
    try:
        hw=pd.read_csv(hw_path); tw=pd.read_csv(tw_path).sort_values('TVT')
    except: return None
    if is_train and 'TVT' not in hw.columns: return None
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0 or len(kn)<10: return None
    if is_train and hw['TVT'].isna().all(): return None
    tw_tvt=tw['TVT'].to_numpy(np.float32); tw_gr=tw['GR'].to_numpy(np.float32)
    if len(tw_tvt)<3: return None

    pf_a,std_a=run_pf_ancc(hw,tw_tvt,tw_gr)
    if len(pf_a)==0: return None
    pf_z,std_z=run_pf_z(hw,tw_tvt,tw_gr)
    pf_use=pf_a.astype(np.float32); std_use=std_a.astype(np.float32)
    has_z=len(pf_z)==len(pf_a) and not np.any(np.isnan(pf_z))

    lk=kn.iloc[-1]; last_tvt=float(lk['TVT_input'])
    gr_full=hw['GR'].astype(float).interpolate(limit_direction='both').fillna(float(np.nanmean(tw_gr)))
    hgr=gr_full.iloc[ev.index[0]:].to_numpy(np.float32)
    kgr=gr_full.iloc[:len(kn)].to_numpy(np.float32)

    # 7 beams (Numba JIT Ã‚Â±2)
    bpaths={}
    for (bs,mc,es,r,tag) in BEAMS:
        bpaths[tag]=beam_search(hgr,tw_tvt,tw_gr,last_tvt,bs,mc,es,r)
    beam_ref=(bpaths['cons']+bpaths['sm5'])/2.

    # Multi-scale NCC Ã¢â€ â€™ score-weighted ensemble
    ktvt=kn['TVT_input'].to_numpy(np.float32)
    sc_res,sc_ens=multi_scale_ncc(kgr,ktvt,hgr,hws=(8,15,25),stride=3)
    sc8,sc8s=sc_res[0]; sc15,sc15s=sc_res[1]; sc25,sc25s=sc_res[2]
    sc_cons=(sc8+sc15+sc25)/3.
    sc_trust=float(np.clip(len(kn)/200.,0.,0.6))
    hyb_ref=(1-sc_trust)*beam_ref+sc_trust*sc_ens  # use ensemble not single

    tw_at_k=np.interp(ktvt,tw_tvt,tw_gr).astype(np.float32)
    a_cal,b_cal=affine_cal(kgr,tw_at_k)
    kmd=kn['MD'].to_numpy(np.float32); kz=kn['Z'].to_numpy(np.float32)
    pfx_rmse=float(np.sqrt(np.mean((kgr-tw_at_k)**2)))
    slp_all=robust_slope(kmd,ktvt); slp_50=robust_slope(kmd[-50:],ktvt[-50:])
    slp_z=robust_slope(kz,ktvt)

    swid=wid if is_train else None
    xy_ev=ev[['X','Y']].to_numpy(np.float64); xy_kn=kn[['X','Y']].to_numpy(np.float64)
    form_ev,knn_d=_FI.impute(xy_ev,self_wid=swid)
    form_kn,_   =_FI.impute(xy_kn,self_wid=swid)
    z_kn=kn['Z'].to_numpy(np.float32); z_ev=ev['Z'].to_numpy(np.float32)

    # Per-formation: segment b_well (early/mid/late/wls) + TVT + known-zone RMSE
    tvt_fs={}; form_rmse={}; form_list=[]
    for fi2,fn in enumerate(FORMATIONS):
        b_full,b_early,b_mid,b_late,b_wls=seg_b_well(ktvt,z_kn,form_kn[:,fi2])
        tvt_f  =(-z_ev+form_ev[:,fi2]+b_full ).astype(np.float32)
        tvt_fw =(-z_ev+form_ev[:,fi2]+b_wls  ).astype(np.float32)
        tvt_f50=(-z_ev+form_ev[:,fi2]+b_late ).astype(np.float32)
        tvt_fs[f'tvtF_{fn}']=tvt_f; tvt_fs[f'tvtFw_{fn}']=tvt_fw
        tvt_fs[f'tvtF50_{fn}']=tvt_f50
        tvt_fs[f'bw_{fn}']=np.float32(b_full); tvt_fs[f'bww_{fn}']=np.float32(b_wls)
        tvt_fs[f'bw50_{fn}']=np.float32(b_late)
        tvt_fs[f'bw_early_{fn}']=np.float32(b_early)   # NEW: early segment
        tvt_fs[f'bw_mid_{fn}']=np.float32(b_mid)       # NEW: mid segment
        form_rmse[fn]=float(np.sqrt(np.mean((ktvt-(-z_kn+form_kn[:,fi2]+b_full))**2)))
        form_list.append(tvt_f)

    fs=np.stack(form_list,1)
    form_mean_d=(fs.mean(1)-last_tvt).astype(np.float32)
    form_std_d =fs.std(1).astype(np.float32)
    form_rng_d =(fs.max(1)-fs.min(1)).astype(np.float32)

    d_ancc,d_std,d_dist=_DI.impute(xy_ev,self_wid=swid)
    d_kn,d_std_kn,_=_DI.impute(xy_kn,self_wid=swid)
    b_vd=ktvt+z_kn-d_kn
    _,b_de,b_dm,b_dl,b_dw=seg_b_well(ktvt,z_kn,d_kn)
    b_d=float(np.median(b_vd))
    tvt_dense  =(-z_ev+d_ancc+b_d  ).astype(np.float32)
    tvt_densew =(-z_ev+d_ancc+b_dw ).astype(np.float32)
    tvt_dense50=(-z_ev+d_ancc+b_dl ).astype(np.float32)
    res_kn=ktvt+z_kn-d_kn
    d_rmse=float(np.sqrt(np.mean(res_kn**2))); d_bias=float(np.mean(res_kn)); d_nb_std=float(np.mean(d_std_kn))

    all_sigs=[pf_use]+[p for p in bpaths.values()]+[sc8,sc15,sc25,sc_ens,tvt_fs['tvtF_ANCC'],tvt_dense]
    sig_mat=np.stack(all_sigs,1)
    sig_std=sig_mat.std(1).astype(np.float32)
    sig_mean=(sig_mat.mean(1)-last_tvt).astype(np.float32)

    gr_s=pd.Series(gr_full.values); rolls={}
    for w in [5,21,51,101]:
        r=gr_s.rolling(w,center=True,min_periods=1)
        rolls[f'grm{w}']=r.mean().iloc[ev.index].values.astype(np.float32)
        rolls[f'grs{w}']=r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1,5,15,30]:
        rolls[f'glag{lag}']=gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32)
        rolls[f'glead{lag}']=gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1=gr_s.diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_d2=gr_s.diff().diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_env=gr_s.rolling(21,center=True,min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg=np.sqrt(np.maximum((gr_s**2).rolling(21,center=True,min_periods=1).mean(),0.)
                   ).iloc[ev.index].values.astype(np.float32)

    hmd=ev['MD'].to_numpy(np.float32); md_since=hmd-float(lk['MD'])
    slp_b_all=(last_tvt+slp_all*md_since).astype(np.float32)
    slp_b_50 =(last_tvt+slp_50 *md_since).astype(np.float32)

    mdd=hw['MD'].diff().replace(0,np.nan)
    dzdmd=(hw['Z'].diff()/mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd=(hw['X'].diff()/mdd).iloc[ev.index].values.astype(np.float32)
    dydmd=(hw['Y'].diff()/mdd).iloc[ev.index].values.astype(np.float32)

    nh=len(ev); frac=(np.arange(nh)/max(nh-1,1)).astype(np.float32)
    def sc(v): return np.full(nh,np.float32(v),np.float32)

    feats={
        'well':wid,'id':[f'{wid}_{i}' for i in ev.index],
        'last_known_tvt':sc(last_tvt),
        'pf_ancc':pf_use,'pf_ancc_std':std_use,
        'pf_ancc_delta':(pf_use-last_tvt).astype(np.float32),
        'pf_z':(pf_z.astype(np.float32) if has_z else sc(last_tvt)),
        'pf_z_delta':((pf_z-last_tvt).astype(np.float32) if has_z else sc(0.)),
        'pf_vs_z':((pf_use-pf_z.astype(np.float32)) if has_z else sc(0.)),
        **{f'__exp514_shared_beam_abs_{t}':p.astype(np.float32) for t,p in bpaths.items()},
        '__exp514_shared_sc8_abs':sc8.astype(np.float32),
        '__exp514_shared_sc15_abs':sc15.astype(np.float32),
        '__exp514_shared_sc25_abs':sc25.astype(np.float32),
        '__exp514_shared_sc_ens_abs':sc_ens.astype(np.float32),
        '__exp514_shared_tvt_dense_abs':tvt_dense.astype(np.float32),
        **{f'beam_{t}_d':(p-np.float32(last_tvt)).astype(np.float32) for t,p in bpaths.items()},
        'beam_mean_d':np.stack([(p-last_tvt) for p in bpaths.values()],1).mean(1).astype(np.float32),
        'beam_std_d': np.stack([(p-last_tvt) for p in bpaths.values()],1).std(1).astype(np.float32),
        'beam_med_d': np.median(np.stack([(p-last_tvt) for p in bpaths.values()],1),1).astype(np.float32),
        'sc8_d':(sc8-np.float32(last_tvt)).astype(np.float32),'sc8_sc':sc8s,
        'sc15_d':(sc15-np.float32(last_tvt)).astype(np.float32),'sc15_sc':sc15s,
        'sc25_d':(sc25-np.float32(last_tvt)).astype(np.float32),'sc25_sc':sc25s,
        'sc_cons_d':(sc_cons-np.float32(last_tvt)).astype(np.float32),
        'sc_ens_d':(sc_ens-np.float32(last_tvt)).astype(np.float32),  # score-weighted ensemble
        'sc_trust':sc(sc_trust),'hyb_d':(hyb_ref-np.float32(last_tvt)).astype(np.float32),
        'sig_std':sig_std,'sig_mean_d':sig_mean,
        **tvt_fs,
        **{f'frm_rmse_{fn}':sc(form_rmse[fn]) for fn in FORMATIONS},
        'form_mean_d':form_mean_d,'form_std_d':form_std_d,'form_rng_d':form_rng_d,
        'spatial_ancc_d':(form_ev[:,0]-np.float32(np.interp(last_tvt,tw_tvt,tw_gr))),
        'spatial_knn_dist':knn_d,
        'dense_ancc':d_ancc,'dense_std':d_std,'dense_dist':d_dist,
        'tvt_dense_d' :(tvt_dense -last_tvt).astype(np.float32),
        'tvt_densew_d':(tvt_densew-last_tvt).astype(np.float32),
        'tvt_dense50_d':(tvt_dense50-last_tvt).astype(np.float32),
        'dense_rmse':sc(d_rmse),'dense_bias':sc(d_bias),'dense_nb_std':sc(d_nb_std),
        'pf_vs_spatial':(pf_use-tvt_fs['tvtF_ANCC']).astype(np.float32),
        'pf_vs_dense':(pf_use-tvt_dense).astype(np.float32),
        'spatial_vs_dense':(tvt_fs['tvtF_ANCC']-tvt_dense).astype(np.float32),
        'beam_vs_spatial':(bpaths['cons']-tvt_fs['tvtF_ANCC']).astype(np.float32),
        'sc_vs_beam':(sc_ens-bpaths['cons']).astype(np.float32),
        'cal_a':sc(a_cal),'cal_b':sc(b_cal),
        'pfx_rmse':sc(pfx_rmse),'known_len':sc(len(kn)),'eval_len':sc(nh),
        'slp_all':sc(slp_all),'slp_50':sc(slp_50),'slp_z':sc(slp_z),
        'slp_b_d_all':(slp_b_all-last_tvt).astype(np.float32),
        'slp_b_d_50': (slp_b_50 -last_tvt).astype(np.float32),
        'ktvt_range':sc(float(np.ptp(ktvt))),'ktvt_std':sc(float(ktvt.std())),
        'md_since':md_since,'frac':frac,'frac2':frac**2,'sqrt_frac':np.sqrt(frac),
        'z':z_ev,
        'dx':(ev['X']-float(lk['X'])).to_numpy(np.float32),
        'dy':(ev['Y']-float(lk['Y'])).to_numpy(np.float32),
        'dz':(z_ev-float(lk['Z'])).astype(np.float32),
        'dxy':np.sqrt((ev['X']-float(lk['X']))**2+(ev['Y']-float(lk['Y']))**2).to_numpy(np.float32),
        'dzdmd':dzdmd,'dxdmd':dxdmd,'dydmd':dydmd,
        'gr':hgr,'gr_d1':gr_d1,'gr_d2':gr_d2,'gr_env':gr_env,'gr_nrg':gr_nrg,
        'gr_vs_tw_anc':hgr-np.float32(np.interp(last_tvt,tw_tvt,tw_gr)),
        'gr_vs_slp_all':hgr-np.interp(slp_b_all,tw_tvt,tw_gr).astype(np.float32),
        **{f'tda{int(o)}' :hgr-np.float32(np.interp(last_tvt+o,tw_tvt,tw_gr)) for o in ANCH_OFFS},
        **{f'tdbc{int(o)}':hgr-np.interp(beam_ref+o,tw_tvt,tw_gr).astype(np.float32) for o in BEAM_OFFS},
        **{f'tdsc{int(o)}':hgr-np.interp(sc_ens+o,tw_tvt,tw_gr).astype(np.float32) for o in SC_OFFS},
        **{f'tdpf{int(o)}':hgr-np.interp(pf_use+o,tw_tvt,tw_gr).astype(np.float32) for o in PF_OFFS},
        'tw_range':sc(float(np.ptp(tw_tvt))),'tw_gr_mean':sc(float(tw_gr.mean())),
    }
    for k,v in rolls.items(): feats[k]=v
    result=pd.DataFrame(feats)
    if is_train:
        if 'TVT' not in ev.columns or ev['TVT'].isna().all(): return None
        result['target']=(ev['TVT'].to_numpy(np.float32)-np.float32(last_tvt))
    return result

def build_dataset(paths,is_train,label):
    args=[(str(p),str(p.parent/f'{p.stem.replace("__horizontal_well","")}__typewell.csv'),is_train)
          for p in paths
          if (p.parent/f'{p.stem.replace("__horizontal_well","")}__typewell.csv').exists()]
    t0=time.time()
    res=Parallel(n_jobs=NCPU,prefer='threads',verbose=3)(
        delayed(build_well)(hp,tp,it) for hp,tp,it in args)
    parts=[r for r in res if r is not None]
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()

# %% [markdown]
# ## 5. Saved ridge artifact inference and runtime Ridge

# %%
_ridge_train_path = CFG.artifacts_path / "data" / "train.csv"
if not _ridge_train_path.is_file():
    raise FileNotFoundError(f"required source ridge feature table is missing: {_ridge_train_path}")
train_df = pd.read_csv(_ridge_train_path, low_memory=False)

test_paths = sorted((CFG.dataset_path / "test").glob("*__horizontal_well.csv"))
test_df = build_dataset(test_paths, is_train=False, label="test")
SP45_SHARED_TEST_FEATURE_FRAME = test_df
SP45_SHARED_IMPUTERS = (FI, DI)

features = [c for c in train_df.columns if c not in {"well", "id", "target"}]
missing_test_features = [column for column in features if column not in test_df]
if missing_test_features:
    raise RuntimeError(f"source ridge test features are missing: {missing_test_features[:40]}")

X = train_df[features]
y = train_df["target"]
g = train_df["well"]
X_test = test_df[features]

# %%
ridge_params = {
    "random_state": 42,
    "alpha": 1.6602834637650032,
    "tol": 0.0005030247295617308,
    "positive": True,
    "fit_intercept": True,
}

pp_params = {"alpha": 1.0, "tau": 85, "w_pf": 0.09}

# %%
oof_preds = {}
test_preds = {}

overall_scores = {}
fold_scores = {}

# %%
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

# %%
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

# %%
oof_preds = pd.DataFrame(oof_preds)
test_preds = pd.DataFrame(test_preds)

# %%
ridge_trainer = Trainer(
    Ridge(**ridge_params),
    task="regression",
    metric=CFG.metric,
    cv=CFG.cv,
    cv_args={"groups": g},
    verbose=True
)

ridge_trainer.fit(oof_preds, y)

ridge_oof_preds = ridge_trainer.oof_preds
ridge_test_preds = ridge_trainer.predict(test_preds)

overall_scores["ridge"] = ridge_trainer.overall_score
fold_scores["ridge"] = ridge_trainer.fold_scores

# %%
def apply_pp(df, md, pd_, alpha, tau, w_pf):
    d = md * (1-w_pf) + pd_ * w_pf
    if tau: 
        d *= (1.-np.exp(-np.maximum(df['md_since'].values,0.) / tau))
        
    return d * alpha

def sg_smooth(df, col, sg_w=17, sg_p=3):
    df = df.copy()
    
    for _, g in df.groupby('well', sort=False):
        v = g[col].values
        n = len(v)
        wl = min(sg_w, n)
        
        if wl % 2 == 0: 
            wl -= 1
            
        if wl >= sg_p + 2: 
            v = savgol_filter(v, wl, sg_p)
            
        df.loc[g.index,col] = v
        
    return df

# %%
base = train_df['last_known_tvt'].values
ytrue = y.values + base

pf_oof = (train_df['pf_ancc'].values - base)

d = apply_pp(train_df, ridge_oof_preds, pf_oof, **pp_params)
ridge_score = root_mean_squared_error(ytrue, base + d)

overall_scores["ridge (pp)"] = ridge_score
fold_scores["ridge (pp)"] = [ridge_score] * CFG.n_splits

# %%
pf_test = test_df['pf_ancc'].values - test_df['last_known_tvt'].values
test_df2 = test_df[['id', 'well', 'md_since']].copy()
test_df2['pred'] = test_df['last_known_tvt'].values + apply_pp(
    test_df,
    ridge_test_preds,
    pf_test,
    **pp_params
)
test_df2 = sg_smooth(test_df2, 'pred')

# %%
sample_sub = pd.read_csv(CFG.dataset_path / "sample_submission.csv")
sub_1 = (sample_sub[['id']].merge(
    test_df2[['id', 'pred']].rename(columns={'pred':'tvt'}),
    on='id', 
    how='left'
))

sub_1['tvt']=sub_1['tvt'].fillna(float(train_df['last_known_tvt'].mean()+train_df['target'].mean()))

# The Ridge prediction is now fully represented by sub_1. Release train-side
# frames, OOF arrays, saved trainer wrappers, and redundant test aliases before
# the 128-seed shared PF starts.
RIDGE_MEMORY_RELEASE_REPORT = _exp514_release_globals(
    (
        '_ridge_train_path', 'train_df', 'test_paths', 'test_df', 'features',
        'X', 'y', 'g', 'X_test', 'oof_preds', 'test_preds', 'trainer',
        'ridge_trainer', 'ridge_oof_preds', 'ridge_test_preds', 'overall_scores',
        'fold_scores', 'base', 'ytrue', 'pf_oof', 'pf_test', 'd', 'ridge_score',
        'test_df2', 'sample_sub', 'ridge_params', 'pp_params',
    ),
    label='ridge_train_and_prediction_intermediates',
)
sub_1

# %%
sample = pd.read_csv(CFG.dataset_path / 'sample_submission.csv')
sample['well']    = sample['id'].str[:8]
sample['row_idx'] = sample['id'].str[9:].astype(int)

train_hw_files = sorted(glob.glob(str(CFG.dataset_path / 'train' / '*__horizontal_well.csv')))
train_wells = [os.path.basename(f).split('__')[0] for f in train_hw_files]

test_hw_files = sorted(glob.glob(str(CFG.dataset_path / 'test' / '*__horizontal_well.csv')))
test_wells = [os.path.basename(f).split('__')[0] for f in test_hw_files]
if not test_wells:
    raise FileNotFoundError('shared likelihood-PF requires at least one dynamic test well')

rows = []
bimodal_report_rows = []
PF_SEED_BRANCH_STATS = {}
if not test_wells:
    raise FileNotFoundError('SP45 requires at least one test well')
_sp45_effective_n_jobs = min(SP45_WELL_N_JOBS, len(test_wells))
_shared_sp45_pipeline_started = time.time()


def _run_sp45_test_well(order, wid, shared_record):
    messages = [f'\nProcessing {order + 1}/{len(test_wells)}: {wid}...']
    hw_te, tw_te = load_well(wid, 'test')

    tvt_phys = None
    hw_tr    = None
    tw_tr    = None

    # Physical model for visible wells
    if wid in train_wells:
        try:
            hw_tr, tw_tr = load_well(wid, 'train')
            hw_te['TVT_input'] = hw_tr['TVT_input'].values
            tvt_phys = tvt_from_contacts(hw_tr, tw_tr)
            messages.append('  Physical model OK')
        except Exception as e:
            messages.append(f'  Physical model failed: {e}')
            tvt_phys = None

    selector_code, selector_variant, selector_n_eval, selector_z_span = selector_well_code(hw_te)

    tw_ref = tw_tr if tw_tr is not None else tw_te

    # Consume the already materialized exp413 x1.0 stable-seed bank.
    # Failure is terminal: no last-known or duplicate legacy PF fallback is allowed.
    pf_by_scale, _seed_branch = shared_likpf_sp45_adapter(shared_record)
    tvt_pf = pf_by_scale['pf_scale_8']
    messages.append(
        f'  Shared exp413 PF bank OK seeds={SHARED_LIKPF_SEEDS} '
        f'scales={SHARED_LIKPF_SCALES}'
    )

    # Beam search ensemble
    try:
        tvt_beam = run_beam_ensemble(hw_te, tw_ref)
        messages.append('  Beam 14-config ensemble OK')
    except Exception as e:
        messages.append(f'  Beam failed: {e}')
        tvt_beam = tvt_pf.copy()

    # Selector blending
    last_known = hw_te['TVT_input'].dropna()
    last_known_tvt = float(last_known.iloc[-1]) if len(last_known) > 0 else float(np.nanmean(tvt_pf))
    tvt_selector, bimodal_info = apply_selector_variant(
        selector_variant, pf_by_scale, tvt_beam, last_known_tvt,
        hw=hw_te, tw=tw_ref, return_info=True,
    )
    bimodal_report = {
        'well': wid,
        'selector_code': int(selector_code),
        'selector_variant': selector_variant,
        'n_eval': float(selector_n_eval),
        'z_span': float(selector_z_span),
        **bimodal_info,
    }
    if bimodal_info.get('bimodal_active'):
        messages.append(
            f'  Selector code={selector_code} variant={selector_variant} '
            f'n_eval={selector_n_eval:.0f} z_span={selector_z_span:.3f} '
            f'bimodal_delta={bimodal_info.get("delta_star", 0.0):.2f}'
        )
    else:
        messages.append(
            f'  Selector code={selector_code} variant={selector_variant} '
            f'n_eval={selector_n_eval:.0f} z_span={selector_z_span:.3f}'
        )

    ws = sample[sample['well'] == wid]
    well_rows = []
    for _, row in ws.iterrows():
        ridx = int(row['row_idx'])
        if tvt_phys is not None:
            tvt_val = float(tvt_phys.iloc[ridx])
        else:
            tvt_val = float(tvt_selector[ridx])
        well_rows.append({'id': row['id'], 'tvt': tvt_val})
    messages.append(f'  Added {len(ws)} rows')
    return {
        'well': str(wid),
        'rows': well_rows,
        'bimodal_report': bimodal_report,
        'seed_branch': _seed_branch,
        'messages': messages,
    }


def _run_shared_likpf_sp45_well(order, wid):
    shared_started = time.time()
    record = _shared_likpf_one_well(
        str(wid),
        'test',
        load_well,
        particles=SHARED_LIKPF_PARTICLES,
        seeds=SHARED_LIKPF_SEEDS,
    )
    shared_seconds = time.time() - shared_started
    sp45_started = time.time()
    well_result = _run_sp45_test_well(order, wid, record)
    sp45_seconds = time.time() - sp45_started
    released_payloads = release_shared_likpf_sp45_payload(record)
    return {
        'order': int(order),
        'well': str(wid),
        'record': record,
        'well_result': well_result,
        'shared_seconds': float(shared_seconds),
        'sp45_seconds': float(sp45_seconds),
        'released_payloads': released_payloads,
    }


_shared_sp45_pipeline_results = Parallel(
    n_jobs=_sp45_effective_n_jobs,
    backend='threading',
)(
    delayed(_run_shared_likpf_sp45_well)(order, wid)
    for order, wid in enumerate(test_wells)
)
_shared_sp45_pipeline_results = sorted(
    _shared_sp45_pipeline_results, key=lambda item: int(item['order'])
)
if [item['well'] for item in _shared_sp45_pipeline_results] != [
    str(well) for well in test_wells
]:
    raise RuntimeError('shared PF/SP45 streaming pipeline changed well order')

SHARED_LIKPF_BANK = {}
for _pipeline_result in _shared_sp45_pipeline_results:
    _record = _pipeline_result['record']
    _well = _pipeline_result['well']
    if any(
        key in _record
        for key in ('sp45_full', 'row_index', 'evaluation_index', 'known_mask')
    ):
        raise RuntimeError(f'shared PF full payload retained after SP45: {_well}')
    SHARED_LIKPF_BANK[_well] = _record
    _well_result = _pipeline_result['well_result']
    for _message in _well_result['messages']:
        print(_message)
    rows.extend(_well_result['rows'])
    bimodal_report_rows.append(_well_result['bimodal_report'])
    if _well_result['seed_branch']:
        PF_SEED_BRANCH_STATS[_well_result['well']] = _well_result['seed_branch']

_shared_sp45_pipeline_elapsed = time.time() - _shared_sp45_pipeline_started
_shared_worker_seconds = float(
    sum(item['shared_seconds'] for item in _shared_sp45_pipeline_results)
)
_sp45_worker_seconds = float(
    sum(item['sp45_seconds'] for item in _shared_sp45_pipeline_results)
)
_combined_worker_seconds = _shared_worker_seconds + _sp45_worker_seconds
_shared_elapsed_share = (
    _shared_sp45_pipeline_elapsed * _shared_worker_seconds / _combined_worker_seconds
    if _combined_worker_seconds > 0.0
    else 0.0
)
_sp45_elapsed_share = _shared_sp45_pipeline_elapsed - _shared_elapsed_share
SHARED_LIKPF_PARALLEL_REPORT = {
    'requested_n_jobs': SHARED_LIKPF_N_JOBS,
    'effective_n_jobs': _sp45_effective_n_jobs,
    'backend': 'threading',
    'wells': len(test_wells),
    'worker_seconds_sum': round(_shared_worker_seconds, 6),
    'elapsed_seconds': round(_shared_elapsed_share, 6),
    'elapsed_semantics': 'proportional_share_of_streaming_pipeline_wall_time',
    'all_well_full_payload_retained': False,
    'max_concurrent_full_payload_wells': _sp45_effective_n_jobs,
}
SP45_WELL_PARALLEL_REPORT = {
    'requested_n_jobs': SP45_WELL_N_JOBS,
    'effective_n_jobs': _sp45_effective_n_jobs,
    'test_wells': len(test_wells),
    'backend': 'threading',
    'worker_seconds_sum': round(_sp45_worker_seconds, 6),
    'elapsed_seconds': round(_sp45_elapsed_share, 6),
    'elapsed_semantics': 'proportional_share_of_streaming_pipeline_wall_time',
}
SHARED_SP45_STREAMING_REPORT = {
    'requested_n_jobs': SHARED_LIKPF_N_JOBS,
    'effective_n_jobs': _sp45_effective_n_jobs,
    'backend': 'threading',
    'wells': len(test_wells),
    'elapsed_seconds': round(_shared_sp45_pipeline_elapsed, 6),
    'full_payload_retention': 'at_most_effective_n_jobs_then_release_after_sp45',
    'retained_for_exp413': ['id', 'likpf_scale_5', 'likpf_mean'],
}
del _shared_sp45_pipeline_results
_shared_sp45_gc_collected = int(_exp514_gc.collect())
SHARED_SP45_STREAMING_REPORT['post_pipeline_gc_collected'] = _shared_sp45_gc_collected
print('shared likelihood-PF producer report:', SHARED_LIKPF_PARALLEL_REPORT)
print('SP45 well-parallel report:', SP45_WELL_PARALLEL_REPORT)
print('shared PF/SP45 streaming report:', SHARED_SP45_STREAMING_REPORT)

if bimodal_report_rows:
    _bimodal_df = pd.DataFrame(bimodal_report_rows)
    _bimodal_df.to_csv('bimodal_selector_report.csv', index=False)
    if 'bimodal_active' in _bimodal_df.columns:
        _active_mask = _bimodal_df['bimodal_active'].astype(str).str.lower().isin(['true', '1', 'yes'])
        _active_cols = [c for c in [
            'well', 'selector_code', 'selector_variant', 'n_eval', 'z_span',
            'delta_star', 'delta_a', 'delta_b', 'p_base', 'p_eff', 'prefix_trust',
            'temperature', 'rho1', 'n_eff', 'score_base', 'score_second', 'j_ratio',
            'prefix_rows', 'j_prefix_base', 'j_prefix_decoy', 'prefix_trust_margin'
        ] if c in _bimodal_df.columns]
        _bimodal_df.loc[_active_mask, _active_cols].to_csv('bimodal_active_wells.csv', index=False)

# %%
sub_2 = pd.DataFrame(rows)

# %%
sub = (
    sub_1.merge(sub_2, on='id', suffixes=('_1', '_2'))
       .assign(tvt=lambda x: SP45_RIDGE_MODEL_WEIGHT * x['tvt_1'] + SP45_SELECTOR_WEIGHT * x['tvt_2'])
       [['id', 'tvt']]
)
sub.to_csv("submission.csv", index=False)
sub

# %% [markdown]
# ## 6. Projection and learned trajectory replay

# %%
# Robust low-order projection post-processing.
# Runs AFTER the 0.3*ridge+0.7*selector blend writes submission.csv; OVERWRITES it with the projected
# version. Per-well robust deg-5 fit of dU = tvt + Z - anchor vs normalized MD -> denoise jitter +
# down-weight wrong-branch outliers. Deterministic; defensive per-well fallback to raw.
import numpy as _np, pandas as _pd
def _robfit(s, y, deg=5):
    if len(s) < deg + 2:
        return y.copy()
    c = _np.polyfit(s, y, deg)
    for _ in range(4):
        r = y - _np.polyval(c, s)
        sc = _np.median(_np.abs(r)) * 1.4826 + 1e-6
        c = _np.polyfit(s, y, deg, w=1.0 / (1.0 + (r / (2.0 * sc)) ** 2))
    return _np.polyval(c, s)
try:
    _base = _pd.read_csv("submission.csv")   # the just-written blended submission
    assert set(['id','tvt']).issubset(_base.columns)
    _base['well'] = _base['id'].str[:8]
    _base['row_idx'] = _base['id'].str[9:].astype(int)
    _out = dict(zip(_base['id'].values, _base['tvt'].astype(float).values))
    _n_ok = 0
    for _wid, _g in _base.groupby('well'):
        try:
            _hw = _pd.read_csv(CFG.dataset_path / 'test' / (_wid + '__horizontal_well.csv'))
            _kn = _hw[_hw['TVT_input'].notna()]
            if len(_kn) < 5:
                continue
            _last = _kn.iloc[-1]
            _anchor = float(_last['TVT_input']) + float(_last['Z'])
            _ps = float(_last['MD']); _end = float(_hw['MD'].iloc[-1])
            _gi = _g.sort_values('row_idx')
            _ri = _gi['row_idx'].values
            _Z = _hw['Z'].values[_ri].astype(float)
            _md = _hw['MD'].values[_ri].astype(float)
            _s = (_md - _ps) / max(_end - _ps, 1e-6)
            _tvt = _gi['tvt'].values.astype(float)
            _fit = _robfit(_s, (_tvt + _Z) - _anchor, int(SP45_PROJECTION_DEGREE))
            _tvt_fit_full = (_anchor + _fit) - _Z
            _tvt_fit = (1.0 - float(SP45_PROJECTION_BLEND_WEIGHT)) * _tvt + float(SP45_PROJECTION_BLEND_WEIGHT) * _tvt_fit_full
            if not _np.all(_np.isfinite(_tvt_fit)):
                continue
            for _rid, _val in zip(_gi['id'].values, _tvt_fit):
                _out[_rid] = float(_val)
            _n_ok += 1
        except Exception as _e:
            print('proj fallback', _wid, _e)
    print('projection applied to', _n_ok, 'wells')
    _final = _base[['id']].copy()
    _final['tvt'] = _final['id'].map(_out).astype(float)
    _final[['id','tvt']].to_csv("submission.csv", index=False)
    print('wrote projected submission.csv', _final.shape)
except Exception as _e:
    print('PROJECTION SKIPPED (kept blended submission):', _e)

# %%
from pathlib import Path as _BlendPath
import pandas as _blend_pd
_sp45_path = _BlendPath('/kaggle/working/submission.csv') if _BlendPath('/kaggle/working').exists() else _BlendPath('submission.csv')
_sp45_df = _blend_pd.read_csv(_sp45_path)
_sp45_df.to_csv((_BlendPath('/kaggle/working') if _BlendPath('/kaggle/working').exists() else _BlendPath('.')) / 'sp45_projection_submission.csv', index=False)
print('saved sp45_projection_submission.csv', _sp45_df.shape, flush=True)


# Learned trajectory inference section follows.

# %%
import os, sys, glob, time, warnings, multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
os.environ.setdefault("SHOW_FIGS", "0")

# ---- environment / paths (Kaggle or local) -------------------------------------
def _find_data():
    for c in ["/kaggle/input/competitions/rogii-wellbore-geology-prediction",
              "/kaggle/input/rogii-wellbore-geology-prediction"]:
        if Path(c).exists() and (Path(c)/"train").exists():
            return Path(c)
    # fallback: find any mounted folder that contains a train/ directory
    for p in glob.glob("/kaggle/input/**/train", recursive=True):
        return Path(p).parent
    return Path(os.environ.get("ROGII_DATA", "."))   # local override for development

class CFG:
    DATA = _find_data()
    OUT  = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    seed = 42
    n_splits = 5
    n_jobs = min(8, multiprocessing.cpu_count())
    # lik-PF
    PF_SEEDS = 128
    PF_PARTICLES = 500
    PF_SCALES = (3., 5., 8., 12.)
    # FAST dev (local smoke test): limit train wells & trees
    FAST = bool(int(os.environ.get("FAST", "0")))
    N_TRAIN_WELLS = int(os.environ.get("N_TRAIN_WELLS", "0"))  # 0 = all
    USE_GPU = os.environ.get("USE_GPU", "auto")
    SHOW_FIGS = os.environ.get("SHOW_FIGS", "1") == "1"   # EDA plots (on in the notebook)

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
def _demo_well():
    """A train well with TVT + a sizable eval zone, for the EDA plots."""
    for w in sorted(p.stem.replace("__horizontal_well", "")
                    for p in (CFG.DATA/"train").glob("*__horizontal_well.csv")):
        try:
            d = pd.read_csv(CFG.DATA/"train"/f"{w}__horizontal_well.csv", usecols=["TVT", "TVT_input"])
        except Exception:
            continue
        if "TVT" in d and d.TVT.notna().any() and d.TVT_input.isna().sum() > 2000:
            return w
    return None
print("DATA:", CFG.DATA, "| OUT:", CFG.OUT, "| cores:", CFG.n_jobs, "| FAST:", CFG.FAST)

def load_well(wid, split="train"):
    base = CFG.DATA / split
    hw = pd.read_csv(base / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(base / f"{wid}__typewell.csv").sort_values("TVT")
    return hw, tw

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float))**2)))

# %%
# ---- single particle filters (ANCC-anchored & Z-velocity-coupled), numba ---------
PF_N = 600; ANCC_N = 600
PF_MOM = 0.993; PF_VN = 0.005; PF_PN = 0.01
PF_GR_SIG_MIN = 10.; PF_GR_SIG_MAX = 60.; PF_GR_SIG_DEF = 30.
PF_GR_WIN = 5; PF_GR_WT = 0.3; PF_RESAMP = 0.5; PF_ROUGH_P = 0.2; PF_ROUGH_V = 0.003
ANCC_ALPHA = 0.998; ANCC_RN = 0.002; ANCC_PN = 0.005; ANCC_IS = 0.3; ANCC_RP = 0.1; ANCC_RR = 0.001

BEAMS = [(10,20.,144.,2,"cons"),(10,8.,64.,2,"loose"),(8,35.,220.,1,"vcons"),
         (10,14.,90.,5,"sm5"),(20,4.,36.,3,"vloose"),(12,12.,100.,3,"mid"),(15,25.,180.,2,"stiff")]

@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t

@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1] = cum[j]+w[j]
    u0 = np.random.uniform(0., 1./N); np2 = np.empty(N); na = np.empty(N); ci = 0
    for j in range(N):
        u = u0+j/N
        while ci < N-1 and cum[ci+1] < u: ci += 1
        np2[j] = pos[ci]+rp*np.random.randn(); na[j] = aux[ci]+rv*np.random.randn()
    return np2, na

@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr); nt = len(tw_gr); MAX = BS*6
    bidx = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30); bcost[0] = 0.; bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64); hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64); cC = np.full(MAX, 1e30); cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]; nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]; cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx+d
                if ni < 0 or ni >= nt: continue
                tot = cost+(gv-tw_gr[ni])**2/es+mc*(d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni: fnd = ci; break
                if fnd >= 0:
                    if tot < cC[fnd]: cC[fnd] = tot; cP[fnd] = bi
                else:
                    if nc < MAX: cI[nc] = ni; cC[nc] = tot; cP[nc] = bi; nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i+1, nc):
                if cC[j] < cC[mi]: mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]; cC[i], cC[mi] = cC[mi], cC[i]; cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]; hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]; bcost[:kept] = cC[:kept]; bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]: best = b
    path = np.zeros(n, np.int64); b = best
    for s in range(n-1, -1, -1): path[s] = hI[s, b]; b = hP[s, b]
    return path

@njit(cache=True)
def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP):
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
    for j in range(N):
        pos[j] = ls+IS*np.random.randn(); rate[j] = ir+0.01*np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v)); pm = md_v[0]-1.
    for i in range(len(md_v)):
        dm = md_v[i]-pm; dm = max(dm, 1.)
        for j in range(N):
            rate[j] = ALPHA*rate[j]+RN*np.random.randn(); pos[j] += rate[j]*dm+PN*np.random.randn()
            tvt_j = pos[j]-z_v[i]; tvt_j = max(tvt_j, vmin-50.); tvt_j = min(tvt_j, vmin+len(gg)*step+50.)
            pos[j] = tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                eg = _interp1(gg, pos[j]-z_v[i], vmin, step); d = (gr_v[i]-eg)/gs
                lk = max(np.exp(-0.5*d*d) if d*d < 600. else 0., 1e-300); w[j] *= lk; ws += w[j]
            if ws > 0.:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
        ne = 0.
        for j in range(N): ne += w[j]*w[j]
        if 1./ne < RESAMP*N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N): w[j] = 1./N
        tv = 0.
        for j in range(N): tv += w[j]*(pos[j]-z_v[i])
        pts[i] = tv; va = 0.
        for j in range(N): va += w[j]*(pos[j]-z_v[i]-tv)**2
        std_[i] = va**0.5; pm = md_v[i]
    return pts, std_

@njit(cache=True)
def _pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step, gs, ip, iv, beta, icpt, zsig, N,
         MOM, VN, PN, GR_WT, RP, RV, RESAMP):
    pos = np.empty(N); vel = np.empty(N); w = np.ones(N)/N
    for j in range(N):
        pos[j] = ip+0.5*np.random.randn(); vel[j] = iv+0.02*np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v)); pm = md_v[0]-1.; pz = z_v[0]-1.
    for i in range(len(md_v)):
        dm = md_v[i]-pm; dm = max(dm, 1.); dzd = (z_v[i]-pz)/dm; ve = beta*dzd+icpt
        for j in range(N):
            vel[j] = MOM*vel[j]+VN*np.random.randn(); pos[j] += vel[j]*dm+PN*np.random.randn()
            pos[j] = max(pos[j], vmin-50.); pos[j] = min(pos[j], vmin+len(gg_p)*step+50.)
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                ep = _interp1(gg_p, pos[j], vmin, step); dp = (gr_v[i]-ep)/gs
                lp = max(np.exp(-0.5*dp*dp) if dp*dp < 600. else 0., 1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es = _interp1(gg_s, pos[j], vmin, step); ds = (gr_sm_v[i]-es)/(gs*1.5)
                    lsm = max(np.exp(-0.5*ds*ds) if ds*ds < 600. else 0., 1e-300); lk = (1.-GR_WT)*lp+GR_WT*lsm
                else: lk = lp
                lk = max(lk, 1e-300); w[j] *= lk; ws += w[j]
            if ws > 0.:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
        ws2 = 0.
        for j in range(N):
            dv = (vel[j]-ve)/max(zsig*2., 0.005); lz = max(np.exp(-0.5*dv*dv) if dv*dv < 600. else 0., 1e-300)
            w[j] *= lz; ws2 += w[j]
        if ws2 > 0.:
            for j in range(N): w[j] /= ws2
        else:
            for j in range(N): w[j] = 1./N
        ne = 0.
        for j in range(N): ne += w[j]*w[j]
        if 1./ne < RESAMP*N:
            pos, vel = _resamp(pos, vel, w, N, RP, RV)
            for j in range(N): w[j] = 1./N
        wm = 0.
        for j in range(N): wm += w[j]*pos[j]
        pts[i] = wm; va = 0.
        for j in range(N): va += w[j]*(pos[j]-wm)**2
        std_[i] = va**0.5; pm = md_v[i]; pz = z_v[i]
    return pts, std_

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax+step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)

def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw.TVT_input.notna() & hw.GR.notna()]
    if len(kn) < 20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn.GR.values-np.interp(kn.TVT_input.values, tw_tvt, tw_gr)),
                         PF_GR_SIG_MIN, PF_GR_SIG_MAX))

def _nn(arr, v):
    i = int(np.searchsorted(arr, v, "left"))
    if i >= len(arr): return len(arr)-1
    if i > 0 and abs(arr[i-1]-v) <= abs(arr[i]-v): return i-1
    return i

def _smooth(vals, fb, r):
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(fb)
    return (s.rolling(r*2+1, center=True, min_periods=1).mean() if r > 0 else s).to_numpy(np.float32)

def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _nn(tw_tvt, start_tvt); sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    return tw_tvt[_beam_jit(sgr, tw_gr.astype(np.float64), si, bs, float(mc), float(es))].astype(np.float32)

def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr); kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    ls = float(kn.TVT_input.iloc[-1]+kn.Z.iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values); dm = np.diff(tail.MD.values); m = dm > 0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    pts, std = _pf_ancc(ev.MD.values.astype(np.float64), ev.Z.values.astype(np.float64), ev.GR.values.astype(np.float64),
                        gg, gmin, gst, gs, ls, ir, N, ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP)
    return pts.astype(np.float32), std.astype(np.float32)

def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr); tw_s = pd.Series(tw_gr).rolling(PF_GR_WIN, center=True, min_periods=1).mean().values.astype(np.float32)
    kna = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    dz_k = np.diff(kna.Z.values); dvt = np.diff(kna.TVT_input.values); dmd_k = np.diff(kna.MD.values); m2 = dmd_k > 0
    if m2.sum() >= 10:
        vz = dz_k[m2]/dmd_k[m2]; vt = dvt[m2]/dmd_k[m2]; A = np.column_stack([vz, np.ones_like(vz)])
        c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
        beta, icpt, zsig = float(c[0]), float(c[1]), max(float(np.std(vt-(c[0]*vz+c[1]))), 0.001)
    else: beta, icpt, zsig = -1., 0., 0.1
    t2 = kna.tail(20); dvt2 = np.diff(t2.TVT_input.values); dmd2 = np.diff(t2.MD.values); m3 = dmd2 > 0
    iv = float(np.median(dvt2[m3]/dmd2[m3])) if m3.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr); gs2, _, _ = _grid(tw_tvt, tw_s)
    gr_sm = hw.GR.rolling(PF_GR_WIN, center=True, min_periods=1).mean()
    pts, std = _pf_z(ev.MD.values.astype(np.float64), ev.Z.values.astype(np.float64), ev.GR.values.astype(np.float64),
                     gr_sm.loc[ev.index].values.astype(np.float64), gg, gs2, gmin, gst, gs,
                     float(kna.TVT_input.iloc[-1]), iv, beta, icpt, zsig, N,
                     PF_MOM, PF_VN, PF_PN, PF_GR_WT, PF_ROUGH_P, PF_ROUGH_V, PF_RESAMP)
    return pts.astype(np.float32), std.astype(np.float32)

def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out = []
    for hw in hws:
        win = 2*hw+1; nk = len(kgr); nh = len(hgr)
        if nk < win+1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk-win+1, stride, dtype=np.int32)
        if len(sts) == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        C = kg[sts[:, None]+np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C-C.mean(1, keepdims=True))/(C.std(1, keepdims=True)+1e-6)
        hp = np.pad(hg, hw, mode="edge"); H = hp[np.arange(nh)[:, None]+np.arange(win)[None, :]].astype(np.float32)
        Hn = (H-H.mean(1, keepdims=True))/(H.std(1, keepdims=True)+1e-6)
        ncc = Hn@Cn.T/win; best = ncc.argmax(1); score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best]+hw, 0, nk-1)].astype(np.float32), score))
    tvts = np.stack([o[0] for o in out], 1); scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3.*scores); sw /= sw.sum(1, keepdims=True)+1e-9
    return out, (tvts*sw).sum(1).astype(np.float32)

# %%
# ---- 128-seed likelihood-weighted particle filter (the workhorse), numba ---------
@njit(cache=True, nogil=True)
def _pf_lik_allseeds(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, n_seeds, seed_base,
                     MOM, VN, PN, RP, RR, RESAMP, init_spr):
    n = len(md_v); preds = np.empty((n_seeds, n)); liks = np.empty(n_seeds); tmax = vmin + len(gg)*step
    for s in range(n_seeds):
        np.random.seed(seed_base + s)
        pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
        for j in range(N):
            pos[j] = ls + init_spr*np.random.randn(); rate[j] = ir + 0.01*np.random.randn()
        log_lik = 0.0; prev_md = md_v[0] - 1.0
        for i in range(n):
            dm = md_v[i] - prev_md
            if dm < 1.0: dm = 1.0
            for j in range(N):
                rate[j] = MOM*rate[j] + VN*np.random.randn(); pos[j] += rate[j]*dm + PN*np.random.randn()
                tvt_j = pos[j] - z_v[i]
                if tvt_j < vmin-100.: tvt_j = vmin-100.
                if tvt_j > tmax+100.: tvt_j = tmax+100.
                pos[j] = tvt_j + z_v[i]
            avg_lk = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j]-z_v[i], vmin, step); d = (gr_v[i]-eg)/gs; dd = d*d
                if dd > 600.: dd = 600.
                lk = np.exp(-0.5*dd)
                if lk < 1e-300: lk = 1e-300
                avg_lk += w[j]*lk; w[j] = w[j]*lk
            if avg_lk < 1e-300: avg_lk = 1e-300
            log_lik += np.log(avg_lk)
            ws = 0.0
            for j in range(N): ws += w[j]
            if ws > 0.0:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
            neff = 0.0
            for j in range(N): neff += w[j]*w[j]
            neff = 1.0/neff
            if neff < RESAMP*N:
                cum = np.empty(N); c = 0.0
                for j in range(N): c += w[j]; cum[j] = c
                u0 = np.random.uniform(0., 1./N); newpos = np.empty(N); newrate = np.empty(N); ci = 0
                for j in range(N):
                    u = u0 + j/N
                    while ci < N-1 and cum[ci] < u: ci += 1
                    newpos[j] = pos[ci] + RP*np.random.randn(); newrate[j] = rate[ci] + RR*np.random.randn()
                for j in range(N): pos[j] = newpos[j]; rate[j] = newrate[j]; w[j] = 1./N
            est = 0.0
            for j in range(N): est += w[j]*(pos[j]-z_v[i])
            preds[s, i] = est; prev_md = md_v[i]
        liks[s] = log_lik
    return preds, liks

def lik_pf(hw, tw, n_particles=CFG.PF_PARTICLES, n_seeds=CFG.PF_SEEDS, scales=CFG.PF_SCALES,
           init_spr=4.5, seed_base=0, with_quality=False):
    """Likelihood-weighted PF ensemble. Returns ({pf_scale_X: pred_eval}, ev_index[, quality])."""
    tw_s = tw.sort_values("TVT"); tw_tvt = tw_s.TVT.values.astype(float)
    tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(float)
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return {}, np.array([]), {}
    last = kn.iloc[-1]; ls = float(last.TVT_input) + float(last.Z)
    tw_at_k = np.interp(kn.TVT_input.values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10., 60.)) * 1.3
    tail = kn.tail(30); dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values); dm = np.diff(tail.MD.values); m = dm > 0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.0
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gr_v = hw.GR.interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)[ev.index]
    preds, liks = _pf_lik_allseeds(ev.MD.values.astype(float), ev.Z.values.astype(float), gr_v,
                                   gg, gmin, gst, gs, ls, ir, n_particles, n_seeds, seed_base,
                                   0.998, 0.002, 0.005, 0.1, 0.001, 0.5, init_spr)
    ln = liks - liks.max(); out = {}
    for sc in scales:
        wts = np.exp(ln/float(sc)); wts /= wts.sum(); out[f"pf_scale_{sc:g}"] = (wts[:, None]*preds).sum(0)
    out["pf_mean"] = preds.mean(0)
    q = {}
    if with_quality:
        q = {"pf_best_ll": float(liks.max())/len(ev), "pf_ll_spread": float(liks.std()),
             "pf_pt_std": preds.std(0).astype(np.float32), "pf_gr_sig": gs}
    return out, ev.index.values, q

# JIT warm-up so timings below are representative
_m = np.linspace(1, 50, 20); _z = np.zeros(20); _g = np.full(20, 50.); _gg = np.linspace(45, 55, 100)
_pf_ancc(_m, _z, _g, _gg, 45., .1, 20., 50., 0., 8, .998, .002, .005, .3, .1, .001, .5)
_pf_z(_m, _z, _g, _g, _gg, _gg, 45., .1, 20., 50., 0., -1., 0., .1, 8, .993, .005, .01, .3, .2, .003, .5)
_beam_jit(np.random.randn(30), np.random.randn(50), 25, 8, 15., 100.)
_pf_lik_allseeds(_m, _z, _g, _gg, 45., .1, 20., 50., 0., 64, 4, 0, .998, .002, .005, .1, .001, .5, 4.5)
print("trackers compiled.")

def fig_tracker_vs_truth(wid):
    import matplotlib.pyplot as plt
    hw, tw = load_well(wid); kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    tw_tvt = tw.TVT.to_numpy(np.float32); tw_gr = tw.GR.to_numpy(np.float32); last = float(kn.TVT_input.iloc[-1])
    pf, _ = run_pf_ancc(hw, tw_tvt, tw_gr); out, _, _ = lik_pf(hw, tw, scales=(3.,))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ev.MD, ev.TVT, lw=2.2, color="black", label="True TVT", zorder=5)
    ax.plot(ev.MD, np.full(len(ev), last), lw=1.1, color="gray", ls=":", label="last-known baseline")
    ax.plot(ev.MD, pf, lw=1.0, color="tab:blue", alpha=.8, label="single particle filter")
    ax.plot(ev.MD, out["pf_scale_3"], lw=1.5, color="crimson", alpha=.9, label="128-seed lik-weighted PF")
    ax.set_xlabel("MD (ft)"); ax.set_ylabel("TVT (ft)"); ax.invert_yaxis(); ax.grid(alpha=.25)
    ax.set_title(f"Well {wid}: trackers vs ground truth â€” the lik-PF resists drift"); ax.legend(loc="best")
    plt.tight_layout(); plt.show()

# %%
PLANE_K = 10; DENSE_SPW = 60; DENSE_K = 20

def robust_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-6: return 0.
    return float(np.polyfit(x[m], y[m], 1)[0])

def affine_cal(kgr, tw_at_k, min_pts=20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1., float(np.nanmean(kgr)-np.nanmean(tw_at_k)) if v.any() else 0.
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1); return float(a), float(b)

def seg_b_well(ktvt, kz, form_col):
    bv = ktvt+kz-form_col; n = len(bv); b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n-50):])) if n >= 5 else b_full
    t1, t2 = n//3, 2*n//3
    b_early = float(np.median(bv[:max(1, t1)])) if t1 > 0 else b_full
    b_mid = float(np.median(bv[t1:max(t1+1, t2)])) if t2 > t1 else b_full
    w = np.exp(0.02*np.arange(n)); w /= w.sum()
    return b_full, b_early, b_mid, b_late, float(np.dot(w, bv))

class FormationPlaneKNN:
    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            try: df = pd.read_csv(data_dir/f"{wid}__horizontal_well.csv", usecols=["X","Y"]+FORMATIONS).dropna()
            except: continue
            if len(df) == 0: continue
            row = {"wid": wid, "x": float(df.X.median()), "y": float(df.Y.median())}
            for c in FORMATIONS: row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows); self.wmap = {w: i for i, w in enumerate(self.df.wid)}
        xy = self.df[["x","y"]].to_numpy(); self.scale = np.where(xy.std(0) < 1e-3, 1., xy.std(0))
        self.tree = cKDTree(xy/self.scale); self.xa = self.df.x.to_numpy(); self.ya = self.df.y.to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)
    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q = xy_q/self.scale; nf = min(k+5, len(self.df)); dist, idx = self.tree.query(q, k=nf, workers=int(globals().get('EXP514_KDTREE_WORKERS', -1)))
        if self_wid in self.wmap: dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ordr = np.argpartition(dist, min(k-1, nf-1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1); ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1./(dk+1e-3), 0.).astype(np.float64)
        xn = self.xa[ik]; yn = self.ya[ik]; fn = self.fa[ik]; wx = w*xn; wy = w*yn
        A = np.zeros((len(q), 3, 3))
        A[:,0,0]=(wx*xn).sum(1); A[:,0,1]=(wx*yn).sum(1); A[:,0,2]=wx.sum(1)
        A[:,1,0]=A[:,0,1]; A[:,1,1]=(wy*yn).sum(1); A[:,1,2]=wy.sum(1)
        A[:,2,0]=A[:,0,2]; A[:,2,1]=A[:,1,2]; A[:,2,2]=w.sum(1)
        A[:,0,0]+=1e-9; A[:,1,1]+=1e-9; A[:,2,2]+=1e-9
        rhs = np.stack([(wx[:,:,None]*fn).sum(1), (wy[:,:,None]*fn).sum(1), (w[:,:,None]*fn).sum(1)], 1)
        try: coef = np.linalg.solve(A, rhs)
        except:
            coef = np.zeros((len(q), 3, 6))
            for r in range(len(q)):
                try: coef[r] = np.linalg.pinv(A[r])@rhs[r]
                except: pass
        Xq = xy_q[:,0]; Yq = xy_q[:,1]
        pred = (Xq[:,None]*coef[:,0,:]+Yq[:,None]*coef[:,1,:]+coef[:,2,:]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)

class DenseANCCImputer:
    def __init__(self, well_ids, data_dir, spw=DENSE_SPW):
        xs, ys, an, wd = [], [], [], []
        for wid in well_ids:
            try: df = pd.read_csv(data_dir/f"{wid}__horizontal_well.csv", usecols=["X","Y","ANCC"]).dropna()
            except: continue
            if len(df) == 0: continue
            ix = np.linspace(0, len(df)-1, min(spw, len(df)), dtype=int); s = df.iloc[ix]
            xs.append(s.X.values); ys.append(s.Y.values); an.append(s.ANCC.values); wd.extend([wid]*len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(an).astype(np.float32); self.wids = np.array(wd)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1., self.xy.std(0)); self.tree = cKDTree(self.xy/self.scale)
    def impute(self, xy_q, self_wid=None, k=DENSE_K, nfetch=5000):
        xy_q = np.atleast_2d(xy_q); q = xy_q/self.scale; nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=int(globals().get('EXP514_KDTREE_WORKERS', -1)))
        if self_wid: dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ordr = np.argpartition(dist, min(k-1, nf-1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1); ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1./(dk+1e-3), 0.); sw = w.sum(1); safe = np.where(sw < 1e-9, 1., sw)
        a = self.ancc[ik]; ap = (a*w).sum(1)/safe; ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var = ((a-ap[:,None])**2*w).sum(1)/safe
        return ap.astype(np.float32), np.sqrt(np.maximum(var, 0.)).astype(np.float32), np.where(vk, dk, np.inf).min(1).astype(np.float32)

_FI = None; _DI = None
ANCH_OFFS = np.array([-80,-40,-20,-10,-5,0,5,10,20,40,80], np.float32)
BEAM_OFFS = np.array([-40,-20,-10,-5,-3,0,3,5,10,20,40], np.float32)
SC_OFFS = np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30], np.float32)
PF_OFFS = SC_OFFS.copy()

# %%
def build_well(hw_path, tw_path, is_train, likpf_map=None):
    global _FI, _DI
    wid = Path(hw_path).stem.replace("__horizontal_well", "")
    try: hw = pd.read_csv(hw_path); tw = pd.read_csv(tw_path).sort_values("TVT")
    except: return None
    if is_train and "TVT" not in hw.columns: return None
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0 or len(kn) < 10: return None
    if is_train and hw.TVT.isna().all(): return None
    tw_tvt = tw.TVT.to_numpy(np.float32); tw_gr = tw.GR.to_numpy(np.float32)
    if len(tw_tvt) < 3: return None
    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a) == 0: return None
    pf_z, std_z = run_pf_z(hw, tw_tvt, tw_gr)
    pf_use = pf_a.astype(np.float32); std_use = std_a.astype(np.float32)
    has_z = len(pf_z) == len(pf_a) and not np.any(np.isnan(pf_z))
    lk = kn.iloc[-1]; last_tvt = float(lk.TVT_input)
    gr_full = hw.GR.astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32); kgr = gr_full.iloc[:len(kn)].to_numpy(np.float32)
    bpaths = {tag: beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r) for (bs, mc, es, r, tag) in BEAMS}
    beam_ref = (bpaths["cons"]+bpaths["sm5"])/2.
    ktvt = kn.TVT_input.to_numpy(np.float32)
    sc_res, sc_ens = multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3)
    sc8, sc8s = sc_res[0]; sc15, sc15s = sc_res[1]; sc25, sc25s = sc_res[2]; sc_cons = (sc8+sc15+sc25)/3.
    sc_trust = float(np.clip(len(kn)/200., 0., 0.6)); hyb_ref = (1-sc_trust)*beam_ref+sc_trust*sc_ens
    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32); a_cal, b_cal = affine_cal(kgr, tw_at_k)
    kmd = kn.MD.to_numpy(np.float32); kz = kn.Z.to_numpy(np.float32)
    pfx_rmse = float(np.sqrt(np.mean((kgr-tw_at_k)**2)))
    slp_all = robust_slope(kmd, ktvt); slp_50 = robust_slope(kmd[-50:], ktvt[-50:]); slp_z = robust_slope(kz, ktvt)
    swid = wid if is_train else None
    xy_ev = ev[["X","Y"]].to_numpy(np.float64); xy_kn = kn[["X","Y"]].to_numpy(np.float64)
    form_ev, knn_d = _FI.impute(xy_ev, self_wid=swid); form_kn, _ = _FI.impute(xy_kn, self_wid=swid)
    z_kn = kn.Z.to_numpy(np.float32); z_ev = ev.Z.to_numpy(np.float32)
    tvt_fs = {}; form_rmse = {}; form_list = []
    for fi2, fn in enumerate(FORMATIONS):
        b_full, b_early, b_mid, b_late, b_wls = seg_b_well(ktvt, z_kn, form_kn[:, fi2])
        tvt_f = (-z_ev+form_ev[:, fi2]+b_full).astype(np.float32)
        tvt_fs[f"tvtF_{fn}"]=tvt_f; tvt_fs[f"tvtFw_{fn}"]=(-z_ev+form_ev[:,fi2]+b_wls).astype(np.float32)
        tvt_fs[f"tvtF50_{fn}"]=(-z_ev+form_ev[:,fi2]+b_late).astype(np.float32)
        tvt_fs[f"bw_{fn}"]=np.float32(b_full); tvt_fs[f"bww_{fn}"]=np.float32(b_wls); tvt_fs[f"bw50_{fn}"]=np.float32(b_late)
        tvt_fs[f"bw_early_{fn}"]=np.float32(b_early); tvt_fs[f"bw_mid_{fn}"]=np.float32(b_mid)
        form_rmse[fn]=float(np.sqrt(np.mean((ktvt-(-z_kn+form_kn[:,fi2]+b_full))**2))); form_list.append(tvt_f)
    fs = np.stack(form_list, 1)
    form_mean_d=(fs.mean(1)-last_tvt).astype(np.float32); form_std_d=fs.std(1).astype(np.float32); form_rng_d=(fs.max(1)-fs.min(1)).astype(np.float32)
    d_ancc, d_std, d_dist = _DI.impute(xy_ev, self_wid=swid); d_kn, d_std_kn, _ = _DI.impute(xy_kn, self_wid=swid)
    _, b_de, b_dm, b_dl, b_dw = seg_b_well(ktvt, z_kn, d_kn); b_d = float(np.median(ktvt+z_kn-d_kn))
    tvt_dense=(-z_ev+d_ancc+b_d).astype(np.float32); tvt_densew=(-z_ev+d_ancc+b_dw).astype(np.float32); tvt_dense50=(-z_ev+d_ancc+b_dl).astype(np.float32)
    res_kn = ktvt+z_kn-d_kn; d_rmse=float(np.sqrt(np.mean(res_kn**2))); d_bias=float(np.mean(res_kn)); d_nb_std=float(np.mean(d_std_kn))
    all_sigs=[pf_use]+list(bpaths.values())+[sc8,sc15,sc25,sc_ens,tvt_fs["tvtF_ANCC"],tvt_dense]
    sig_mat=np.stack(all_sigs,1); sig_std=sig_mat.std(1).astype(np.float32); sig_mean=(sig_mat.mean(1)-last_tvt).astype(np.float32)
    gr_s=pd.Series(gr_full.values); rolls={}
    for w in [5,21,51,101]:
        r=gr_s.rolling(w,center=True,min_periods=1); rolls[f"grm{w}"]=r.mean().iloc[ev.index].values.astype(np.float32); rolls[f"grs{w}"]=r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1,5,15,30]:
        rolls[f"glag{lag}"]=gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32); rolls[f"glead{lag}"]=gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1=gr_s.diff().fillna(0.).iloc[ev.index].values.astype(np.float32); gr_d2=gr_s.diff().diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_env=gr_s.rolling(21,center=True,min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg=np.sqrt(np.maximum((gr_s**2).rolling(21,center=True,min_periods=1).mean(),0.)).iloc[ev.index].values.astype(np.float32)
    hmd=ev.MD.to_numpy(np.float32); md_since=hmd-float(lk.MD)
    slp_b_all=(last_tvt+slp_all*md_since).astype(np.float32); slp_b_50=(last_tvt+slp_50*md_since).astype(np.float32)
    mdd=hw.MD.diff().replace(0,np.nan)
    dzdmd=(hw.Z.diff()/mdd).iloc[ev.index].values.astype(np.float32); dxdmd=(hw.X.diff()/mdd).iloc[ev.index].values.astype(np.float32); dydmd=(hw.Y.diff()/mdd).iloc[ev.index].values.astype(np.float32)
    nh=len(ev); frac=(np.arange(nh)/max(nh-1,1)).astype(np.float32)
    def sc(v): return np.full(nh, np.float32(v), np.float32)
    feats={"well":wid,"id":[f"{wid}_{i}" for i in ev.index],"last_known_tvt":sc(last_tvt),
        "pf_ancc":pf_use,"pf_ancc_std":std_use,"pf_ancc_delta":(pf_use-last_tvt).astype(np.float32),
        "pf_z":(pf_z.astype(np.float32) if has_z else sc(last_tvt)),"pf_z_delta":((pf_z-last_tvt).astype(np.float32) if has_z else sc(0.)),
        "pf_vs_z":((pf_use-pf_z.astype(np.float32)) if has_z else sc(0.)),
        **{f"beam_{t}_d":(p-np.float32(last_tvt)).astype(np.float32) for t,p in bpaths.items()},
        "beam_mean_d":np.stack([(p-last_tvt) for p in bpaths.values()],1).mean(1).astype(np.float32),
        "beam_std_d":np.stack([(p-last_tvt) for p in bpaths.values()],1).std(1).astype(np.float32),
        "beam_med_d":np.median(np.stack([(p-last_tvt) for p in bpaths.values()],1),1).astype(np.float32),
        "sc8_d":(sc8-np.float32(last_tvt)).astype(np.float32),"sc8_sc":sc8s,"sc15_d":(sc15-np.float32(last_tvt)).astype(np.float32),"sc15_sc":sc15s,
        "sc25_d":(sc25-np.float32(last_tvt)).astype(np.float32),"sc25_sc":sc25s,"sc_cons_d":(sc_cons-np.float32(last_tvt)).astype(np.float32),
        "sc_ens_d":(sc_ens-np.float32(last_tvt)).astype(np.float32),"sc_trust":sc(sc_trust),"hyb_d":(hyb_ref-np.float32(last_tvt)).astype(np.float32),
        "sig_std":sig_std,"sig_mean_d":sig_mean,**tvt_fs,**{f"frm_rmse_{fn}":sc(form_rmse[fn]) for fn in FORMATIONS},
        "form_mean_d":form_mean_d,"form_std_d":form_std_d,"form_rng_d":form_rng_d,
        "spatial_ancc_d":(form_ev[:,0]-np.float32(np.interp(last_tvt,tw_tvt,tw_gr))),"spatial_knn_dist":knn_d,
        "dense_ancc":d_ancc,"dense_std":d_std,"dense_dist":d_dist,"tvt_dense_d":(tvt_dense-last_tvt).astype(np.float32),
        "tvt_densew_d":(tvt_densew-last_tvt).astype(np.float32),"tvt_dense50_d":(tvt_dense50-last_tvt).astype(np.float32),
        "dense_rmse":sc(d_rmse),"dense_bias":sc(d_bias),"dense_nb_std":sc(d_nb_std),
        "pf_vs_spatial":(pf_use-tvt_fs["tvtF_ANCC"]).astype(np.float32),"pf_vs_dense":(pf_use-tvt_dense).astype(np.float32),
        "spatial_vs_dense":(tvt_fs["tvtF_ANCC"]-tvt_dense).astype(np.float32),"beam_vs_spatial":(bpaths["cons"]-tvt_fs["tvtF_ANCC"]).astype(np.float32),
        "sc_vs_beam":(sc_ens-bpaths["cons"]).astype(np.float32),"cal_a":sc(a_cal),"cal_b":sc(b_cal),
        "pfx_rmse":sc(pfx_rmse),"known_len":sc(len(kn)),"eval_len":sc(nh),"slp_all":sc(slp_all),"slp_50":sc(slp_50),"slp_z":sc(slp_z),
        "slp_b_d_all":(slp_b_all-last_tvt).astype(np.float32),"slp_b_d_50":(slp_b_50-last_tvt).astype(np.float32),
        "ktvt_range":sc(float(np.ptp(ktvt))),"ktvt_std":sc(float(ktvt.std())),"md_since":md_since,"frac":frac,"frac2":frac**2,"sqrt_frac":np.sqrt(frac),
        "z":z_ev,"dx":(ev.X-float(lk.X)).to_numpy(np.float32),"dy":(ev.Y-float(lk.Y)).to_numpy(np.float32),"dz":(z_ev-float(lk.Z)).astype(np.float32),
        "dxy":np.sqrt((ev.X-float(lk.X))**2+(ev.Y-float(lk.Y))**2).to_numpy(np.float32),"dzdmd":dzdmd,"dxdmd":dxdmd,"dydmd":dydmd,
        "gr":hgr,"gr_d1":gr_d1,"gr_d2":gr_d2,"gr_env":gr_env,"gr_nrg":gr_nrg,
        "gr_vs_tw_anc":hgr-np.float32(np.interp(last_tvt,tw_tvt,tw_gr)),"gr_vs_slp_all":hgr-np.interp(slp_b_all,tw_tvt,tw_gr).astype(np.float32),
        **{f"tda{int(o)}":hgr-np.float32(np.interp(last_tvt+o,tw_tvt,tw_gr)) for o in ANCH_OFFS},
        **{f"tdbc{int(o)}":hgr-np.interp(beam_ref+o,tw_tvt,tw_gr).astype(np.float32) for o in BEAM_OFFS},
        **{f"tdsc{int(o)}":hgr-np.interp(sc_ens+o,tw_tvt,tw_gr).astype(np.float32) for o in SC_OFFS},
        **{f"tdpf{int(o)}":hgr-np.interp(pf_use+o,tw_tvt,tw_gr).astype(np.float32) for o in PF_OFFS},
        "tw_range":sc(float(np.ptp(tw_tvt))),"tw_gr_mean":sc(float(tw_gr.mean()))}
    for k,v in rolls.items(): feats[k]=v
    res = pd.DataFrame(feats)
    if is_train: res["target"]=(ev.TVT.to_numpy(np.float32)-np.float32(last_tvt))
    return res

def init_imputers(train_wids):
    global _FI, _DI
    _FI = FormationPlaneKNN(train_wids, CFG.DATA/"train"); _DI = DenseANCCImputer(train_wids, CFG.DATA/"train")

def _likpf_rows(wid, split):
    hw, tw = load_well(wid, split)
    out, idx, _ = lik_pf(hw, tw)
    if not len(out): return None
    d = {"id": [f"{wid}_{i}" for i in idx]}
    for k, v in out.items():
        d["likpf_" + k.replace("pf_scale_", "scale_").replace("pf_mean", "mean")] = v.astype(np.float32)
    return pd.DataFrame(d)

def build_likpf(wids, split):
    # threads are safe here: the lik-PF numba kernel is compiled with nogil=True, so it
    # releases the GIL and parallelises across threads (no pickling of numba code needed).
    res = Parallel(n_jobs=CFG.n_jobs, prefer="threads")(delayed(_likpf_rows)(w, split) for w in wids)
    return pd.concat([r for r in res if r is not None], ignore_index=True)

def build_features(wids, split, is_train):
    paths = [CFG.DATA/split/f"{w}__horizontal_well.csv" for w in wids]
    res = Parallel(n_jobs=CFG.n_jobs, prefer="threads")(
        delayed(build_well)(str(p), str(p.parent/f"{p.stem.replace('__horizontal_well','')}__typewell.csv"), is_train)
        for p in paths if (p.parent/f"{p.stem.replace('__horizontal_well','')}__typewell.csv").exists())
    parts = [r for r in res if r is not None]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def add_likpf_features(df, likpf):
    if df["id"].duplicated().any() or likpf["id"].duplicated().any():
        raise ValueError("likelihood-PF feature alignment requires unique IDs")
    aligned = (
        likpf.assign(id=likpf["id"].astype(str))
        .set_index("id")
        .reindex(df["id"].astype(str))
    )
    for c in [c for c in likpf.columns if c != "id"]:
        df[c] = aligned[c].to_numpy(copy=False)
        df[c] = df[c].fillna(df["last_known_tvt"])
        df[c+"_d"] = (df[c]-df["last_known_tvt"]).astype(np.float32)
    return df

# %%
def _device():
    if CFG.USE_GPU == "cpu": return "cpu", "CPU"
    if CFG.USE_GPU == "gpu": return "gpu", "GPU"
    try:  # detect a real NVIDIA GPU (Kaggle GPU accelerator) via nvidia-smi
        import subprocess
        if subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0:
            return "gpu", "GPU"
    except Exception:
        pass
    return "cpu", "CPU"

def lgb_configs(dev):
    base = dict(boosting_type="gbdt", objective="regression", verbose=-1, n_jobs=-1, max_bin=255)
    if dev == "gpu": base.update(device_type="gpu", gpu_use_dp=False)
    n = 600 if CFG.FAST else 5000
    return [
        dict(**base, num_leaves=255, min_child_samples=15, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05, learning_rate=0.03, n_estimators=n, seed=123),
        dict(**base, num_leaves=64, min_child_samples=40, subsample=0.474, subsample_freq=1,
             colsample_bytree=0.393, reg_lambda=95.75, reg_alpha=10.79, min_child_weight=0.24,
             learning_rate=0.0093, n_estimators=min(2*n, 10000), random_state=0),
        dict(**base, num_leaves=64, min_child_samples=40, subsample=0.474, subsample_freq=1,
             colsample_bytree=0.393, reg_lambda=95.75, reg_alpha=10.79, min_child_weight=0.24,
             learning_rate=0.0093, n_estimators=min(2*n, 10000), random_state=29),
    ]

def cb_configs(dev):
    tt = "GPU" if dev == "gpu" else "CPU"
    n = 800 if CFG.FAST else 8000
    return [
        dict(iterations=n, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254,
             loss_function="RMSE", task_type=tt, od_type="Iter", od_wait=300, verbose=0, learning_rate=0.02, random_seed=7),
        dict(iterations=n, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254,
             loss_function="RMSE", task_type=tt, od_type="Iter", od_wait=300, verbose=0, learning_rate=0.03, random_seed=123),
    ]

def train_stack(train_df, test_df, features):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from catboost import CatBoostRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge
    dev, devname = _device(); print("device:", devname)
    X = train_df[features].values.astype(np.float32); y = train_df["target"].values.astype(np.float32)
    g = train_df["well"].values; Xt = test_df[features].values.astype(np.float32)
    cv = GroupKFold(CFG.n_splits); oof_cols = {}; test_cols = {}
    def run(name, make, fit_kw, is_lgb):
        # LightGBM: slice to best_iteration_ via num_iteration. CatBoost: use_best_model
        # already trims to the best tree, and its predict() takes no num_iteration kwarg.
        oof = np.zeros(len(train_df)); tp = np.zeros(len(test_df))
        for tr, va in cv.split(X, y, groups=g):
            m = make(); m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], **fit_kw)
            if is_lgb:
                it = m.best_iteration_
                oof[va] = m.predict(X[va], num_iteration=it); tp += m.predict(Xt, num_iteration=it) / CFG.n_splits
            else:
                oof[va] = m.predict(X[va]); tp += m.predict(Xt) / CFG.n_splits
        oof_cols[name] = oof; test_cols[name] = tp
        print(f"  {name}: OOF RMSE={rmse(y, oof):.4f}", flush=True)
    for i, p in enumerate(lgb_configs(dev)):
        run(f"lgb{i}", lambda p=p: LGBMRegressor(**p),
            dict(eval_metric="rmse", callbacks=[early_stopping(250, verbose=False), log_evaluation(0)]), True)
    for i, p in enumerate(cb_configs(dev)):
        run(f"cb{i}", lambda p=p: CatBoostRegressor(**p),
            dict(early_stopping_rounds=250, use_best_model=True), False)
    OOF = pd.DataFrame(oof_cols); TEST = pd.DataFrame(test_cols)
    rid = Ridge(alpha=1.66, positive=True, fit_intercept=True); meta = np.zeros(len(train_df))
    for tr, va in cv.split(OOF.values, y, groups=g):
        rid.fit(OOF.values[tr], y[tr]); meta[va] = rid.predict(OOF.values[va])
    rid.fit(OOF.values, y); meta_test = rid.predict(TEST.values)
    print(f"  ridge-stack OOF RMSE={rmse(y, meta):.4f}")
    return meta, meta_test, OOF, TEST

# %%
class PP:   # tuned on 773-well GroupKFold OOF (Nelder-Mead + grid; the optimum is flat)
    alpha = 1.0         # global scale on the learned delta (tuned ~1.0)
    tau = 85.0          # warm-up length in ft: damps the first feet after PS (tuned ~90)
    w_pf = 0.0          # blending the model with the single PF no longer helps once lik-PF is a feature
    w_sub1 = 0.60       # weight on the learned model; lik-PF gets 1-w_sub1. CV optimum ~0.68 (flat
                        # 0.55-0.68); 0.60 is a small hedge toward the drift-robust lik-PF for LB transfer.
    sub2_scale = "scale_5"   # which likelihood-scale of the lik-PF to use as sub2 (3/5/8 ~equivalent)
    sg_win = 61         # per-well Savitzky-Golay smoothing window (effect is small, ~0.01 ft)
    sg_poly = 3

def warmup(md_since, tau): return 1.-np.exp(-np.maximum(md_since, 0.)/tau) if tau > 1e-6 else 1.0

def make_prediction(df, model_delta, likpf):
    last = df["last_known_tvt"].values.astype(float)
    pf_delta = df["pf_ancc"].values.astype(float) - last
    lp = df[f"likpf_{PP.sub2_scale}"].values.astype(float) - last
    sub1 = PP.alpha*warmup(df["md_since"].values.astype(float), PP.tau)*(model_delta*(1-PP.w_pf)+pf_delta*PP.w_pf)
    delta = PP.w_sub1*sub1 + (1-PP.w_sub1)*lp
    pred = last + delta
    # per-well Savitzky-Golay smoothing
    out = pred.copy(); dfx = df.reset_index(drop=True)
    for _, idx in dfx.groupby("well", sort=False).groups.items():
        pos = dfx.index.get_indexer(idx); v = pred[pos]; n = len(v); wl = min(PP.sg_win, n)
        if wl % 2 == 0: wl -= 1
        if wl >= PP.sg_poly+2: out[pos] = savgol_filter(v, wl, PP.sg_poly)
    return out

# %%
def _find_models():
    """Find pre-trained trajectory boosters from explicit roots first."""
    roots = [Path(p) for p in globals().get('LEARNED_MODEL_ROOTS', ()) if str(p).strip()]
    for d in roots:
        if (d / "features.json").exists() and list(d.glob("lgb*.pkl")):
            return d
    d = CFG.OUT / "models"
    return d if (d/"features.json").exists() and list(d.glob("lgb*.pkl")) else None


# The SP45 ridge path has already generated the expensive deterministic test
# feature block. HJYACT keeps its own stochastic pf_ancc/pf_z draw, then
# rebuilds only columns that depend on that draw.
HJYACT_PF_REGENERATED_COLUMNS = (
    "pf_ancc", "pf_ancc_std", "pf_ancc_delta", "pf_z", "pf_z_delta",
    "pf_vs_z", "sig_std", "sig_mean_d", "pf_vs_spatial", "pf_vs_dense",
    *tuple(f"tdpf{int(offset)}" for offset in PF_OFFS),
)
HJYACT_AUXILIARY_SHARED_COLUMNS = tuple(
    column
    for column in SP45_SHARED_TEST_FEATURE_FRAME.columns
    if column.startswith("__exp514_shared_")
)
HJYACT_DETERMINISTIC_REUSE_MANIFEST = None


def _hjyact_pf_only_well(wid, split, deterministic_well):
    horizontal, typewell = load_well(str(wid), str(split))
    typewell = typewell.sort_values("TVT")
    known = horizontal[horizontal["TVT_input"].notna()]
    evaluation = horizontal[horizontal["TVT_input"].isna()]
    expected_ids = [f"{wid}_{int(index)}" for index in evaluation.index]
    deterministic_well = deterministic_well.reset_index(drop=True)
    if deterministic_well["id"].astype(str).tolist() != expected_ids:
        raise ValueError(f"SP45/HJYACT deterministic row order mismatch for well {wid}")
    if len(evaluation) == 0 or len(known) < 10:
        raise ValueError(f"HJYACT PF-only refresh received ineligible well {wid}")

    typewell_tvt = typewell["TVT"].to_numpy(np.float32)
    typewell_gr = typewell["GR"].to_numpy(np.float32)
    pf_ancc, pf_ancc_std = run_pf_ancc(horizontal, typewell_tvt, typewell_gr)
    pf_z, _ = run_pf_z(horizontal, typewell_tvt, typewell_gr)
    if len(pf_ancc) != len(expected_ids):
        raise ValueError(f"HJYACT pf_ancc row mismatch for well {wid}")
    pf_use = pf_ancc.astype(np.float32)
    std_use = pf_ancc_std.astype(np.float32)
    has_z = len(pf_z) == len(pf_ancc) and not np.any(np.isnan(pf_z))
    last_tvt = float(known["TVT_input"].iloc[-1])
    last_values = np.full(len(evaluation), np.float32(last_tvt), np.float32)
    pf_z_use = pf_z.astype(np.float32) if has_z else last_values

    required_auxiliary = {
        *[f"__exp514_shared_beam_abs_{tag}" for _, _, _, _, tag in BEAMS],
        "__exp514_shared_sc8_abs",
        "__exp514_shared_sc15_abs",
        "__exp514_shared_sc25_abs",
        "__exp514_shared_sc_ens_abs",
        "__exp514_shared_tvt_dense_abs",
    }
    if missing := required_auxiliary - set(deterministic_well.columns):
        raise ValueError(f"SP45/HJYACT auxiliary feature columns missing: {sorted(missing)}")
    signals = [pf_use]
    signals.extend(
        deterministic_well[f"__exp514_shared_beam_abs_{tag}"].to_numpy(np.float32)
        for _, _, _, _, tag in BEAMS
    )
    signals.extend(
        deterministic_well[column].to_numpy(np.float32)
        for column in (
            "__exp514_shared_sc8_abs",
            "__exp514_shared_sc15_abs",
            "__exp514_shared_sc25_abs",
            "__exp514_shared_sc_ens_abs",
            "tvtF_ANCC",
            "__exp514_shared_tvt_dense_abs",
        )
    )
    signal_matrix = np.stack(signals, axis=1)
    gr_full = (
        horizontal["GR"].astype(float).interpolate(limit_direction="both")
        .fillna(float(np.nanmean(typewell_gr)))
    )
    hidden_gr = gr_full.iloc[evaluation.index[0]:].to_numpy(np.float32)
    if len(hidden_gr) != len(evaluation):
        raise ValueError(f"HJYACT PF-only hidden GR row mismatch for well {wid}")

    updates = {
        "id": expected_ids,
        "pf_ancc": pf_use,
        "pf_ancc_std": std_use,
        "pf_ancc_delta": (pf_use - last_tvt).astype(np.float32),
        "pf_z": pf_z_use,
        "pf_z_delta": ((pf_z - last_tvt).astype(np.float32) if has_z else np.zeros(len(evaluation), np.float32)),
        "pf_vs_z": ((pf_use - pf_z.astype(np.float32)) if has_z else np.zeros(len(evaluation), np.float32)),
        "sig_std": signal_matrix.std(axis=1).astype(np.float32),
        "sig_mean_d": (signal_matrix.mean(axis=1) - last_tvt).astype(np.float32),
        "pf_vs_spatial": (
            pf_use - deterministic_well["tvtF_ANCC"].to_numpy(np.float32)
        ).astype(np.float32),
        "pf_vs_dense": (
            pf_use
            - deterministic_well["__exp514_shared_tvt_dense_abs"].to_numpy(np.float32)
        ).astype(np.float32),
    }
    for offset in PF_OFFS:
        updates[f"tdpf{int(offset)}"] = (
            hidden_gr
            - np.interp(pf_use + offset, typewell_tvt, typewell_gr).astype(np.float32)
        ).astype(np.float32)
    return pd.DataFrame(updates)


def build_hjyact_features_from_sp45(wids, split):
    started = time.time()
    base = SP45_SHARED_TEST_FEATURE_FRAME
    ordered_wells = [str(well) for well in wids]
    if set(base["well"].astype(str)) != set(ordered_wells):
        raise ValueError("SP45/HJYACT deterministic feature well set mismatch")
    by_well = {
        str(well): group.reset_index(drop=True)
        for well, group in base.groupby("well", sort=False)
    }
    effective_n_jobs = min(CFG.n_jobs, len(ordered_wells))
    pf_blocks = Parallel(n_jobs=effective_n_jobs, prefer="threads")(
        delayed(_hjyact_pf_only_well)(well, split, by_well[well])
        for well in ordered_wells
    )
    pf_frame = pd.concat(pf_blocks, ignore_index=True)
    del pf_blocks, by_well
    _exp514_gc.collect()
    if pf_frame["id"].duplicated().any() or len(pf_frame) != len(base):
        raise ValueError("HJYACT PF-only refresh ID contract failed")
    # SP45 has finished consuming this frame. Move its sole ownership into
    # HJYACT and update only the stochastic PF-dependent columns in place.
    result = base
    result.reset_index(drop=True, inplace=True)
    if result["id"].astype(str).tolist() != pf_frame["id"].astype(str).tolist():
        raise ValueError("HJYACT PF-only refresh merge order changed")
    for column in HJYACT_PF_REGENERATED_COLUMNS:
        if column not in pf_frame:
            raise ValueError(f"HJYACT PF-only refresh column missing: {column}")
        result[column] = pf_frame[column].to_numpy(np.float32)
    result.drop(columns=list(HJYACT_AUXILIARY_SHARED_COLUMNS), inplace=True)
    reused_columns = [
        column
        for column in result.columns
        if column not in set(HJYACT_PF_REGENERATED_COLUMNS)
    ]
    globals()["HJYACT_DETERMINISTIC_REUSE_MANIFEST"] = {
        "producer": "sp45_test_feature_frame",
        "consumers": ["sp45_ridge", "hjyact_learned", "exp413"],
        "wells": len(ordered_wells),
        "rows": len(result),
        "effective_pf_refresh_threads": effective_n_jobs,
        "imputer_instance_reused": True,
        "deterministic_generation_count": 1,
        "hjyact_full_build_features_calls": 0,
        "reused_column_count": len(reused_columns),
        "reused_columns": reused_columns,
        "pf_regenerated_columns": list(HJYACT_PF_REGENERATED_COLUMNS),
        "auxiliary_columns_dropped": list(HJYACT_AUXILIARY_SHARED_COLUMNS),
        "ownership_transfer": "sp45_to_hjyact_in_place",
        "full_frame_deep_copy_count": 0,
        "elapsed_pf_refresh_seconds": round(time.time() - started, 6),
    }
    globals()["SP45_SHARED_TEST_FEATURE_FRAME"] = None
    return result

HJYACT_SHARED_FEATURE_RUNTIME_SECONDS = None
HJYACT_LEARNED_TOTAL_RUNTIME_SECONDS = None


def main():
    import json
    import joblib

    t0 = time.time()
    train_wids = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA / "train").glob("*__horizontal_well.csv"))
    test_wids = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA / "test").glob("*__horizontal_well.csv"))
    if CFG.N_TRAIN_WELLS:
        train_wids = train_wids[: CFG.N_TRAIN_WELLS]
    print(f"train wells: {len(train_wids)} | test wells: {len(test_wids)}")
    globals()['_FI'], globals()['_DI'] = SP45_SHARED_IMPUTERS

    print("building lik-PF + reusing SP45 deterministic features (test)...", flush=True)
    likpf_test = build_likpf(test_wids, "test")
    shared_started = time.time()
    shared = build_hjyact_features_from_sp45(test_wids, "test")
    globals()["HJYACT_SHARED_FEATURE_RUNTIME_SECONDS"] = time.time() - shared_started
    test_df = add_likpf_features(shared, likpf_test)
    test_df.reset_index(drop=True, inplace=True)
    del likpf_test
    _exp514_gc.collect()

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
    globals()["HJYACT_LEARNED_TOTAL_RUNTIME_SECONDS"] = time.time() - t0
    print(f"submission.csv written ({len(sample)} rows) in {HJYACT_LEARNED_TOTAL_RUNTIME_SECONDS:.0f}s")
    return sample, None, test_df


sub, cv_final, HJYACT_SHARED_FEATURE_FRAME = main()
sub.head()

# %%
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

# %%
from pathlib import Path as _FinalBlendPath
import numpy as _final_np
import pandas as _final_pd

_WORK = _FinalBlendPath('/kaggle/working') if _FinalBlendPath('/kaggle/working').exists() else _FinalBlendPath('.')
_BLEND_WEIGHTS_SP45 = tuple(float(w) for w in SP45_BLEND_CANDIDATE_WEIGHTS)
_SELECTED_SP45_WEIGHT = float(SP45_BLEND_WEIGHT)
_INPUT_FILES = {
    'learned': _WORK / 'submission.csv',
    'sp45': _WORK / 'sp45_projection_submission.csv',
}


def _read_submission_frame(path, label):
    frame = _final_pd.read_csv(path)
    missing = {'id', 'tvt'} - set(frame.columns)
    if missing:
        raise RuntimeError(f'{label} submission is missing columns: {sorted(missing)}')

    frame = frame[['id', 'tvt']].copy()
    frame['id'] = frame['id'].astype(str)
    frame['tvt'] = frame['tvt'].astype(float)

    if not _final_np.isfinite(frame['tvt'].to_numpy(dtype=float)).all():
        raise RuntimeError(f'Non-finite values in {label} tvt')
    return frame


def _merge_blend_inputs(sp45, learned):
    merged = sp45.rename(columns={'tvt': 'tvt_sp45'}).merge(
        learned.rename(columns={'tvt': 'tvt_learned'}),
        on='id',
        how='inner',
    )
    if len(merged) != len(sp45) or len(merged) != len(learned):
        raise RuntimeError(
            f'Blend id mismatch: sp45={len(sp45)}, learned={len(learned)}, merged={len(merged)}'
        )
    return merged


def _weighted_submission(merged, w_sp45):
    w_learned = 1.0 - float(w_sp45)
    out = merged[['id']].copy()
    out['tvt'] = (
        float(w_sp45) * merged['tvt_sp45'].astype(float)
        + w_learned * merged['tvt_learned'].astype(float)
    )
    return out


def _candidate_report_row(candidate, merged, file_name, w_sp45):
    diff = candidate['tvt'].to_numpy(dtype=float) - merged['tvt_sp45'].to_numpy(dtype=float)
    return {
        'file': file_name,
        'w_sp45': float(w_sp45),
        'w_learned': float(1.0 - w_sp45),
        'rows': int(len(candidate)),
        'mean_tvt': float(candidate['tvt'].mean()),
        'std_tvt': float(candidate['tvt'].std()),
        'rmse_vs_sp45': float(_final_np.sqrt(_final_np.mean(diff * diff))),
        'p95_abs_vs_sp45': float(_final_np.quantile(_final_np.abs(diff), 0.95)),
    }


_learned = _read_submission_frame(_INPUT_FILES['learned'], 'learned')
_learned.to_csv(_WORK / 'learned_trajectory_submission.csv', index=False)
_sp45 = _read_submission_frame(_INPUT_FILES['sp45'], 'sp45')
_merged = _merge_blend_inputs(_sp45, _learned)

_report_rows = []
for _w_sp45 in _BLEND_WEIGHTS_SP45:
    _candidate = _weighted_submission(_merged, _w_sp45)
    _name = f'submission_sp45_learned_w{_w_sp45:.2f}.csv'
    _candidate.to_csv(_WORK / _name, index=False)
    _report_rows.append(_candidate_report_row(_candidate, _merged, _name, _w_sp45))

_final_name = f'submission_sp45_learned_w{_SELECTED_SP45_WEIGHT:.2f}.csv'
_final = _final_pd.read_csv(_WORK / _final_name)
_final.to_csv(_WORK / 'submission.csv', index=False)

_report = _final_pd.DataFrame(_report_rows)
_report.to_csv(_WORK / 'sp45_learned_blend_report.csv', index=False)
print(_report.to_string(index=False), flush=True)
print('wrote final submission.csv from', _final_name, _final.shape, flush=True)

# %% [markdown]
# ## 7. Guarded overlap and final hjyact-v2 layers

# %%
if not bool(globals().get('RUN_GUARDED_OVERLAP_OVERRIDE', True)):
    print('guarded overlap override disabled; keeping current submission.csv')
else:
    # Same-well lookup is powerful only when the train copy and current test copy
    # are compatible. Validate every candidate against the visible prefix before
    # replacing any rows. The formation priority list lets the guard choose a
    # better contact surface without trusting it blindly.
    import os as _ov_os, glob as _ov_glob
    import numpy as _ov_np, pandas as _ov_pd
    from pathlib import Path as _OvPath

    def _ov_contact_ref_priority():
        raw = globals().get('GUARDED_OVERRIDE_REF_COLS', None)
        if raw is None:
            raw = (globals().get('GUARDED_OVERRIDE_REF_COL', 'EGFDU'),)
        refs = []
        for ref in raw:
            ref = str(ref).strip()
            if ref and ref not in refs:
                refs.append(ref)
        primary = str(globals().get('GUARDED_OVERRIDE_REF_COL', 'EGFDU')).strip()
        if primary and primary not in refs:
            refs.insert(0, primary)
        return tuple(refs or ('EGFDU',))

    def _ov_tvt_from_contacts(hw_tr, tw_tr, ref_col):
        if ref_col not in hw_tr.columns:
            return None
        if 'Geology' not in tw_tr.columns or 'TVT' not in tw_tr.columns:
            return None
        tw_g = tw_tr.dropna(subset=['Geology', 'TVT'])
        ref_vals = tw_g.loc[tw_g['Geology'].astype(str) == str(ref_col), 'TVT']
        if ref_vals.empty:
            return None
        ref_tvt = float(ref_vals.min())
        raw = ref_tvt - (hw_tr['Z'].to_numpy(dtype=float) - hw_tr[ref_col].to_numpy(dtype=float))
        offset = float(_ov_np.nanmean(hw_tr['TVT'].to_numpy(dtype=float) - raw))
        return raw + offset

    def _ov_best_contact_reconstruction(hw_te, hw_tr, tw_tr):
        min_phys = int(globals().get('GUARDED_OVERRIDE_MIN_VALID_PHYS_ROWS', 100))
        min_known = int(globals().get('GUARDED_OVERRIDE_MIN_KNOWN_PREFIX_ROWS', 50))
        rmse_limit = float(globals().get('GUARDED_OVERRIDE_PREFIX_RMSE_LIMIT', 1.0))
        primary_ref = str(globals().get('GUARDED_OVERRIDE_REF_COL', 'EGFDU')).strip()
        known = hw_te[hw_te['TVT_input'].notna()].copy()
        best = None
        reasons = []
        for ref_col in _ov_contact_ref_priority():
            phys = _ov_tvt_from_contacts(hw_tr, tw_tr, ref_col)
            if phys is None:
                reasons.append(f'{ref_col}:no_ref')
                continue
            md_raw = hw_tr['MD'].to_numpy(dtype=float)
            m_fin = _ov_np.isfinite(phys) & _ov_np.isfinite(md_raw)
            valid_rows = int(m_fin.sum())
            if valid_rows < min_phys:
                reasons.append(f'{ref_col}:valid_rows={valid_rows}')
                continue
            order = _ov_np.argsort(md_raw[m_fin])
            md_tr = md_raw[m_fin][order]
            ph_tr = phys[m_fin][order]
            comparable = known[(known['MD'] >= md_tr[0]) & (known['MD'] <= md_tr[-1])]
            if len(comparable) < min_known:
                reasons.append(f'{ref_col}:known_rows={len(comparable)}')
                continue
            pred_known = _ov_np.interp(comparable['MD'].to_numpy(dtype=float), md_tr, ph_tr)
            rk = float(_ov_np.sqrt(_ov_np.mean((pred_known - comparable['TVT_input'].to_numpy(dtype=float)) ** 2)))
            if not _ov_np.isfinite(rk):
                reasons.append(f'{ref_col}:rmse=nan')
                continue
            candidate = dict(
                ref_col=str(ref_col),
                rmse=rk,
                md_tr=md_tr,
                ph_tr=ph_tr,
                valid_phys_rows=valid_rows,
                comparable_known_rows=int(len(comparable)),
            )
            if best is None or rk < best['rmse']:
                best = candidate
            if str(ref_col) == primary_ref and rk <= rmse_limit:
                break
        if best is None:
            return None, ';'.join(reasons) or 'no_candidate'
        if best['rmse'] > rmse_limit:
            return None, f"best_ref={best['ref_col']} rmse={best['rmse']:.3f}>{rmse_limit:.3f}"
        return best, ''

    try:
        _W = _OvPath('/kaggle/working') if _OvPath('/kaggle/working').exists() else _OvPath('.')
        _DATA = None
        for _c in [_OvPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction'),
                   _OvPath('/kaggle/input/rogii-wellbore-geology-prediction')]:
            if _c.exists() and (_c / 'train').exists():
                _DATA = _c; break
        if _DATA is None:
            for _p in _ov_glob.glob('/kaggle/input/**/train/*__horizontal_well.csv', recursive=True):
                _DATA = _OvPath(_p).parent.parent; break
        if _DATA is None:
            raise RuntimeError('could not locate competition train/test folders')
        _sub = _ov_pd.read_csv(_W / 'submission.csv')
        _sub['well'] = _sub['id'].str[:8]; _sub['row_idx'] = _sub['id'].str[9:].astype(int)
        _pred = dict(zip(_sub['id'].astype(str), _sub['tvt'].astype(float)))
        _train_wells = set(_ov_os.path.basename(f).split('__')[0]
                           for f in _ov_glob.glob(str(_DATA / 'train' / '*__horizontal_well.csv')))
        _n_ok = _n_skip = 0
        _report_rows = []
        for _wid, _g in _sub.groupby('well'):
            if _wid not in _train_wells:
                continue
            try:
                _hw_te = _ov_pd.read_csv(_DATA / 'test' / (_wid + '__horizontal_well.csv'))
                _hw_tr = _ov_pd.read_csv(_DATA / 'train' / (_wid + '__horizontal_well.csv'))
                _tw_tr = _ov_pd.read_csv(_DATA / 'train' / (_wid + '__typewell.csv'))
                _best, _reason = _ov_best_contact_reconstruction(_hw_te, _hw_tr, _tw_tr)
                if _best is None:
                    print('override SKIP %s %s' % (_wid, _reason))
                    _report_rows.append({'well': _wid, 'status': 'skip', 'reason': _reason, 'rows_total': int(len(_g))})
                    _n_skip += 1
                    continue
                _md_tr = _best['md_tr']; _ph_tr = _best['ph_tr']
                _md_te = _hw_te['MD'].to_numpy(dtype=float)
                _n_row = 0
                for _rid, _ri in zip(_g['id'].astype(str).values, _g['row_idx'].values):
                    _ri = int(_ri)
                    if 0 <= _ri < len(_md_te):
                        _m = float(_md_te[_ri])
                        if _md_tr[0] <= _m <= _md_tr[-1]:
                            _pred[_rid] = float(_ov_np.interp(_m, _md_tr, _ph_tr)); _n_row += 1
                print('override OK   %s ref=%s known-prefix rmse=%.4f rows overridden=%d/%d' % (
                    _wid, _best['ref_col'], _best['rmse'], _n_row, len(_g)))
                _report_rows.append({
                    'well': _wid,
                    'status': 'override',
                    'ref_col': _best['ref_col'],
                    'known_prefix_rmse': float(_best['rmse']),
                    'valid_phys_rows': int(_best['valid_phys_rows']),
                    'comparable_known_rows': int(_best['comparable_known_rows']),
                    'rows_overridden': int(_n_row),
                    'rows_total': int(len(_g)),
                    'reason': '',
                })
                _n_ok += 1
            except Exception as _e:
                print('override fallback %s: %s' % (_wid, _e))
                _report_rows.append({'well': _wid, 'status': 'error', 'reason': str(_e), 'rows_total': int(len(_g))})
                _n_skip += 1
        _new = _sub['id'].astype(str).map(_pred).astype(float)
        assert _new.notna().all(), 'override produced NaN, aborting'
        _sub['tvt'] = _new
        _sub[['id', 'tvt']].to_csv(_W / 'submission.csv', index=False)
        _ov_pd.DataFrame(_report_rows).to_csv(_W / 'guarded_overlap_override_report.csv', index=False)
        print('GUARDED override done: overridden=%d skipped=%d (skipped = kept the blend)' % (_n_ok, _n_skip))
    except Exception as _e:
        print('GUARDED override skipped entirely (kept the blend):', _e)

# %%
# Visible-prefix calibration overlay.
# It runs AFTER the final blend and guarded contact override.
# The self-verified anchor remains the default trajectory; this layer only makes a per-well move
# when the visible-prefix backtest says a geology/PF candidate beats the default tracker.
import os as _gold_os
import glob as _gold_glob
import json as _gold_json
import time as _gold_time
import hashlib as _gold_hashlib
from pathlib import Path as _GoldPath

import numpy as _gold_np
import pandas as _gold_pd

_GOLD_ENABLE = _gold_os.environ.get('ROGII_GOLD_PREFIX_CAL', '1') == '1'
_GOLD_PROFILE = _gold_os.environ.get('ROGII_GOLD_PROFILE', 'balanced').strip().lower()
_GOLD_INCLUDE_PF = _gold_os.environ.get('ROGII_GOLD_INCLUDE_PF', '1') == '1'
_GOLD_CAL_SEEDS = int(_gold_os.environ.get('ROGII_GOLD_CAL_SEEDS', '24'))
_GOLD_FINAL_SEEDS = int(_gold_os.environ.get('ROGII_GOLD_FINAL_SEEDS', '48'))
_GOLD_PARTICLES = int(_gold_os.environ.get('ROGII_GOLD_PARTICLES', '350'))
_GOLD_CUT_FRACS = tuple(float(x) for x in _gold_os.environ.get('ROGII_GOLD_CUT_FRACS', '0.50,0.65,0.75').split(',') if x.strip())
_GOLD_MAX_WELLS = int(_gold_os.environ.get('ROGII_GOLD_MAX_WELLS', '1000000'))
_GOLD_SKIP_BIMODAL = _gold_os.environ.get('ROGII_GOLD_SKIP_BIMODAL', '1') == '1'
_GOLD_VP_SKIP_REQUIRES_LOW_TRUST = _gold_os.environ.get('ROGII_GOLD_VP_SKIP_REQUIRES_LOW_TRUST', '0') == '1'
_GOLD_VP_LOW_TRUST_THRESHOLD = float(_gold_os.environ.get('ROGII_GOLD_VP_LOW_TRUST_THRESHOLD', '0.25'))
_GOLD_CONTACT_OVERRIDE = _gold_os.environ.get('ROGII_GOLD_CONTACT_OVERRIDE', '0') == '1'
_GOLD_FINAL_SELECTION = _gold_os.environ.get('ROGII_GOLD_FINAL_SELECTION', 'self_verified_anchor').strip().lower()

_GOLD_PROFILES = {
    'conservative': dict(min_gain=1.00, max_best=12.0, min_consistency=0.67, base=0.06, gain_scale=0.12, margin_scale=0.04, quality_bonus=0.02, cap=0.22, clip_base=8.0, clip_gain=3.0, clip_max=18.0, delta_soft=22.0, p95_hard=55.0),
    'balanced':     dict(min_gain=1.00, max_best=12.0, min_consistency=0.80, min_margin=0.10, base=0.08, gain_scale=0.20, margin_scale=0.06, quality_bonus=0.04, cap=0.36, clip_base=10.0, clip_gain=4.5, clip_max=28.0, delta_soft=30.0, p95_hard=75.0),
    'aggressive':   dict(min_gain=0.25, max_best=15.0, min_consistency=0.34, base=0.12, gain_scale=0.32, margin_scale=0.10, quality_bonus=0.06, cap=0.56, clip_base=14.0, clip_gain=7.0, clip_max=45.0, delta_soft=42.0, p95_hard=110.0),
}
if _GOLD_PROFILE not in _GOLD_PROFILES:
    print(f'Unknown ROGII_GOLD_PROFILE={_GOLD_PROFILE!r}; using balanced')
    _GOLD_PROFILE = 'balanced'


def _gold_work_dir():
    return _GoldPath('/kaggle/working') if _GoldPath('/kaggle/working').exists() else _GoldPath('.')


def _gold_find_data():
    candidates = []
    obj = globals().get('CFG')
    if obj is not None:
        for attr in ('dataset_path', 'DATA'):
            if hasattr(obj, attr):
                candidates.append(_GoldPath(getattr(obj, attr)))
    candidates.extend([
        _GoldPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction'),
        _GoldPath('/kaggle/input/rogii-wellbore-geology-prediction'),
        _GoldPath('.'),
    ])
    for c in candidates:
        try:
            if (c / 'train').exists() and (c / 'test').exists() and (c / 'sample_submission.csv').exists():
                return c
        except Exception:
            pass
    for p in _gold_glob.glob('/kaggle/input/**/sample_submission.csv', recursive=True):
        c = _GoldPath(p).parent
        if (c / 'train').exists() and (c / 'test').exists():
            return c
    raise RuntimeError('Could not locate ROGII data directory')


def _gold_split_ids(df):
    out = df.copy()
    parts = out['id'].astype(str).str.rsplit('_', n=1, expand=True)
    if parts.shape[1] != 2:
        raise RuntimeError('Unexpected id format; expected well_rowindex')
    out['well'] = parts[0]
    out['row_idx'] = parts[1].astype(int)
    return out


def _gold_rmse(a, b):
    a = _gold_np.asarray(a, dtype=float)
    b = _gold_np.asarray(b, dtype=float)
    m = _gold_np.isfinite(a) & _gold_np.isfinite(b)
    if int(m.sum()) == 0:
        return float('inf')
    d = a[m] - b[m]
    return float(_gold_np.sqrt(_gold_np.mean(d * d)))


def _gold_sha256(path):
    h = _gold_hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _gold_truthy(series):
    if series is None:
        return _gold_pd.Series([], dtype=bool)
    if getattr(series, 'dtype', None) == bool:
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({'1', 'true', 'yes', 'y'})


def _gold_load_bimodal_skip_wells(work_dir, requires_low_trust=False, trust_threshold=0.25):
    paths = [
        _GoldPath(work_dir) / 'bimodal_active_wells.csv',
        _GoldPath(work_dir) / 'bimodal_selector_report.csv',
    ]
    frames = []
    for path in paths:
        if not path.exists():
            continue
        try:
            df = _gold_pd.read_csv(path)
        except Exception as e:
            print('Could not read bimodal report:', path, e)
            continue
        if 'well' not in df.columns:
            continue
        if 'bimodal_active' in df.columns:
            active = _gold_truthy(df['bimodal_active'])
        else:
            active = _gold_pd.Series(True, index=df.index)
        out = df.loc[active].copy()
        if len(out):
            frames.append(out)
    if not frames:
        return set(), _gold_pd.DataFrame()
    report = _gold_pd.concat(frames, ignore_index=True).drop_duplicates(subset=['well'], keep='first')
    if requires_low_trust:
        if 'prefix_trust' in report.columns:
            trust = _gold_pd.to_numeric(report['prefix_trust'], errors='coerce')
            report = report[trust.fillna(0.0) <= float(trust_threshold)].copy()
        else:
            report = report.iloc[0:0].copy()
    wells = set(report['well'].astype(str))
    report['visible_prefix_action'] = 'keep_bimodal_hedge'
    report['vp_requires_low_trust'] = bool(requires_low_trust)
    report['vp_low_trust_threshold'] = float(trust_threshold)
    return wells, report


def _gold_robust_poly_predict(x_known, y_known, x_all, deg):
    x_known = _gold_np.asarray(x_known, dtype=float)
    y_known = _gold_np.asarray(y_known, dtype=float)
    x_all = _gold_np.asarray(x_all, dtype=float)
    m = _gold_np.isfinite(x_known) & _gold_np.isfinite(y_known)
    x_known = x_known[m]
    y_known = y_known[m]
    if len(x_known) < 3:
        fill = float(_gold_np.nanmedian(y_known)) if len(y_known) else 0.0
        return _gold_np.full_like(x_all, fill, dtype=float)
    deg = int(min(max(1, deg), len(x_known) - 1))
    x0 = float(x_known[0])
    xs = float(_gold_np.nanmax(x_known) - _gold_np.nanmin(x_known))
    if (not _gold_np.isfinite(xs)) or xs < 1e-6:
        xs = 1.0
    xk = (x_known - x0) / xs
    xa = (x_all - x0) / xs
    try:
        coef = _gold_np.polyfit(xk, y_known, deg)
        for _ in range(5):
            fit = _gold_np.polyval(coef, xk)
            res = y_known - fit
            sc = 1.4826 * float(_gold_np.nanmedian(_gold_np.abs(res - _gold_np.nanmedian(res)))) + 1e-6
            weights = 1.0 / (1.0 + (res / (2.5 * sc)) ** 2)
            coef = _gold_np.polyfit(xk, y_known, deg, w=weights)
        return _gold_np.polyval(coef, xa).astype(float)
    except Exception:
        return _gold_np.full_like(x_all, float(_gold_np.nanmedian(y_known)), dtype=float)


def _gold_variant_grid():
    variants = set()
    try:
        variants.update(SELECTOR_BIN_VARIANTS.values())
        variants.add(SELECTOR_GLOBAL_VARIANT)
    except Exception:
        pass
    for scale in (3, 5, 8, 12):
        for hold in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
            variants.add(f'pf_scale_{scale:g}_hold_{hold:g}')
        for beam in (0.05, 0.10, 0.20, 0.30):
            for hold in (0.0, 0.05, 0.10, 0.15, 0.20):
                variants.add(f'pf_scale_{scale:g}_beam_{beam:g}_hold_{hold:g}')
    return sorted(variants)


def _gold_contact_ref_priority():
    raw = globals().get('GUARDED_OVERRIDE_REF_COLS', None)
    if raw is None:
        raw = (globals().get('GUARDED_OVERRIDE_REF_COL', 'EGFDU'),)
    refs = []
    for ref in raw:
        ref = str(ref).strip()
        if ref and ref not in refs:
            refs.append(ref)
    primary = str(globals().get('GUARDED_OVERRIDE_REF_COL', 'EGFDU')).strip()
    if primary and primary not in refs:
        refs.insert(0, primary)
    return tuple(refs or ('EGFDU',))


def _gold_tvt_from_contacts(hw_tr, tw_tr, ref_col='EGFDU'):
    if ref_col not in hw_tr.columns:
        return _gold_np.full(len(hw_tr), _gold_np.nan, dtype=float)
    tw_g = tw_tr.dropna(subset=['Geology', 'TVT']) if {'Geology', 'TVT'}.issubset(tw_tr.columns) else tw_tr.copy()
    if 'Geology' not in tw_g.columns or 'TVT' not in tw_g.columns:
        return _gold_np.full(len(hw_tr), _gold_np.nan, dtype=float)
    ref_vals = tw_g.loc[tw_g['Geology'].astype(str) == str(ref_col), 'TVT']
    if ref_vals.empty:
        return _gold_np.full(len(hw_tr), _gold_np.nan, dtype=float)
    ref_tvt = float(ref_vals.min())
    raw = ref_tvt - (hw_tr['Z'].to_numpy(dtype=float) - hw_tr[ref_col].to_numpy(dtype=float))
    offset = float(_gold_np.nanmean(hw_tr['TVT'].to_numpy(dtype=float) - raw))
    return raw + offset


def _gold_contact_candidate(wid, hw, data_dir):
    out = {}
    try:
        hw_tr_path = data_dir / 'train' / f'{wid}__horizontal_well.csv'
        tw_tr_path = data_dir / 'train' / f'{wid}__typewell.csv'
        if not hw_tr_path.exists() or not tw_tr_path.exists():
            return out
        hw_tr = _gold_pd.read_csv(hw_tr_path)
        tw_tr = _gold_pd.read_csv(tw_tr_path)
        primary_ref = _gold_contact_ref_priority()[0]
        for ref_col in _gold_contact_ref_priority():
            phys = _gold_tvt_from_contacts(hw_tr, tw_tr, ref_col=ref_col)
            md = hw_tr['MD'].to_numpy(dtype=float)
            m = _gold_np.isfinite(md) & _gold_np.isfinite(phys)
            if int(m.sum()) < 100:
                continue
            order = _gold_np.argsort(md[m])
            md_s = md[m][order]
            ph_s = phys[m][order]
            pred = _gold_np.interp(hw['MD'].to_numpy(dtype=float), md_s, ph_s, left=_gold_np.nan, right=_gold_np.nan)
            name = 'contact_md_lookup' if str(ref_col) == str(primary_ref) else f'contact_md_lookup_{str(ref_col).lower()}'
            out[name] = pred.astype(float)
    except Exception:
        return out
    return out


def _gold_poly_candidates(hw_masked):
    out = {}
    tvt = hw_masked['TVT_input'].to_numpy(dtype=float)
    md = hw_masked['MD'].to_numpy(dtype=float)
    z = hw_masked['Z'].to_numpy(dtype=float)
    kn = _gold_np.flatnonzero(_gold_np.isfinite(tvt) & _gold_np.isfinite(md) & _gold_np.isfinite(z))
    if len(kn) < 30:
        return out
    u = tvt + z
    for tail in (80, 160, 320, 640, 1000000):
        sel = kn[-min(int(tail), len(kn)):]
        if len(sel) < 30:
            continue
        tag = 'all' if tail >= 1000000 else f'tail{tail}'
        for deg in (1, 2, 3):
            if len(sel) < deg + 12:
                continue
            uhat = _gold_robust_poly_predict(md[sel], u[sel], md, deg)
            out[f'poly_u_deg{deg}_{tag}'] = (uhat - z).astype(float)
    return out


def _gold_surface_candidates(hw_masked, wid, data_dir):
    out = {}
    tvt = hw_masked['TVT_input'].to_numpy(dtype=float)
    z = hw_masked['Z'].to_numpy(dtype=float)
    xy = hw_masked[['X', 'Y']].to_numpy(dtype=float)
    kn = _gold_np.isfinite(tvt) & _gold_np.isfinite(z) & _gold_np.isfinite(xy).all(axis=1)
    if int(kn.sum()) < 30:
        return out
    formations = list(globals().get('FORMATIONS', ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']))
    fi = globals().get('_FI', globals().get('FI', None))
    di = globals().get('_DI', globals().get('DI', None))
    surf_names = []
    try:
        if fi is not None:
            form_all, _ = fi.impute(xy, self_wid=None)
            form_all = _gold_np.asarray(form_all, dtype=float)
            for i, fn in enumerate(formations[:form_all.shape[1]]):
                f = form_all[:, i]
                good = kn & _gold_np.isfinite(f)
                if int(good.sum()) < 30:
                    continue
                b_med = float(_gold_np.nanmedian(tvt[good] + z[good] - f[good]))
                out[f'surface_{fn}_median'] = (-z + f + b_med).astype(float)
                surf_names.append(f'surface_{fn}_median')
                if callable(globals().get('seg_b_well')):
                    try:
                        b_full, _, _, b_late, b_wls = seg_b_well(
                            tvt[good].astype(_gold_np.float32),
                            z[good].astype(_gold_np.float32),
                            f[good].astype(_gold_np.float32),
                        )
                        out[f'surface_{fn}_full'] = (-z + f + float(b_full)).astype(float)
                        out[f'surface_{fn}_late'] = (-z + f + float(b_late)).astype(float)
                        out[f'surface_{fn}_wls'] = (-z + f + float(b_wls)).astype(float)
                        surf_names.extend([f'surface_{fn}_full', f'surface_{fn}_late', f'surface_{fn}_wls'])
                    except Exception:
                        pass
    except Exception as e:
        print('surface imputer skipped', wid, e)
    try:
        if di is not None:
            dense, _, _ = di.impute(xy, self_wid=None)
            dense = _gold_np.asarray(dense, dtype=float)
            good = kn & _gold_np.isfinite(dense)
            if int(good.sum()) >= 30:
                b_med = float(_gold_np.nanmedian(tvt[good] + z[good] - dense[good]))
                out['dense_ancc_median'] = (-z + dense + b_med).astype(float)
                surf_names.append('dense_ancc_median')
                if callable(globals().get('seg_b_well')):
                    try:
                        b_full, _, _, b_late, b_wls = seg_b_well(
                            tvt[good].astype(_gold_np.float32),
                            z[good].astype(_gold_np.float32),
                            dense[good].astype(_gold_np.float32),
                        )
                        out['dense_ancc_full'] = (-z + dense + float(b_full)).astype(float)
                        out['dense_ancc_late'] = (-z + dense + float(b_late)).astype(float)
                        out['dense_ancc_wls'] = (-z + dense + float(b_wls)).astype(float)
                        surf_names.extend(['dense_ancc_full', 'dense_ancc_late', 'dense_ancc_wls'])
                    except Exception:
                        pass
    except Exception as e:
        print('dense imputer skipped', wid, e)
    ens_names = [n for n in surf_names if n in out]
    if len(ens_names) >= 2:
        errs = _gold_np.array([_gold_rmse(out[n][kn], tvt[kn]) for n in ens_names], dtype=float)
        finite = _gold_np.isfinite(errs)
        if int(finite.sum()) >= 2:
            names = [n for n, ok in zip(ens_names, finite) if ok]
            errs = errs[finite]
            weights = 1.0 / _gold_np.maximum(errs, 0.25) ** 2
            weights = weights / weights.sum()
            mat = _gold_np.vstack([out[n] for n in names])
            out['surface_weighted_prefix'] = (weights[:, None] * mat).sum(axis=0).astype(float)
    out.update(_gold_contact_candidate(wid, hw_masked, data_dir))
    return out


def _gold_pf_candidates(hw_masked, tw, variants, n_seeds, n_particles):
    out = {}
    if not _GOLD_INCLUDE_PF:
        return out
    if not callable(globals().get('run_pf_lik_ensemble_scales')) or not callable(globals().get('apply_selector_variant')):
        return out
    kn = hw_masked[hw_masked['TVT_input'].notna()]
    ev = hw_masked[hw_masked['TVT_input'].isna()]
    if len(kn) < 30 or len(ev) == 0:
        return out
    try:
        pf_by_scale = run_pf_lik_ensemble_scales(
            hw_masked,
            tw,
            scales=tuple(globals().get('SELECTOR_SCALES', (3.0, 5.0, 8.0, 12.0))),
            n_particles=int(n_particles),
            n_seeds=int(n_seeds),
        )
        try:
            tvt_beam = run_beam_ensemble(hw_masked, tw)
        except Exception:
            tvt_beam = pf_by_scale.get('pf_mean')
            if tvt_beam is None:
                tvt_beam = next(iter(pf_by_scale.values()))
        last_known_tvt = float(kn['TVT_input'].iloc[-1])
        for variant in variants:
            try:
                pred = apply_selector_variant(variant, pf_by_scale, tvt_beam, last_known_tvt, hw=hw_masked, tw=tw)
                if pred is not None and len(pred) == len(hw_masked):
                    out['pf|' + variant] = _gold_np.asarray(pred, dtype=float)
            except Exception:
                pass
    except Exception as e:
        print('PF calibration skipped:', e)
    return out


def _gold_candidate_pool(wid, hw_masked, tw, data_dir, variants, include_pf=True, n_seeds=24, n_particles=350):
    pool = {}
    pool.update(_gold_poly_candidates(hw_masked))
    pool.update(_gold_surface_candidates(hw_masked, wid, data_dir))
    if include_pf:
        pool.update(_gold_pf_candidates(hw_masked, tw, variants, n_seeds=n_seeds, n_particles=n_particles))
    clean = {}
    for name, pred in pool.items():
        arr = _gold_np.asarray(pred, dtype=float)
        if len(arr) == len(hw_masked) and _gold_np.isfinite(arr).sum() >= max(20, len(hw_masked) // 20):
            clean[name] = arr
    return clean


def _gold_default_pf_name(hw):
    try:
        return 'pf|' + selector_well_code(hw)[1]
    except Exception:
        try:
            return 'pf|' + SELECTOR_GLOBAL_VARIANT
        except Exception:
            return None


def _gold_calibrate_well(wid, hw, tw, data_dir, variants):
    tvt = hw['TVT_input'].to_numpy(dtype=float)
    is_known = _gold_np.isfinite(tvt)
    is_hidden = ~is_known
    if not bool(is_hidden.any()):
        return None
    first_hidden = int(_gold_np.flatnonzero(is_hidden)[0])
    known_prefix = _gold_np.flatnonzero(is_known & (_gold_np.arange(len(hw)) < first_hidden))
    if len(known_prefix) < 140:
        return dict(well=wid, status='skip_short_prefix', known_prefix=int(len(known_prefix)))
    cuts = []
    for frac in _GOLD_CUT_FRACS:
        cut_pos = int(round(len(known_prefix) * float(frac)))
        cut_pos = max(50, min(cut_pos, len(known_prefix) - 35))
        if cut_pos <= 0 or cut_pos >= len(known_prefix):
            continue
        cutoff_idx = int(known_prefix[cut_pos - 1])
        hold_idx = known_prefix[cut_pos:]
        if len(hold_idx) >= 35:
            cuts.append((float(frac), cutoff_idx, hold_idx))
    if not cuts:
        return dict(well=wid, status='skip_no_holdout', known_prefix=int(len(known_prefix)))
    scores = {}
    cut_rows = []
    default_name = _gold_default_pf_name(hw)
    for frac, cutoff_idx, hold_idx in cuts:
        hw_m = hw.copy(deep=True)
        hw_m.loc[hw_m.index > cutoff_idx, 'TVT_input'] = _gold_np.nan
        pool = _gold_candidate_pool(
            wid, hw_m, tw, data_dir, variants,
            include_pf=_GOLD_INCLUDE_PF,
            n_seeds=_GOLD_CAL_SEEDS,
            n_particles=_GOLD_PARTICLES,
        )
        y = tvt[hold_idx]
        row = {'well': wid, 'cut_frac': frac, 'holdout_rows': int(len(hold_idx)), 'candidates': int(len(pool))}
        local = []
        for name, pred in pool.items():
            err = _gold_rmse(pred[hold_idx], y)
            if _gold_np.isfinite(err):
                scores.setdefault(name, []).append(err)
                local.append((err, name))
        local.sort()
        if local:
            row['best_name'] = local[0][1]
            row['best_rmse'] = float(local[0][0])
            if default_name in pool:
                row['default_rmse'] = float(_gold_rmse(pool[default_name][hold_idx], y))
            else:
                row['default_rmse'] = float('nan')
        cut_rows.append(row)
    if not scores:
        return dict(well=wid, status='skip_no_scores', known_prefix=int(len(known_prefix)))
    agg = {}
    for name, vals in scores.items():
        vals = _gold_np.asarray(vals, dtype=float)
        agg[name] = float(_gold_np.nanmedian(vals) + 0.10 * _gold_np.nanstd(vals))
    ordered = sorted((v, k) for k, v in agg.items() if _gold_np.isfinite(v))
    if not ordered:
        return dict(well=wid, status='skip_nonfinite_scores', known_prefix=int(len(known_prefix)))
    best_score, best_name = ordered[0]
    second_score = ordered[1][0] if len(ordered) > 1 else best_score
    if default_name is not None and default_name in agg:
        default_score = float(agg[default_name])
    else:
        pf_scores = [v for k, v in agg.items() if k.startswith('pf|')]
        default_score = float(_gold_np.nanmedian(pf_scores)) if pf_scores else float(second_score)
    consistency = 0.0
    comparable = 0
    for row in cut_rows:
        if _gold_np.isfinite(row.get('default_rmse', _gold_np.nan)):
            comparable += 1
            if row.get('best_rmse', float('inf')) <= row['default_rmse'] - 0.25:
                consistency += 1.0
    if comparable:
        consistency /= comparable
    else:
        winners = [r.get('best_name') for r in cut_rows if r.get('best_name')]
        consistency = float(sum(w == best_name for w in winners)) / max(1, len(winners))
    return dict(
        well=wid,
        status='ok',
        known_prefix=int(len(known_prefix)),
        cuts=int(len(cut_rows)),
        candidate_count=int(len(agg)),
        best_name=best_name,
        best_score=float(best_score),
        second_score=float(second_score),
        default_name=default_name,
        default_score=float(default_score),
        gain=float(default_score - best_score),
        rank_margin=float(second_score - best_score),
        consistency=float(consistency),
        cut_rows=cut_rows,
    )


def _gold_alpha(report, delta_rmse, delta_p95, profile_name):
    p = _GOLD_PROFILES[profile_name]
    if report.get('status') != 'ok':
        return 0.0
    gain = float(report.get('gain', 0.0))
    best = float(report.get('best_score', float('inf')))
    margin = float(report.get('rank_margin', 0.0))
    consistency = float(report.get('consistency', 0.0))
    if ((not _gold_np.isfinite(best)) or best > p['max_best'] or gain < p['min_gain'] or consistency < p['min_consistency'] or margin < p.get('min_margin', 0.0)):
        return 0.0
    alpha = p['base']
    alpha += p['gain_scale'] * min(max(gain, 0.0), 5.0) / 5.0
    alpha += p['margin_scale'] * min(max(margin, 0.0), 3.0) / 3.0
    if best <= 5.0:
        alpha += p['quality_bonus']
    best_name = str(report.get('best_name', ''))
    if (best_name.startswith('surface_') or best_name.startswith('dense_') or best_name.startswith('poly_') or best_name.startswith('contact_')) and consistency >= 0.67:
        alpha += 0.03 if profile_name != 'aggressive' else 0.06
    if _gold_np.isfinite(delta_rmse) and delta_rmse > p['delta_soft']:
        alpha *= max(0.20, p['delta_soft'] / max(delta_rmse, 1e-6))
    if _gold_np.isfinite(delta_p95) and delta_p95 > p['p95_hard']:
        return 0.0
    return float(min(p['cap'], max(0.0, alpha * 1.30)))


def _gold_profile_output(base_sub, candidate_by_id, reports_by_well, profile_name):
    prof = _GOLD_PROFILES[profile_name]
    out = base_sub.copy()
    move_rows = []
    for wid, rep in reports_by_well.items():
        ids = out.loc[out['well'] == wid, 'id'].astype(str).tolist()
        if not ids:
            continue
        if rep.get('status') != 'ok':
            row = dict(rep)
            row.update(dict(
                profile=profile_name,
                alpha=0.0,
                delta_rmse_vs_base=float('nan'),
                delta_p95_vs_base=float('nan'),
                max_move_clip=0.0,
                applied_rows=0,
                mean_abs_move=0.0,
                max_abs_move=0.0,
                apply_status=str(rep.get('status', 'skipped')),
            ))
            move_rows.append(row)
            continue
        cand = _gold_np.array([candidate_by_id.get(i, _gold_np.nan) for i in ids], dtype=float)
        idx = out.index[out['well'] == wid].to_numpy()
        base = out.loc[idx, 'tvt'].to_numpy(dtype=float)
        ok = _gold_np.isfinite(cand) & _gold_np.isfinite(base)
        if int(ok.sum()) != len(base):
            rep = dict(rep)
            rep['apply_status'] = 'skip_nonfinite_candidate'
            move_rows.append(rep)
            continue
        diff = cand - base
        delta_rmse = float(_gold_np.sqrt(_gold_np.mean(diff * diff))) if len(diff) else float('nan')
        delta_p95 = float(_gold_np.quantile(_gold_np.abs(diff), 0.95)) if len(diff) else float('nan')
        alpha = _gold_alpha(rep, delta_rmse, delta_p95, profile_name)
        gain = max(0.0, float(rep.get('gain', 0.0)))
        max_move = min(prof['clip_max'], prof['clip_base'] + prof['clip_gain'] * _gold_np.sqrt(gain + 1e-9))
        ramp = 1.0 - _gold_np.exp(-_gold_np.arange(len(diff), dtype=float) / max(80.0, 0.12 * max(1, len(diff))))
        move = _gold_np.clip(alpha * ramp * diff, -max_move, max_move)
        out.loc[idx, 'tvt'] = base + move
        row = dict(rep)
        row.update(dict(
            profile=profile_name,
            alpha=float(alpha),
            delta_rmse_vs_base=float(delta_rmse),
            delta_p95_vs_base=float(delta_p95),
            max_move_clip=float(max_move),
            applied_rows=int(len(idx)),
            mean_abs_move=float(_gold_np.mean(_gold_np.abs(move))) if len(move) else 0.0,
            max_abs_move=float(_gold_np.max(_gold_np.abs(move))) if len(move) else 0.0,
            apply_status='applied' if alpha > 0 else 'kept_base',
        ))
        move_rows.append(row)
    return out, move_rows


def _gold_best_contact_reconstruction(hw_te, hw_tr, tw_tr):
    min_phys = int(globals().get('GUARDED_OVERRIDE_MIN_VALID_PHYS_ROWS', 100))
    min_known = int(globals().get('GUARDED_OVERRIDE_MIN_KNOWN_PREFIX_ROWS', 50))
    rmse_limit = float(globals().get('GUARDED_OVERRIDE_PREFIX_RMSE_LIMIT', 1.0))
    primary_ref = str(globals().get('GUARDED_OVERRIDE_REF_COL', 'EGFDU')).strip()
    known = hw_te[hw_te['TVT_input'].notna()].copy()
    best = None
    reasons = []
    for ref_col in _gold_contact_ref_priority():
        phys = _gold_tvt_from_contacts(hw_tr, tw_tr, ref_col=ref_col)
        md_raw = hw_tr['MD'].to_numpy(dtype=float)
        m = _gold_np.isfinite(phys) & _gold_np.isfinite(md_raw)
        valid_rows = int(m.sum())
        if valid_rows < min_phys:
            reasons.append(f'{ref_col}:valid_rows={valid_rows}')
            continue
        order = _gold_np.argsort(md_raw[m])
        md_tr = md_raw[m][order]
        ph_tr = phys[m][order]
        comparable = known[(known['MD'] >= md_tr[0]) & (known['MD'] <= md_tr[-1])]
        if len(comparable) < min_known:
            reasons.append(f'{ref_col}:known_rows={len(comparable)}')
            continue
        rk = _gold_rmse(
            _gold_np.interp(comparable['MD'].to_numpy(dtype=float), md_tr, ph_tr),
            comparable['TVT_input'].to_numpy(dtype=float),
        )
        if not _gold_np.isfinite(rk):
            reasons.append(f'{ref_col}:rmse=nan')
            continue
        candidate = dict(
            ref_col=str(ref_col),
            rmse=float(rk),
            md_tr=md_tr,
            ph_tr=ph_tr,
            valid_phys_rows=valid_rows,
            comparable_known_rows=int(len(comparable)),
        )
        if best is None or rk < best['rmse']:
            best = candidate
        if str(ref_col) == primary_ref and rk <= rmse_limit:
            break
    if best is None:
        return None, ';'.join(reasons) or 'no_candidate'
    if best['rmse'] > rmse_limit:
        return None, f"best_ref={best['ref_col']} rmse={best['rmse']:.3f}>{rmse_limit:.3f}"
    return best, ''


def _gold_reapply_guarded_contact_override(sub_df, data_dir):
    sub = _gold_split_ids(sub_df[['id', 'tvt']])
    pred = dict(zip(sub['id'].astype(str), sub['tvt'].astype(float)))
    train_wells = set(p.stem.replace('__horizontal_well', '') for p in (data_dir / 'train').glob('*__horizontal_well.csv'))
    n_ok = 0
    n_skip = 0
    rows = []
    for wid, g in sub.groupby('well'):
        if wid not in train_wells:
            continue
        try:
            hw_te = _gold_pd.read_csv(data_dir / 'test' / f'{wid}__horizontal_well.csv')
            hw_tr = _gold_pd.read_csv(data_dir / 'train' / f'{wid}__horizontal_well.csv')
            tw_tr = _gold_pd.read_csv(data_dir / 'train' / f'{wid}__typewell.csv')
            best, reason = _gold_best_contact_reconstruction(hw_te, hw_tr, tw_tr)
            if best is None:
                n_skip += 1
                rows.append({'well': wid, 'status': 'skip', 'reason': reason, 'rows_total': int(len(g))})
                continue
            md_tr = best['md_tr']
            ph_tr = best['ph_tr']
            md_te = hw_te['MD'].to_numpy(dtype=float)
            n_row = 0
            for rid, ri in zip(g['id'].astype(str).values, g['row_idx'].values):
                ri = int(ri)
                if 0 <= ri < len(md_te):
                    mte = float(md_te[ri])
                    if md_tr[0] <= mte <= md_tr[-1]:
                        pred[rid] = float(_gold_np.interp(mte, md_tr, ph_tr))
                        n_row += 1
            print('gold contact override OK %s ref=%s rmse=%.4f rows=%d/%d' % (wid, best['ref_col'], best['rmse'], n_row, len(g)))
            rows.append({
                'well': wid,
                'status': 'override',
                'ref_col': best['ref_col'],
                'known_prefix_rmse': float(best['rmse']),
                'valid_phys_rows': int(best['valid_phys_rows']),
                'comparable_known_rows': int(best['comparable_known_rows']),
                'rows_overridden': int(n_row),
                'rows_total': int(len(g)),
                'reason': '',
            })
            n_ok += 1
        except Exception as e:
            print('gold contact override fallback', wid, e)
            rows.append({'well': wid, 'status': 'error', 'reason': str(e), 'rows_total': int(len(g))})
            n_skip += 1
    sub['tvt'] = sub['id'].astype(str).map(pred).astype(float)
    try:
        _gold_pd.DataFrame(rows).to_csv(_GOLD_WORK / 'gold_contact_override_report.csv', index=False)
    except Exception:
        pass
    print('gold contact override done: overridden=%d skipped=%d' % (n_ok, n_skip))
    return sub[['id', 'tvt']]


def _gold_validate_and_write(sub, sample, path):
    out = sub[['id', 'tvt']].copy()
    out['id'] = out['id'].astype(str)
    out['tvt'] = out['tvt'].astype(float)
    if list(out.columns) != ['id', 'tvt']:
        raise RuntimeError('bad output columns')
    if len(out) != len(sample):
        raise RuntimeError('bad output length')
    if not out['id'].equals(sample['id'].astype(str)):
        raise RuntimeError('id order mismatch')
    if not _gold_np.isfinite(out['tvt'].to_numpy(dtype=float)).all():
        raise RuntimeError('non-finite tvt in output')
    out.to_csv(path, index=False)
    return out


if not _GOLD_ENABLE:
    print('Visible-prefix calibration disabled; keeping current submission.csv')
else:
    _gold_t0 = _gold_time.time()
    _GOLD_WORK = _gold_work_dir()
    _GOLD_DATA = _gold_find_data()
    _gold_sample = _gold_pd.read_csv(_GOLD_DATA / 'sample_submission.csv')[['id']].copy()
    _gold_sample['id'] = _gold_sample['id'].astype(str)
    _gold_base = _gold_pd.read_csv(_GOLD_WORK / 'submission.csv')[['id', 'tvt']].copy()
    _gold_base['id'] = _gold_base['id'].astype(str)
    _gold_base['tvt'] = _gold_base['tvt'].astype(float)
    _gold_validate_and_write(_gold_base, _gold_sample, _GOLD_WORK / 'submission_self_verified_anchor.csv')
    _gold_base = _gold_split_ids(_gold_base)
    _gold_variants = _gold_variant_grid()
    print('Visible-prefix calibration:', dict(
        profile=_GOLD_PROFILE,
        include_pf=_GOLD_INCLUDE_PF,
        cal_seeds=_GOLD_CAL_SEEDS,
        final_seeds=_GOLD_FINAL_SEEDS,
        particles=_GOLD_PARTICLES,
        cut_fracs=_GOLD_CUT_FRACS,
        variants=len(_gold_variants),
        skip_bimodal=_GOLD_SKIP_BIMODAL,
        vp_skip_requires_low_trust=_GOLD_VP_SKIP_REQUIRES_LOW_TRUST,
        vp_low_trust_threshold=_GOLD_VP_LOW_TRUST_THRESHOLD,
        contact_override=_GOLD_CONTACT_OVERRIDE,
        final_selection=_GOLD_FINAL_SELECTION,
    ))

    _gold_bimodal_skip_wells, _gold_bimodal_skip_report = (
        _gold_load_bimodal_skip_wells(
            _GOLD_WORK,
            requires_low_trust=_GOLD_VP_SKIP_REQUIRES_LOW_TRUST,
            trust_threshold=_GOLD_VP_LOW_TRUST_THRESHOLD,
        ) if _GOLD_SKIP_BIMODAL else (set(), _gold_pd.DataFrame())
    )
    if len(_gold_bimodal_skip_report):
        _gold_bimodal_skip_report.to_csv(_GOLD_WORK / 'gold_prefix_bimodal_guard_report.csv', index=False)
    if _gold_bimodal_skip_wells:
        print('Visible-prefix bimodal guard wells:', sorted(_gold_bimodal_skip_wells), flush=True)

    from joblib import Parallel as _GoldParallel, delayed as _gold_delayed

    def _gold_run_one_well(_task):
        _wi, _wid = _task
        _worker_started = _gold_time.time()
        try:
            from threadpoolctl import threadpool_limits as _gold_threadpool_limits
            with _gold_threadpool_limits(limits=1):
                _hw_path = _GOLD_DATA / 'test' / f'{_wid}__horizontal_well.csv'
                _tw_path = _GOLD_DATA / 'test' / f'{_wid}__typewell.csv'
                if not _hw_path.exists() or not _tw_path.exists():
                    return dict(
                        order=_wi, well=_wid,
                        report=dict(well=_wid, status='skip_missing_files'),
                        cut_rows=[], candidate_by_id={},
                        elapsed_sec=float(_gold_time.time() - _worker_started),
                    )
                _hw = _gold_pd.read_csv(_hw_path)
                _tw = _gold_pd.read_csv(_tw_path)
                if _wid in _gold_bimodal_skip_wells:
                    return dict(
                        order=_wi, well=_wid,
                        report=dict(
                            well=_wid,
                            status='skip_bimodal_hedge',
                            bimodal_prefix_guard=True,
                            reason='visible-prefix commit disabled for active bimodal selector well',
                        ),
                        cut_rows=[], candidate_by_id={},
                        elapsed_sec=float(_gold_time.time() - _worker_started),
                    )
                _rep = _gold_calibrate_well(_wid, _hw, _tw, _GOLD_DATA, _gold_variants)
                if _rep is None:
                    _rep = dict(well=_wid, status='skip_none')
                _cut_rows = _rep.pop('cut_rows', []) if isinstance(_rep, dict) else []
                _candidate_by_id = {}
                if _rep.get('status') == 'ok':
                    _best_name = _rep['best_name']
                    _need_pf_final = str(_best_name).startswith('pf|')
                    _pool_final = _gold_candidate_pool(
                        _wid, _hw, _tw, _GOLD_DATA, _gold_variants,
                        include_pf=_need_pf_final,
                        n_seeds=_GOLD_FINAL_SEEDS,
                        n_particles=_GOLD_PARTICLES,
                    )
                    if _best_name not in _pool_final and _need_pf_final:
                        _pool_final = _gold_candidate_pool(
                            _wid, _hw, _tw, _GOLD_DATA, _gold_variants,
                            include_pf=False,
                            n_seeds=0,
                            n_particles=_GOLD_PARTICLES,
                        )
                    if _best_name in _pool_final:
                        _g = _gold_base[_gold_base['well'] == _wid]
                        _arr = _pool_final[_best_name]
                        for _rid, _ri in zip(
                            _g['id'].astype(str).values,
                            _g['row_idx'].astype(int).values,
                        ):
                            if 0 <= int(_ri) < len(_arr) and _gold_np.isfinite(_arr[int(_ri)]):
                                _candidate_by_id[_rid] = float(_arr[int(_ri)])
                        _rep['final_candidate_available'] = True
                    else:
                        _rep['final_candidate_available'] = False
                        _rep['status'] = 'skip_no_final_candidate'
                return dict(
                    order=_wi, well=_wid, report=_rep,
                    cut_rows=_cut_rows, candidate_by_id=_candidate_by_id,
                    elapsed_sec=float(_gold_time.time() - _worker_started),
                )
        except Exception as _e:
            return dict(
                order=_wi, well=_wid,
                report=dict(well=_wid, status='error', error=repr(_e)),
                cut_rows=[], candidate_by_id={},
                elapsed_sec=float(_gold_time.time() - _worker_started),
            )

    _gold_reports = []
    _gold_cut_reports = []
    _gold_candidate_by_id = {}
    _gold_wells = list(_gold_base['well'].drop_duplicates())[:_GOLD_MAX_WELLS]
    _gold_requested_processes = 4
    _gold_effective_processes = min(_gold_requested_processes, len(_gold_wells))
    globals()['EXP514_KDTREE_WORKERS'] = 1
    _gold_parallel_started = _gold_time.time()
    _gold_tasks = list(enumerate(_gold_wells, 1))
    if _gold_effective_processes > 1:
        _gold_results = _GoldParallel(
            n_jobs=_gold_effective_processes,
            backend='multiprocessing',
            batch_size=1,
            pre_dispatch=_gold_effective_processes,
        )(
            _gold_delayed(_gold_run_one_well)(_task)
            for _task in _gold_tasks
        )
        _gold_backend = 'multiprocessing'
    else:
        _gold_results = [_gold_run_one_well(_task) for _task in _gold_tasks]
        _gold_backend = 'sequential'
    _gold_results = sorted(_gold_results, key=lambda item: int(item['order']))
    if [int(item['order']) for item in _gold_results] != list(range(1, len(_gold_wells) + 1)):
        raise RuntimeError('Gold process result order/coverage mismatch')
    if [str(item['well']) for item in _gold_results] != [str(well) for well in _gold_wells]:
        raise RuntimeError('Gold process well order changed')
    for _result in _gold_results:
        _rep = _result['report']
        if _rep.get('status') == 'error':
            raise RuntimeError(f"Gold process worker failed for {_result['well']}: {_rep.get('error')}")
        _gold_reports.append(_rep)
        _gold_cut_reports.extend(_result['cut_rows'])
        for _rid, _value in _result['candidate_by_id'].items():
            if _rid in _gold_candidate_by_id:
                raise RuntimeError(f'Gold process duplicate candidate id: {_rid}')
            _gold_candidate_by_id[_rid] = float(_value)
        print(
            '[gold %d/%d] %s report:' % (
                int(_result['order']), len(_gold_wells), _result['well']
            ),
            {k: _rep.get(k) for k in ['status', 'best_name', 'best_score', 'default_score', 'gain', 'consistency']},
            flush=True,
        )
    GOLD_WELL_PARALLEL_REPORT = {
        'requested_processes': _gold_requested_processes,
        'effective_processes': _gold_effective_processes,
        'backend': _gold_backend,
        'wells': len(_gold_wells),
        'result_merge_order': 'input_well_order',
        'inner_blas_threads': 1,
        'inner_kdtree_workers': 1,
        'elapsed_seconds': round(_gold_time.time() - _gold_parallel_started, 6),
        'worker_elapsed_seconds': [round(float(item['elapsed_sec']), 6) for item in _gold_results],
    }
    _gold_report_df = _gold_pd.DataFrame(_gold_reports)
    _gold_report_df.to_csv(_GOLD_WORK / 'gold_prefix_calibration_report.csv', index=False)
    if _gold_cut_reports:
        _gold_pd.DataFrame(_gold_cut_reports).to_csv(_GOLD_WORK / 'gold_prefix_cut_report.csv', index=False)
    _reports_by_well = {r.get('well'): r for r in _gold_reports if isinstance(r, dict) and r.get('well')}

    _profile_summaries = {}
    for _profile_name in ('conservative', 'balanced', 'aggressive'):
        _profile_sub, _move_rows = _gold_profile_output(_gold_base, _gold_candidate_by_id, _reports_by_well, _profile_name)
        if _GOLD_CONTACT_OVERRIDE:
            _profile_sub = _gold_reapply_guarded_contact_override(_profile_sub[['id', 'tvt']], _GOLD_DATA)
        else:
            _profile_sub = _profile_sub[['id', 'tvt']].copy()
        _path = _GOLD_WORK / f'submission_gold_prefix_{_profile_name}.csv'
        _profile_sub = _gold_validate_and_write(_profile_sub, _gold_sample, _path)
        _move_df = _gold_pd.DataFrame(_move_rows)
        _move_df.to_csv(_GOLD_WORK / f'gold_prefix_moves_{_profile_name}.csv', index=False)
        _profile_summaries[_profile_name] = dict(
            file=str(_path),
            rows=int(len(_profile_sub)),
            sha256=_gold_sha256(_path),
            applied_wells=int((_move_df.get('apply_status') == 'applied').sum()) if 'apply_status' in _move_df else 0,
            mean_abs_move=float(_move_df['mean_abs_move'].mean()) if 'mean_abs_move' in _move_df and len(_move_df) else 0.0,
            max_abs_move=float(_move_df['max_abs_move'].max()) if 'max_abs_move' in _move_df and len(_move_df) else 0.0,
        )

    if _GOLD_FINAL_SELECTION in {'self_verified_anchor', 'anchor', 'self_verified'}:
        _chosen_path = _GOLD_WORK / 'submission_self_verified_anchor.csv'
    elif _GOLD_FINAL_SELECTION in {'profile', 'gold_profile', 'visible_prefix'}:
        _chosen_path = _GOLD_WORK / f'submission_gold_prefix_{_GOLD_PROFILE}.csv'
    else:
        raise ValueError(f'Unknown ROGII_GOLD_FINAL_SELECTION={_GOLD_FINAL_SELECTION!r}')
    _chosen = _gold_pd.read_csv(_chosen_path)
    _chosen = _gold_validate_and_write(_chosen, _gold_sample, _GOLD_WORK / 'submission.csv')
    _audit = dict(
        selected_profile=_GOLD_PROFILE,
        selected_sha256=_gold_sha256(_GOLD_WORK / 'submission.csv'),
        self_verified_anchor_sha256=_gold_sha256(_GOLD_WORK / 'submission_self_verified_anchor.csv'),
        elapsed_sec=float(_gold_time.time() - _gold_t0),
        wells=int(len(_gold_wells)),
        candidates_with_final_values=int(len(_gold_candidate_by_id)),
        bimodal_guard_enabled=bool(_GOLD_SKIP_BIMODAL),
        bimodal_guard_requires_low_trust=bool(_GOLD_VP_SKIP_REQUIRES_LOW_TRUST),
        bimodal_guard_low_trust_threshold=float(_GOLD_VP_LOW_TRUST_THRESHOLD),
        bimodal_guard_wells=sorted(_gold_bimodal_skip_wells),
        contact_override_enabled=bool(_GOLD_CONTACT_OVERRIDE),
        final_selection=_GOLD_FINAL_SELECTION,
        chosen_path=str(_chosen_path),
        profiles=_profile_summaries,
        well_parallel=GOLD_WELL_PARALLEL_REPORT,
    )
    with open(_GOLD_WORK / 'gold_prefix_submission_audit.json', 'w', encoding='utf-8') as f:
        _gold_json.dump(_audit, f, indent=2, sort_keys=True)
    print('Visible-prefix selected submission.csv:', _audit, flush=True)

# %%
# Optional saved-model correction on top of the current trajectory submission.
if not bool(globals().get('RUN_MODEL_PACKAGE_CORRECTION', False)):
    print('Model package correction skipped.')
else:
    import importlib.util
    import inspect
    import json
    import pickle
    import sys
    from pathlib import Path as _MPPath
    from typing import Any as _MPAny

    import numpy as _mp_np
    import pandas as _mp_pd

    _mp_work = _MPPath(globals().get('OUTPUT_DIR', _MPPath('/kaggle/working')))
    _mp_final_output = _MPPath(globals().get('FINAL_SUBMISSION_OUTPUT', _mp_work / 'submission.csv'))
    _mp_sample_path = globals().get('SAMPLE_SUBMISSION', _MPPath(globals().get('COMPETITION_DATA_ROOT', '/kaggle/input/competitions/rogii-wellbore-geology-prediction')) / 'sample_submission.csv')
    _mp_data_dir = _MPPath(globals().get('DATA_DIR', globals().get('COMPETITION_DATA_ROOT', '/kaggle/input/competitions/rogii-wellbore-geology-prediction')))
    if not _mp_final_output.exists():
        raise RuntimeError(f'Base submission for model package correction was not produced: {_mp_final_output}')
    _mp_sample = _mp_pd.read_csv(_mp_sample_path)[['id']]
    _mp_base = _mp_sample.merge(_mp_pd.read_csv(_mp_final_output)[['id', 'tvt']], on='id', how='left')
    if _mp_base['tvt'].isna().any():
        raise RuntimeError('Base submission has missing sample ids before model package correction.')
    _mp_base.to_csv(_mp_work / 'submission_before_model_package.csv', index=False)


    def _mp_read_json(path: _MPPath):
        with _MPPath(path).open() as f:
            return json.load(f)

    def _mp_manifest_path(manifest: dict, key: str, default: str) -> str:
        value = manifest.get(key, default)
        if isinstance(value, str) and value.strip():
            return value
        raise RuntimeError(f'Manifest field {key!r} must be a relative file path string.')

    def _mp_prediction_column(entry: dict) -> str:
        if entry.get('prediction_column'):
            return str(entry['prediction_column'])
        branch_name = entry.get('branch_name')
        model_name = entry.get('model_name')
        if not branch_name or not model_name:
            raise RuntimeError(f'Model entry needs prediction_column or branch_name/model_name: {entry}')
        return f'pred_delta_{branch_name}_{model_name}'

    def _mp_find_package_root() -> _MPPath | None:
        roots = [
            _MPPath(path)
            for path in globals().get('MODEL_PACKAGE_ROOTS', [])
            if str(path).strip()
        ]
        for root in roots:
            candidates = [root]
            candidates += [root / 'rogii_model_package', root / 'rogii_artifacts']
            for candidate in candidates:
                if (candidate / 'metadata' / 'model_package_manifest.json').exists():
                    return candidate
        if bool(globals().get('MODEL_PACKAGE_ALLOW_AUTO_SEARCH', False)):
            input_root = _MPPath('/kaggle/input')
            if input_root.exists():
                for manifest_path in input_root.glob('**/metadata/model_package_manifest.json'):
                    return manifest_path.parents[1]
        return None

    def _mp_validate_submission_ids(df: _mp_pd.DataFrame, sample: _mp_pd.DataFrame, label: str) -> _mp_pd.DataFrame:
        if not {'id', 'tvt'}.issubset(df.columns):
            raise RuntimeError(f'{label}: expected columns id,tvt; got {list(df.columns)}')
        frame = df[['id', 'tvt']].copy()
        frame['id'] = frame['id'].astype(str)
        sample_ids = sample[['id']].copy()
        sample_ids['id'] = sample_ids['id'].astype(str)
        if frame['id'].duplicated().any():
            dup = frame.loc[frame['id'].duplicated(), 'id'].head(10).tolist()
            raise RuntimeError(f'{label}: duplicate ids: {dup}')
        aligned = sample_ids.merge(frame, on='id', how='left')
        if aligned['tvt'].isna().any():
            bad = aligned.loc[aligned['tvt'].isna(), 'id'].head(10).tolist()
            raise RuntimeError(f'{label}: missing predictions after alignment; examples={bad}')
        aligned['tvt'] = _mp_pd.to_numeric(aligned['tvt'], errors='coerce')
        if aligned['tvt'].isna().any() or not _mp_np.isfinite(aligned['tvt'].to_numpy(dtype=float)).all():
            raise RuntimeError(f'{label}: non-finite tvt values')
        return aligned[['id', 'tvt']]

    def _mp_load_feature_builder(package_root: _MPPath):
        feature_dir = package_root / 'feature_builders'
        for import_root in [package_root, feature_dir]:
            key = str(import_root)
            if key not in sys.path:
                sys.path.insert(0, key)
        sys.modules.pop('rogii_model_package_feature_builder', None)
        for path in [feature_dir / 'build_features.py', feature_dir / 'feature_builder.py']:
            if path.exists():
                spec = importlib.util.spec_from_file_location('rogii_model_package_feature_builder', path)
                if spec is None or spec.loader is None:
                    raise RuntimeError(f'Could not import feature builder: {path}')
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                for fn_name in ['build_features', 'build_tail_features', 'make_features']:
                    if hasattr(module, fn_name):
                        return getattr(module, fn_name), path
        raise RuntimeError('Model package has no feature builder file.')

    def _mp_call_feature_builder(builder, *, data_dir: _MPPath, sample: _mp_pd.DataFrame, package_root: _MPPath, manifest: dict) -> _mp_pd.DataFrame:
        possible_kwargs = {
            'data_dir': data_dir,
            'competition_root': data_dir,
            'sample_submission': sample,
            'sample': sample,
            'package_root': package_root,
            'manifest': manifest,
            'config': manifest,
        }
        sig = inspect.signature(builder)
        kwargs = {name: value for name, value in possible_kwargs.items() if name in sig.parameters}
        features = builder(**kwargs)
        if not isinstance(features, _mp_pd.DataFrame):
            raise RuntimeError('Feature builder must return a pandas DataFrame.')
        if 'id' not in features.columns:
            raise RuntimeError('Feature frame must include id.')
        features = features.copy()
        features['id'] = features['id'].astype(str)
        sample_ids = sample[['id']].copy()
        sample_ids['id'] = sample_ids['id'].astype(str)
        if features['id'].duplicated().any():
            dup = features.loc[features['id'].duplicated(), 'id'].head(10).tolist()
            raise RuntimeError(f'Feature frame contains duplicate ids: {dup}')
        missing = sorted(set(sample_ids['id']) - set(features['id']))
        extra = sorted(set(features['id']) - set(sample_ids['id']))
        if missing or extra:
            raise RuntimeError(f'Feature frame id mismatch: missing={len(missing)}, extra={len(extra)}, examples={missing[:10]}')
        return sample_ids.merge(features, on='id', how='left')

    def _mp_feature_columns_for_model(feature_columns, entry: dict) -> list[str]:
        if isinstance(entry.get('feature_columns'), list):
            return list(entry['feature_columns'])
        feature_set = entry.get('feature_set')
        if isinstance(feature_columns, list):
            return list(feature_columns)
        if isinstance(feature_columns, dict):
            if feature_set and isinstance(feature_columns.get(feature_set), list):
                return list(feature_columns[feature_set])
            if isinstance(feature_columns.get('columns'), list):
                return list(feature_columns['columns'])
        raise RuntimeError(f'Could not resolve feature columns for model entry: {entry}')

    def _mp_load_model(package_root: _MPPath, entry: dict):
        model_type = entry.get('model_type')
        path = package_root / entry['path']
        if model_type == 'lightgbm_booster':
            import lightgbm as lgb
            return lgb.Booster(model_file=str(path))
        if model_type == 'xgboost_json':
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(str(path))
            return booster
        if model_type == 'catboost_cbm':
            from catboost import CatBoostRegressor
            model = CatBoostRegressor()
            model.load_model(str(path))
            return model
        if model_type in {'lightgbm_sklearn_pickle', 'xgboost_pickle', 'sklearn_pickle'}:
            try:
                import joblib
                return joblib.load(path)
            except Exception:
                with path.open('rb') as f:
                    return pickle.load(f)
        if model_type == 'torch_tcn':
            import torch
            try:
                return torch.load(path, map_location='cpu', weights_only=False)
            except TypeError:
                return torch.load(path, map_location='cpu')
        raise RuntimeError(f'Unsupported model_type={model_type!r}')

    def _mp_first_existing_column(frame: _mp_pd.DataFrame, names: list[str]) -> str | None:
        for name in names:
            if name in frame.columns:
                return name
        return None

    def _mp_build_tcn_module(torch, nn, n_features: int, config: dict):
        class TCNBlock(nn.Module):
            def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
                super().__init__()
                padding = dilation * (kernel_size - 1) // 2
                self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
                self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
                self.act = nn.GELU()
                self.drop = nn.Dropout(float(dropout))
                self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)
            def forward(self, x):
                residual = self.skip(x)
                y1 = self.drop(self.act(self.conv1(x)))
                y2 = self.drop(self.act(self.conv2(y1)))
                if y2.shape[-1] != residual.shape[-1]:
                    min_len = min(y2.shape[-1], residual.shape[-1])
                    y2 = y2[..., :min_len]
                    residual = residual[..., :min_len]
                return self.act(y2 + residual)
        class TCNRegressor(nn.Module):
            def __init__(self):
                super().__init__()
                blocks = []
                in_ch = int(n_features)
                channels = int(config.get('channels', 64))
                kernel_size = int(config.get('kernel_size', 5))
                dropout = float(config.get('dropout', 0.0))
                for i in range(int(config.get('blocks', 6))):
                    blocks.append(TCNBlock(in_ch, channels, kernel_size=kernel_size, dilation=2**i, dropout=dropout))
                    in_ch = channels
                self.net = nn.Sequential(*blocks)
                self.head = nn.Conv1d(channels, 1, 1)
            def forward(self, x):
                return self.head(self.net(x)).squeeze(1)
        return TCNRegressor()

    def _mp_predict_torch_tcn(payload: dict, frame: _mp_pd.DataFrame, columns: list[str], entry: dict) -> _mp_np.ndarray:
        import torch
        from torch import nn
        X = frame[columns].replace([_mp_np.inf, -_mp_np.inf], _mp_np.nan).to_numpy(dtype=_mp_np.float32)
        standardizer = payload.get('standardizer', {}) or {}
        mean = _mp_np.asarray(standardizer.get('mean'), dtype=_mp_np.float32)
        scale = _mp_np.asarray(standardizer.get('scale'), dtype=_mp_np.float32)
        if mean.shape[0] != X.shape[1] or scale.shape[0] != X.shape[1]:
            raise RuntimeError(f'torch_tcn standardizer shape mismatch for {entry.get("prediction_column")}')
        X = (X - mean.reshape(1, -1)) / _mp_np.maximum(scale.reshape(1, -1), 1e-6)
        X = _mp_np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(_mp_np.float32)
        group_col = entry.get('sequence_group_column') or _mp_first_existing_column(frame, ['well_id', 'well', 'WELL'])
        order_col = entry.get('sequence_order_column') or _mp_first_existing_column(frame, ['row_index', 'row', 'sample_index', 'MD'])
        tmp = _mp_pd.DataFrame({'_pos': _mp_np.arange(len(frame), dtype=int)})
        tmp['_group'] = frame[group_col].astype(str).to_numpy() if group_col and group_col in frame.columns else frame['id'].astype(str).str.rsplit('_', n=1).str[0].to_numpy()
        tmp['_order'] = _mp_pd.to_numeric(frame[order_col], errors='coerce').to_numpy(dtype=float) if order_col and order_col in frame.columns else _mp_np.arange(len(frame), dtype=float)
        device = torch.device('cpu')
        model = _mp_build_tcn_module(torch, nn, len(columns), payload.get('config', {}) or {}).to(device)
        model.load_state_dict(payload['state_dict'])
        model.eval()
        pred = _mp_np.full(len(frame), _mp_np.nan, dtype=_mp_np.float32)
        with torch.no_grad():
            for _, part in tmp.groupby('_group', sort=False):
                ordered = part.sort_values('_order')
                idx = ordered['_pos'].to_numpy(dtype=int)
                xt = torch.from_numpy(X[idx].T[None, :, :].copy()).to(device)
                pred[idx] = model(xt).detach().cpu().numpy().reshape(-1)[:len(idx)].astype(_mp_np.float32)
        if not _mp_np.isfinite(pred).all():
            raise RuntimeError(f'torch_tcn produced non-finite predictions for {entry.get("prediction_column")}')
        return pred.astype(float)

    def _mp_feature_matrix(frame: _mp_pd.DataFrame, columns: list[str], entry: dict, manifest: dict) -> _mp_pd.DataFrame:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise RuntimeError(f'Feature frame missing {len(missing)} columns; examples={missing[:10]}')
        X_df = frame[columns].replace([_mp_np.inf, -_mp_np.inf], _mp_np.nan)
        fill_value = entry.get('fillna', None)
        policy = str(entry.get('missing_value_policy', manifest.get('missing_value_policy', 'native'))).lower()
        if fill_value is not None:
            X_df = X_df.fillna(float(fill_value))
        elif policy in {'native', 'none', 'null'}:
            pass
        elif policy in {'zero', 'fill_zero'}:
            X_df = X_df.fillna(0.0)
        else:
            raise RuntimeError(f'Unsupported missing_value_policy={policy!r}')
        return X_df

    def _mp_predict_model(model, model_type: str, frame: _mp_pd.DataFrame, columns: list[str], entry: dict, manifest: dict) -> _mp_np.ndarray:
        X_df = _mp_feature_matrix(frame, columns, entry, manifest)
        if model_type == 'torch_tcn':
            pred = _mp_predict_torch_tcn(model, frame, columns, entry)
        elif model_type == 'xgboost_json':
            import xgboost as xgb
            pred = model.predict(xgb.DMatrix(X_df.to_numpy(dtype=_mp_np.float32)))
        else:
            pred = model.predict(X_df)
        pred = _mp_np.asarray(pred, dtype=float)
        if pred.ndim > 1:
            pred = pred.reshape(len(frame), -1)[:, 0]
        if len(pred) != len(frame):
            raise RuntimeError(f'Model prediction length mismatch: got {len(pred)}, expected {len(frame)}')
        if not _mp_np.isfinite(pred).all():
            raise RuntimeError(f'Model {entry.get("prediction_column")} produced non-finite predictions.')
        return pred

    def _mp_weights_from_keys_and_coef(keys, coef, label: str) -> dict[str, float]:
        keys = list(keys)
        coef = list(coef)
        if len(keys) != len(coef):
            raise RuntimeError(f'{label} result_keys and coef length mismatch: {len(keys)} != {len(coef)}')
        return {str(k): float(v) for k, v in zip(keys, coef)}

    def _mp_normalize_weights(blend_config: dict) -> dict[str, float]:
        if isinstance(blend_config.get('weights'), dict):
            return {str(k): float(v) for k, v in blend_config['weights'].items()}
        if isinstance(blend_config.get('model_weights'), dict):
            return {str(k): float(v) for k, v in blend_config['model_weights'].items()}
        weights = {}
        for row in blend_config.get('models', []):
            if 'prediction_column' in row and 'weight' in row:
                weights[str(row['prediction_column'])] = float(row['weight'])
        if weights:
            return weights
        if 'result_keys' in blend_config and 'coef' in blend_config:
            return _mp_weights_from_keys_and_coef(blend_config['result_keys'], blend_config['coef'], 'blend_config')
        stacker = blend_config.get('stacker')
        if isinstance(stacker, dict) and 'result_keys' in stacker and 'coef' in stacker:
            return _mp_weights_from_keys_and_coef(stacker['result_keys'], stacker['coef'], 'blend_config.stacker')
        raise RuntimeError('blend_config.json must contain weights/model_weights/models or result_keys/coef.')

    def _mp_blend_intercept(blend_config: dict) -> float:
        for key in ['intercept', 'bias', 'blend_intercept']:
            if key in blend_config:
                return float(blend_config[key])
        stacker = blend_config.get('stacker')
        if isinstance(stacker, dict):
            for key in ['intercept', 'bias']:
                if key in stacker:
                    return float(stacker[key])
        return 0.0

    def _mp_apply_delta_postprocess(delta: _mp_np.ndarray, blend_config: dict, features: _mp_pd.DataFrame) -> _mp_np.ndarray:
        post = blend_config.get('postprocess', {}) or {}
        out = delta.astype(float).copy()
        tau = post.get('fade_tau_md', post.get('tau', None))
        if tau is not None:
            md_col = _mp_first_existing_column(features, ['md_since_ps', 'md_since', 'md_delta', 'MD_since', 'md_from_start'])
            if md_col is None:
                raise RuntimeError('postprocess.fade_tau_md was set, but no md_since column is available.')
            md_since = _mp_pd.to_numeric(features[md_col], errors='coerce').to_numpy(dtype=float)
            out *= 1.0 - _mp_np.exp(-_mp_np.maximum(md_since, 0.0) / float(tau))
        out *= float(post.get('alpha', 1.0))
        return out

    def _mp_apply_savgol(tvt: _mp_np.ndarray, blend_config: dict, features: _mp_pd.DataFrame) -> _mp_np.ndarray:
        post = blend_config.get('postprocess', {}) or {}
        window = int(post.get('savgol_window', 0) or 0)
        if window <= 2:
            return tvt
        if window % 2 == 0:
            window += 1
        poly = int(post.get('savgol_poly', 2) or 2)
        from scipy.signal import savgol_filter
        out = tvt.astype(float).copy()
        group_col = _mp_first_existing_column(features, ['well_id', 'well', 'WELL'])
        row_col = _mp_first_existing_column(features, ['row_index', 'row', 'sample_index'])
        tmp = _mp_pd.DataFrame({'_pos': _mp_np.arange(len(out), dtype=int), '_tvt': out})
        tmp['_group'] = features[group_col].astype(str).to_numpy() if group_col else features['id'].astype(str).str.rsplit('_', n=1).str[0].to_numpy()
        tmp['_order'] = _mp_pd.to_numeric(features[row_col], errors='coerce').to_numpy(dtype=float) if row_col else _mp_np.arange(len(out), dtype=float)
        for _, grp in tmp.groupby('_group', sort=False):
            if len(grp) < max(window, poly + 2):
                continue
            order = grp.sort_values('_order')
            w = min(window, len(order) if len(order) % 2 == 1 else len(order) - 1)
            if w < poly + 2 or w <= 2:
                continue
            out[order['_pos'].to_numpy(dtype=int)] = savgol_filter(order['_tvt'].to_numpy(dtype=float), window_length=w, polyorder=min(poly, w - 1), mode='interp')
        return out

    def _mp_build_submission() -> tuple[_mp_pd.DataFrame, _mp_pd.DataFrame, _mp_pd.DataFrame, dict]:
        package_root = _mp_find_package_root()
        if package_root is None:
            if bool(globals().get('MODEL_PACKAGE_REQUIRE', True)):
                raise RuntimeError('Model package dataset was not found.')
            return None, _mp_pd.DataFrame(), _mp_pd.DataFrame(), {}
        manifest = _mp_read_json(package_root / 'metadata' / 'model_package_manifest.json')
        blend_config = _mp_read_json(package_root / _mp_manifest_path(manifest, 'blend_config', 'stacking/blend_config.json'))
        feature_columns_config = _mp_read_json(package_root / _mp_manifest_path(manifest, 'feature_columns', 'feature_builders/feature_columns.json'))
        builder, builder_path = _mp_load_feature_builder(package_root)
        feature_frame = _mp_call_feature_builder(
            builder,
            data_dir=_mp_data_dir,
            sample=_mp_sample,
            package_root=package_root,
            manifest=manifest,
        )
        predictions = _mp_pd.DataFrame({'id': feature_frame['id'].to_numpy()})
        report_rows = []
        for entry in manifest.get('models', []):
            pred_col = _mp_prediction_column(entry)
            model_type = entry.get('model_type')
            if model_type == 'direct_feature':
                source_col = entry.get('feature_column')
                if source_col not in feature_frame.columns:
                    raise RuntimeError(f'direct_feature source column is missing: {source_col}')
                pred = _mp_pd.to_numeric(feature_frame[source_col], errors='coerce').to_numpy(dtype=float)
            else:
                columns = _mp_feature_columns_for_model(feature_columns_config, entry)
                model = _mp_load_model(package_root, entry)
                pred = _mp_predict_model(model, model_type, feature_frame, columns, entry, manifest)
            if not _mp_np.isfinite(pred).all():
                raise RuntimeError(f'Non-finite predictions from {pred_col}')
            predictions[pred_col] = pred
            report_rows.append({
                'prediction_column': pred_col,
                'model_type': model_type,
                'pred_mean': float(_mp_np.mean(pred)),
                'pred_std': float(_mp_np.std(pred)),
                'pred_min': float(_mp_np.min(pred)),
                'pred_max': float(_mp_np.max(pred)),
            })
        weights = _mp_normalize_weights(blend_config)
        missing_cols = [col for col in weights if col not in predictions.columns]
        if missing_cols:
            raise RuntimeError(f'Blend config references missing prediction columns: {missing_cols}')
        pred_value = _mp_np.full(len(predictions), _mp_blend_intercept(blend_config), dtype=float)
        for col, weight in weights.items():
            pred_value += float(weight) * predictions[col].to_numpy(dtype=float)
        target_space = blend_config.get('target_space') or blend_config.get('prediction_space') or manifest.get('target_space', 'delta')
        if target_space == 'delta':
            if 'last_known_TVT' not in feature_frame.columns:
                raise RuntimeError('Delta-space blend requires last_known_TVT in feature frame.')
            pred_value = _mp_apply_delta_postprocess(pred_value, blend_config, feature_frame)
            tvt = feature_frame['last_known_TVT'].to_numpy(dtype=float) + pred_value
        elif target_space == 'tvt':
            tvt = pred_value
        else:
            raise RuntimeError(f'Unsupported target_space={target_space!r}')
        tvt = _mp_apply_savgol(tvt, blend_config, feature_frame)
        clip_min = globals().get('TVT_CLIP_MIN', None) if globals().get('TVT_CLIP_MIN', None) is not None else blend_config.get('tvt_clip_min')
        clip_max = globals().get('TVT_CLIP_MAX', None) if globals().get('TVT_CLIP_MAX', None) is not None else blend_config.get('tvt_clip_max')
        if clip_min is not None or clip_max is not None:
            tvt = _mp_np.clip(tvt, -_mp_np.inf if clip_min is None else float(clip_min), _mp_np.inf if clip_max is None else float(clip_max))
        submission = _mp_validate_submission_ids(_mp_pd.DataFrame({'id': feature_frame['id'].to_numpy(), 'tvt': tvt}), _mp_sample, 'model_package_submission')
        info = {
            'package_root': str(package_root),
            'feature_builder': str(builder_path),
            'target_space': target_space,
            'weight_sum': float(sum(weights.values())),
            'postprocess': json.dumps(blend_config.get('postprocess', {}) or {}),
        }
        weight_report = _mp_pd.DataFrame([{'prediction_column': k, 'weight': v} for k, v in weights.items()])
        return submission, _mp_pd.DataFrame(report_rows), weight_report, info

    _mp_pkg_sub, _mp_pred_report, _mp_weight_report, _mp_info = _mp_build_submission()
    if _mp_pkg_sub is None:
        _mp_pd.Series({'model_package_available': False}).to_csv(_mp_work / 'model_package_correction_summary.csv')
    else:
        _mp_pkg_sub.to_csv(_mp_work / 'submission_model_package_only.csv', index=False)
        _mp_pred_report.to_csv(_mp_work / 'model_package_prediction_report.csv', index=False)
        _mp_weight_report.to_csv(_mp_work / 'model_package_blend_weights.csv', index=False)
        _mp_merged = _mp_base.rename(columns={'tvt': 'tvt_base'}).merge(
            _mp_pkg_sub.rename(columns={'tvt': 'tvt_model_package'}), on='id', how='inner'
        )
        if len(_mp_merged) != len(_mp_sample):
            raise RuntimeError('Model package blend id mismatch.')
        _base_v = _mp_merged['tvt_base'].to_numpy(dtype=float)
        _pkg_v = _mp_merged['tvt_model_package'].to_numpy(dtype=float)
        _signed_diff = _pkg_v - _base_v
        _diff = _mp_np.abs(_signed_diff)
        _p95 = float(_mp_np.quantile(_diff, 0.95))
        _diff_rmse = float(_mp_np.sqrt(_mp_np.mean(_signed_diff * _signed_diff)))
        _active_signal = bool(_mp_np.isfinite(_diff_rmse) and _diff_rmse > 1e-9)
        _mp_pd.Series({
            'model_package_available': True,
            'active_signal': _active_signal,
            'mean_signed_model_package_diff': float(_mp_np.mean(_signed_diff)),
            'mean_abs_model_package_diff': float(_mp_np.mean(_diff)),
            'p95_abs_model_package_diff': _p95,
            'max_abs_model_package_diff': float(_mp_np.max(_diff)),
            'model_package_diff_rmse': _diff_rmse,
            'model_package_corr_with_base': float(_mp_np.corrcoef(_base_v, _pkg_v)[0, 1]) if len(_base_v) > 1 else float('nan'),
        }).to_csv(_mp_work / 'model_package_active_signal_summary.csv')
        _disable_limit = globals().get('MODEL_PACKAGE_DIFF_P95_DISABLE', None)
        _disabled = _disable_limit is not None and _p95 > float(_disable_limit)
        _selected_gmax = float(globals().get('MODEL_PACKAGE_GATED_MAX_WEIGHT', 0.005))
        _scale = float(globals().get('MODEL_PACKAGE_GATED_SCALE', 4.0))
        _candidates = list(float(x) for x in globals().get('MODEL_PACKAGE_GATED_CANDIDATES', (0.003, 0.005, 0.010)))
        if not any(abs(x - _selected_gmax) < 1e-12 for x in _candidates):
            _candidates.append(_selected_gmax)
        _rows = []
        for _gmax in sorted(set(round(x, 12) for x in _candidates)):
            _gate = float(_gmax) / (1.0 + (_diff / _scale) ** 2)
            _out = _mp_merged[['id']].copy()
            _out['tvt'] = (1.0 - _gate) * _base_v + _gate * _pkg_v
            _move = _mp_np.abs(_out['tvt'].to_numpy(dtype=float) - _base_v)
            _name = f'submission_model_package_gated_{int(round(_gmax * 1000)):03d}.csv'
            _out.to_csv(_mp_work / _name, index=False)
            _rows.append({
                'file': _name,
                'gated_max_weight': float(_gmax),
                'scale': float(_scale),
                'selected_for_submission_csv': bool(abs(_gmax - _selected_gmax) < 1e-12 and not _disabled),
                'gate_mean': float(_mp_np.mean(_gate)),
                'gate_p95': float(_mp_np.quantile(_gate, 0.95)),
                'gate_max': float(_mp_np.max(_gate)),
                'mean_abs_model_package_diff': float(_mp_np.mean(_diff)),
                'p95_abs_model_package_diff': _p95,
                'max_abs_model_package_diff': float(_mp_np.max(_diff)),
                'mean_abs_final_move': float(_mp_np.mean(_move)),
                'p95_abs_final_move': float(_mp_np.quantile(_move, 0.95)),
                'max_abs_final_move': float(_mp_np.max(_move)),
                'disabled_by_diff_guard': bool(_disabled),
            })
        _report = _mp_pd.DataFrame(_rows)
        _report.to_csv(_mp_work / 'model_package_correction_report.csv', index=False)
        _mp_pd.Series({
            **_mp_info,
            'model_package_available': True,
            'active_signal': _active_signal,
            'mean_abs_model_package_diff': float(_mp_np.mean(_diff)),
            'p95_abs_model_package_diff': _p95,
            'model_package_diff_rmse': _diff_rmse,
            'disabled_by_diff_guard': bool(_disabled),
        }).to_csv(_mp_work / 'model_package_correction_summary.csv')
        if _disabled:
            _mp_base.to_csv(_mp_final_output, index=False)
            globals()['FINAL_BASE_SOURCE_LABEL'] = 'model_package_disabled'
            globals()['FINAL_MODEL_PACKAGE_AUTO_DISABLED_REASON'] = f'model package p95 diff {_p95:.3f} > {float(_disable_limit):.3f}'
            print(globals()['FINAL_MODEL_PACKAGE_AUTO_DISABLED_REASON'])
        else:
            _final_name = f'submission_model_package_gated_{int(round(_selected_gmax * 1000)):03d}.csv'
            _final = _mp_pd.read_csv(_mp_work / _final_name)
            _final.to_csv(_mp_final_output, index=False)
            globals()['FINAL_BASE_SOURCE_LABEL'] = f'model_package_gated_{int(round(_selected_gmax * 1000)):03d}'
            print('wrote final submission.csv from', _final_name, _final.shape, flush=True)
        globals()['FINAL_SELECTED_BASE_SOURCE'] = _mp_final_output
        globals()['FINAL_MODEL_PACKAGE_SOURCE_LABEL'] = globals()['FINAL_BASE_SOURCE_LABEL']
        globals()['FINAL_MODEL_PACKAGE_AVAILABLE'] = True
        display(_report)

# %%
# Guarded PF seed-branch midpoint hedge computed entirely in this notebook.
import numpy as _bh_np
import pandas as _bh_pd
from pathlib import Path as _BhPath

_BH_STRENGTH = 0.60
_BH_MIN_MASS = 0.25
_BH_SEP_LOW = 4.00
_BH_SEP_HIGH = 40.00
_BH_CAP = 2.00
_BH_SKIP_EXISTING = False
_BH_WORK = _BhPath('/kaggle/working') if _BhPath('/kaggle/working').exists() else _BhPath('.')
_BH_SUB = _BH_WORK / 'submission.csv'

_bh_skip = set()
if _BH_SKIP_EXISTING:
    for _name in ('gold_contact_override_report.csv', 'guarded_overlap_override_report.csv'):
        _path = _BH_WORK / _name
        if not _path.exists():
            continue
        try:
            _report = _bh_pd.read_csv(_path)
            if {'well', 'status'}.issubset(_report.columns):
                _active = _report['status'].astype(str).str.lower().isin(['override', 'applied'])
                _bh_skip.update(_report.loc[_active, 'well'].astype(str))
        except Exception:
            pass

    _gold_moves = _BH_WORK / 'gold_prefix_moves_balanced.csv'
    if _gold_moves.exists():
        try:
            _report = _bh_pd.read_csv(_gold_moves)
            _alpha = _bh_pd.to_numeric(_report.get('alpha', 0.0), errors='coerce').fillna(0.0)
            if 'well' in _report.columns:
                _bh_skip.update(_report.loc[_alpha > 0.0, 'well'].astype(str))
        except Exception:
            pass

_bh_rows = []
if _BH_SUB.exists():
    _sub = _bh_pd.read_csv(_BH_SUB)
    if list(_sub.columns) != ['id', 'tvt']:
        raise RuntimeError('branch hedge expected id,tvt submission schema')
    _sub.to_csv(_BH_WORK / 'submission_before_branch_hedge.csv', index=False)
    _well = _sub['id'].astype(str).str.split('_', n=1).str[0]
    _row = _bh_pd.to_numeric(_sub['id'].astype(str).str.rsplit('_', n=1).str[-1], errors='coerce')
    _tvt = _bh_pd.to_numeric(_sub['tvt'], errors='coerce').to_numpy(dtype=float)
    for _wid, _stats in sorted((globals().get('PF_SEED_BRANCH_STATS', {}) or {}).items()):
        _reason = 'not_qualified'
        _shift = 0.0
        _moved = 0
        try:
            _low = float(_stats['center_low'])
            _high = float(_stats['center_high'])
            _mass_low = float(_stats['mass_low'])
            _mass_high = float(_stats['mass_high'])
            _weighted = float(_stats['weighted_center'])
            _sep = abs(_high - _low)
            _minor = min(_mass_low, _mass_high)
            if str(_wid) in _bh_skip:
                _reason = 'skip_existing_route'
            elif _minor < _BH_MIN_MASS:
                _reason = 'skip_minor_mass'
            elif not (_BH_SEP_LOW <= _sep <= _BH_SEP_HIGH):
                _reason = 'skip_separation'
            else:
                _target = 0.5 * (_low + _high)
                _shift = float(_bh_np.clip(_BH_STRENGTH * (_target - _weighted), -_BH_CAP, _BH_CAP))
                _eval_rows = set(int(x) for x in _stats.get('eval_rows', []))
                _mask = (_well == str(_wid)).to_numpy()
                if _eval_rows:
                    _mask &= _row.isin(_eval_rows).to_numpy()
                if abs(_shift) >= 0.01 and bool(_mask.any()):
                    _tvt[_mask] += _shift
                    _moved = int(_mask.sum())
                    _reason = 'applied'
                else:
                    _reason = 'skip_zero_or_missing_rows'
            _bh_rows.append(dict(
                well=str(_wid), reason=_reason, center_low=_low, center_high=_high,
                mass_low=_mass_low, mass_high=_mass_high, separation=_sep,
                weighted_center=_weighted, shift=_shift, moved_rows=_moved,
            ))
        except Exception as _exc:
            _bh_rows.append(dict(well=str(_wid), reason='error', error=repr(_exc)))
    if not _bh_np.isfinite(_tvt).all():
        raise RuntimeError('branch hedge produced non-finite predictions')
    _sub['tvt'] = _tvt
    _sub.to_csv(_BH_SUB, index=False)

_bh_pd.DataFrame(_bh_rows).to_csv(_BH_WORK / 'pf_seed_branch_hedge_report.csv', index=False)
print('PF seed-branch hedge:', {r: sum(x.get('reason') == r for x in _bh_rows) for r in set(x.get('reason') for x in _bh_rows)})

# %%
# Final submission audit: verify the final file after all enabled correction layers.
import hashlib as _audit_hashlib
import json as _audit_json
import pandas as _audit_pd
import numpy as _audit_np
from pathlib import Path as _AuditPath

_AUDIT_WORK = _AuditPath('/kaggle/working') if _AuditPath('/kaggle/working').exists() else _AuditPath('.')
_AUDIT_SUBMISSION = _AUDIT_WORK / 'submission.csv'

# CFG may be redefined by the learned trajectory section (using .DATA instead of .dataset_path)
_AUDIT_DATA = getattr(CFG, 'DATA', getattr(CFG, 'dataset_path', _AuditPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')))
_AUDIT_SAMPLE = _AUDIT_DATA / 'sample_submission.csv'


def _sha256_file(path):
    h = _audit_hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _build_submission_audit(sub_path, sample_path):
    sub = _audit_pd.read_csv(sub_path)
    sample = _audit_pd.read_csv(sample_path)
    if list(sub.columns) != ['id', 'tvt']:
        raise RuntimeError(f'Unexpected submission columns: {list(sub.columns)}')
    if len(sub) != len(sample):
        raise RuntimeError(f'Unexpected row count: submission={len(sub)} sample={len(sample)}')
    if not sub['id'].astype(str).equals(sample['id'].astype(str)):
        raise RuntimeError('Submission id order does not match sample_submission.csv')
    tvt = sub['tvt'].to_numpy(dtype=float)
    if not _audit_np.isfinite(tvt).all():
        raise RuntimeError('Submission contains non-finite tvt values')
    return {
        'rows': int(len(sub)),
        'columns': list(sub.columns),
        'id_order_matches_sample': True,
        'tvt_min': float(_audit_np.min(tvt)),
        'tvt_max': float(_audit_np.max(tvt)),
        'tvt_mean': float(_audit_np.mean(tvt)),
        'tvt_std': float(_audit_np.std(tvt)),
        'sha256_submission_csv': _sha256_file(sub_path),
    }


_audit = _build_submission_audit(_AUDIT_SUBMISSION, _AUDIT_SAMPLE)
with open(_AUDIT_WORK / 'submission_audit.json', 'w', encoding='utf-8') as f:
    _audit_json.dump(_audit, f, indent=2, sort_keys=True)

# Keep named copies for manual inspection; Kaggle still submits submission.csv.
_latest_valid = _audit_pd.read_csv(_AUDIT_SUBMISSION)
_latest_valid.to_csv(_AUDIT_WORK / 'submission_audit_copy.csv', index=False)
_latest_valid.to_csv(_AUDIT_WORK / 'latest_valid_submission.csv', index=False)
with open(_AUDIT_WORK / 'latest_valid_submission.json', 'w', encoding='utf-8') as f:
    _audit_json.dump(_audit, f, indent=2, sort_keys=True)
print('Submission audit:', _audit, flush=True)

# %% [markdown]
# ## 8. Embedded hidden-safe exp413 runtime

# %%
def generate_dynamic_exp413_prediction(
    shared_deterministic_frame=None,
    reuse_tracker=None,
    shared_likpf_bank=None,
):
    import gc
    import gzip
    import hashlib
    import importlib.util
    import json
    import shutil
    import sys
    import time
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    import yaml
    from IPython.display import display
    from joblib import Parallel as JoblibParallel
    from joblib import delayed as joblib_delayed
    from numba import set_num_threads as numba_set_num_threads
    from exp413_runtime.settings import EXPERIMENT_NAME, ExperimentPaths, load_config

    from src.candidate_selector_pipeline import (
        ShapeState,
        build_candidate_long_features,
        build_compact_meta,
        build_raw_context,
        candidate_contract_sha,
        candidate_ids,
        current_test_bundle_from_wide,
        fill_current_test_anchor,
        load_feature_schema,
        read_yaml,
        sha256_file,
        validate_current_test_native_confidence,
        validate_inference_feature_missingness,
        write_json,
    )
    from src.signed_residual_meta import (
        build_signed_compact_meta,
        signed_compact_feature_names,
    )

    STARTED_AT = time.time()
    PACKAGE_DIR = Path("exp413_runtime")
    if not (PACKAGE_DIR / "config.yaml").exists():
        raise FileNotFoundError("embedded exp413 runtime config is missing")
    paths = ExperimentPaths()
    paths.require_kaggle_runtime()
    paths.ensure_output_dirs()
    config = load_config()
    # exp510 execution authorization activates the already-approved exp413 v4
    # inference path in memory. The vendored parent config remains immutable.
    config["inference"]["run_enabled"] = True
    parent_config = yaml.safe_load(
        (PACKAGE_DIR / "inputs/parent_exp335_config.yaml").read_text()
    )
    output_dir = paths.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    KAGGLE_INPUT_ROOT = Path("/kaggle/input")


    def get_nested(mapping: dict[str, Any], dotted: str, default: Any = None) -> Any:
        value: Any = mapping
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value


    def import_file(name: str, candidates: list[Path]) -> Any:
        path = next((item for item in candidates if item.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Cannot resolve {name}: {candidates}")
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


    def sha256_gzip_decompressed(path: Path) -> str:
        digest = hashlib.sha256()
        with gzip.open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


    def source_record(path: Path) -> dict[str, Any]:
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


    def resolve_unique_source(filename: str, path_token: str) -> Path:
        matches = [
            path
            for path in sorted(KAGGLE_INPUT_ROOT.rglob(filename))
            if path_token in str(path)
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one {filename} under source token {path_token}, got {matches}"
            )
        return matches[0]


    def copy_trusted_source(source: Path, target_dir: Path, module_name: str) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{module_name}.py"
        shutil.copy2(source, target)
        return target


    def parse_identity(frame: pd.DataFrame) -> pd.DataFrame:
        ids = frame["id"].astype(str)
        split = ids.str.rsplit("_", n=1, expand=True)
        if split.shape[1] != 2:
            raise ValueError("candidate id must use <well>_<row_idx>")
        return pd.DataFrame(
            {
                "id": ids,
                "well": split[0].astype(str),
                "well_row_idx": pd.to_numeric(split[1], errors="raise").astype(np.int32),
            }
        )


    def finalize_primitive_confidence(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        excluded = {"id", "well", "well_row_idx", "candidate_tvt", "confidence_valid"}
        native_fields = [column for column in result if column not in excluded]
        available: list[np.ndarray] = []
        for field in native_fields:
            values = pd.to_numeric(result[field], errors="coerce").to_numpy(np.float32)
            result[field] = values
            available.append(np.isfinite(values))
        candidate_finite = np.isfinite(result["candidate_tvt"].to_numpy(np.float32))
        result["confidence_valid"] = (
            candidate_finite & np.logical_or.reduce(available)
            if available
            else np.zeros(len(result), dtype=bool)
        )
        return result


    def standard_primitive(
        frame: pd.DataFrame,
        value: Any,
        *,
        confidence: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        result = parse_identity(frame)
        result["candidate_tvt"] = np.asarray(value, dtype=np.float32)
        for field, field_value in (confidence or {}).items():
            result[field] = np.asarray(field_value, dtype=np.float32)
        return finalize_primitive_confidence(result)


    hmm_parallel_reports: list[dict[str, Any]] = []


    def generate_hmm_primitive(
        *,
        list_well_ids: Callable[[str | Path], list[str]],
        load_well: Callable[[str, str | Path], tuple[pd.DataFrame, pd.DataFrame]],
        run_hmm2: Callable[..., dict[str, Any]],
        test_dir: Path,
        hmm_params: dict[str, Any],
        self_gr: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        well_ids = list_well_ids(test_dir)
        if not well_ids:
            raise ValueError("HMM raw-test generation found no wells")
        effective_n_jobs = min(EXP413_WELL_N_JOBS, len(well_ids))
        parallel_started = time.time()

        def run_one_well(well: str) -> pd.DataFrame | None:
            # The HMM core already releases the GIL. One Numba worker per outer
            # well avoids nested 4x4 oversubscription while preserving arithmetic.
            numba_set_num_threads(1)
            horizontal, typewell = load_well(well, test_dir)
            known = pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy()
            if not known.any():
                raise ValueError(f"raw test well {well} has no finite TVT_input prefix")
            expected_eval = np.flatnonzero(~known).astype(np.int64)
            if len(expected_eval) == 0:
                return None
            kwargs = dict(hmm_params)
            if self_gr is not None:
                kwargs.update(
                    {
                        "self_gr_config": dict(self_gr["surface"]),
                        "self_gr_alpha": float(self_gr["alpha"]),
                        "self_gr_clip": float(self_gr["clip"]),
                        "self_gr_mode": str(self_gr["mode"]),
                    }
                )
            inference = run_hmm2(horizontal, typewell, **kwargs)
            actual_eval = np.asarray(inference["ev_index"], dtype=np.int64)
            if not np.array_equal(actual_eval, expected_eval):
                raise ValueError(f"HMM eval identity mismatch for well {well}")
            row = pd.DataFrame(
                {
                    "id": [f"{well}_{int(index)}" for index in actual_eval],
                    "well": str(well),
                    "well_row_idx": actual_eval.astype(np.int32),
                    "candidate_tvt": np.asarray(inference["mean_eval"], dtype=np.float32),
                    "sigma_tvt": np.asarray(inference["std_eval"], dtype=np.float32),
                    "source_loglik": np.full(
                        len(actual_eval), np.float32(inference["loglik"]), dtype=np.float32
                    ),
                    "loglik_per_row": np.full(
                        len(actual_eval),
                        np.float32(float(inference["loglik"]) / len(actual_eval)),
                        dtype=np.float32,
                    ),
                }
            )
            if self_gr is not None:
                row["candidate_finite_source"] = np.isfinite(
                    np.asarray(inference["mean_eval"], dtype=np.float32)
                ).astype(np.float32)
                row["selfgr_quality"] = np.asarray(inference["self_gr_quality"], dtype=np.float32)
                row["selfgr_peak_tvt"] = np.asarray(
                    inference["self_gr_peak_tvt"], dtype=np.float32
                )
                row["score_margin"] = np.asarray(
                    inference["self_gr_peak_gap"], dtype=np.float32
                )
                row["selfgr_typewell_agreement"] = np.asarray(
                    inference["self_gr_typewell_agreement"], dtype=np.float32
                )
                row["selfgr_valid"] = np.asarray(inference["self_gr_valid"], dtype=np.float32)
            return row

        well_results = JoblibParallel(
            n_jobs=effective_n_jobs,
            backend="threading",
        )(
            joblib_delayed(run_one_well)(well)
            for well in well_ids
        )
        rows = [row for row in well_results if row is not None]
        if not rows:
            raise ValueError("HMM raw-test generation produced no rows")
        result = finalize_primitive_confidence(pd.concat(rows, ignore_index=True))
        if result.duplicated("id").any() or not np.isfinite(result["candidate_tvt"]).all():
            raise ValueError("HMM raw-test output violates duplicate/finite contract")
        hmm_parallel_reports.append(
            {
                "requested_n_jobs": EXP413_WELL_N_JOBS,
                "effective_n_jobs": effective_n_jobs,
                "test_wells": len(well_ids),
                "backend": "threading",
                "numba_threads_per_well": 1,
                "elapsed_seconds": round(time.time() - parallel_started, 3),
                "self_gr": self_gr is not None,
            }
        )
        return result


    def generate_k16_primitive(
        module: Any,
        *,
        train_dir: Path,
        test_dir: Path,
        source_config: dict[str, Any],
        frame_content_sha256: Callable[[pd.DataFrame], str],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        return run_exp413_k16_haswell_subprocess(
            module,
            train_dir=train_dir,
            test_dir=test_dir,
            source_config=source_config,
            finalize_primitive_confidence=finalize_primitive_confidence,
            frame_content_sha256=frame_content_sha256,
        )
        params = module.params_from_config(source_config)  # pragma: no cover - retained source body
        max_train = get_nested(source_config, "inference.max_train_wells")
        max_test = get_nested(source_config, "inference.max_test_wells")
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
            raise FileNotFoundError("exp226 K16 requires non-empty train and test wells")
        fields = module.build_fields(train_wells, params)
        kappa = module.fit_kappa(train_wells, fields, params)
        print("exp226 kappa:", np.round(kappa, 3), flush=True)
        rows: list[pd.DataFrame] = []
        well_summaries: list[dict[str, Any]] = []
        for order, well in enumerate(test_wells, start=1):
            inference = module.predict_well(well, fields, kappa, params)
            row_idx = np.arange(well.s + 1, well.s + well.n + 1, dtype=np.int32)
            if len(row_idx) != len(inference.pred) or len(inference.pred) != len(inference.delta):
                raise ValueError(f"exp226 K16 row contract mismatch for well={well.wid}")
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
        result = finalize_primitive_confidence(pd.concat(rows, ignore_index=True))
        if result.duplicated("id").any() or not np.isfinite(
            result[["candidate_tvt", "geometry_gr_delta"]].to_numpy()
        ).all():
            raise ValueError("exp226 K16 output violates duplicate/finite confidence contract")
        return result, {
            "train_wells": len(train_wells),
            "test_wells": len(test_wells),
            "rows": len(result),
            "kappa": [float(value) for value in np.asarray(kappa).ravel()],
            "well_summaries": well_summaries,
            "prediction_and_confidence_content_sha256": frame_content_sha256(result),
        }


    # [embedded markdown boundary]
    # ## 2. User authorization and saved-model contracts
    #
    # Stage D primary gate PASSを必要条件とし、2026-07-29のユーザー指示を
    # 保存modelによるCPU推論、予測監査生成物、Kaggle Notebook outputとしての
    # submission.csv生成までの承認として扱う。外部competition submitへは拡張しない。

    # [embedded code boundary]
    inference_cfg = dict(config["inference"])
    if not bool(config["authorization"]["inference_implementation_approved"]):
        raise RuntimeError("exp413 inference implementation is not approved")
    if not bool(config["authorization"]["inference_run_approved"]):
        raise RuntimeError("exp413 inference run is not approved")
    if not bool(inference_cfg.get("run_enabled")):
        raise RuntimeError("exp413 inference run flag is disabled")
    if (
        inference_cfg.get("status")
        != "user_authorized_2026_07_29_kaggle_submission_output"
    ):
        raise RuntimeError("exp413 CPU inference does not have the fixed user authorization")
    if not bool(inference_cfg.get("stage_d_primary_gate_passed")):
        raise RuntimeError("Stage D primary gate PASS is required for exp413 inference")
    if str(inference_cfg.get("runtime")) != "kaggle_cpu":
        raise RuntimeError("exp413 inference must use Kaggle CPU")
    if not bool(config["authorization"]["submission_file_generation_approved"]):
        raise RuntimeError("submission.csv generation is not approved")
    if not bool(inference_cfg.get("generate_submission_file")):
        raise RuntimeError("submission.csv generation flag is disabled")
    if bool(inference_cfg.get("submit_to_kaggle")):
        raise RuntimeError("the inference notebook must not call the Kaggle submit API")
    if bool(inference_cfg.get("competition_submit_authorized")):
        raise RuntimeError("external Kaggle competition submit must remain unauthorized")
    if int(inference_cfg.get("booster_training_count", -1)) != 0:
        raise RuntimeError("inference must train zero boosters")

    candidate_contract_path = PACKAGE_DIR / "inputs/exp264_candidate_contract.yaml"
    candidate_contract = read_yaml(candidate_contract_path)
    names = candidate_ids(candidate_contract)
    if len(names) != 12:
        raise ValueError("exp413 inference requires exactly 12 candidates")
    sample = pd.read_csv(paths.sample_submission_path, dtype={"id": str})
    if paths.submission_path.exists():
        raise RuntimeError(
            f"submission output must not pre-exist before inference: "
            f"{paths.submission_path}"
        )

    stage_c_manifest_path = resolve_unique_source(
        "nested_selector_model_manifest.json",
        "exp413-scale5-likpf-selector-train",
    )
    stage_c_root = stage_c_manifest_path.parent
    selector_schema_path = resolve_unique_source(
        "feature_schema.json",
        "exp413-scale5-likpf-replacement-preflight",
    )
    compact_schema_path = resolve_unique_source(
        "compact_meta_schema.json",
        "exp413-scale5-likpf-replacement-preflight",
    )
    selector_catalog_path = PACKAGE_DIR / str(inference_cfg["selector_feature_catalog"])
    expected_stage_c_files = {
        stage_c_manifest_path: inference_cfg["nested_selector_model_manifest_sha256"],
        selector_schema_path: inference_cfg["selector_feature_schema_file_sha256"],
        compact_schema_path: inference_cfg["parent_compact_schema_file_sha256"],
    }
    for artifact_path, expected_sha in expected_stage_c_files.items():
        if sha256_file(artifact_path) != str(expected_sha):
            raise ValueError(f"Stage C contract SHA mismatch: {artifact_path.name}")
    if sha256_file(selector_catalog_path) != str(
        inference_cfg["selector_feature_catalog_sha256"]
    ):
        raise ValueError("Stage A selector feature catalog SHA mismatch")

    stage_c_manifest = json.loads(stage_c_manifest_path.read_text())
    selector_schema = load_feature_schema(selector_schema_path)
    selector_features = [str(item) for item in selector_schema["features"]]
    compact_schema = json.loads(compact_schema_path.read_text())
    parent_compact_features = [str(item) for item in compact_schema["features"]]
    if stage_c_manifest.get("candidate_order") != names:
        raise ValueError("Stage C candidate order differs from exp264 contract")
    if int(stage_c_manifest.get("model_count", -1)) != int(
        inference_cfg["parent_selector_model_count"]
    ):
        raise ValueError("Stage C manifest must contain 40 selector models")
    if stage_c_manifest.get("feature_schema_sha256") != selector_schema.get(
        "feature_schema_sha256"
    ):
        raise ValueError("Stage C selector schema logical SHA mismatch")
    if compact_schema.get("compact_meta_schema_sha256") != str(
        inference_cfg["parent_compact_schema_logical_sha256"]
    ):
        raise ValueError("Stage C compact schema logical SHA mismatch")
    if len(selector_features) != int(inference_cfg["expected_selector_feature_count"]):
        raise ValueError("Stage C selector feature count mismatch")
    if len(parent_compact_features) != int(
        inference_cfg["expected_parent_compact_feature_count"]
    ):
        raise ValueError("Stage C compact feature count mismatch")

    selector_catalog = pd.read_csv(selector_catalog_path)
    selected_mask = selector_catalog["selected"].astype(str).str.lower().eq("true")
    selected_catalog = selector_catalog.loc[selected_mask].copy()
    catalog_features = selected_catalog["feature"].astype(str).tolist()
    if catalog_features != selector_features or selected_catalog["feature"].duplicated().any():
        raise ValueError("Stage A selected feature catalog differs from Stage C schema")
    selected_catalog["missing_rate"] = pd.to_numeric(
        selected_catalog["missing_rate"], errors="raise"
    )
    training_missing_rate_by_feature = dict(
        zip(
            catalog_features,
            selected_catalog["missing_rate"].astype(float),
            strict=True,
        )
    )
    training_sparse_feature_count = int(
        (selected_catalog["missing_rate"] > 0.0).sum()
    )
    if training_sparse_feature_count != int(
        inference_cfg["selector_training_sparse_feature_count"]
    ):
        raise ValueError("Stage A selector sparse-feature count mismatch")

    selector_models: dict[int, dict[str, list[lgb.Booster]]] = {
        outer: {"pred_abs_error": [], "p_within10": []} for outer in range(5)
    }
    selector_model_audit: list[dict[str, Any]] = []
    for item in stage_c_manifest["models"]:
        outer = int(item["downstream_outer_fold"])
        objective = str(item["objective"])
        model_path = stage_c_root / str(item["path"])
        if sha256_file(model_path) != str(item["sha256"]):
            raise ValueError(f"Stage C selector model SHA mismatch: {model_path.name}")
        booster = lgb.Booster(model_file=str(model_path))
        if list(booster.feature_name()) != selector_features:
            raise ValueError(f"Stage C selector feature order mismatch: {model_path.name}")
        selector_models[outer][objective].append(booster)
        selector_model_audit.append(
            {
                "outer_fold": outer,
                "inner_fold": int(item["inner_fold"]),
                "objective": objective,
                "file": model_path.name,
                "sha256": str(item["sha256"]),
                "best_iteration": int(item["best_iteration"]),
            }
        )
    for outer, by_objective in selector_models.items():
        for objective, models in by_objective.items():
            if len(models) != int(
                inference_cfg["parent_selector_models_per_outer_objective"]
            ):
                raise ValueError(f"Stage C model coverage mismatch: outer={outer} {objective}")

    signed_manifest_path = resolve_unique_source(
        "signed_selector_model_manifest.json",
        "exp413-scale5-likpf-signed-train",
    )
    if sha256_file(signed_manifest_path) != str(
        inference_cfg["signed_selector_model_manifest_sha256"]
    ):
        raise ValueError("Stage S signed-selector model manifest SHA mismatch")
    signed_schema_path = resolve_unique_source(
        "signed_compact_schema.json",
        "exp413-scale5-likpf-signed-train",
    )
    if sha256_file(signed_schema_path) != str(
        inference_cfg["signed_compact_schema_file_sha256"]
    ):
        raise ValueError("Stage S signed compact schema file SHA mismatch")
    signed_manifest = json.loads(signed_manifest_path.read_text())
    signed_schema = json.loads(signed_schema_path.read_text())
    signed_compact_features = [str(item) for item in signed_schema["features"]]
    if signed_manifest.get("candidate_order") != names:
        raise ValueError("Stage S candidate order differs from exp413 contract")
    if int(signed_manifest.get("model_count", -1)) != int(
        inference_cfg["signed_selector_model_count"]
    ):
        raise ValueError("Stage S manifest must contain exactly 20 signed selectors")
    if signed_manifest.get("feature_schema_sha256") != selector_schema.get(
        "feature_schema_sha256"
    ):
        raise ValueError("Stage S selector feature schema differs from corrected Stage A")
    if signed_schema.get("signed_compact_schema_sha256") != str(
        inference_cfg["signed_compact_schema_logical_sha256"]
    ):
        raise ValueError("Stage S signed compact logical schema SHA mismatch")
    if signed_compact_features != signed_compact_feature_names(candidate_contract):
        raise ValueError("Stage S signed compact schema differs from source contract")
    if len(signed_compact_features) != int(
        inference_cfg["expected_signed_compact_feature_count"]
    ):
        raise ValueError("Stage S signed compact feature count mismatch")

    signed_selector_models: dict[int, list[lgb.Booster]] = {
        outer: [] for outer in range(5)
    }
    signed_selector_model_audit: list[dict[str, Any]] = []
    for item in signed_manifest["models"]:
        outer = int(item["downstream_outer_fold"])
        model_path = signed_manifest_path.parent / str(item["path"])
        if sha256_file(model_path) != str(item["sha256"]):
            raise ValueError(f"Stage S signed-selector model SHA mismatch: {model_path.name}")
        booster = lgb.Booster(model_file=str(model_path))
        if list(booster.feature_name()) != selector_features:
            raise ValueError(
                f"Stage S signed-selector feature order mismatch: {model_path.name}"
            )
        signed_selector_models[outer].append(booster)
        signed_selector_model_audit.append(
            {
                "outer_fold": outer,
                "inner_fold": int(item["inner_fold"]),
                "objective": str(item["objective"]),
                "file": model_path.name,
                "sha256": str(item["sha256"]),
                "best_iteration": int(item["best_iteration"]),
            }
        )
    for outer, models in signed_selector_models.items():
        if len(models) != int(inference_cfg["signed_selector_models_per_outer"]):
            raise ValueError(f"Stage S signed model coverage mismatch: outer={outer}")

    stage_d_manifest_path = resolve_unique_source(
        "stage_d_model_manifest.json",
        "exp413-scale5-likpf-downstream-train",
    )
    if sha256_file(stage_d_manifest_path) != str(inference_cfg["tvt_model_manifest_sha256"]):
        raise ValueError("Stage D model manifest SHA mismatch")
    stage_d_manifest = json.loads(stage_d_manifest_path.read_text())
    stage_d_rows = [
        dict(item)
        for item in stage_d_manifest["models"]
        if str(item["variant"]) == str(inference_cfg["tvt_model_variant"])
    ]
    if len(stage_d_rows) != int(inference_cfg["tvt_model_count"]):
        raise ValueError("Stage D inference requires exactly 15 replacement models")
    resolved_tvt_models: list[tuple[dict[str, Any], Path]] = []
    for item in stage_d_rows:
        model_path = stage_d_manifest_path.parent / str(item["path"])
        if sha256_file(model_path) != str(item["sha256"]):
            raise ValueError(f"Stage D TVT model SHA mismatch: {model_path.name}")
        resolved_tvt_models.append((item, model_path))

    schema_probe = lgb.Booster(model_file=str(resolved_tvt_models[0][1]))
    final_feature_columns = list(schema_probe.feature_name())
    del schema_probe
    base_feature_count = int(inference_cfg["expected_base_feature_count"])
    base_feature_columns = final_feature_columns[:base_feature_count]
    model_compact_features = final_feature_columns[base_feature_count:]
    source_base_catalog_path = PACKAGE_DIR / str(
        inference_cfg["source_base_feature_catalog"]
    )
    if sha256_file(source_base_catalog_path) != str(
        inference_cfg["source_base_feature_catalog_sha256"]
    ):
        raise ValueError("exp218 source 380 feature catalog SHA mismatch")
    source_base_catalog = pd.read_csv(source_base_catalog_path)
    source_base_columns = source_base_catalog["feature"].astype(str).tolist()
    if len(source_base_columns) != int(inference_cfg["expected_source_base_feature_count"]):
        raise ValueError("exp218 source feature count mismatch")
    if len(source_base_columns) != len(set(source_base_columns)):
        raise ValueError("exp218 source feature catalog contains duplicates")
    base_allowlist_path = PACKAGE_DIR / str(inference_cfg["base_feature_allowlist"])
    if sha256_file(base_allowlist_path) != str(
        inference_cfg["base_feature_allowlist_sha256"]
    ):
        raise ValueError("clean 273 base feature allowlist SHA mismatch")
    base_allowlist = pd.read_csv(base_allowlist_path)["feature"].astype(str).tolist()
    if len(base_allowlist) != base_feature_count or len(base_allowlist) != len(
        set(base_allowlist)
    ):
        raise ValueError("clean 273 base feature allowlist count/uniqueness mismatch")
    if base_feature_columns != base_allowlist:
        raise ValueError("Stage D model base feature order differs from clean 273 allowlist")
    safe_catalog = source_base_catalog.loc[
        source_base_catalog["fold_safe"].astype(str).str.lower().eq("true")
        & source_base_catalog["hidden_safe"].astype(str).str.lower().eq("true"),
        "feature",
    ].astype(str).tolist()
    if safe_catalog != base_allowlist:
        raise ValueError("clean 273 allowlist differs from source feature safety catalog")
    expected_model_compact = parent_compact_features + signed_compact_features
    if model_compact_features != expected_model_compact:
        raise ValueError(
            "Stage D model compact feature order differs from saved74 + signed23 schema"
        )
    if len(final_feature_columns) != int(inference_cfg["expected_final_feature_count"]):
        raise ValueError("Stage D final feature count mismatch")

    display(
        {
            "experiment": EXPERIMENT_NAME,
            "route": config["experiment"]["route"],
            "authorization": inference_cfg["authorization_scope"],
            "stage_d_primary_gate_passed": True,
            "retained_tail_readout": inference_cfg["retained_tail_readout"],
            "candidate_count": len(names),
            "parent_selector_models": len(selector_model_audit),
            "signed_selector_models": len(signed_selector_model_audit),
            "tvt_models": len(resolved_tvt_models),
            "base_features": len(base_feature_columns),
            "parent_compact_features": len(parent_compact_features),
            "signed_compact_features": len(signed_compact_features),
            "final_features": len(final_feature_columns),
            "booster_training_count": 0,
            "runtime": "kaggle_cpu",
            "competition_submit_authorized": False,
        }
    )

    # [embedded markdown boundary]
    # ## 3. Exp263 hidden-safe 12-candidate regeneration and scale5 replacement
    #
    # 保存済みpublic-test row artifactは使わない。exp263が固定したsource file名・Kaggle source token・
    # parameterを用い、PF/Beam/likPF、exact/self-GR HMM、K16をraw testから再生成する。
    # likPFは同じ128 seed trajectoryから既に生成されるtemperature-5列をsemantic
    # `likpf_mean`へ移し、旧arithmetic meanは差分監査後にcandidate/model入力から除外する。

    # [embedded code boundary]
    exp263_source_dir = PACKAGE_DIR / "inputs/exp263_source"
    sys.path.insert(0, str(exp263_source_dir))
    from candidate_cache_builder import (  # noqa: E402
        assemble_stage1_current_test_parity,
        attach_stage1_current_test_confidence,
    )
    from candidate_cache_contract import (  # noqa: E402
        PAIR_SHORTLIST,
        RAWTEST_CORE_CANDIDATE_IDS,
        STAGE1_NATIVE_CONFIDENCE_FIELDS,
        validate_contract,
    )
    from candidate_cache_loader import frame_content_sha256  # noqa: E402

    exp263_config = yaml.safe_load((exp263_source_dir / "config.yaml").read_text())
    stage1 = dict(exp263_config["stage1"])
    generation = dict(stage1["raw_test_generation"])
    validate_contract()
    rawtest_pairs = [pair for pair in PAIR_SHORTLIST if pair.tier == "raw-test"]
    if len(RAWTEST_CORE_CANDIDATE_IDS) != 6 or len(rawtest_pairs) != 5:
        raise ValueError("exp263 Stage 1 deployability tier count mismatch")
    expected_confidence_contract = {
        candidate_id: ["confidence_valid", *fields]
        for candidate_id, fields in STAGE1_NATIVE_CONFIDENCE_FIELDS.items()
    }
    if stage1["confidence_output"]["required_fields_by_primitive"] != (
        expected_confidence_contract
    ):
        raise ValueError("exp263 Stage 1 native-confidence contract mismatch")

    source_work = Path("/tmp/exp413_trusted_upstream_sources")
    if source_work.exists():
        shutil.rmtree(source_work)
    source_specs = {
        "exp263_public_replay_source": generation["pf_replay"],
        "exp263_exact_hmm_source": generation["exact_hmm"],
        "exp263_selfgr_hmm_source": generation["selfgr_hmm_a070"],
        "exp263_k16_source": generation["exp226_k16"],
    }
    resolved_sources: dict[str, Path] = {}
    for module_name, source_spec in source_specs.items():
        source = resolve_unique_source(
            str(source_spec["source_filename"]), str(source_spec["source_path_token"])
        )
        resolved_sources[module_name] = source
        copy_trusted_source(source, source_work, module_name)
    sys.path.insert(0, str(source_work))
    observed_replay_source_sha = sha256_file(
        resolved_sources['exp263_public_replay_source']
    )
    if observed_replay_source_sha != EXP073_REPLAY_SOURCE_SHA256:
        raise ValueError(
            'shared likelihood-PF replay source SHA mismatch: '
            f'{observed_replay_source_sha} != {EXP073_REPLAY_SOURCE_SHA256}'
        )

    import exp263_k16_source as k16_module  # noqa: E402
    from exp263_exact_hmm_source import list_well_ids as exact_list_well_ids  # noqa: E402
    from exp263_exact_hmm_source import load_well as exact_load_well  # noqa: E402
    from exp263_exact_hmm_source import run_hmm2 as exact_run_hmm2  # noqa: E402
    import exp263_public_replay_source as replay_source  # noqa: E402
    from exp263_public_replay_source import (  # noqa: E402
        configure_public_runtime,
        list_test_wells as replay_list_test_wells,
        stable_seed as replay_stable_seed,
    )
    from exp263_selfgr_hmm_source import (  # noqa: E402
        list_well_ids as selfgr_list_well_ids,
    )
    from exp263_selfgr_hmm_source import load_well as selfgr_load_well  # noqa: E402
    from exp263_selfgr_hmm_source import run_hmm2 as selfgr_run_hmm2  # noqa: E402

    stage0_manifest_path = resolve_unique_source(
        "cache_manifest.json", "exp263-last-anchor-pair-cache-train"
    )
    if sha256_file(stage0_manifest_path) != str(
        stage1["stage0_manifest"]["expected_manifest_sha256"]
    ):
        raise ValueError("exp263 Stage 0 manifest SHA mismatch")

    pf_config = generation["pf_replay"]
    replacement_likpf = dict(config["replacement"]["likelihood_pf"])
    if int(pf_config["pf_seeds"]) != int(replacement_likpf["seeds"]):
        raise ValueError("current-test likelihood-PF seed count differs from exp413")
    if int(pf_config["pf_particles"]) != int(replacement_likpf["particles"]):
        raise ValueError("current-test likelihood-PF particle count differs from exp413")
    if float(replacement_likpf["seed_weighting_scale"]) != 5.0:
        raise ValueError("exp413 current-test seed weighting scale must remain 5.0")
    configure_public_runtime(
        data_dir=paths.raw_data_dir,
        output_dir=output_dir / "pf_replay",
        n_jobs=int(pf_config["n_jobs"]),
        pf_seeds=int(pf_config["pf_seeds"]),
        pf_particles=int(pf_config["pf_particles"]),
        fast=bool(pf_config["fast"]),
        use_gpu=str(pf_config["use_gpu"]),
    )
    if shared_deterministic_frame is None or reuse_tracker is None:
        raise RuntimeError("exp413 requires the in-memory hjyact deterministic candidate frame")
    # The caller transfers sole ownership with globals().pop(). Reuse the
    # frame directly; the HJYACT component and reuse SHA records are already frozen.
    pf_frame = shared_deterministic_frame
    shared_deterministic_frame = None
    pf_frame.reset_index(drop=True, inplace=True)
    pf_frame["id"] = pf_frame["id"].astype(str)
    pf_frame["well"] = pf_frame["well"].astype(str)
    route_started = time.time()
    test_wells = replay_list_test_wells()
    if not test_wells:
        raise ValueError("exp413 route-specific PF found no test wells")
    if set(pf_frame["well"]) != set(test_wells):
        raise ValueError("shared deterministic frame well set differs from dynamic raw test")

    likpf_columns = [column for column in pf_frame if column.startswith("likpf_")]
    pf_frame.drop(columns=likpf_columns, inplace=True)
    id_to_position = {value: index for index, value in enumerate(pf_frame["id"].astype(str))}
    route_pf_effective_n_jobs = min(EXP413_WELL_N_JOBS, len(test_wells))

    # The seeded wrappers are source-identical except for releasing the GIL.
    # Each well still owns its stable seed and private Numba RNG state.
    from numba import njit as numba_njit

    for _seeded_name in ("_pf_ancc_seeded", "_pf_z_seeded"):
        _seeded_dispatcher = getattr(replay_source, _seeded_name)
        setattr(
            replay_source,
            _seeded_name,
            numba_njit(cache=False, nogil=True)(_seeded_dispatcher.py_func),
        )

    def run_route_pf_well(well: str) -> dict[str, Any] | None:
        horizontal, typewell = replay_source.load_well(well, "test")
        typewell = typewell.sort_values("TVT")
        known = horizontal["TVT_input"].notna()
        evaluation = horizontal[~known]
        expected_ids = [f"{well}_{int(index)}" for index in evaluation.index]
        if not expected_ids:
            return None
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
        return {
            "well": str(well),
            "positions": positions,
            "updates": updates,
            "record": {
                "well": str(well),
                "rows": len(expected_ids),
                "pf_ancc_seed": int(replay_stable_seed("pf_ancc", well)),
                "pf_z_seed": int(replay_stable_seed("pf_z", well)),
            },
        }

    route_pf_results = JoblibParallel(
        n_jobs=route_pf_effective_n_jobs,
        backend="threading",
    )(
        joblib_delayed(run_route_pf_well)(well)
        for well in test_wells
    )
    route_pf_records = []
    for route_pf_result in route_pf_results:
        if route_pf_result is None:
            continue
        positions = route_pf_result["positions"]
        for column, values in route_pf_result["updates"].items():
            pf_frame.loc[positions, column] = np.asarray(values, dtype=np.float32)
        reuse_tracker.mark_exp413_hit(route_pf_result["well"])
        route_pf_records.append(route_pf_result["record"])

    # All route-PF updates have been applied to pf_frame. Drop the per-well
    # result list and ID lookup before materializing the compact likPF columns.
    route_pf_results = None
    route_pf_result = None
    id_to_position = None
    positions = None
    values = None
    _exp514_gc.collect()

    if shared_likpf_bank is None:
        raise RuntimeError('exp413 requires the precomputed shared likelihood-PF bank')
    likpf_test = shared_likpf_exp413_adapter(shared_likpf_bank, test_wells)
    _likpf_test_rows = len(likpf_test)
    _likpf_aligned = (
        likpf_test.assign(id=likpf_test['id'].astype(str))
        .set_index('id')
        .reindex(pf_frame['id'].astype(str))
    )
    for likpf_column in [column for column in likpf_test.columns if column != 'id']:
        if _likpf_aligned[likpf_column].isna().any():
            raise ValueError(f'shared exp413 likelihood-PF coverage failed: {likpf_column}')
        pf_frame[likpf_column] = _likpf_aligned[likpf_column].to_numpy(
            dtype=np.float32, copy=False
        )
        pf_frame[likpf_column + '_d'] = (
            pf_frame[likpf_column] - pf_frame['last_known_tvt']
        ).astype(np.float32)
    del likpf_test, _likpf_aligned
    _exp514_gc.collect()
    pf_frame.reset_index(drop=True, inplace=True)
    pf_meta = {
        "shared_deterministic_dag_reused": True,
        "test_wells": len(test_wells),
        "test_rows": len(pf_frame),
        "test_likpf_rows": int(_likpf_test_rows),
        "route_specific_pf_records": route_pf_records,
        "well_parallel": {
            "requested_n_jobs": EXP413_WELL_N_JOBS,
            "effective_n_jobs": route_pf_effective_n_jobs,
            "backend": "threading",
            "numba_seeded_wrappers_nogil": True,
        },
        "elapsed_feature_seconds": round(time.time() - route_started, 3),
    }
    pf_frame["id"] = pf_frame["id"].astype(str)
    pf_frame["well"] = pf_frame["well"].astype(str)
    if pf_frame.empty:
        raise ValueError("current-test likelihood-PF produced no rows")
    if len(pf_frame) != len(sample):
        raise ValueError("current-test likelihood-PF row count differs from sample submission")
    if pf_frame["id"].duplicated().any():
        raise ValueError("current-test likelihood-PF contains duplicate IDs")
    if set(pf_frame["id"]) != set(sample["id"]):
        raise ValueError("current-test likelihood-PF IDs differ from sample submission")
    if pf_frame["well"].nunique() < 1:
        raise ValueError("current-test likelihood-PF produced no wells")
    required_pf = {
        "id",
        "well",
        "last_known_tvt",
        "likpf_mean",
        "likpf_mean_d",
        "likpf_scale_5",
        "likpf_scale_5_d",
        "pf_ancc",
        "pf_ancc_std",
        "beam_mean_d",
        "beam_std_d",
    }
    if missing_pf := required_pf - set(pf_frame.columns):
        raise ValueError(f"exp073 raw-test replay columns missing: {sorted(missing_pf)}")
    old_mean = pf_frame["likpf_mean"].to_numpy(np.float32).copy()
    scale5 = pf_frame["likpf_scale_5"].to_numpy(np.float32).copy()
    if not np.isfinite(scale5).all():
        raise ValueError("current-test scale5 likelihood-PF contains non-finite values")
    changed_rows = int(np.not_equal(old_mean, scale5).sum())
    if changed_rows == 0:
        raise ValueError("current-test scale5 replacement changed zero rows")
    pf_frame["likpf_mean"] = scale5
    pf_frame["likpf_mean_d"] = pf_frame["likpf_scale_5_d"].to_numpy(np.float32)
    scale5_delta_roundtrip_max_abs = float(
        np.max(
            np.abs(
                pf_frame["likpf_mean"].to_numpy(np.float32)
                - (
                    pf_frame["last_known_tvt"].to_numpy(np.float32)
                    + pf_frame["likpf_mean_d"].to_numpy(np.float32)
                )
            )
        )
    )
    if scale5_delta_roundtrip_max_abs > float(
        config["replacement"]["parent_old_mean_parity_max_abs_ft"]
    ):
        raise ValueError("scale5 absolute/delta replacement parity failed")
    test_wells = replay_list_test_wells()
    seed_namespace = "SHA256(likpf::test::<well>)"
    if seed_namespace != str(inference_cfg["stable_seed_namespace"]):
        raise ValueError("current-test stable-seed namespace differs from frozen contract")
    stable_seed_records = [
        {
            "well": str(well),
            "seed_base": int(replay_stable_seed("likpf", "test", well)),
        }
        for well in test_wells
    ]
    if len({item["seed_base"] for item in stable_seed_records}) != len(stable_seed_records):
        raise ValueError("current-test stable per-well likelihood-PF seeds are not unique")
    replacement_pf_audit = {
        "semantic_slot": "likpf_mean",
        "value_source": "likpf_scale_5_x1p0",
        "rows": int(len(pf_frame)),
        "wells": int(pf_frame["well"].nunique()),
        "changed_rows_vs_arithmetic_mean": changed_rows,
        "particles": int(pf_config["pf_particles"]),
        "seeds": int(pf_config["pf_seeds"]),
        "temperature": 5.0,
        "gr_scale_multiplier": 1.0,
        "seed_namespace": seed_namespace,
        "thread_schedule_independent": True,
        "absolute_delta_roundtrip_max_abs_ft": scale5_delta_roundtrip_max_abs,
        "stable_seed_records": stable_seed_records,
        "arithmetic_mean_content_sha256": frame_content_sha256(
            pd.DataFrame(
                {
                    "id": pf_frame["id"].astype(str),
                    "well": pf_frame["well"].astype(str),
                    "candidate_tvt": old_mean,
                }
            )
        ),
        "scale5_content_sha256": frame_content_sha256(
            pd.DataFrame(
                {
                    "id": pf_frame["id"].astype(str),
                    "well": pf_frame["well"].astype(str),
                    "candidate_tvt": scale5,
                }
            )
        ),
    }

    k16_source_config = resolved_sources["exp263_k16_source"].parent / str(
        generation["exp226_k16"]["source_config_filename"]
    )
    if not k16_source_config.exists():
        raise FileNotFoundError(f"exp226 source config missing: {k16_source_config}")
    k16_frame, k16_summary = generate_k16_primitive(
        k16_module,
        train_dir=paths.train_data_dir,
        test_dir=paths.test_data_dir,
        source_config=yaml.safe_load(k16_source_config.read_text()),
        frame_content_sha256=frame_content_sha256,
    )
    exact_config = generation["exact_hmm"]
    exact_frame = generate_hmm_primitive(
        list_well_ids=exact_list_well_ids,
        load_well=exact_load_well,
        run_hmm2=exact_run_hmm2,
        test_dir=paths.test_data_dir,
        hmm_params=dict(exact_config["params"]),
    )
    selfgr_config = generation["selfgr_hmm_a070"]
    selfgr_frame = generate_hmm_primitive(
        list_well_ids=selfgr_list_well_ids,
        load_well=selfgr_load_well,
        run_hmm2=selfgr_run_hmm2,
        test_dir=paths.test_data_dir,
        hmm_params=dict(exact_config["params"]),
        self_gr=dict(selfgr_config),
    )
    primitive_frames = {
        "exp226_k16": k16_frame,
        "selfgr_hmm_a070": selfgr_frame,
        "likpf_mean": standard_primitive(
            pf_frame,
            pf_frame["likpf_mean"].to_numpy(np.float32),
        ),
        "exact_hmm": exact_frame,
        "pf_ancc": standard_primitive(
            pf_frame,
            pf_frame["pf_ancc"],
            confidence={"sigma_tvt": pf_frame["pf_ancc_std"]},
        ),
        "beam_mean": standard_primitive(
            pf_frame,
            pf_frame["last_known_tvt"].to_numpy(np.float32)
            + pf_frame["beam_mean_d"].to_numpy(np.float32),
            confidence={"beam_family_std": pf_frame["beam_std_d"]},
        ),
    }
    formula_frame, max_abs_formula = assemble_stage1_current_test_parity(primitive_frames)
    formula_frame = attach_stage1_current_test_confidence(formula_frame, primitive_frames)
    formula_frame = formula_frame.sort_values(["well", "well_row_idx"], kind="stable").reset_index(
        drop=True
    )
    formula_path = output_dir / "current_test_formula_parity.parquet"
    formula_frame.to_parquet(formula_path, index=False, compression="zstd")
    confidence_parity = validate_current_test_native_confidence(
        formula_frame, candidate_contract
    )
    if int(confidence_parity["required_column_count"]) != int(
        inference_cfg["required_namespaced_confidence_column_count"]
    ):
        raise ValueError("exp263 current-test confidence column count mismatch")
    if sum(column.startswith("confidence__") for column in formula_frame) != 21:
        raise ValueError("exp263 Stage 1 must export exactly 21 confidence columns")
    if set(formula_frame["id"].astype(str)) != set(sample["id"]):
        raise ValueError("generated exp263 candidate IDs differ from sample submission")

    source_audit = {name: source_record(path) for name, path in resolved_sources.items()}
    source_audit["exp263_stage0_manifest"] = source_record(stage0_manifest_path)
    primitive_content_sha = {
        candidate_id: frame_content_sha256(frame)
        for candidate_id, frame in primitive_frames.items()
    }
    display(
        {
            "rows": len(formula_frame),
            "wells": int(formula_frame["well"].nunique()),
            "primitive_count": len(primitive_frames),
            "pair_count": len(rawtest_pairs),
            "candidate_count": len(names),
            "confidence_columns": 21,
            "formula_max_abs_error": float(max_abs_formula),
            "replacement_pf": replacement_pf_audit,
        }
    )
    display(formula_frame[["id", "well", *names]].head())

    # [embedded markdown boundary]
    # ## 4. Candidate-long context, parent compact, and signed compact features
    #
    # candidate-long matrixはchunkごとに一度だけ作る。同じmatrixへouter-fold別8 parent selectorと
    # 4 signed selectorを適用し、saved74とsigned23を同じcandidate/order/outer契約で生成する。

    # [embedded code boundary]
    bundle = current_test_bundle_from_wide(formula_frame, candidate_contract)
    fill_current_test_anchor(bundle, paths.test_data_dir)
    feature_cfg = dict(parent_config["features"])
    feature_cfg["primary_domain"] = candidate_contract["legal_domains"][
        "primitive_pair_bank"
    ]["candidates"]
    feature_cfg["fixed_domain"] = candidate_contract["legal_domains"][
        "primitive_fixed_bank"
    ]["candidates"]
    raw_context, truth = build_raw_context(
        bundle.base, paths.test_data_dir, feature_cfg, require_truth=False
    )
    if truth is not None:
        raise RuntimeError("current-test selector context unexpectedly contains truth")

    shape_state = ShapeState.from_bundle(bundle.base, bundle.values)
    chunk_size = int(
        parent_config["model"]["selector"]["training"]["predict_base_row_chunk_size"]
    )
    compact_parts: dict[int, list[pd.DataFrame]] = {outer: [] for outer in range(5)}
    signed_compact_parts: dict[int, list[pd.DataFrame]] = {
        outer: [] for outer in range(5)
    }
    signed_top1_parity_max = 0.0
    score_sample: pd.DataFrame | None = None
    selector_missing_count_by_feature = np.zeros(len(selector_features), dtype=np.int64)
    selector_missing_count_by_candidate = np.zeros(len(names), dtype=np.int64)
    selector_long_row_count = 0
    for start in range(0, len(bundle.base), chunk_size):
        stop = min(start + chunk_size, len(bundle.base))
        indices = np.arange(start, stop, dtype=np.int64)
        long_frame, metadata = build_candidate_long_features(
            bundle,
            raw_context,
            indices,
            feature_cfg,
            shape_state=shape_state,
            expected_features=selector_features,
        )
        matrix = long_frame.to_numpy(np.float32, copy=False)
        missingness_chunk = validate_inference_feature_missingness(
            long_frame,
            training_missing_rate_by_feature,
            context=f"current-test selector rows {start}:{stop}",
        )
        selector_missing_count_by_feature += missingness_chunk["missing_count"].to_numpy(
            np.int64
        )
        missing_tensor = np.isnan(matrix).reshape(
            len(indices), len(names), len(selector_features)
        )
        selector_missing_count_by_candidate += missing_tensor.sum(axis=(0, 2)).astype(
            np.int64
        )
        selector_long_row_count += len(long_frame)
        confidence_valid = metadata["confidence_valid"].to_numpy(bool).reshape(
            len(indices), len(names)
        )
        for outer in range(5):
            p = np.mean(
                [
                    model.predict(matrix, num_iteration=model.best_iteration)
                    for model in selector_models[outer]["p_within10"]
                ],
                axis=0,
            ).reshape(len(indices), len(names)).astype(np.float32)
            e = np.mean(
                [
                    model.predict(matrix, num_iteration=model.best_iteration)
                    for model in selector_models[outer]["pred_abs_error"]
                ],
                axis=0,
            ).reshape(len(indices), len(names)).astype(np.float32)
            e = np.maximum(e, 0.0)
            if not np.isfinite(e).all() or not np.isfinite(p).all():
                raise ValueError(f"Stage C selector scores are non-finite for outer fold {outer}")
            parent_compact = build_compact_meta(
                bundle.base.iloc[indices].reset_index(drop=True),
                bundle.values[indices],
                e,
                p,
                bundle.available[indices],
                confidence_valid,
                candidate_contract,
            )
            compact_parts[outer].append(parent_compact)
            signed_prediction = np.mean(
                [
                    model.predict(matrix, num_iteration=model.best_iteration)
                    for model in signed_selector_models[outer]
                ],
                axis=0,
            ).reshape(len(indices), len(names)).astype(np.float32)
            if not np.isfinite(signed_prediction).all():
                raise ValueError(
                    f"Stage S signed selector scores are non-finite for outer fold {outer}"
                )
            signed_compact, signed_evidence = build_signed_compact_meta(
                bundle.base.iloc[indices].reset_index(drop=True),
                bundle.values[indices],
                signed_prediction,
                parent_compact,
                candidate_contract,
                top1_value_atol=float(
                    parent_config["guards"]["stage_s"]["saved_top1_value_parity_atol"]
                ),
            )
            signed_compact_parts[outer].append(signed_compact)
            signed_top1_parity_max = max(
                signed_top1_parity_max,
                float(signed_evidence["top1_value_parity_max_abs_error"]),
            )
            if outer == 0 and score_sample is None:
                take = min(
                    len(metadata), int(inference_cfg["score_sample_rows"])
                )
                score_sample = metadata.iloc[:take].copy()
                score_sample["pred_abs_error"] = e.reshape(-1)[:take]
                score_sample["p_within10"] = p.reshape(-1)[:take]
                score_sample["pred_signed_residual"] = signed_prediction.reshape(-1)[:take]
                score_sample["downstream_outer_fold"] = np.int8(outer)
            del parent_compact, signed_prediction, signed_compact, signed_evidence
        del long_frame, metadata, matrix, confidence_valid, missingness_chunk, missing_tensor
        gc.collect()

    selector_missingness = selected_catalog[
        ["feature", "group", "missing_rate"]
    ].rename(columns={"missing_rate": "training_missing_rate"})
    selector_missingness["current_missing_count"] = selector_missing_count_by_feature
    selector_missingness["current_missing_rate"] = (
        selector_missing_count_by_feature.astype(np.float64) / float(selector_long_row_count)
    )
    selector_missingness["structural_missingness"] = selector_missingness[
        "feature"
    ].str.startswith(("conf__", "formula__"))
    all_missing_current = selector_missingness.loc[
        selector_missingness["current_missing_rate"].ge(1.0), "feature"
    ].tolist()
    if all_missing_current:
        raise ValueError(
            f"current-test selector features became all-missing: {all_missing_current[:20]}"
        )
    selector_missingness_path = output_dir / "selector_missingness_current_test.csv"
    selector_missingness.to_csv(selector_missingness_path, index=False)

    selector_candidate_missingness = pd.DataFrame(
        {
            "candidate_id": names,
            "missing_count": selector_missing_count_by_candidate,
            "missing_rate": selector_missing_count_by_candidate.astype(np.float64)
            / float(len(bundle.base) * len(selector_features)),
        }
    )
    selector_candidate_missingness_path = (
        output_dir / "selector_missingness_by_candidate_current_test.csv"
    )
    selector_candidate_missingness.to_csv(selector_candidate_missingness_path, index=False)
    display(
        {
            "selector_training_sparse_features": training_sparse_feature_count,
            "selector_current_sparse_features": int(
                selector_missingness["current_missing_count"].gt(0).sum()
            ),
            "selector_current_missing_cells": int(selector_missing_count_by_feature.sum()),
            "selector_infinite_cells": 0,
        }
    )
    display(
        selector_missingness.sort_values(
            ["current_missing_rate", "feature"], ascending=[False, True]
        ).head(40)
    )
    display(selector_candidate_missingness)

    compact_by_outer: dict[int, pd.DataFrame] = {}
    compact_sha: dict[str, str] = {}
    signed_compact_by_outer: dict[int, pd.DataFrame] = {}
    signed_compact_sha: dict[str, str] = {}
    for outer in range(5):
        compact = pd.concat(compact_parts[outer], ignore_index=True)
        if len(compact) != len(bundle.base):
            raise ValueError(f"compact row coverage mismatch for outer fold {outer}")
        if [
            column for column in compact if column.startswith("selector__")
        ] != parent_compact_features:
            raise ValueError(f"compact schema mismatch for outer fold {outer}")
        if not np.isfinite(
            compact[parent_compact_features].to_numpy(np.float32)
        ).all():
            raise ValueError(f"compact features are non-finite for outer fold {outer}")
        compact_path = output_dir / f"parent_compact_current_test_outer{outer}.parquet"
        compact.to_parquet(compact_path, index=False, compression="zstd")
        compact_sha[str(outer)] = sha256_file(compact_path)
        compact_by_outer[outer] = compact
        signed_compact = pd.concat(signed_compact_parts[outer], ignore_index=True)
        if len(signed_compact) != len(bundle.base):
            raise ValueError(f"signed compact row coverage mismatch for outer fold {outer}")
        if [
            column for column in signed_compact if column.startswith("selector__")
        ] != signed_compact_features:
            raise ValueError(f"signed compact schema mismatch for outer fold {outer}")
        if not np.isfinite(
            signed_compact[signed_compact_features].to_numpy(np.float32)
        ).all():
            raise ValueError(
                f"signed compact features are non-finite for outer fold {outer}"
            )
        signed_path = output_dir / f"signed_compact_current_test_outer{outer}.parquet"
        signed_compact.to_parquet(signed_path, index=False, compression="zstd")
        signed_compact_sha[str(outer)] = sha256_file(signed_path)
        signed_compact_by_outer[outer] = signed_compact
    if score_sample is None:
        raise RuntimeError("selector score sample was not generated")
    score_sample_path = output_dir / "candidate_score_sample_outer0.parquet"
    score_sample.to_parquet(score_sample_path, index=False, compression="zstd")
    del compact_parts, signed_compact_parts, score_sample
    gc.collect()
    display(compact_by_outer[0].head())
    display(signed_compact_by_outer[0].head())

    # [embedded markdown boundary]
    # ## 5. Exp218 current-test clean 273-feature surface
    #
    # exp263 replay frameを共通baseとし、anchor、U projection、exp145 learned likelihood、GRWRを
    # current testから再計算する。保存済みpublic-test feature artifactは入力に使わず、Stage Dモデルの
    # clean 273 allowlistと列順が一致する特徴だけを使う。

    # [embedded code boundary]
    exp218 = import_file(
        "exp264_inference_exp218",
        [
            PACKAGE_DIR / "inputs/exp218_source/gr_wavelet_rotation_confidence_features_on_exp148.py",
            Path(
                "experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/"
                "gr_wavelet_rotation_confidence_features_on_exp148.py"
            ),
        ],
    )
    exp218_config = yaml.safe_load(
        (PACKAGE_DIR / "inputs/exp218_source/config.yaml").read_text()
    )
    exp145_source_dir = PACKAGE_DIR / "inputs/exp145_source"
    exp145_settings = import_file(
        "exp264_inference_exp145_settings",
        [exp145_source_dir / "settings.py"],
    )
    original_settings_module = sys.modules.get("settings")
    sys.modules["settings"] = exp145_settings
    exp145 = import_file(
        "exp264_inference_exp145",
        [exp145_source_dir / "learned_likelihood_rawtest_feature_generator_parity.py"],
    )

    exp145_config = exp145.load_config()
    exp145_candidates = exp145.candidate_specs_from_config(exp145_config)
    learned_source_frame = exp145.ensure_candidate_value_columns(
        pf_frame, exp145_candidates
    )
    learned_cache_path = output_dir / "exp263_replay_for_exp145.csv.gz"
    learned_source_frame.to_csv(learned_cache_path, index=False, compression="gzip")
    del learned_source_frame
    _exp514_gc.collect()
    learned_output_dir = output_dir / "exp145_current_test"
    sys.modules["settings"] = exp145_settings
    learned_generator_summary = exp145.run_generator(
        output_dir=learned_output_dir,
        mode="rawtest",
        train_cache_path=None,
        rawtest_cache_path=learned_cache_path,
        exp111_schema_path=None,
        exp111_manifest_path=None,
        exp112_schema_path=None,
        max_rows=None,
    )
    if original_settings_module is not None:
        sys.modules["settings"] = original_settings_module
    else:
        sys.modules.pop("settings", None)
    if not bool(learned_generator_summary["generated_schema"]["schema_parity_pass"]):
        raise ValueError("exp145 current-test learned feature schema parity failed")
    learned_feature_path = Path(
        learned_generator_summary["outputs"]["rawtest_ml_features"]["path"]
    )
    learned_source = pd.read_csv(learned_feature_path, dtype={"id": str, "well": str})

    test_frame, anchor_meta = exp218.add_inference_anchor_columns(
        pf_frame, paths.test_data_dir
    )
    pf_frame = None
    _exp514_gc.collect()
    projection_cfg = get_nested(exp218_config, "model.u_projection", {}) or {}
    projection, _, _ = exp218.build_u_projection_features(
        test_frame,
        source_specs=dict(projection_cfg.get("sources") or {}),
        degree=int(projection_cfg.get("degree", 3)),
        robust_iters=int(projection_cfg.get("robust_iters", 3)),
        clip_sigma=float(projection_cfg.get("clip_sigma", 4.0)),
    )
    projection_columns = [column for column in projection if column not in {"id", "well"}]
    exp218._assign_aligned_float32_columns(
        test_frame, projection.reset_index(drop=True), projection_columns
    )
    if not exp218.learned_feature_keys_match(learned_source, test_frame):
        raise ValueError("dynamic exp145 learned-feature keys differ from exp263 replay test")
    learned, _, _ = exp218.build_learned_likelihood_features(
        learned_source,
        test_frame,
        get_nested(exp218_config, "model.learned_likelihood_features", {}) or {},
    )
    learned_columns = [column for column in learned if column not in {"id", "well"}]
    exp218._assign_aligned_float32_columns(
        test_frame, learned.reset_index(drop=True), learned_columns
    )
    grwr, _, _, grwr_meta = exp218.build_gr_wavelet_rotation_confidence_features(
        test_frame,
        train_dir=paths.test_data_dir,
        config=get_nested(exp218_config, "model.gr_wavelet_rotation_confidence_features", {})
        or {},
    )
    grwr_columns = [column for column in grwr if column not in {"id", "well"}]
    exp218._assign_aligned_float32_columns(
        test_frame, grwr.reset_index(drop=True), grwr_columns
    )
    missing_source_base = [column for column in source_base_columns if column not in test_frame]
    if missing_source_base:
        raise ValueError(
            "raw-test exp218 surface missing source features: "
            f"{missing_source_base[:40]}"
        )
    for start in range(0, len(source_base_columns), 32):
        columns = source_base_columns[start : start + 32]
        if not np.isfinite(test_frame[columns].to_numpy(np.float32, copy=False)).all():
            raise ValueError(f"raw-test exp218 features contain non-finite values: {columns}")
    if set(test_frame["id"].astype(str)) != set(formula_frame["id"].astype(str)):
        raise ValueError("exp218 and exp263 current-test ID sets differ")
    display(
        {
            "rows": len(test_frame),
            "wells": int(test_frame["well"].nunique()),
            "source_base_feature_count": len(source_base_columns),
            "base_feature_count": len(base_feature_columns),
            "learned_schema_parity": True,
        }
    )
    del projection, learned_source, learned, grwr
    gc.collect()

    # [embedded markdown boundary]
    # ## 6. Exp413 Stage D saved-booster CPU inference
    #
    # 各TVT modelは学習時と同じouter foldのnested74 + signed23だけを使う。GPUで学習したtext modelを
    # CPU predictorで読み、3 config × 5 foldの15 residual predictionを等重み平均する。

    # [embedded code boundary]
    pred_delta = np.zeros(len(test_frame), dtype=np.float32)
    component_predictions: dict[str, np.ndarray] = {}
    tvt_model_audit: list[dict[str, Any]] = []
    for outer in range(5):
        compact = compact_by_outer[outer]
        signed_compact = signed_compact_by_outer[outer]
        aligned_compact = test_frame[["id"]].merge(
            compact[["id", *parent_compact_features]],
            on="id",
            how="left",
            validate="one_to_one",
        )
        aligned_signed = test_frame[["id"]].merge(
            signed_compact[["id", *signed_compact_features]],
            on="id",
            how="left",
            validate="one_to_one",
        )
        if aligned_compact[parent_compact_features].isna().any().any():
            raise ValueError(
                f"parent compact alignment introduced missing values for outer fold {outer}"
            )
        if aligned_signed[signed_compact_features].isna().any().any():
            raise ValueError(
                f"signed compact alignment introduced missing values for outer fold {outer}"
            )
        matrix_frame = pd.concat(
            [
                test_frame[base_feature_columns].reset_index(drop=True),
                aligned_compact[parent_compact_features].reset_index(drop=True),
                aligned_signed[signed_compact_features].reset_index(drop=True),
            ],
            axis=1,
        )
        if list(matrix_frame.columns) != final_feature_columns:
            raise ValueError(f"Stage D feature order mismatch for outer fold {outer}")
        matrix = matrix_frame.to_numpy(np.float32, copy=False)
        if not np.isfinite(matrix).all():
            raise ValueError(f"Stage D feature matrix is non-finite for outer fold {outer}")
        fold_models = [
            (item, model_path)
            for item, model_path in resolved_tvt_models
            if int(item["outer_fold"]) == outer
        ]
        if len(fold_models) != 3:
            raise ValueError(
                f"Stage D outer fold {outer} must have three replacement models"
            )
        for item, model_path in fold_models:
            booster = lgb.Booster(model_file=str(model_path))
            if list(booster.feature_name()) != final_feature_columns:
                raise ValueError(f"Stage D model feature schema mismatch: {model_path.name}")
            prediction = booster.predict(
                matrix, num_iteration=int(item["best_iteration"])
            ).astype(np.float32)
            if not np.isfinite(prediction).all():
                raise ValueError(f"Stage D model prediction is non-finite: {model_path.name}")
            key = f"pred_delta__{item['model']}__outer{outer}"
            component_predictions[key] = prediction
            pred_delta += prediction / np.float32(len(resolved_tvt_models))
            tvt_model_audit.append(
                {
                    "model": str(item["model"]),
                    "config_index": int(item["config_index"]),
                    "outer_fold": outer,
                    "selector_score_outer_fold": outer,
                    "file": model_path.name,
                    "sha256": str(item["sha256"]),
                    "best_iteration": int(item["best_iteration"]),
                }
            )
            del booster, prediction
            gc.collect()
        del compact, signed_compact, aligned_compact, aligned_signed, matrix_frame, matrix
        gc.collect()
    if len(tvt_model_audit) != 15:
        raise ValueError("Stage D inference did not use all 15 replacement models")
    pred_tvt = test_frame["last_known_tvt"].to_numpy(np.float32) + pred_delta
    if not np.isfinite(pred_tvt).all():
        raise ValueError("final Stage D TVT prediction contains non-finite values")
    predictions = pd.DataFrame(
        {
            "id": test_frame["id"].astype(str),
            "well": test_frame["well"].astype(str),
            "last_known_tvt": test_frame["last_known_tvt"].to_numpy(np.float32),
            "pred_delta": pred_delta,
            "pred_tvt": pred_tvt,
            **component_predictions,
        }
    )

    # [embedded markdown boundary]
    # ## 7. Prediction, submission, and reproducibility outputs
    #
    # sample submissionのID順へstrict joinした`id,tvt`だけを
    # `/kaggle/working/submission.csv`へ保存する。予測監査CSVは追加列を含む。
    # competition submit APIは呼ばない。

    # [embedded code boundary]
    prediction_contract = sample[["id"]].merge(
        predictions[["id", "pred_tvt"]], on="id", how="left", validate="one_to_one"
    )
    if len(prediction_contract) != len(sample) or not prediction_contract["id"].equals(
        sample["id"]
    ):
        raise ValueError("prediction row/order contract failed")
    if prediction_contract["pred_tvt"].isna().any() or not np.isfinite(
        prediction_contract["pred_tvt"]
    ).all():
        raise ValueError("prediction finite contract failed")
    submission = prediction_contract.rename(columns={"pred_tvt": "tvt"})
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(
            f"unexpected sample submission columns: {list(sample.columns)}"
        )
    if list(submission.columns) != list(sample.columns):
        raise ValueError("submission column contract failed")
    if submission["id"].duplicated().any():
        raise ValueError("submission contains duplicate IDs")
    submission.to_csv(paths.submission_path, index=False)
    if not paths.submission_path.exists():
        raise RuntimeError("submission.csv was not written to the Kaggle working directory")

    prediction_path = output_dir / "exp413_current_test_predictions.csv.gz"
    feature_schema_path = output_dir / "exp413_inference_feature_schema.csv"
    predictions.to_csv(prediction_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "feature_index": np.arange(len(final_feature_columns), dtype=np.int32),
            "feature": final_feature_columns,
            "feature_group": [
                (
                    "exp218_base"
                    if index < len(base_feature_columns)
                    else (
                        "replacement_nested_compact"
                        if index
                        < len(base_feature_columns) + len(parent_compact_features)
                        else "signed_residual_compact"
                    )
                )
                for index in range(len(final_feature_columns))
            ],
        }
    ).to_csv(feature_schema_path, index=False)

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "route": config["experiment"]["route"],
        "status": "cpu_inference_completed_with_kaggle_submission_output",
        "authorization": {
            "status": inference_cfg["status"],
            "scope": inference_cfg["authorization_scope"],
            "stage_d_primary_gate_passed": True,
            "generate_submission_file": True,
            "competition_submit_performed": False,
            "competition_submit_authorized": False,
        },
        "runtime": "kaggle_cpu",
        "runtime_seconds": round(time.time() - STARTED_AT, 3),
        "rows": int(len(predictions)),
        "wells": int(predictions["well"].nunique()),
        "candidate_count": len(names),
        "namespaced_confidence_column_count": 21,
        "selector_feature_count": len(selector_features),
        "selector_missingness": {
            "training_sparse_feature_count": training_sparse_feature_count,
            "current_sparse_feature_count": int(
                selector_missingness["current_missing_count"].gt(0).sum()
            ),
            "current_missing_cell_count": int(selector_missing_count_by_feature.sum()),
            "infinite_cell_count": 0,
            "zero_imputation_performed": False,
        },
        "parent_compact_feature_count": len(parent_compact_features),
        "signed_compact_feature_count": len(signed_compact_features),
        "base_feature_count": len(base_feature_columns),
        "source_base_feature_count": len(source_base_columns),
        "final_feature_count": len(final_feature_columns),
        "parent_selector_model_count": len(selector_model_audit),
        "signed_selector_model_count": len(signed_selector_model_audit),
        "tvt_model_count": len(tvt_model_audit),
        "booster_training_count": 0,
        "submission_file_generated": True,
        "external_submission_performed": False,
        "max_abs_formula_parity": float(max_abs_formula),
        "signed_top1_value_parity_max_abs_error": float(signed_top1_parity_max),
        "confidence_parity": confidence_parity,
        "prediction_stats": {
            "min": float(pred_tvt.min()),
            "max": float(pred_tvt.max()),
            "mean": float(pred_tvt.mean()),
            "std": float(pred_tvt.std()),
        },
        "stage_d_gate_evidence": {
            "saved_exp335_rmse": float(inference_cfg["stage_d_saved_exp335_rmse"]),
            "replacement_rmse": float(inference_cfg["stage_d_variant_rmse"]),
            "gain_ft": float(inference_cfg["stage_d_gain_ft"]),
            "nonworse_folds": int(inference_cfg["stage_d_nonworse_folds"]),
            "maximum_scope_delta_rmse_ft": float(
                inference_cfg["stage_d_maximum_scope_delta_rmse_ft"]
            ),
            "by_well_delta_p95": float(inference_cfg["stage_d_by_well_delta_p95"]),
            "worst_well_delta_rmse": float(
                inference_cfg["stage_d_worst_well_delta_rmse"]
            ),
            "primary_gate_passed": True,
        },
        "source_audit": source_audit,
        "pf_generation": pf_meta,
        "hmm_well_parallel": hmm_parallel_reports,
        "replacement_pf": replacement_pf_audit,
        "exp226_generation": k16_summary,
        "exp145_generation": learned_generator_summary,
        "exp218_anchor": anchor_meta,
        "exp218_grwr": exp218._jsonable(grwr_meta),
        "primitive_content_sha256": primitive_content_sha,
        "parent_selector_models": selector_model_audit,
        "signed_selector_models": signed_selector_model_audit,
        "tvt_models": tvt_model_audit,
        "sha256": {
            "candidate_contract": candidate_contract_sha(candidate_contract),
            "exp263_formula_parquet": sha256_file(formula_path),
            "stage_c_model_manifest": sha256_file(stage_c_manifest_path),
            "stage_s_signed_model_manifest": sha256_file(signed_manifest_path),
            "stage_s_signed_compact_schema": sha256_file(signed_schema_path),
            "selector_feature_schema": sha256_file(selector_schema_path),
            "selector_feature_catalog": sha256_file(selector_catalog_path),
            "source_base_feature_catalog": sha256_file(source_base_catalog_path),
            "base_feature_allowlist": sha256_file(base_allowlist_path),
            "selector_missingness_current_test": sha256_file(selector_missingness_path),
            "selector_missingness_by_candidate_current_test": sha256_file(
                selector_candidate_missingness_path
            ),
            "parent_compact_meta_schema": sha256_file(compact_schema_path),
            "stage_d_model_manifest": sha256_file(stage_d_manifest_path),
            "parent_compact_parquet_by_outer": compact_sha,
            "signed_compact_parquet_by_outer": signed_compact_sha,
            "candidate_score_sample": sha256_file(score_sample_path),
            "exp145_replay_cache_decompressed": sha256_gzip_decompressed(learned_cache_path),
            "predictions_decompressed": sha256_gzip_decompressed(prediction_path),
            "predictions_file": sha256_file(prediction_path),
            "feature_schema": sha256_file(feature_schema_path),
            "submission": sha256_file(paths.submission_path),
        },
        "notes": [
            "All 12 candidates and 21 native-confidence columns are regenerated from raw test in this run.",
            "The likpf_mean semantic slot is sourced from the temperature-5 aggregation of the same 128 stable per-well seed trajectories.",
            "The arithmetic seed mean is retained only for replacement parity audit and is excluded from candidate/model input.",
            "Selector NaN values are preserved exactly as trained; no zero imputation is performed.",
            "The Stage A catalog guards training-dense features and structural confidence/formula missing rates.",
            "Each Stage D model receives replacement nested74 and signed23 features from its matching downstream outer fold.",
            "All 40 replacement selectors, 20 signed selectors, and 15 TVT models are SHA-verified; no model is fitted.",
            "No public-test row artifact, saved selector score CSV, hard selector, Viterbi, or candidate softmax average participates in prediction.",
            "submission.csv is generated by this Kaggle Notebook in /kaggle/working; external competition submission remains unauthorized.",
            "The Stage D primary gate PASS and report-only by-well tail degradation remain recorded separately.",
        ],
    }
    write_json(output_dir / "inference_metrics.json", metrics)
    write_json(output_dir / "reproducibility_manifest_inference.json", metrics)
    write_json(paths.metrics_path, metrics)
    if not paths.submission_path.exists():
        raise RuntimeError("Kaggle submission output is missing after inference")
    display(submission.head(20))
    display(submission["tvt"].describe())
    display(metrics)
    print("Generated artifacts:")
    for artifact_path in [
        formula_path,
        prediction_path,
        feature_schema_path,
        selector_missingness_path,
        selector_candidate_missingness_path,
        output_dir / "inference_metrics.json",
        paths.submission_path,
    ]:
        print(f"- {artifact_path} ({artifact_path.stat().st_size} bytes)")
    print("submission.csv generated: True")
    print("external submission performed: False")

    return int(len(predictions)), dict(metrics), Path(prediction_path)

# %% [markdown]
# ## 9. Shared-PF ledger, fixed blend, and reproducibility outputs

# %%
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
visible_reference_checks["hjyact_parent_exact_match_required"] = False
if visible_reference_checks["sample_id_order_match"]:
    observed_hjyact_sha = sha256_file(source_submission_path)
    visible_reference_checks["hjyact_submission_sha256"] = observed_hjyact_sha
    visible_reference_checks["hjyact_parent_submission_match"] = (
        observed_hjyact_sha == SOURCE_VISIBLE_FINAL_SHA256
    )
    visible_reference_checks["hjyact_candidate_submission_match"] = (
        observed_hjyact_sha == STAGE_D_VISIBLE_HJYACT_CANDIDATE_SHA256
    )
    visible_reference_checks["hjyact_submission_match"] = visible_reference_checks[
        "hjyact_candidate_submission_match"
    ]
    if not visible_reference_checks["hjyact_candidate_submission_match"]:
        raise RuntimeError(
            "visible exp514 HJYACT candidate witness failed: "
            f"{observed_hjyact_sha} != {STAGE_D_VISIBLE_HJYACT_CANDIDATE_SHA256}"
        )
else:
    visible_reference_checks["hjyact_parent_submission_match"] = None
    visible_reference_checks["hjyact_candidate_submission_match"] = None
    visible_reference_checks["hjyact_submission_match"] = None

if HJYACT_COMPONENT_PATH.exists():
    HJYACT_COMPONENT_PATH.unlink()
shutil.move(str(source_submission_path), HJYACT_COMPONENT_PATH)
if FINAL_SUBMISSION_PATH.exists():
    raise RuntimeError("submission.csv must be absent before exp413 component regeneration")

HJYACT_SHARED_FEATURE_WELL_COUNT = int(
    HJYACT_SHARED_FEATURE_FRAME["well"].nunique()
)
EXP413_PRE_RELEASE_REPORT = _exp514_release_globals(
    (
        "PF_SEED_BRANCH_STATS", "rows", "bimodal_report_rows", "sub_1",
        "sub_2", "sub", "cv_final", "_bimodal_df", "_active_mask",
        "_active_cols", "train_hw_files", "test_hw_files",
    ),
    label="consumed_sp45_and_hjyact_intermediates_before_exp413",
)
exp413_prediction_rows_memory, exp413_metrics, exp413_prediction_path = generate_dynamic_exp413_prediction(
    shared_deterministic_frame=globals().pop("HJYACT_SHARED_FEATURE_FRAME"),
    reuse_tracker=CANDIDATE_REUSE_TRACKER,
    shared_likpf_bank=SHARED_LIKPF_BANK,
)
if not exp413_prediction_path.is_file():
    raise RuntimeError("exp413 did not produce its declared prediction artifact")
exp413_predictions = pd.read_csv(exp413_prediction_path, compression="gzip", dtype={"id": str})
if "pred_tvt" not in exp413_predictions:
    raise RuntimeError("exp413 prediction artifact is missing pred_tvt")
if exp413_prediction_rows_memory != len(exp413_predictions):
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
component_frame.to_csv(WORKING_DIR / "exp514_component_readout.csv", index=False)
reuse_manifest = CANDIDATE_REUSE_TRACKER.manifest()
(WORKING_DIR / "candidate_reuse_manifest.json").write_text(
    json.dumps(reuse_manifest, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)

shared_likpf_manifest = finalize_shared_likpf_manifest(
    SHARED_LIKPF_BANK,
    test_wells,
)
SHARED_BANK_FINAL_RELEASE_REPORT = _exp514_release_globals(
    ("SHARED_LIKPF_BANK",),
    label="consumed_shared_likpf_manifest_records",
)
shared_likpf_manifest['parallel_report'] = SHARED_LIKPF_PARALLEL_REPORT
shared_likpf_manifest['jit_warmup_seconds'] = SHARED_LIKPF_JIT_WARMUP_SECONDS
SHARED_LIKPF_MANIFEST_PATH = WORKING_DIR / 'exp514_shared_likpf_manifest.json'
SHARED_LIKPF_MANIFEST_PATH.write_text(
    json.dumps(shared_likpf_manifest, indent=2, sort_keys=True, default=str) + '\n',
    encoding='utf-8',
)

model_manifest = {
    "experiment": "exp514_exp413_likpf_seed_bank_reuse_on_exp512",
    "legacy_directory_suffix": "10pct_hedge",
    "parent_experiment": "exp512_hjyact_v2_final_10pct_hedge_on_exp413",
    "actual_formula": "0.50 * exp413 + 0.50 * hjyact_v2_final",
    "route": "ensemble",
    "new_booster_training_count": 0,
    "runtime_ridge_fit_count": 5,
    "saved_model_file_count": 83,
    "contained_estimator_count": 103,
    "component_model_inventory": {
        "exp413": {"parent_selectors": 40, "signed_selectors": 20, "tvt_models": 15},
        "hjyact": {
            "trainer_wrapper_files": 5,
            "trainer_fold_estimators": 25,
            "learned_trajectory_models": 3,
            "model_package_models": 0,
        },
    },
    "model_package_correction_enabled": RUN_MODEL_PACKAGE_CORRECTION,
    "hjyact_input_audit": HJYACT_INPUT_AUDIT,
    "exp413_metrics": exp413_metrics,
}
(WORKING_DIR / "exp514_model_manifest.json").write_text(
    json.dumps(model_manifest, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)

metrics = {
    "experiment": "exp514_exp413_likpf_seed_bank_reuse_on_exp512",
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
    "wells": HJYACT_SHARED_FEATURE_WELL_COUNT,
    "new_booster_training_count": 0,
    "runtime_ridge_fit_count": 5,
    "runtime_optimizations": {
        "sp45_well_parallel": SP45_WELL_PARALLEL_REPORT,
        "shared_likpf_sp45_streaming": SHARED_SP45_STREAMING_REPORT,
        "ridge_memory_release": RIDGE_MEMORY_RELEASE_REPORT,
        "pre_exp413_memory_release": EXP413_PRE_RELEASE_REPORT,
        "shared_bank_final_release": SHARED_BANK_FINAL_RELEASE_REPORT,
        "dataframe_ownership_transfer": {
            "sp45_to_hjyact": True,
            "hjyact_to_exp413": True,
            "exp413_returned_prediction_frame": False,
        },
        "exp413_well_n_jobs": EXP413_WELL_N_JOBS,
        "model_package_correction_enabled": RUN_MODEL_PACKAGE_CORRECTION,
    },
    "submission_file_generated": True,
    "external_submission_performed": False,
    "candidate_reuse": {
        "node_count": len(SHARED_NODE_COLUMNS),
        "record_count": len(reuse_manifest["records"]),
        "fallback_to_duplicate_generation": False,
    },
    "shared_likelihood_pf": shared_likpf_manifest,
    "hjyact_deterministic_feature_reuse": HJYACT_DETERMINISTIC_REUSE_MANIFEST,
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
        "shared_likelihood_pf_manifest": sha256_file(SHARED_LIKPF_MANIFEST_PATH),
        "model_manifest": sha256_file(WORKING_DIR / "exp514_model_manifest.json"),
        "submission": sha256_file(FINAL_SUBMISSION_PATH),
    },
}
(WORKING_DIR / "metrics.json").write_text(
    json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n",
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
(WORKING_DIR / "exp514_reproducibility_manifest.json").write_text(
    json.dumps(reproducibility_manifest, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)
print("exp514 shared-PF fixed 50/50 submission generated:", FINAL_SUBMISSION_PATH, submission.shape)
print("external submission performed: False")
display(submission.head(20))
display(metrics)

# %% [markdown]
# ## 10. Stage D visible runtime and 200-well estimate

# %%
import resource as _stage_d_resource

_stage_d_visible_wells = int(len(test_wells))
_stage_d_visible_rows = int(len(submission))
if _stage_d_visible_wells < 1 or _stage_d_visible_rows < 1:
    raise RuntimeError("Stage D visible runtime report requires nonempty wells and rows")

_stage_d_gold_path = WORKING_DIR / "gold_prefix_submission_audit.json"
if bool(_GOLD_ENABLE) and not _stage_d_gold_path.is_file():
    raise RuntimeError("Stage D enabled Gold stage did not write its runtime audit")
_stage_d_gold_seconds = 0.0
if _stage_d_gold_path.is_file():
    _stage_d_gold_seconds = float(
        json.loads(_stage_d_gold_path.read_text(encoding="utf-8"))["elapsed_sec"]
    )

if HJYACT_LEARNED_TOTAL_RUNTIME_SECONDS is None:
    raise RuntimeError("Stage D learned trajectory total runtime was not recorded")

_stage_d_v2_equivalence_path = (
    WORKING_DIR / "exp514_stage_d_v2_output_equivalence.json"
)
if visible_reference_checks["sample_id_order_match"]:
    _stage_d_v2_equivalence_targets = {
        "gold_balanced": (
            WORKING_DIR / "submission_gold_prefix_balanced.csv",
            STAGE_D_V2_GOLD_BALANCED_SHA256,
        ),
        "hjyact_component": (
            HJYACT_COMPONENT_PATH,
            STAGE_D_VISIBLE_HJYACT_CANDIDATE_SHA256,
        ),
        "exp413_component": (
            EXP413_COMPONENT_PATH,
            STAGE_D_V2_EXP413_COMPONENT_SHA256,
        ),
        "component_readout": (
            WORKING_DIR / "exp514_component_readout.csv",
            STAGE_D_V2_COMPONENT_READOUT_SHA256,
        ),
        "final_submission": (
            FINAL_SUBMISSION_PATH,
            STAGE_D_V2_FINAL_SUBMISSION_SHA256,
        ),
    }
    _stage_d_v2_equivalence = {}
    for _stage_d_name, (_stage_d_path, _stage_d_expected_sha) in (
        _stage_d_v2_equivalence_targets.items()
    ):
        if not _stage_d_path.is_file():
            raise FileNotFoundError(
                f"Stage D v2 equivalence target missing: {_stage_d_name}: {_stage_d_path}"
            )
        _stage_d_observed_sha = sha256_file(_stage_d_path)
        _stage_d_match = _stage_d_observed_sha == _stage_d_expected_sha
        _stage_d_v2_equivalence[_stage_d_name] = {
            "path": str(_stage_d_path),
            "expected_sha256": _stage_d_expected_sha,
            "observed_sha256": _stage_d_observed_sha,
            "exact_match": bool(_stage_d_match),
        }
        if not _stage_d_match:
            raise RuntimeError(
                "Stage D v4 runtime-only output parity failed for "
                f"{_stage_d_name}: {_stage_d_observed_sha} != {_stage_d_expected_sha}"
            )
    _stage_d_v2_equivalence_manifest = {
        "reference_kernel": "kentookumura/exp514-shared-likpf-stage-d-visible",
        "reference_kernel_version": 2,
        "status": "PASS",
        "all_exact": True,
        "targets": _stage_d_v2_equivalence,
    }
else:
    _stage_d_v2_equivalence_manifest = {
        "reference_kernel": "kentookumura/exp514-shared-likpf-stage-d-visible",
        "reference_kernel_version": 2,
        "status": "SKIPPED_HIDDEN_DYNAMIC",
        "all_exact": None,
        "targets": {},
        "reason": "visible output SHA witnesses do not apply to hidden dynamic IDs",
    }
_stage_d_v2_equivalence_path.write_text(
    json.dumps(
        _stage_d_v2_equivalence_manifest,
        indent=2,
        sort_keys=True,
        default=str,
    )
    + "\n",
    encoding="utf-8",
)

_stage_d_parallel_stages = [
    {
        "name": "shared_likpf_sp45_streaming_pipeline",
        "seconds": float(SHARED_SP45_STREAMING_REPORT["elapsed_seconds"]),
        "scaling": "parallel_4",
    },
    {
        "name": "exp413_full_after_shared_pf",
        "seconds": float(exp413_metrics["runtime_seconds"]),
        "scaling": "parallel_4_mixed",
    },
    {
        "name": "gold_visible_prefix",
        "seconds": float(_stage_d_gold_seconds),
        "scaling": "parallel_4_process",
    },
]
_stage_d_sequential_stages = [
    {
        "name": "hjyact_learned_trajectory_total",
        "seconds": float(HJYACT_LEARNED_TOTAL_RUNTIME_SECONDS),
        "scaling": "sequential_visible_throughput_conservative",
    },
]
_stage_d_all_stages = _stage_d_parallel_stages + _stage_d_sequential_stages
if any(item["seconds"] < 0.0 for item in _stage_d_all_stages):
    raise RuntimeError("Stage D stage runtime cannot be negative")

_stage_d_started = float(
    globals().get("_KAGGLE_BOOTSTRAP_STARTED", STAGE_D_VISIBLE_STARTED)
)
_stage_d_total_seconds = float(time.time() - _stage_d_started)
_stage_d_known_seconds = float(sum(item["seconds"] for item in _stage_d_all_stages))
_stage_d_fixed_overhead_seconds = float(max(0.0, _stage_d_total_seconds - _stage_d_known_seconds))
_stage_d_target_wells = 200
_stage_d_parallel_workers = 4

_stage_d_lower_seconds = _stage_d_fixed_overhead_seconds
_stage_d_upper_seconds = _stage_d_fixed_overhead_seconds
for _stage_d_item in _stage_d_parallel_stages:
    _stage_d_lower_seconds += (
        _stage_d_item["seconds"] * _stage_d_target_wells / _stage_d_parallel_workers
    )
    _stage_d_upper_seconds += (
        _stage_d_item["seconds"] * _stage_d_target_wells / _stage_d_visible_wells
    )
for _stage_d_item in _stage_d_sequential_stages:
    _stage_d_scaled = (
        _stage_d_item["seconds"] * _stage_d_target_wells / _stage_d_visible_wells
    )
    _stage_d_lower_seconds += _stage_d_scaled
    _stage_d_upper_seconds += _stage_d_scaled

_stage_d_peak_rss_mib = float(
    _stage_d_resource.getrusage(_stage_d_resource.RUSAGE_SELF).ru_maxrss / 1024.0
)
_stage_d_estimated_pass = bool(_stage_d_upper_seconds <= 32400.0)
_stage_d_report = {
    "experiment": EXPERIMENT_NAME,
    "stage": "stage_d_submission_ready_visible_test",
    "status": "PASS" if FINAL_SUBMISSION_PATH.is_file() else "FAIL",
    "runtime_estimate_status": (
        "estimated_pass_not_hidden_runtime_guarantee"
        if _stage_d_estimated_pass
        else "estimated_fail"
    ),
    "base_candidate_source_sha256": STAGE_D_BASE_CANDIDATE_SHA256,
    "generator_sha256": STAGE_D_GENERATOR_SHA256,
    "visible": {
        "wells": _stage_d_visible_wells,
        "rows": _stage_d_visible_rows,
        "total_seconds_including_bootstrap": round(_stage_d_total_seconds, 6),
        "peak_parent_process_rss_mib": round(_stage_d_peak_rss_mib, 3),
        "peak_rss_scope": "parent_process_only_excludes_child_subprocess_peak",
    },
    "stage_timings": _stage_d_all_stages,
    "runtime_optimizations": {
        "gold_well_parallel": GOLD_WELL_PARALLEL_REPORT,
        "hjyact_deterministic_feature_reuse": HJYACT_DETERMINISTIC_REUSE_MANIFEST,
        "ridge_memory_release": RIDGE_MEMORY_RELEASE_REPORT,
        "pre_exp413_memory_release": EXP413_PRE_RELEASE_REPORT,
        "shared_bank_final_release": SHARED_BANK_FINAL_RELEASE_REPORT,
        "shared_likpf_sp45_streaming": SHARED_SP45_STREAMING_REPORT,
        "dataframe_ownership_transfer": True,
    },
    "v2_output_equivalence": _stage_d_v2_equivalence_manifest,
    "fixed_overhead_seconds": round(_stage_d_fixed_overhead_seconds, 6),
    "target_hidden_wells": _stage_d_target_wells,
    "estimation": {
        "parallel_worker_reference": _stage_d_parallel_workers,
        "parallel_lower_formula": "stage_seconds_times_200_div_4",
        "parallel_upper_formula": "stage_seconds_times_200_div_visible_wells",
        "sequential_formula": "stage_seconds_times_200_div_visible_wells",
        "fixed_overhead_policy": "add_once",
        "lower_seconds": round(_stage_d_lower_seconds, 6),
        "upper_seconds": round(_stage_d_upper_seconds, 6),
        "lower_hours": round(_stage_d_lower_seconds / 3600.0, 6),
        "upper_hours": round(_stage_d_upper_seconds / 3600.0, 6),
        "upper_limit_seconds": 32400,
        "hidden_runtime_observed": False,
        "uncertainty": "high_visible_workload_may_not_match_hidden",
    },
    "submission": {
        "generated": bool(FINAL_SUBMISSION_PATH.is_file()),
        "rows": int(len(submission)),
        "sha256": sha256_file(FINAL_SUBMISSION_PATH),
        "external_submission_performed": False,
    },
}
_stage_d_report_path = WORKING_DIR / "exp514_stage_d_visible_runtime_report.json"
_stage_d_report_path.write_text(
    json.dumps(_stage_d_report, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)
metrics["stage_d_visible_runtime"] = _stage_d_report
metrics["sha256"]["stage_d_visible_runtime_report"] = sha256_file(
    _stage_d_report_path
)
metrics["sha256"]["stage_d_v2_output_equivalence"] = sha256_file(
    _stage_d_v2_equivalence_path
)
(WORKING_DIR / "metrics.json").write_text(
    json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)
_stage_d_repro_path = WORKING_DIR / "exp514_reproducibility_manifest.json"
if _stage_d_repro_path.is_file():
    _stage_d_repro = json.loads(_stage_d_repro_path.read_text(encoding="utf-8"))
    _stage_d_repro["stage_d_visible_runtime"] = _stage_d_report
    _stage_d_repro_path.write_text(
        json.dumps(_stage_d_repro, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

print("Stage D visible runtime report:", json.dumps(_stage_d_report, sort_keys=True))
print("Stage D external submission performed: False")
display(_stage_d_report)
