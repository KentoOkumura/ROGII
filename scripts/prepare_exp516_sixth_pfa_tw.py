from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP_NAME = "exp516_sixth_place_pfa_tw_late_submit"
EXP_DIR = ROOT / "experiments" / EXP_NAME
REFERENCE_NOTEBOOK = (
    ROOT
    / "docs/notebooks/rogii-wellbore-geology-prediction/solution_6th"
    / "k256net__public20th-private6th-pf-pf-pf-pf-and-bagging"
    / "public20th-private6th-pf-pf-pf-pf-and-bagging.ipynb"
)
PUBLIC_CONFIG = EXP_DIR / "pf_banks_config_v96_public.json"
OUTPUT_SOURCE = EXP_DIR / f"{EXP_NAME}_compact_selfcontained_inference.py"

EXPECTED_REFERENCE_SHA = "b44f7889d6abdf9b027d33cb6c6b45f23902d609fae8d06f332914017784c924"
EXPECTED_CONFIG_SHA = "80e973d5f5e0e39be758a03f399cdd3d81d9e79320da8db6fbddbc25c2a202f3"
CHECKPOINT_SHA = {
    "stageA_enccapaug_f0.pt": "9ce3763b14a68ae5f05e78467ae4faa5b696ab714c270fd129a6e7d468cdd007",
    "stageA_enccapaug_f1.pt": "d6b1dc0956e47b778bb2089c644d71c9006fdf8948a9cc962c6a40d27dccdfec",
    "stageA_enccapaug_f2.pt": "2f312ce8de05feab9d8480a7971d91eb4e8289cce7e654f895f1b63d95cf5208",
    "stageA_enccapaug_f3.pt": "8bf70f128c12a86e6f65177912d7ee90b5d24a7a67caf91f40df992bdd2c599f",
    "stageA_enccapaug_f4.pt": "b438d113b8fc07d8e9842cb66d01cfff2971ef4a5d213841574ec2680b2ab170",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def writefile_cell(notebook: dict, filename: str) -> str:
    prefix = f"%%writefile {filename}\n"
    matches: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source_raw = cell.get("source", "")
        source = "".join(source_raw) if isinstance(source_raw, list) else str(source_raw)
        if source.startswith(prefix):
            matches.append(source[len(prefix) :])
    if len(matches) != 1:
        raise RuntimeError(f"expected one {filename} writefile cell, found {len(matches)}")
    return matches[0]


def strip_main_block(source: str) -> str:
    marker = '\nif __name__ == "__main__":'
    pos = source.rfind(marker)
    if pos < 0:
        raise RuntimeError("public helper has no terminal __main__ block")
    return source[:pos].rstrip() + "\n"


def adapt_emission_paths(source: str) -> str:
    old = (
        'PROJ = Path(r"C:\\Users\\kosaka256\\Documents\\rogii_claude")\n'
        'DATA_DIR = PROJ / "rogii-wellbore-geology-prediction"\n'
    )
    new = (
        'PROJ = Path(os.environ.get("ROGII_PROJ", "."))\n'
        'DATA_DIR = Path(os.environ.get("ROGII_DATA", str(PROJ / "rogii-wellbore-geology-prediction")))\n'
    )
    if source.count(old) != 1:
        raise RuntimeError("unexpected public nn_emission path resolver")
    return source.replace(old, new)


def markdown_cell(title: str, body: str) -> str:
    lines = ["# %% [markdown]", f"# ## {title}"]
    lines.extend(f"# {line}" if line else "#" for line in body.splitlines())
    return "\n".join(lines) + "\n\n"


def code_cell(source: str) -> str:
    return "# %%\n" + source.rstrip() + "\n\n"


def build_source(anchor_source: str, emission_source: str, pf_source: str, config_text: str) -> str:
    anchor_sha = sha256_bytes(anchor_source.encode())
    config_text_sha = sha256_bytes(config_text.encode())
    emission_raw_sha = sha256_bytes(emission_source.encode())
    emission_adapted = adapt_emission_paths(strip_main_block(emission_source))
    emission_adapted_sha = sha256_bytes(emission_adapted.encode())
    pf_core = strip_main_block(pf_source)
    pf_sha = sha256_bytes(pf_source.encode())

    parts = [
        "# %% [markdown]\n"
        "# # LATE SUBMIT — exp516 6th-place pfA × twGR faithful replay\n"
        "#\n"
        "# This is a post-competition reproduction audit, not an official-place submission.\n"
        "# It replays only the published standalone pfA × typewell-GR component.\n\n",
        "# %% [markdown]\n"
        "# ## Contents\n"
        "# 1. Imports and immutable source contract\n"
        "# 2. Runtime, input, and checkpoint guards\n"
        "# 3. Public GR-free anchor generator\n"
        "# 4. Generate fold-safe GR-free hidden-test anchor\n"
        "# 5. Public learned-emission helpers\n"
        "# 6. Generate hidden-test similarity bands\n"
        "# 7. Public GPU particle-filter and whole-smoother engine\n"
        "# 8. Run pfA × twGR only\n"
        "# 9. Sample-ID alignment, manifests, and LATE SUBMIT output\n\n",
        markdown_cell(
            "1. Imports and immutable source contract",
            "The public notebook/config/checkpoint identities are frozen before any prediction. "
            "Reported author scores are external references, never outputs of this run.",
        ),
        code_cell(
            f'''from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EXPERIMENT = "{EXP_NAME}"
LATE_SUBMISSION_PHASE = "post_competition_late_submission"
PUBLIC_KERNEL = "k256net/public20th-private6th-pf-pf-pf-pf-and-bagging"
PUBLIC_KERNEL_ID_NO = 126919690
PUBLIC_NOTEBOOK_SHA256 = "{EXPECTED_REFERENCE_SHA}"
PUBLIC_CONFIG_SHA256 = "{EXPECTED_CONFIG_SHA}"
PUBLIC_CONFIG_TEXT_SHA256 = "{config_text_sha}"
PUBLIC_ANCHOR_SOURCE_SHA256 = "{anchor_sha}"
PUBLIC_EMISSION_SOURCE_SHA256 = "{emission_raw_sha}"
ADAPTED_EMISSION_SOURCE_SHA256 = "{emission_adapted_sha}"
PUBLIC_PF_SOURCE_SHA256 = "{pf_sha}"
PF_BANK = "pfA"
PF_REPRESENTATION = "tw"
PF_GENERATION_SEED = 4423098
PF_N_PARTICLES = 600
PF_N_SEEDS = 32
PF_SMOOTH_MODE = "full"
PF_WELL_CHUNK = 40
EXPECTED_CHECKPOINT_SHA256 = {CHECKPOINT_SHA!r}
PUBLIC_CONFIG_JSON = {config_text!r}


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def logical_csv_sha(frame: pd.DataFrame) -> str:
    return sha256_text(frame.to_csv(index=False, lineterminator="\\n"))
'''
        ),
        markdown_cell(
            "2. Runtime, input, and checkpoint guards",
            "The current competition train/test and sample submission are discovered dynamically. "
            "The five public encoder files must match frozen SHA256 values; there is no emission-off fallback.",
        ),
        code_cell(
            '''def resolve_competition_root() -> Path:
    roots = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    ]
    roots.extend(path.parent for path in Path("/kaggle/input").rglob("sample_submission.csv"))
    local = Path("data/raw").resolve()
    roots.append(local)
    for root in roots:
        if (root / "train").is_dir() and (root / "test").is_dir() and (root / "sample_submission.csv").is_file():
            return root
    raise FileNotFoundError("competition root with train/test/sample_submission.csv was not found")


def resolve_public_checkpoints(input_root: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for name, expected_sha in EXPECTED_CHECKPOINT_SHA256.items():
        exact = []
        for path in sorted(input_root.rglob(name)):
            if path.is_file() and sha256_path(path) == expected_sha:
                exact.append(path)
        if len(exact) != 1:
            raise RuntimeError(f"expected exactly one exact public checkpoint {name}, found {len(exact)}")
        resolved[name] = exact[0]
    return resolved


DATA_ROOT = resolve_competition_root()
SAMPLE_PATH = DATA_ROOT / "sample_submission.csv"
WORK_ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".").resolve()
ART_ROOT = WORK_ROOT / "exp516_public_pfa_artifacts"
NN50_ROOT = ART_ROOT / "nn50"
ART_ROOT.mkdir(parents=True, exist_ok=True)
NN50_ROOT.mkdir(parents=True, exist_ok=True)

if hashlib.sha256(PUBLIC_CONFIG_JSON.encode("utf-8")).hexdigest() != PUBLIC_CONFIG_TEXT_SHA256:
    raise RuntimeError("embedded public v96 config SHA drift")
config_path = ART_ROOT / "pf_banks_config.json"
config_path.write_text(PUBLIC_CONFIG_JSON, encoding="utf-8")

checkpoints = resolve_public_checkpoints(Path("/kaggle/input"))
for name, source in checkpoints.items():
    target = NN50_ROOT / name
    shutil.copy2(source, target)
    if sha256_path(target) != EXPECTED_CHECKPOINT_SHA256[name]:
        raise RuntimeError(f"checkpoint copy SHA mismatch: {name}")

gpu_count = torch.cuda.device_count()
gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
if gpu_count != 2 or not all("T4" in name for name in gpu_names):
    raise RuntimeError(f"official faithful run requires T4 x2, found {gpu_count}: {gpu_names}")

os.environ.update({
    "ROGII_DATA": str(DATA_ROOT),
    "ROGII_PROJ": str(WORK_ROOT),
    "ROGII_ART95": str(ART_ROOT),
    "ROGII_ART97": str(ART_ROOT),
    "PF_NGPU": "2",
    "PF_WELL_CHUNK": str(PF_WELL_CHUNK),
    "FULL_VRAM_GB": "8.0",
    "PYTHONUNBUFFERED": "1",
})

sample = pd.read_csv(SAMPLE_PATH)
if len(sample.columns) != 2 or list(sample.columns)[:1] != ["id"]:
    raise RuntimeError(f"sample submission must contain id plus one target, got {list(sample.columns)}")
TARGET_COLUMN = str(sample.columns[1])
if sample["id"].astype(str).duplicated().any():
    raise RuntimeError("sample submission contains duplicate ids")
print({
    "experiment": EXPERIMENT,
    "phase": LATE_SUBMISSION_PHASE,
    "data_root": str(DATA_ROOT),
    "sample_rows": int(len(sample)),
    "gpu_names": gpu_names,
    "public_kernel": PUBLIC_KERNEL,
})
'''
        ),
        markdown_cell(
            "3. Public GR-free anchor generator",
            "This source is copied verbatim from the public submission notebook. It trains the fold-safe field/GRU anchor "
            "with five folds and three seeds; it is executed in a subprocess because the public script terminates with os._exit(0).",
        ),
        code_cell(f"PUBLIC_ANCHOR_SOURCE = {anchor_source!r}\n"),
        markdown_cell(
            "4. Generate fold-safe GR-free hidden-test anchor",
            "The saved fold artifact contains the five field/pool states and 15 GRU state dicts. "
            "The current hidden test anchor is regenerated from runtime inputs; no visible-test anchor is reused.",
        ),
        code_cell(
            '''anchor_script = WORK_ROOT / "exp516_public_gen_grfree_anchor.py"
anchor_script.write_text(PUBLIC_ANCHOR_SOURCE, encoding="utf-8")
if sha256_path(anchor_script) != PUBLIC_ANCHOR_SOURCE_SHA256:
    raise RuntimeError("public anchor source SHA mismatch after materialization")

anchor_path = ART_ROOT / "grfree_anchor_test.pkl"
fold_artifact_path = ART_ROOT / "grfree_fold_art.pkl"
anchor_env = dict(os.environ)
anchor_env.update({
    "GRF_SPLIT": "test",
    "GRF_ANCHOR_OUT": str(anchor_path),
    "GRFREE_FOLD_ART": str(fold_artifact_path),
    "FORCE_FOLD_ART": "1",
    "FT_EP": "10",
    "SEEDS": "3",
    "SPLIT_SEED": "42",
})
anchor_started = time.perf_counter()
subprocess.run([sys.executable, str(anchor_script)], env=anchor_env, check=True)
anchor_seconds = time.perf_counter() - anchor_started
if not anchor_path.is_file() or not fold_artifact_path.is_file():
    raise RuntimeError("public anchor generator did not produce both anchor and fold artifact")
anchor_payload = pickle.loads(anchor_path.read_bytes())
os.environ["NN_SPLIT"] = "test"
os.environ["V93_ANCHOR_PKL"] = str(anchor_path)
print({
    "anchor_wells": len(anchor_payload),
    "anchor_seconds": anchor_seconds,
    "anchor_sha256": sha256_path(anchor_path),
    "fold_artifact_sha256": sha256_path(fold_artifact_path),
})
'''
        ),
        markdown_cell(
            "5. Public learned-emission helpers",
            "The encoder architecture and similarity-band construction are the public implementation. "
            "Only the hard-coded Windows project/data resolver is replaced by ROGII_PROJ/ROGII_DATA for Kaggle portability.",
        ),
        code_cell(emission_adapted),
        markdown_cell(
            "6. Generate hidden-test similarity bands",
            "All five frozen public encoders are averaged for test wells around the GR-free anchor at 0.5-ft spacing over ±45 ft.",
        ),
        code_cell(
            '''similarity_started = time.perf_counter()
SIMD = build_sims(verify_cache=False, split="test")
similarity_seconds = time.perf_counter() - similarity_started
similarity_path = ART_ROOT / "sim_grfree_test_v97.pkl"
if not similarity_path.is_file():
    raise RuntimeError("public learned-emission builder did not save similarity artifact")
if len(SIMD) != len(anchor_payload):
    missing = sorted(set(anchor_payload) - set(SIMD))
    raise RuntimeError(f"learned emission missing anchor wells: {missing[:10]}")
print({
    "similarity_wells": len(SIMD),
    "similarity_seconds": similarity_seconds,
    "similarity_sha256": sha256_path(similarity_path),
})
'''
        ),
        markdown_cell(
            "7. Public GPU particle-filter and whole-smoother engine",
            "This is the public pf_banks_v95 numerical engine with only its terminal standalone smoke block removed. "
            "The public v96 config and freshly generated anchor are loaded from ART_ROOT.",
        ),
        code_cell(pf_core),
        markdown_cell(
            "8. Run pfA × twGR only",
            "Exactly one bank and one representation are allowed. Each well uses 600 particles × 32 seeds, "
            "full ancestral smoothing, physical anchor mult 20, and learned-emission weight 0.01.",
        ),
        code_cell(
            '''public_config = json.loads(config_path.read_text(encoding="utf-8"))
if public_config["bank_order"] != ["pf_1", "pf_2", "pf_3", "r0_seed32", "r1_seed32", "pfA"]:
    raise RuntimeError("unexpected public v96 bank order")
P = bank_param(PF_BANK)
expected_contract = {
    "n_particles": PF_N_PARTICLES,
    "smooth_mode": PF_SMOOTH_MODE,
    "anchor_mult": 20.0,
    "grid_step": 0.2,
}
for key, expected in expected_contract.items():
    actual = P.get(key)
    if actual != expected:
        raise RuntimeError(f"pfA contract drift for {key}: {actual!r} != {expected!r}")
if N_SEED != PF_N_SEEDS or P["_w_nn"] != 0.01 or not P["_physics"]:
    raise RuntimeError("pfA seed/emission/anchor contract drift")

sample_keys = sample["id"].astype(str)
parsed = sample_keys.str.rsplit("_", n=1, expand=True)
if parsed.shape[1] != 2:
    raise RuntimeError("sample ids must end in an underscore-separated row index")
sample_wells = parsed[0].astype(str)
try:
    sample_rows = parsed[1].astype(int)
except Exception as exc:
    raise RuntimeError("sample id row suffix is not integer") from exc

test_dir = DATA_ROOT / "test"
available_wells = sorted({path.name.split("__", 1)[0] for path in test_dir.glob("*__horizontal_well.csv")})
requested_wells = sorted(sample_wells.unique())
if set(requested_wells) - set(available_wells):
    raise RuntimeError(f"sample wells missing from runtime test files: {sorted(set(requested_wells)-set(available_wells))}")

names: list[str] = []
inps: list[dict] = []
eval_rows_by_well: dict[str, np.ndarray] = {}
for wid in requested_wells:
    hw = pd.read_csv(test_dir / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(test_dir / f"{wid}__typewell.csv").sort_values("TVT")
    for column in ["MD", "Z", "GR", "TVT_input"]:
        if column not in hw.columns:
            raise RuntimeError(f"{wid} horizontal well missing {column}")
    for column in ["TVT", "GR"]:
        if column not in tw.columns:
            raise RuntimeError(f"{wid} typewell missing {column}")
    x = build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P)
    if x is None:
        raise RuntimeError(f"{wid} has no hidden suffix for PF")
    x = attach_anchor(x, wid, P["_physics"])
    similarity = SIMD.get(wid)
    if similarity is None or len(similarity.get("st", [])) != len(x["md"]):
        raise RuntimeError(f"{wid} learned-emission similarity missing or row-mismatched")
    x["_sim"] = similarity["sim"]
    x["_st"] = similarity["st"]
    names.append(wid)
    inps.append(x)
    eval_rows_by_well[wid] = np.flatnonzero(hw["TVT_input"].isna().to_numpy())

pf_started = time.perf_counter()
outs = run_smoother_ext(
    inps,
    P,
    seed=PF_GENERATION_SEED,
    n_seeds=PF_N_SEEDS,
    chunk=PF_WELL_CHUNK,
    w_nn=P["_w_nn"],
)
pf_seconds = time.perf_counter() - pf_started
if len(outs) != len(names) or any(output is None for output in outs):
    raise RuntimeError("PF did not return exactly one output for every requested well")

prediction_lookup: dict[str, tuple[float, float, float]] = {}
for wid, rows, output in zip(names, (eval_rows_by_well[w] for w in names), outs):
    mean = np.asarray(output["mean"], dtype=np.float64)
    std = np.asarray(output["std"], dtype=np.float64)
    if len(mean) != len(rows) or len(std) != len(rows):
        raise RuntimeError(f"{wid} PF output length mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError(f"{wid} PF output contains non-finite values")
    loglik = float(output["loglik"])
    for row_idx, mean_value, std_value in zip(rows, mean, std):
        prediction_lookup[f"{wid}_{int(row_idx)}"] = (float(mean_value), float(std_value), loglik)

expected_ids = set(sample_keys)
actual_ids = set(prediction_lookup)
if expected_ids != actual_ids:
    raise RuntimeError(
        f"PF/sample id mismatch: missing={len(expected_ids-actual_ids)} extra={len(actual_ids-expected_ids)}"
    )
print({
    "bank": PF_BANK,
    "representation": PF_REPRESENTATION,
    "wells": len(names),
    "rows": len(prediction_lookup),
    "particles": PF_N_PARTICLES,
    "seeds": PF_N_SEEDS,
    "smooth_mode": PF_SMOOTH_MODE,
    "pf_seconds": pf_seconds,
})
'''
        ),
        markdown_cell(
            "9. Sample-ID alignment, manifests, and LATE SUBMIT output",
            "The runtime sample submission is the sole schema/order authority. The direct pfA smoothed mean is written without "
            "fusion, gain, or visible-test branch. Logical candidate content and final submission receive SHA256 manifests.",
        ),
        code_cell(
            '''candidate = pd.DataFrame({
    "id": sample_keys,
    "pfa_tw_mean": [prediction_lookup[key][0] for key in sample_keys],
    "pfa_tw_std": [prediction_lookup[key][1] for key in sample_keys],
    "pfa_tw_loglik": [prediction_lookup[key][2] for key in sample_keys],
})
submission = sample[["id"]].copy()
submission[TARGET_COLUMN] = candidate["pfa_tw_mean"].to_numpy(dtype=np.float64)

if len(submission) != len(sample) or not submission["id"].equals(sample["id"]):
    raise RuntimeError("submission id/order/row contract failed")
if submission["id"].astype(str).duplicated().any():
    raise RuntimeError("submission contains duplicate ids")
if not np.isfinite(submission[TARGET_COLUMN].to_numpy(dtype=np.float64)).all():
    raise RuntimeError("submission contains non-finite predictions")

candidate_content = candidate.to_csv(index=False, lineterminator="\\n")
candidate_gzip_path = WORK_ROOT / "pfa_tw_candidate.csv.gz"
with gzip.open(candidate_gzip_path, "wt", encoding="utf-8", newline="") as handle:
    handle.write(candidate_content)
submission_path = WORK_ROOT / "submission.csv"
submission.to_csv(submission_path, index=False, lineterminator="\\n")

checkpoint_manifest = [
    {
        "name": name,
        "source_path": str(checkpoints[name]),
        "sha256": sha256_path(NN50_ROOT / name),
        "bytes": int((NN50_ROOT / name).stat().st_size),
    }
    for name in sorted(checkpoints)
]
model_manifest = {
    "experiment": EXPERIMENT,
    "late_submission_phase": LATE_SUBMISSION_PHASE,
    "public_encoder_models": checkpoint_manifest,
    "anchor_fold_artifact": {
        "path": str(fold_artifact_path),
        "sha256": sha256_path(fold_artifact_path),
        "bytes": int(fold_artifact_path.stat().st_size),
        "folds": 5,
        "seeds_per_fold": 3,
        "model_count": 15,
        "loss": "masked_huber_delta_8ft",
    },
    "pf": {
        "bank": PF_BANK,
        "representation": PF_REPRESENTATION,
        "config_sha256": sha256_path(config_path),
        "particles": PF_N_PARTICLES,
        "seeds": PF_N_SEEDS,
        "smooth_mode": PF_SMOOTH_MODE,
        "generation_seed": PF_GENERATION_SEED,
    },
}
(WORK_ROOT / "model_manifest.json").write_text(json.dumps(model_manifest, indent=2, sort_keys=True), encoding="utf-8")

manifest = {
    "experiment": EXPERIMENT,
    "late_submission": True,
    "late_submission_phase": LATE_SUBMISSION_PHASE,
    "submission_message": "LATE SUBMIT | exp516 | 6th-place pfA x twGR standalone faithful replay | fixed v1",
    "source": {
        "public_kernel": PUBLIC_KERNEL,
        "public_kernel_id_no": PUBLIC_KERNEL_ID_NO,
        "public_notebook_sha256": PUBLIC_NOTEBOOK_SHA256,
        "public_anchor_source_sha256": PUBLIC_ANCHOR_SOURCE_SHA256,
        "public_emission_source_sha256": PUBLIC_EMISSION_SOURCE_SHA256,
        "adapted_emission_source_sha256": ADAPTED_EMISSION_SOURCE_SHA256,
        "public_pf_source_sha256": PUBLIC_PF_SOURCE_SHA256,
        "public_config_sha256": PUBLIC_CONFIG_SHA256,
        "public_config_embedded_text_sha256": PUBLIC_CONFIG_TEXT_SHA256,
    },
    "runtime": {
        "gpu_names": gpu_names,
        "anchor_seconds": anchor_seconds,
        "similarity_seconds": similarity_seconds,
        "pf_seconds": pf_seconds,
        "well_count": len(names),
        "row_count": len(candidate),
    },
    "artifacts": {
        "anchor_sha256": sha256_path(anchor_path),
        "anchor_fold_artifact_sha256": sha256_path(fold_artifact_path),
        "similarity_sha256": sha256_path(similarity_path),
        "candidate_decompressed_content_sha256": sha256_text(candidate_content),
        "candidate_raw_gzip_sha256": sha256_path(candidate_gzip_path),
        "prediction_content_sha256": logical_csv_sha(candidate[["id", "pfa_tw_mean"]]),
        "submission_sha256": sha256_path(submission_path),
    },
    "submission_contract": {
        "columns": list(submission.columns),
        "rows": int(len(submission)),
        "duplicate_ids": int(submission["id"].astype(str).duplicated().sum()),
        "missing_predictions": int(submission[TARGET_COLUMN].isna().sum()),
        "finite": bool(np.isfinite(submission[TARGET_COLUMN].to_numpy(dtype=np.float64)).all()),
        "sample_id_order_exact": bool(submission["id"].equals(sample["id"])),
    },
    "prediction_summary": {
        "mean": float(submission[TARGET_COLUMN].mean()),
        "std": float(submission[TARGET_COLUMN].std()),
        "min": float(submission[TARGET_COLUMN].min()),
        "max": float(submission[TARGET_COLUMN].max()),
    },
    "deterministic_anchor": False,
    "deterministic_reason": "GPU fixed-environment rerun equality not yet tested",
}
(WORK_ROOT / "exp516_execution_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
)
print(json.dumps({
    "LATE_SUBMIT": True,
    "submission": str(submission_path),
    "submission_sha256": manifest["artifacts"]["submission_sha256"],
    "candidate_content_sha256": manifest["artifacts"]["candidate_decompressed_content_sha256"],
    "contract": manifest["submission_contract"],
}, indent=2))
'''
        ),
    ]
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if sha256_path(REFERENCE_NOTEBOOK) != EXPECTED_REFERENCE_SHA:
        raise RuntimeError("public reference notebook SHA drift")
    if sha256_path(PUBLIC_CONFIG) != EXPECTED_CONFIG_SHA:
        raise RuntimeError("public v96 config SHA drift")
    notebook = json.loads(REFERENCE_NOTEBOOK.read_text(encoding="utf-8"))
    anchor_source = writefile_cell(notebook, "gen_grfree_anchor.py")
    emission_source = writefile_cell(notebook, "nn_emission_v97.py")
    pf_source = writefile_cell(notebook, "pf_banks_v95.py")
    generated = build_source(
        anchor_source=anchor_source,
        emission_source=emission_source,
        pf_source=pf_source,
        config_text=PUBLIC_CONFIG.read_text(encoding="utf-8"),
    )

    if args.check:
        if not OUTPUT_SOURCE.is_file() or OUTPUT_SOURCE.read_text(encoding="utf-8") != generated:
            raise SystemExit(f"generated source drift: run {Path(__file__).relative_to(ROOT)}")
        print(f"PASS {OUTPUT_SOURCE.relative_to(ROOT)} {sha256_bytes(generated.encode())}")
        return
    OUTPUT_SOURCE.write_text(generated, encoding="utf-8")
    print(f"wrote {OUTPUT_SOURCE.relative_to(ROOT)}")
    print(f"sha256 {sha256_bytes(generated.encode())}")


if __name__ == "__main__":
    main()
