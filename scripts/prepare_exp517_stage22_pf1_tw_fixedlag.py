from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP_NAME = "exp517_stage22_pf1_tw_fixedlag192_late_submit"
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


def markdown_cell(title: str, body: str) -> str:
    lines = ["# %% [markdown]", f"# ## {title}"]
    lines.extend(f"# {line}" if line else "#" for line in body.splitlines())
    return "\n".join(lines) + "\n\n"


def code_cell(source: str) -> str:
    return "# %%\n" + source.rstrip() + "\n\n"


def build_source(pf_source: str, config_text: str) -> str:
    config_text_sha = sha256_bytes(config_text.encode())
    pf_core = strip_main_block(pf_source)
    pf_sha = sha256_bytes(pf_source.encode())

    parts = [
        "# %% [markdown]\n"
        "# # LATE SUBMIT — exp517 stage 2-2 pf_1 × twGR fixed-lag-192 proxy\n"
        "#\n"
        "# This is a post-competition component audit, not an official-place submission.\n"
        "# The published stage 2-2 score used five PF inputs plus a tabular model; this notebook emits one PF path only.\n\n",
        "# %% [markdown]\n"
        "# ## Contents\n"
        "# 1. Imports and immutable proxy contract\n"
        "# 2. Runtime and dynamic input guards\n"
        "# 3. Public GPU particle-filter and fixed-lag smoother engine\n"
        "# 4. Run pf_1 × twGR × fixed-lag 192 only\n"
        "# 5. Sample-ID alignment, manifests, and LATE SUBMIT output\n\n",
        markdown_cell(
            "1. Imports and immutable proxy contract",
            "The public final notebook and v96 config are SHA-pinned. Stage 2-2 exact parameters are not public, so the final "
            "released pf_1 parameters are an explicitly approved proxy. No author stage 2-2 score is treated as this run's target.",
        ),
        code_cell(
            f'''from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EXPERIMENT = "{EXP_NAME}"
LATE_SUBMISSION_PHASE = "post_competition_late_submission"
FIDELITY = "proxy"
PUBLIC_KERNEL = "k256net/public20th-private6th-pf-pf-pf-pf-and-bagging"
PUBLIC_KERNEL_ID_NO = 126919690
PUBLIC_NOTEBOOK_SHA256 = "{EXPECTED_REFERENCE_SHA}"
PUBLIC_CONFIG_SHA256 = "{EXPECTED_CONFIG_SHA}"
PUBLIC_CONFIG_TEXT_SHA256 = "{config_text_sha}"
PUBLIC_PF_SOURCE_SHA256 = "{pf_sha}"
PF_BANK = "pf_1"
PF_REPRESENTATION = "tw"
PF_GENERATION_SEED = 4423098
PF_N_PARTICLES = 600
PF_N_SEEDS = 32
PF_SMOOTH_MODE = "fixedlag"
PF_SMOOTH_LAG = 192
PF_WELL_CHUNK = 40
PF_PHYSICS_ENABLED = False
PF_EMISSION_WEIGHT = 0.0
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
            "2. Runtime and dynamic input guards",
            "The current competition test and sample submission are discovered dynamically. An empty anchor payload is "
            "materialized only because the released PF module loads that file at import; pf_1 is non-physical and never consumes it.",
        ),
        code_cell(
            '''def resolve_competition_root() -> Path:
    roots = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    ]
    roots.extend(path.parent for path in Path("/kaggle/input").rglob("sample_submission.csv"))
    roots.append(Path("data/raw").resolve())
    for root in roots:
        if (root / "test").is_dir() and (root / "sample_submission.csv").is_file():
            return root
    raise FileNotFoundError("competition root with test/sample_submission.csv was not found")


DATA_ROOT = resolve_competition_root()
SAMPLE_PATH = DATA_ROOT / "sample_submission.csv"
WORK_ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".").resolve()
ART_ROOT = WORK_ROOT / "exp517_stage22_pf1_artifacts"
ART_ROOT.mkdir(parents=True, exist_ok=True)

if hashlib.sha256(PUBLIC_CONFIG_JSON.encode("utf-8")).hexdigest() != PUBLIC_CONFIG_TEXT_SHA256:
    raise RuntimeError("embedded public v96 config SHA drift")
config_path = ART_ROOT / "pf_banks_config.json"
config_path.write_text(PUBLIC_CONFIG_JSON, encoding="utf-8")
empty_anchor_path = ART_ROOT / "empty_anchor.pkl"
empty_anchor_path.write_bytes(pickle.dumps({}, protocol=4))

gpu_count = torch.cuda.device_count()
gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
if gpu_count != 2 or not all("T4" in name for name in gpu_names):
    raise RuntimeError(f"fixed proxy run requires T4 x2, found {gpu_count}: {gpu_names}")

os.environ.update({
    "ROGII_DATA": str(DATA_ROOT),
    "ROGII_PROJ": str(WORK_ROOT),
    "ROGII_ART95": str(ART_ROOT),
    "V93_ANCHOR_PKL": str(empty_anchor_path),
    "PF_NGPU": "2",
    "PF_WELL_CHUNK": str(PF_WELL_CHUNK),
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
    "fidelity": FIDELITY,
    "data_root": str(DATA_ROOT),
    "sample_rows": int(len(sample)),
    "gpu_names": gpu_names,
    "public_kernel": PUBLIC_KERNEL,
})
'''
        ),
        markdown_cell(
            "3. Public GPU particle-filter and fixed-lag smoother engine",
            "This is the released pf_banks_v95 numerical engine with only its terminal smoke block removed. The runtime path "
            "below forces its fixed-lag branch; the included whole-interval implementation is not executed.",
        ),
        code_cell(pf_core),
        markdown_cell(
            "4. Run pf_1 × twGR × fixed-lag 192 only",
            "Exactly one final-public pf_1 parameter set and one typewell-GR representation are allowed. Anchor, learned "
            "emission, self/neighbor representations, full smoothing, tabular fusion, and post-processing are disabled.",
        ),
        code_cell(
            '''public_config = json.loads(config_path.read_text(encoding="utf-8"))
if public_config["bank_order"] != ["pf_1", "pf_2", "pf_3", "r0_seed32", "r1_seed32", "pfA"]:
    raise RuntimeError("unexpected public v96 bank order")
P = bank_param(PF_BANK)
P["smooth_mode"] = PF_SMOOTH_MODE
P["smooth_lag"] = PF_SMOOTH_LAG
P["_physics"] = PF_PHYSICS_ENABLED
P["_w_nn"] = PF_EMISSION_WEIGHT
P["_ps_combo_tau"] = 0.0
expected_contract = {
    "n_particles": PF_N_PARTICLES,
    "smooth_mode": PF_SMOOTH_MODE,
    "smooth_lag": PF_SMOOTH_LAG,
    "use_anchor": False,
    "use_phys": False,
    "_physics": False,
    "_w_nn": 0.0,
}
for key, expected in expected_contract.items():
    actual = P.get(key)
    if actual != expected:
        raise RuntimeError(f"pf_1 proxy contract drift for {key}: {actual!r} != {expected!r}")
if N_SEED != PF_N_SEEDS:
    raise RuntimeError(f"PF seed-count drift: {N_SEED} != {PF_N_SEEDS}")

sample_keys = sample["id"].astype(str)
parsed = sample_keys.str.rsplit("_", n=1, expand=True)
if parsed.shape[1] != 2:
    raise RuntimeError("sample ids must end in an underscore-separated row index")
sample_wells = parsed[0].astype(str)
try:
    parsed_row_ids = parsed[1].astype(int)
except Exception as exc:
    raise RuntimeError("sample id row suffix is not integer") from exc
if parsed_row_ids.isna().any():
    raise RuntimeError("sample id row suffix contains missing values")

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
    x = attach_anchor(x, wid, physics=False)
    if "_sim" in x or "_st" in x:
        raise RuntimeError(f"{wid} unexpectedly contains learned-emission inputs")
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
    w_nn=PF_EMISSION_WEIGHT,
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
    "smooth_lag": PF_SMOOTH_LAG,
    "physics": PF_PHYSICS_ENABLED,
    "emission_weight": PF_EMISSION_WEIGHT,
    "pf_seconds": pf_seconds,
})
'''
        ),
        markdown_cell(
            "5. Sample-ID alignment, manifests, and LATE SUBMIT output",
            "The runtime sample submission is the sole schema/order authority. The direct fixed-lag pf_1 mean is written "
            "without fusion, gain, or visible-test branch, with logical candidate and submission SHA manifests.",
        ),
        code_cell(
            '''candidate = pd.DataFrame({
    "id": sample_keys,
    "pf1_tw_fl192_mean": [prediction_lookup[key][0] for key in sample_keys],
    "pf1_tw_fl192_std": [prediction_lookup[key][1] for key in sample_keys],
    "pf1_tw_fl192_loglik": [prediction_lookup[key][2] for key in sample_keys],
})
submission = sample[["id"]].copy()
submission[TARGET_COLUMN] = candidate["pf1_tw_fl192_mean"].to_numpy(dtype=np.float64)

if len(submission) != len(sample) or not submission["id"].equals(sample["id"]):
    raise RuntimeError("submission id/order/row contract failed")
if submission["id"].astype(str).duplicated().any():
    raise RuntimeError("submission contains duplicate ids")
if not np.isfinite(submission[TARGET_COLUMN].to_numpy(dtype=np.float64)).all():
    raise RuntimeError("submission contains non-finite predictions")

candidate_content = candidate.to_csv(index=False, lineterminator="\\n")
candidate_gzip_path = WORK_ROOT / "pf1_tw_fl192_candidate.csv.gz"
with gzip.open(candidate_gzip_path, "wt", encoding="utf-8", newline="") as handle:
    handle.write(candidate_content)
submission_path = WORK_ROOT / "submission.csv"
submission.to_csv(submission_path, index=False, lineterminator="\\n")

component_manifest = {
    "experiment": EXPERIMENT,
    "fidelity": FIDELITY,
    "learned_models": [],
    "pf": {
        "bank": PF_BANK,
        "representation": PF_REPRESENTATION,
        "public_config_raw_sha256": PUBLIC_CONFIG_SHA256,
        "runtime_config_text_sha256": sha256_path(config_path),
        "particles": PF_N_PARTICLES,
        "seeds": PF_N_SEEDS,
        "smooth_mode": PF_SMOOTH_MODE,
        "smooth_lag": PF_SMOOTH_LAG,
        "physics": PF_PHYSICS_ENABLED,
        "emission_weight": PF_EMISSION_WEIGHT,
        "generation_seed": PF_GENERATION_SEED,
    },
}
(WORK_ROOT / "component_manifest.json").write_text(
    json.dumps(component_manifest, indent=2, sort_keys=True), encoding="utf-8"
)

manifest = {
    "experiment": EXPERIMENT,
    "fidelity": FIDELITY,
    "late_submission": True,
    "late_submission_phase": LATE_SUBMISSION_PHASE,
    "submission_message": "LATE SUBMIT | exp517 | stage2-2 pf1 x twGR fixedlag192 proxy | fixed v1",
    "source": {
        "public_kernel": PUBLIC_KERNEL,
        "public_kernel_id_no": PUBLIC_KERNEL_ID_NO,
        "public_notebook_sha256": PUBLIC_NOTEBOOK_SHA256,
        "public_pf_source_sha256": PUBLIC_PF_SOURCE_SHA256,
        "public_config_raw_sha256": PUBLIC_CONFIG_SHA256,
        "public_config_embedded_text_sha256": PUBLIC_CONFIG_TEXT_SHA256,
    },
    "runtime": {
        "gpu_names": gpu_names,
        "pf_seconds": pf_seconds,
        "well_count": len(names),
        "row_count": len(candidate),
    },
    "artifacts": {
        "candidate_decompressed_content_sha256": sha256_text(candidate_content),
        "candidate_raw_gzip_sha256": sha256_path(candidate_gzip_path),
        "prediction_content_sha256": logical_csv_sha(candidate[["id", "pf1_tw_fl192_mean"]]),
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
    "published_stage22_system_reference": {
        "cv": 7.50,
        "public": 6.724,
        "private": 7.404,
        "comparable_to_this_pf_only_proxy": False,
        "reason": "published result used five PF inputs plus a tabular model",
    },
    "deterministic_anchor": False,
    "deterministic_reason": "GPU fixed-environment rerun equality not tested",
}
(WORK_ROOT / "exp517_execution_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
)
print(json.dumps({
    "LATE_SUBMIT": True,
    "fidelity": FIDELITY,
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
    pf_source = writefile_cell(notebook, "pf_banks_v95.py")
    generated = build_source(
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
