from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP_NAME = "exp517_stage22_pf1_tw_fixedlag192_late_submit"
EXP_DIR = ROOT / "experiments" / EXP_NAME
REFERENCE_NOTEBOOK = (
    ROOT
    / "docs/notebooks/rogii-wellbore-geology-prediction/solution_6th"
    / "k256net__public20th-private6th-pf-pf-pf-pf-and-bagging"
    / "public20th-private6th-pf-pf-pf-pf-and-bagging.ipynb"
)
PUBLIC_CONFIG = EXP_DIR / "pf_banks_config_v96_public.json"
PUBLIC_TABULAR_SOURCE = (
    ROOT
    / "experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit"
    / "public_notebook_replay_audit.py"
)
OUTPUT_TRAIN = EXP_DIR / f"{EXP_NAME}_stage22_v2_compact_selfcontained_train.py"
OUTPUT_INFERENCE = EXP_DIR / f"{EXP_NAME}_stage22_v2_compact_selfcontained_inference.py"

EXPECTED_REFERENCE_SHA = "b44f7889d6abdf9b027d33cb6c6b45f23902d609fae8d06f332914017784c924"
EXPECTED_CONFIG_SHA = "80e973d5f5e0e39be758a03f399cdd3d81d9e79320da8db6fbddbc25c2a202f3"
BANKS = ["pf_1", "pf_2", "pf_3", "r0_seed32", "r1_seed32"]


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
        raw = cell.get("source", "")
        source = "".join(raw) if isinstance(raw, list) else str(raw)
        if source.startswith(prefix):
            matches.append(source[len(prefix) :])
    if len(matches) != 1:
        raise RuntimeError(f"expected one {filename} writefile cell, found {len(matches)}")
    return matches[0]


def strip_main_block(source: str) -> str:
    marker = '\nif __name__ == "__main__":'
    pos = source.rfind(marker)
    if pos < 0:
        raise RuntimeError("public PF helper has no terminal __main__ block")
    return source[:pos].rstrip() + "\n"


def extract_tabular_core(source: str) -> str:
    marker = "\nPUBLIC_SOURCE_PROVENANCE = {"
    pos = source.find(marker)
    if pos < 0:
        raise RuntimeError("public tabular source marker was not found")
    core = source[:pos]
    core = core.replace("from __future__ import annotations\n", "", 1)
    old = "    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)"
    new = (
        "    _stage22 = STAGE22_PF_CACHE[wid][\"pf_1\"]\n"
        "    pf_a = np.asarray(_stage22[\"mean\"], dtype=np.float32)\n"
        "    std_a = np.asarray(_stage22[\"std\"], dtype=np.float32)"
    )
    if core.count(old) != 1:
        raise RuntimeError("public build_well PF call did not match exactly once")
    return core.replace(old, new, 1).rstrip() + "\n"


def markdown(title: str, body: str) -> str:
    out = ["# %% [markdown]", f"# ## {title}"]
    out.extend(f"# {line}" if line else "#" for line in body.splitlines())
    return "\n".join(out) + "\n\n"


def code(source: str) -> str:
    return "# %%\n" + source.rstrip() + "\n\n"


def header(kind: str) -> str:
    return (
        "# ---\n"
        "# jupyter:\n"
        "#   kernelspec:\n"
        "#     display_name: Python 3\n"
        "#     language: python\n"
        "#     name: python3\n"
        "#   language_info:\n"
        "#     name: python\n"
        "# jupytext:\n"
        "#   text_representation:\n"
        "#     extension: .py\n"
        "#     format_name: percent\n"
        "#     format_version: '1.3'\n"
        "# ---\n\n"
        "# %% [markdown]\n"
        f"# # exp517 stage 2-2 five-PF fixed-lag-192 tabular {kind} — corrected v2\n"
        "#\n"
        "# v1の1-PF direct提出は契約不一致の失敗履歴として保持する。\n"
        "# この候補は同じexp517内で、公開writeupの5 PF + smoother + tabular契約を実装する。\n\n"
    )


def common_runtime(config_text: str, pf_sha: str, tabular_sha: str) -> str:
    config_text_sha = sha256_bytes(config_text.encode())
    return f'''from __future__ import annotations

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
IMPLEMENTATION_VERSION = "stage22_corrected_v2"
FIDELITY = "historical_contract_reconstruction"
PUBLIC_KERNEL = "k256net/public20th-private6th-pf-pf-pf-pf-and-bagging"
PUBLIC_KERNEL_ID_NO = 126919690
PUBLIC_NOTEBOOK_SHA256 = "{EXPECTED_REFERENCE_SHA}"
PUBLIC_CONFIG_SHA256 = "{EXPECTED_CONFIG_SHA}"
PUBLIC_CONFIG_TEXT_SHA256 = "{config_text_sha}"
PUBLIC_PF_SOURCE_SHA256 = "{pf_sha}"
PUBLIC_TABULAR_SOURCE_SHA256 = "{tabular_sha}"
PF_BANKS = {BANKS!r}
PF_REPRESENTATION = "tw"
PF_GENERATION_SEED = 4423098
PF_N_SEEDS = 32
PF_SMOOTH_MODE = "fixedlag"
PF_SMOOTH_LAG = 192
PF_WELL_CHUNK = 40
PF_OFFSETS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], dtype=np.float32)
PUBLIC_CONFIG_JSON = {config_text!r}
STAGE22_PF_CACHE: dict[str, dict[str, dict[str, object]]] = {{}}


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_competition_root() -> Path:
    roots = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path("data/raw").resolve(),
    ]
    if Path("/kaggle/input").is_dir():
        roots.extend(path.parent for path in Path("/kaggle/input").rglob("sample_submission.csv"))
    for root in roots:
        if (root / "train").is_dir() and (root / "test").is_dir() and (root / "sample_submission.csv").is_file():
            return root
    raise FileNotFoundError("competition root with train/test/sample_submission.csv was not found")


DATA_ROOT = resolve_competition_root()
WORK_ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".").resolve()
ART_ROOT = WORK_ROOT / "exp517_stage22_v2_runtime"
ART_ROOT.mkdir(parents=True, exist_ok=True)
if sha256_text(PUBLIC_CONFIG_JSON) != PUBLIC_CONFIG_TEXT_SHA256:
    raise RuntimeError("embedded public config text SHA drift")
config_path = ART_ROOT / "pf_banks_config.json"
config_path.write_text(PUBLIC_CONFIG_JSON, encoding="utf-8")
empty_anchor_path = ART_ROOT / "empty_anchor.pkl"
empty_anchor_path.write_bytes(pickle.dumps({{}}, protocol=4))

gpu_count = torch.cuda.device_count()
gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
if gpu_count < 1:
    raise RuntimeError("stage2-2 corrected run requires a Kaggle GPU")
os.environ.update({{
    "ROGII_DATA": str(DATA_ROOT),
    "ROGII_OUT": str(WORK_ROOT),
    "ROGII_PROJ": str(WORK_ROOT),
    "ROGII_ART95": str(ART_ROOT),
    "V93_ANCHOR_PKL": str(empty_anchor_path),
    "PF_NGPU": str(gpu_count),
    "PF_WELL_CHUNK": str(PF_WELL_CHUNK),
    "PS_COMBO_TAU": "0",
    "USE_GPU": "gpu",
    "PYTHONUNBUFFERED": "1",
}})
'''


def shared_stage22_functions() -> str:
    return '''def stage22_bank_param(bank: str) -> dict:
    if bank not in PF_BANKS:
        raise KeyError(bank)
    p = bank_param(bank)
    p["smooth_mode"] = PF_SMOOTH_MODE
    p["smooth_lag"] = PF_SMOOTH_LAG
    p["use_anchor"] = False
    p["use_phys"] = False
    p["robust_nu"] = 0.0
    p["temper_beta"] = 1.0
    p["_physics"] = False
    p["_w_nn"] = 0.0
    p["_ps_combo_tau"] = 0.0
    return p


def list_wells(split: str) -> list[str]:
    return sorted(path.name.split("__", 1)[0] for path in (DATA_ROOT / split).glob("*__horizontal_well.csv"))


def generate_stage22_pf_cache(split: str, wells: list[str]) -> tuple[pd.DataFrame, dict]:
    STAGE22_PF_CACHE.clear()
    meta: dict[str, dict[str, object]] = {}
    for wid in wells:
        hw = pd.read_csv(DATA_ROOT / split / f"{wid}__horizontal_well.csv")
        tw = pd.read_csv(DATA_ROOT / split / f"{wid}__typewell.csv").sort_values("TVT")
        rows = np.flatnonzero(hw["TVT_input"].isna().to_numpy())
        if not len(rows):
            continue
        known = hw.loc[hw["TVT_input"].notna(), "TVT_input"]
        if len(known) < 10:
            continue
        gr = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw["GR"]))).to_numpy()
        meta[wid] = {
            "rows": rows,
            "last_known_tvt": float(known.iloc[-1]),
            "horizontal_gr": gr[rows].astype(np.float32),
            "tw_tvt": tw["TVT"].to_numpy(np.float64),
            "tw_gr": tw["GR"].to_numpy(np.float64),
        }
        STAGE22_PF_CACHE[wid] = {}

    t0 = time.perf_counter()
    bank_seconds: dict[str, float] = {}
    for bank in PF_BANKS:
        p = stage22_bank_param(bank)
        names: list[str] = []
        inps: list[dict] = []
        for wid in meta:
            hw = pd.read_csv(DATA_ROOT / split / f"{wid}__horizontal_well.csv")
            tw = pd.read_csv(DATA_ROOT / split / f"{wid}__typewell.csv").sort_values("TVT")
            x = build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), p)
            if x is None:
                raise RuntimeError(f"{wid}/{bank}: no PF suffix")
            x = attach_anchor(x, wid, physics=False)
            x.pop("_sim", None); x.pop("_st", None)
            names.append(wid); inps.append(x)
        started = time.perf_counter()
        outputs = run_smoother_ext(
            inps, p, seed=PF_GENERATION_SEED, n_seeds=PF_N_SEEDS,
            chunk=PF_WELL_CHUNK, w_nn=0.0,
        )
        bank_seconds[bank] = time.perf_counter() - started
        if len(outputs) != len(names) or any(value is None for value in outputs):
            raise RuntimeError(f"{bank}: incomplete PF outputs")
        for wid, output in zip(names, outputs):
            mean = np.asarray(output["mean"], dtype=np.float32)
            std = np.asarray(output["std"], dtype=np.float32)
            if len(mean) != len(meta[wid]["rows"]) or not np.isfinite(mean).all() or not np.isfinite(std).all():
                raise RuntimeError(f"{wid}/{bank}: invalid PF output")
            STAGE22_PF_CACHE[wid][bank] = {"mean": mean, "std": std, "loglik": float(output["loglik"])}
        print({"bank": bank, "seconds": bank_seconds[bank], "wells": len(names)}, flush=True)

    parts: list[pd.DataFrame] = []
    for wid, info in meta.items():
        rows = np.asarray(info["rows"], dtype=np.int64)
        last = float(info["last_known_tvt"])
        hgr = np.asarray(info["horizontal_gr"], dtype=np.float32)
        tw_tvt = np.asarray(info["tw_tvt"], dtype=np.float64)
        tw_gr = np.asarray(info["tw_gr"], dtype=np.float64)
        values: dict[str, object] = {
            "id": [f"{wid}_{int(row)}" for row in rows],
            "well": wid,
        }
        for idx, bank in enumerate(PF_BANKS, start=1):
            output = STAGE22_PF_CACHE[wid][bank]
            mean = np.asarray(output["mean"], dtype=np.float32)
            values[f"pf_ancc_{idx}"] = mean
            values[f"pf_ancc_std_{idx}"] = np.asarray(output["std"], dtype=np.float32)
            values[f"pf_ancc_delta_{idx}"] = (mean - np.float32(last)).astype(np.float32)
            for offset in PF_OFFSETS:
                values[f"tdpf{int(offset)}_{idx}"] = (
                    hgr - np.interp(mean + float(offset), tw_tvt, tw_gr).astype(np.float32)
                ).astype(np.float32)
        parts.append(pd.DataFrame(values))
    frame = pd.concat(parts, ignore_index=True)
    if frame["id"].duplicated().any():
        raise RuntimeError("PF feature frame contains duplicate ids")
    manifest = {
        "split": split,
        "banks": PF_BANKS,
        "wells": int(frame["well"].nunique()),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "bank_seconds": bank_seconds,
        "total_seconds": time.perf_counter() - t0,
    }
    return frame, manifest


def augment_stage22_frame(base: pd.DataFrame, pf_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if len(base) != len(pf_frame) or not base["id"].astype(str).equals(pf_frame["id"].astype(str)):
        pf_frame = base[["id"]].merge(pf_frame, on="id", how="left", validate="one_to_one")
    if pf_frame.drop(columns=["id", "well"], errors="ignore").isna().any().any():
        raise RuntimeError("PF feature alignment produced missing values")
    # The public train CSV is 7.4 GB on disk. Mutate the loaded frame in place so
    # the corrected run does not hold a second full public frame in RAM.
    out = base
    last = out["last_known_tvt"].to_numpy(np.float32)
    spatial = out["tvtF_ANCC"].to_numpy(np.float32)
    dense = last + out["tvt_dense_d"].to_numpy(np.float32)
    pf_z = out["pf_z"].to_numpy(np.float32)
    added: dict[str, np.ndarray] = {}
    for idx, bank in enumerate(PF_BANKS, start=1):
        mean = pf_frame[f"pf_ancc_{idx}"].to_numpy(np.float32)
        for column in [f"pf_ancc_{idx}", f"pf_ancc_std_{idx}", f"pf_ancc_delta_{idx}"]:
            added[column] = pf_frame[column].to_numpy(np.float32)
        added[f"pf_vs_z_{idx}"] = (mean - pf_z).astype(np.float32)
        added[f"pf_vs_spatial_{idx}"] = (mean - spatial).astype(np.float32)
        added[f"pf_vs_dense_{idx}"] = (mean - dense).astype(np.float32)
        for offset in PF_OFFSETS:
            column = f"tdpf{int(offset)}_{idx}"
            added[column] = pf_frame[column].to_numpy(np.float32)
    overlap = sorted(set(added).intersection(out.columns))
    if overlap:
        raise RuntimeError(f"stage2 suffixed feature collision: {overlap[:20]}")
    out = pd.concat([out, pd.DataFrame(added, index=out.index)], axis=1)

    out["pf_ancc"] = out["pf_ancc_1"].to_numpy(np.float32)
    out["pf_ancc_std"] = out["pf_ancc_std_1"].to_numpy(np.float32)
    out["pf_ancc_delta"] = out["pf_ancc_delta_1"].to_numpy(np.float32)
    out["pf_vs_z"] = out["pf_vs_z_1"].to_numpy(np.float32)
    out["pf_vs_spatial"] = out["pf_vs_spatial_1"].to_numpy(np.float32)
    out["pf_vs_dense"] = out["pf_vs_dense_1"].to_numpy(np.float32)
    for offset in PF_OFFSETS:
        out[f"tdpf{int(offset)}"] = out[f"tdpf{int(offset)}_1"].to_numpy(np.float32)

    candidate_paths = [out[f"pf_ancc_{idx}"].to_numpy(np.float32) for idx in range(1, 6)]
    signal_paths = candidate_paths + [
        last + out[f"beam_{tag}_d"].to_numpy(np.float32) for *_, tag in BEAMS
    ] + [
        last + out["sc8_d"].to_numpy(np.float32),
        last + out["sc15_d"].to_numpy(np.float32),
        last + out["sc25_d"].to_numpy(np.float32),
        last + out["sc_ens_d"].to_numpy(np.float32),
        spatial,
        dense,
    ]
    signal_matrix = np.stack(signal_paths, axis=1)
    out["sig_std"] = signal_matrix.std(axis=1).astype(np.float32)
    out["sig_mean_d"] = (signal_matrix.mean(axis=1) - last).astype(np.float32)

    excluded = {"id", "well", "target"}
    features = [column for column in out.columns if column not in excluded]
    for column in features:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(np.float32)
    nonfinite = [column for column in features if not np.isfinite(out[column].to_numpy(np.float32)).all()]
    if nonfinite:
        raise RuntimeError(f"stage2 feature frame contains non-finite columns: {nonfinite[:20]}")
    return out, features


def apply_public_postprocess(df: pd.DataFrame, model_delta: np.ndarray, pf_delta: np.ndarray) -> np.ndarray:
    delta = np.asarray(model_delta, float) * 0.91 + np.asarray(pf_delta, float) * 0.09
    delta *= 1.0 - np.exp(-np.maximum(df["md_since"].to_numpy(float), 0.0) / 85.0)
    return delta


def sg_smooth_by_well(df: pd.DataFrame, values: np.ndarray, window: int = 17, poly: int = 3) -> np.ndarray:
    result = np.asarray(values, float).copy()
    for _, group in df.groupby("well", sort=False):
        idx = group.index.to_numpy()
        width = min(window, len(idx))
        if width % 2 == 0:
            width -= 1
        if width >= poly + 2:
            result[idx] = savgol_filter(result[idx], width, poly)
    return result
'''


def train_orchestration() -> str:
    return '''def resolve_public_train_csv() -> Path:
    candidates = [
        Path("/kaggle/input/wellbore-geology-prediction-artifacts/data/train.csv"),
        Path("/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts/data/train.csv"),
    ]
    if Path("/kaggle/input").is_dir():
        candidates.extend(Path("/kaggle/input").glob("**/wellbore-geology-prediction-artifacts/data/train.csv"))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Ravaghi public artifact data/train.csv was not found")


from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.linear_model import Ridge

train_wells = list_wells("train")
pf_frame, pf_runtime = generate_stage22_pf_cache("train", train_wells)
STAGE22_PF_CACHE.clear()
public_train_path = resolve_public_train_csv()
print({"public_train_csv": str(public_train_path), "bytes": public_train_path.stat().st_size}, flush=True)
train_df = pd.read_csv(public_train_path, low_memory=False)
train_df, features = augment_stage22_frame(train_df, pf_frame)
del pf_frame

expected_rows = 3_783_989
expected_wells = 773
if len(train_df) != expected_rows or train_df["well"].nunique() != expected_wells:
    raise RuntimeError(f"public train coverage drift: rows={len(train_df)} wells={train_df['well'].nunique()}")

model_dir = WORK_ROOT / "exp517_stage22_v2_model"
model_dir.mkdir(parents=True, exist_ok=True)
X = train_df[features].to_numpy(np.float32)
y = train_df["target"].to_numpy(np.float32)
groups = train_df["well"].astype(str).to_numpy()
ids = train_df["id"].astype(str).to_numpy()
base = train_df["last_known_tvt"].to_numpy(np.float32)
pf_delta = train_df["pf_ancc_delta_1"].to_numpy(np.float32)

lgb_params = [
    dict(boosting_type="gbdt", num_leaves=255, min_child_samples=15, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05, objective="regression", verbose=-1,
         n_jobs=-1, device_type="gpu", gpu_use_dp=False, max_bin=255, learning_rate=0.03,
         n_estimators=5000, seed=123),
    dict(n_jobs=-1, verbose=-1, reg_alpha=10.788188919840913, subsample=0.47437582748953966,
         num_leaves=64, reg_lambda=95.75401894533888, n_estimators=10000, random_state=0,
         boosting_type="gbdt", learning_rate=0.00934485794382918,
         colsample_bytree=0.39283351290380497, min_child_weight=0.24081152127177283,
         min_child_samples=40, device="gpu"),
    dict(n_jobs=-1, verbose=-1, reg_alpha=10.788188919840913, subsample=0.47437582748953966,
         num_leaves=64, reg_lambda=95.75401894533888, n_estimators=10000, random_state=29,
         boosting_type="gbdt", learning_rate=0.00934485794382918,
         colsample_bytree=0.39283351290380497, min_child_weight=0.24081152127177283,
         min_child_samples=40, device="gpu"),
]
cb_params = [
    dict(iterations=8000, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254,
         loss_function="RMSE", task_type="GPU", devices="0", od_type="Iter", od_wait=300,
         verbose=0, learning_rate=0.02, random_seed=7),
    dict(iterations=8000, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254,
         loss_function="RMSE", task_type="GPU", devices="0", od_type="Iter", od_wait=300,
         verbose=0, learning_rate=0.03, random_seed=123),
]

cv = GroupKFold(n_splits=5)
splits = list(cv.split(X, y, groups=groups))
base_oof = np.zeros((len(train_df), 5), dtype=np.float32)
model_records: list[dict] = []

for config_index, params in enumerate(lgb_params):
    for fold, (tr, va) in enumerate(splits):
        model = LGBMRegressor(**params)
        model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="rmse",
                  callbacks=[early_stopping(250, verbose=False), log_evaluation(0)])
        best = int(model.best_iteration_ or params["n_estimators"])
        base_oof[va, config_index] = model.predict(X[va], num_iteration=best).astype(np.float32)
        filename = f"lightgbm_{config_index}_fold{fold}.txt"
        model.booster_.save_model(str(model_dir / filename), num_iteration=best)
        model_records.append({"family": "lightgbm", "config": config_index, "fold": fold,
                              "best_iteration": best, "file": filename})
        print({"family": "lightgbm", "config": config_index, "fold": fold,
               "rmse": rmse(y[va], base_oof[va, config_index]), "best_iteration": best}, flush=True)

for config_index, params in enumerate(cb_params):
    column = 3 + config_index
    for fold, (tr, va) in enumerate(splits):
        model = CatBoostRegressor(**params)
        model.fit(X[tr], y[tr], eval_set=(X[va], y[va]), early_stopping_rounds=250, use_best_model=True)
        base_oof[va, column] = model.predict(X[va]).astype(np.float32)
        filename = f"catboost_{config_index}_fold{fold}.cbm"
        model.save_model(str(model_dir / filename))
        model_records.append({"family": "catboost", "config": config_index, "fold": fold,
                              "best_iteration": int(model.get_best_iteration()), "file": filename})
        print({"family": "catboost", "config": config_index, "fold": fold,
               "rmse": rmse(y[va], base_oof[va, column]),
               "best_iteration": int(model.get_best_iteration())}, flush=True)

ridge_oof = np.zeros(len(train_df), dtype=np.float32)
ridge_records: list[dict] = []
for fold, (tr, va) in enumerate(splits):
    ridge = Ridge(alpha=1.6602834637650032, tol=0.0005030247295617308,
                  positive=True, fit_intercept=True, random_state=42)
    ridge.fit(base_oof[tr], y[tr])
    ridge_oof[va] = ridge.predict(base_oof[va]).astype(np.float32)
    filename = f"ridge_fold{fold}.npz"
    np.savez(model_dir / filename, coef=ridge.coef_.astype(np.float64), intercept=np.float64(ridge.intercept_))
    ridge_records.append({"fold": fold, "file": filename})

pp_delta = apply_public_postprocess(train_df, ridge_oof, pf_delta)
pred_tvt = base.astype(np.float64) + pp_delta
true_tvt = base.astype(np.float64) + y.astype(np.float64)
cv_ridge = rmse(y, ridge_oof)
cv_pp = rmse(true_tvt, pred_tvt)
cv_pp_sg = rmse(true_tvt, sg_smooth_by_well(train_df, pred_tvt))
fold_metrics = []
for fold, (_, va) in enumerate(splits):
    fold_metrics.append({"fold": fold, "rows": int(len(va)),
                         "ridge_rmse": rmse(y[va], ridge_oof[va]),
                         "postprocess_rmse": rmse(true_tvt[va], pred_tvt[va])})

oof_path = WORK_ROOT / "exp517_stage22_v2_oof.csv.gz"
pd.DataFrame({"id": ids, "well": groups, "target_tvt": true_tvt,
              "last_known_tvt": base, "pred_delta_ridge": ridge_oof,
              "pred_tvt_postprocess": pred_tvt}).to_csv(oof_path, index=False, compression="gzip")

for record in model_records:
    record["sha256"] = sha256_path(model_dir / record["file"])
for record in ridge_records:
    record["sha256"] = sha256_path(model_dir / record["file"])
manifest = {
    "experiment": EXPERIMENT,
    "implementation_version": IMPLEMENTATION_VERSION,
    "fidelity": FIDELITY,
    "method_contract": {
        "input": "Ravaghi public base frame + five original-Optuna twGR PF trajectories",
        "target": "TVT - last_known_tvt",
        "output": "row residual",
        "loss": "LightGBM/CatBoost RMSE + positive Ridge stack",
        "decode": "0.91 ridge + 0.09 pf_1; tau85 fade; SG17/3 at inference",
        "context_unit": "one well suffix; PF fixed-lag 192; row tabular; GroupKFold by well",
    },
    "source": {"public_notebook_sha256": PUBLIC_NOTEBOOK_SHA256,
               "public_config_sha256": PUBLIC_CONFIG_SHA256,
               "public_pf_source_sha256": PUBLIC_PF_SOURCE_SHA256,
               "public_tabular_source_sha256": PUBLIC_TABULAR_SOURCE_SHA256,
               "public_train_csv": str(public_train_path),
               "public_train_csv_sha256": sha256_path(public_train_path)},
    "features": features,
    "feature_count": len(features),
    "rows": len(train_df),
    "wells": int(train_df["well"].nunique()),
    "pf_runtime": pf_runtime,
    "lgb_params": lgb_params,
    "cb_params": cb_params,
    "models": model_records,
    "ridge_models": ridge_records,
    "execution_count": {"scientific_variants": 1, "pf_banks": 5, "representations": 1,
                        "lightgbm_configs": 3, "catboost_configs": 2, "folds": 5,
                        "base_models": 25, "ridge_models": 5, "control_reruns": 0},
    "metrics": {"ridge_rmse": cv_ridge, "postprocess_rmse": cv_pp,
                "postprocess_sg_diagnostic_rmse": cv_pp_sg, "published_stage22_cv": 7.50,
                "absolute_delta_from_published": abs(cv_pp - 7.50), "folds": fold_metrics},
    "oof": {"file": oof_path.name, "sha256": sha256_path(oof_path)},
    "gpu_names": gpu_names,
}
(model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
(WORK_ROOT / "exp517_stage22_v2_metrics.json").write_text(
    json.dumps(manifest["metrics"], indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"stage22_v2": True, "cv_postprocess": cv_pp,
                  "published_cv": 7.50, "model_count": len(model_records),
                  "ridge_count": len(ridge_records), "feature_count": len(features)}, indent=2))
'''


def inference_orchestration() -> str:
    return '''from catboost import CatBoostRegressor
from lightgbm import Booster


def resolve_model_dir() -> Path:
    candidates = []
    if Path("/kaggle/input").is_dir():
        candidates.extend(Path("/kaggle/input").glob("**/exp517_stage22_v2_model/manifest.json"))
    candidates.append(Path("exp517_stage22_v2_model/manifest.json").resolve())
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one stage22 v2 model manifest, found {matches}")
    return matches[0].parent


model_dir = resolve_model_dir()
manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
if manifest.get("implementation_version") != IMPLEMENTATION_VERSION:
    raise RuntimeError("model implementation version mismatch")
expected_counts = {"base_models": 25, "ridge_models": 5, "control_reruns": 0}
for key, expected in expected_counts.items():
    if manifest["execution_count"].get(key) != expected:
        raise RuntimeError(f"model count drift for {key}")
for record in manifest["models"] + manifest["ridge_models"]:
    if sha256_path(model_dir / record["file"]) != record["sha256"]:
        raise RuntimeError(f"model SHA mismatch: {record['file']}")

train_wells = list_wells("train")
test_wells = list_wells("test")
pf_frame, pf_runtime = generate_stage22_pf_cache("test", test_wells)
init_imputers(train_wells)
test_df = build_features(test_wells, "test", is_train=False).reset_index(drop=True)
test_df, features = augment_stage22_frame(test_df, pf_frame)
if features != manifest["features"]:
    missing = [column for column in manifest["features"] if column not in features]
    extra = [column for column in features if column not in manifest["features"]]
    raise RuntimeError(f"feature schema drift: missing={missing[:10]} extra={extra[:10]}")

X = test_df[features].to_numpy(np.float32)
base_predictions = np.zeros((len(test_df), 5), dtype=np.float32)
for record in manifest["models"]:
    path = model_dir / record["file"]
    if record["family"] == "lightgbm":
        model = Booster(model_file=str(path))
        prediction = model.predict(X, num_iteration=int(record["best_iteration"]))
        column = int(record["config"])
    elif record["family"] == "catboost":
        model = CatBoostRegressor()
        model.load_model(str(path))
        prediction = model.predict(X)
        column = 3 + int(record["config"])
    else:
        raise RuntimeError(record["family"])
    base_predictions[:, column] += np.asarray(prediction, np.float32) / 5.0

ridge_predictions = np.zeros(len(test_df), dtype=np.float64)
for record in manifest["ridge_models"]:
    params = np.load(model_dir / record["file"])
    ridge_predictions += (base_predictions @ params["coef"] + float(params["intercept"])) / 5.0

pf_delta = test_df["pf_ancc_delta_1"].to_numpy(np.float32)
delta = apply_public_postprocess(test_df, ridge_predictions, pf_delta)
raw_tvt = test_df["last_known_tvt"].to_numpy(np.float64) + delta
pred_tvt = sg_smooth_by_well(test_df, raw_tvt)
prediction_frame = pd.DataFrame({"id": test_df["id"].astype(str), "tvt": pred_tvt})
if prediction_frame["id"].duplicated().any() or not np.isfinite(prediction_frame["tvt"]).all():
    raise RuntimeError("invalid inference predictions")

sample = pd.read_csv(DATA_ROOT / "sample_submission.csv")
target_column = str(sample.columns[1])
submission = sample[["id"]].merge(prediction_frame.rename(columns={"tvt": target_column}),
                                   on="id", how="left", validate="one_to_one")
if len(submission) != len(sample) or not submission["id"].equals(sample["id"]):
    raise RuntimeError("sample row/order contract failed")
if submission[target_column].isna().any() or not np.isfinite(submission[target_column]).all():
    raise RuntimeError("submission contains missing/non-finite values")
submission_path = WORK_ROOT / "submission.csv"
submission.to_csv(submission_path, index=False, lineterminator="\\n")

candidate_path = WORK_ROOT / "exp517_stage22_v2_fivepf_test_features.csv.gz"
keep = ["id"] + [column for column in test_df.columns if column.startswith("pf_ancc_")]
test_df[keep].to_csv(candidate_path, index=False, compression="gzip")
execution = {
    "experiment": EXPERIMENT,
    "implementation_version": IMPLEMENTATION_VERSION,
    "fidelity": FIDELITY,
    "late_submission": True,
    "submission_message": "LATE SUBMIT | exp517 | corrected stage2-2 5PF fixedlag192 tabular v2",
    "model_manifest_sha256": sha256_path(model_dir / "manifest.json"),
    "pf_runtime": pf_runtime,
    "test_wells": int(test_df["well"].nunique()),
    "test_rows": int(len(test_df)),
    "feature_count": len(features),
    "submission": {"rows": len(submission), "duplicate_ids": int(submission["id"].duplicated().sum()),
                   "missing": int(submission[target_column].isna().sum()),
                   "finite": bool(np.isfinite(submission[target_column]).all()),
                   "sample_order_exact": bool(submission["id"].equals(sample["id"])),
                   "sha256": sha256_path(submission_path)},
    "candidate_sha256": sha256_path(candidate_path),
    "gpu_names": gpu_names,
}
(WORK_ROOT / "exp517_stage22_v2_execution_manifest.json").write_text(
    json.dumps(execution, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(execution, indent=2, sort_keys=True))
'''


def build_source(kind: str, pf_source: str, tabular_core: str, config_text: str) -> str:
    pf_core = strip_main_block(pf_source)
    pf_sha = sha256_bytes(pf_source.encode())
    tabular_sha = sha256_path(PUBLIC_TABULAR_SOURCE)
    parts = [
        header(kind),
        markdown(
            "1. Method contract",
            "input=Ravaghi public feature frame + five original Optuna PF trajectories; "
            "target=TVT-last_known_tvt; output=row residual; loss=GBDT RMSE + Ridge; "
            "decode=public PF blend/fade/SG; context=well suffix fixed-lag192 + row tabular.",
        ),
        code(common_runtime(config_text, pf_sha, tabular_sha)),
        markdown(
            "2. Released GPU PF engine",
            "The released source is embedded and SHA-pinned. Runtime overrides disable all post-stage2 likelihood/anchor/emission changes.",
        ),
        code(pf_core),
        markdown(
            "3. Public tabular feature engine",
            "The strict public Ravaghi replay is embedded. Only its base PF call is redirected to the corrected pf_1 fixed-lag cache.",
        ),
        code(tabular_core),
        markdown(
            "4. Five-PF feature and decode contract",
            "All five banks are generated; the public one-PF feature family is repeated per bank and pf_1 remains the unsuffixed compatibility alias.",
        ),
        code(shared_stage22_functions()),
        markdown("5. Orchestration", "Train only the corrected scientific variant." if kind == "train" else "Load the saved manifest, regenerate hidden-test features, and align to runtime sample IDs."),
        code(train_orchestration() if kind == "train" else inference_orchestration()),
    ]
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if sha256_path(REFERENCE_NOTEBOOK) != EXPECTED_REFERENCE_SHA:
        raise RuntimeError("public reference notebook SHA drift")
    if sha256_path(PUBLIC_CONFIG) != EXPECTED_CONFIG_SHA:
        raise RuntimeError("public config SHA drift")
    notebook = json.loads(REFERENCE_NOTEBOOK.read_text(encoding="utf-8"))
    pf_source = writefile_cell(notebook, "pf_banks_v95.py")
    tabular_core = extract_tabular_core(PUBLIC_TABULAR_SOURCE.read_text(encoding="utf-8"))
    config_text = PUBLIC_CONFIG.read_text(encoding="utf-8")
    outputs = {
        OUTPUT_TRAIN: build_source("train", pf_source, tabular_core, config_text),
        OUTPUT_INFERENCE: build_source("inference", pf_source, tabular_core, config_text),
    }
    if args.check:
        for path, generated in outputs.items():
            if not path.is_file() or path.read_text(encoding="utf-8") != generated:
                raise SystemExit(f"generated source drift: run {Path(__file__).relative_to(ROOT)}")
            print(f"PASS {path.relative_to(ROOT)} {sha256_bytes(generated.encode())}")
        return
    for path, generated in outputs.items():
        path.write_text(generated, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")
        print(f"sha256 {sha256_bytes(generated.encode())}")


if __name__ == "__main__":
    main()
