from __future__ import annotations

import gzip
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from settings import ExperimentPaths, get_nested

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - Kaggle image is expected to have torch.
    raise ImportError("exp088 requires PyTorch. Run this on a Kaggle image with torch.") from exc


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return int(hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest(), 16) % (2**31)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_content(path: Path) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmse(error: np.ndarray | pd.Series) -> float:
    arr = np.asarray(error, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(arr))))


def find_first_existing(candidates: list[str | Path], label: str) -> Path:
    checked: list[str] = []
    for candidate in candidates:
        path = Path(candidate)
        checked.append(str(path))
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(f"No existing {label} path. Checked: {checked}")


def candidate_paths(config: dict[str, Any], key: str) -> list[str]:
    value = get_nested(config, f"audit.candidates.{key}")
    if not isinstance(value, list) or not value:
        raise KeyError(f"audit.candidates.{key} must be a non-empty list")
    return [str(item) for item in value]


def read_available_columns(path: Path, desired: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    selected = [column for column in desired if column in set(header.columns)]
    missing = sorted(set(desired) - set(selected))
    if missing:
        print(f"[warn] {path.name} missing columns: {missing}", flush=True)
    return pd.read_csv(path, usecols=selected)


def load_anchor_predictions(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    selected_mode = str(get_nested(config, "audit.selected_mode"))
    selected_model = str(get_nested(config, "audit.selected_model"))
    max_rows = get_nested(config, "audit.max_rows")
    usecols = [
        "id",
        "well",
        "mode",
        "model",
        "target_tvt",
        "last_known_tvt",
        "target_delta",
        "pred_delta",
        "pred_tvt",
    ]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=1_000_000):
        selected = chunk[
            chunk["mode"].eq(selected_mode) & chunk["model"].eq(selected_model)
        ].copy()
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        raise ValueError(f"No rows for mode={selected_mode} model={selected_model} in {path}")
    frame = pd.concat(chunks, ignore_index=True)
    if max_rows:
        frame = frame.head(int(max_rows)).copy()
    frame["baseline_error"] = frame["pred_tvt"].astype(float) - frame["target_tvt"].astype(float)
    frame["correction_target"] = -frame["baseline_error"]
    return frame


def load_feature_frame(path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    configured = [str(value) for value in get_nested(config, "model.sequence_features") or []]
    required = ["id", "well", "MD", "md_since", "md_from_ps", "eval_len"]
    frame = read_available_columns(path, required + configured)
    available_features = [feature for feature in configured if feature in frame.columns]
    if not available_features:
        raise ValueError("No configured sequence features were found in the feature cache")
    return frame, available_features


def add_distance_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    distance_column = "md_since" if "md_since" in out.columns else "md_from_ps"
    distance = pd.to_numeric(out.get(distance_column), errors="coerce")
    bins = [-np.inf, 50, 250, 1000, 2500, np.inf]
    labels = ["0000_0050", "0050_0250", "0250_1000", "1000_2500", "2500_plus"]
    out["distance_bucket"] = pd.cut(distance, bins=bins, labels=labels)
    out["distance_bucket"] = out["distance_bucket"].astype("string").fillna("unknown")
    return out


def stable_well_fold(well: str, n_folds: int) -> int:
    return stable_seed("well_fold", well) % n_folds


def prepare_model_frame(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    frame = predictions.merge(features, on=["id", "well"], how="inner", validate="one_to_one")
    missing_rows = int(len(predictions) - len(frame))
    if len(frame) != len(predictions):
        max_missing_fraction = float(
            get_nested(config, "audit.max_feature_join_missing_fraction") or 0.02
        )
        missing_fraction = missing_rows / max(len(predictions), 1)
        print(
            json.dumps(
                {
                    "warning": "feature_join_lost_rows",
                    "prediction_rows": int(len(predictions)),
                    "joined_rows": int(len(frame)),
                    "missing_rows": missing_rows,
                    "missing_fraction": missing_fraction,
                    "max_missing_fraction": max_missing_fraction,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if missing_fraction > max_missing_fraction:
            raise ValueError(
                "Feature join lost too many rows: "
                f"predictions={len(predictions)} joined={len(frame)} "
                f"missing_fraction={missing_fraction:.6f}"
            )

    sort_column = "md_since" if "md_since" in frame.columns else "md_from_ps"
    if sort_column not in frame.columns:
        sort_column = "MD" if "MD" in frame.columns else "id"
    frame = frame.sort_values(["well", sort_column, "id"]).reset_index(drop=True)
    frame["row_in_well"] = frame.groupby("well", sort=False).cumcount()
    frame.attrs["feature_join_missing_rows"] = missing_rows
    n_folds = int(get_nested(config, "validation.n_folds") or 5)
    frame["sequence_fold"] = frame["well"].astype(str).map(
        lambda value: stable_well_fold(value, n_folds)
    )
    frame = add_distance_bucket(frame)

    for feature in feature_columns:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce").astype(np.float32)
    numeric = [
        "target_tvt",
        "last_known_tvt",
        "target_delta",
        "pred_delta",
        "pred_tvt",
        "baseline_error",
        "correction_target",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    return frame


@dataclass
class WellArrays:
    well: str
    fold: int
    ids: np.ndarray
    features: np.ndarray
    target: np.ndarray
    target_tvt: np.ndarray
    baseline_pred: np.ndarray
    distance_bucket: np.ndarray


def fit_normalizer(
    frame: pd.DataFrame,
    feature_columns: list[str],
    folds: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    train = frame[frame["sequence_fold"].isin(folds)]
    values = train[feature_columns].to_numpy(dtype=np.float32, copy=True)
    mean = np.nanmean(values, axis=0).astype(np.float32)
    std = np.nanstd(values, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where((np.isfinite(std)) & (std > 1e-6), std, 1.0).astype(np.float32)
    return mean, std


def build_well_arrays(
    frame: pd.DataFrame,
    feature_columns: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> list[WellArrays]:
    arrays: list[WellArrays] = []
    for well, group in frame.groupby("well", sort=False):
        raw = group[feature_columns].to_numpy(dtype=np.float32, copy=True)
        values = (raw - mean) / std
        values = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
        arrays.append(
            WellArrays(
                well=str(well),
                fold=int(group["sequence_fold"].iloc[0]),
                ids=group["id"].astype(str).to_numpy(),
                features=values,
                target=group["correction_target"].to_numpy(dtype=np.float32),
                target_tvt=group["target_tvt"].to_numpy(dtype=np.float32),
                baseline_pred=group["pred_tvt"].to_numpy(dtype=np.float32),
                distance_bucket=group["distance_bucket"].astype(str).to_numpy(),
            )
        )
    return arrays


class SequenceWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        wells: list[WellArrays],
        indices: list[tuple[int, int]],
        context_window: int,
    ) -> None:
        self.wells = wells
        self.indices = indices
        self.context_window = context_window
        feature_count = wells[0].features.shape[1] if wells else 0
        self.input_features = feature_count + 1

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        well_index, pos = self.indices[index]
        well = self.wells[well_index]
        start = max(0, pos - self.context_window + 1)
        window = well.features[start : pos + 1]
        x = np.zeros((self.context_window, self.input_features), dtype=np.float32)
        x[-len(window) :, : well.features.shape[1]] = window
        x[-len(window) :, -1] = 1.0
        y = np.asarray(well.target[pos], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y.reshape(()))


class SequencePredictDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        wells: list[WellArrays],
        indices: list[tuple[int, int]],
        context_window: int,
    ) -> None:
        self.wells = wells
        self.indices = indices
        self.context_window = context_window
        feature_count = wells[0].features.shape[1] if wells else 0
        self.input_features = feature_count + 1

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        well_index, pos = self.indices[index]
        well = self.wells[well_index]
        start = max(0, pos - self.context_window + 1)
        window = well.features[start : pos + 1]
        x = np.zeros((self.context_window, self.input_features), dtype=np.float32)
        x[-len(window) :, : well.features.shape[1]] = window
        x[-len(window) :, -1] = 1.0
        return torch.from_numpy(x)


class GRUResidualModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(self.dropout(out[:, -1])).squeeze(-1)


class TCNResidualModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float, layers: int) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        channels = input_size
        for layer in range(layers):
            dilation = 2**layer
            padding = dilation
            blocks.extend(
                [
                    nn.Conv1d(
                        channels,
                        hidden_size,
                        kernel_size=3,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            channels = hidden_size
        self.net = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.transpose(1, 2)
        out = self.net(out)
        out = out[..., : x.shape[1]]
        return self.head(out[:, :, -1]).squeeze(-1)


def make_model(variant: dict[str, Any], input_size: int) -> nn.Module:
    model_type = str(variant.get("type", "gru")).lower()
    hidden_size = int(variant.get("hidden_size", 32))
    dropout = float(variant.get("dropout", 0.05))
    if model_type == "gru":
        return GRUResidualModel(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
    if model_type == "tcn":
        return TCNResidualModel(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            layers=int(variant.get("layers", 3)),
        )
    raise ValueError(f"Unknown sequence model type: {model_type}")


def make_indices(
    wells: list[WellArrays],
    folds: set[int],
    stride: int,
    max_windows: int | None,
    seed: int,
) -> list[tuple[int, int]]:
    indices: list[tuple[int, int]] = []
    for well_index, well in enumerate(wells):
        if well.fold not in folds:
            continue
        indices.extend((well_index, pos) for pos in range(0, len(well.target), stride))
    if max_windows and len(indices) > max_windows:
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(indices), size=int(max_windows), replace=False)
        indices = [indices[int(idx)] for idx in np.sort(selected)]
    return indices


def train_one_fold(
    variant: dict[str, Any],
    wells: list[WellArrays],
    train_folds: set[int],
    valid_fold: int,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    name = str(variant["name"])
    context_window = int(
        variant.get("context_window") or get_nested(config, "model.context_window") or 64
    )
    batch_size = int(variant.get("batch_size") or get_nested(config, "model.batch_size") or 1024)
    epochs = int(variant.get("epochs") or get_nested(config, "model.epochs") or 2)
    lr = float(variant.get("lr") or get_nested(config, "model.lr") or 0.001)
    weight_decay = float(
        variant.get("weight_decay") or get_nested(config, "model.weight_decay") or 0.0
    )
    train_stride = int(variant.get("train_stride") or get_nested(config, "model.train_stride") or 4)
    valid_stride = int(variant.get("valid_stride") or get_nested(config, "model.valid_stride") or 1)
    max_train_windows = get_nested(config, "model.max_train_windows_per_fold")
    max_valid_windows = get_nested(config, "model.max_valid_windows_per_fold")
    seed = stable_seed(get_nested(config, "validation.seed") or 42, name, valid_fold)
    seed_everything(seed)

    train_indices = make_indices(
        wells,
        train_folds,
        train_stride,
        int(max_train_windows) if max_train_windows else None,
        seed,
    )
    valid_indices = make_indices(
        wells,
        {valid_fold},
        valid_stride,
        int(max_valid_windows) if max_valid_windows else None,
        seed + 1,
    )
    if not train_indices or not valid_indices:
        raise ValueError(f"Empty train/valid indices for {name} fold {valid_fold}")

    dataset = SequenceWindowDataset(wells, train_indices, context_window)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )
    model = make_model(variant, dataset.input_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=float(variant.get("huber_beta", 5.0)))
    history: list[dict[str, Any]] = []

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        seen = 0
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(variant.get("grad_clip", 1.0)))
            optimizer.step()
            rows = int(x.shape[0])
            loss_sum += float(loss.detach().cpu()) * rows
            seen += rows
        history.append(
            {
                "variant": name,
                "fold": int(valid_fold),
                "epoch": int(epoch + 1),
                "train_rows": int(seen),
                "train_loss": float(loss_sum / max(seen, 1)),
            }
        )

    pred_dataset = SequencePredictDataset(wells, valid_indices, context_window)
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    correction_values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x in pred_loader:
            x = x.to(device=device, dtype=torch.float32)
            correction_values.append(model(x).detach().cpu().numpy().astype(np.float32))
    correction = np.concatenate(correction_values)
    clip = variant.get("correction_clip", get_nested(config, "model.correction_clip"))
    if clip:
        correction = np.clip(correction, -float(clip), float(clip)).astype(np.float32)

    row_ids: list[str] = []
    target_tvt: list[float] = []
    baseline_pred: list[float] = []
    buckets: list[str] = []
    for well_index, pos in valid_indices:
        well = wells[well_index]
        row_ids.append(str(well.ids[pos]))
        target_tvt.append(float(well.target_tvt[pos]))
        baseline_pred.append(float(well.baseline_pred[pos]))
        buckets.append(str(well.distance_bucket[pos]))

    valid_frame = pd.DataFrame(
        {
            "id": row_ids,
            "target_tvt": np.asarray(target_tvt, dtype=np.float32),
            "baseline_pred_tvt": np.asarray(baseline_pred, dtype=np.float32),
            "distance_bucket": buckets,
        }
    )
    return (
        valid_frame["id"].to_numpy(),
        valid_frame["target_tvt"].to_numpy(dtype=np.float32),
        (valid_frame["baseline_pred_tvt"].to_numpy(dtype=np.float32) + correction).astype(
            np.float32
        ),
        history,
    )


def metric_rows(frame: pd.DataFrame, prediction_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y = frame["target_tvt"].to_numpy(dtype=np.float64)
    baseline_rmse = rmse(frame["baseline_pred_tvt"].to_numpy(dtype=np.float64) - y)
    for column in prediction_columns:
        pred = frame[column].to_numpy(dtype=np.float64)
        error = pred - y
        rows.append(
            {
                "prediction": column,
                "rows": int(len(frame)),
                "wells": int(frame["well"].nunique()) if "well" in frame.columns else None,
                "rmse_tvt": rmse(error),
                "mae_tvt": float(np.mean(np.abs(error))),
                "error_mean": float(np.mean(error)),
                "rmse_delta_vs_baseline": rmse(error) - baseline_rmse,
            }
        )
    return pd.DataFrame(rows)


def bucket_metrics(frame: pd.DataFrame, prediction_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket, group in frame.groupby("distance_bucket", observed=False):
        y = group["target_tvt"].to_numpy(dtype=np.float64)
        baseline_rmse = rmse(group["baseline_pred_tvt"].to_numpy(dtype=np.float64) - y)
        for column in prediction_columns:
            pred = group[column].to_numpy(dtype=np.float64)
            error = pred - y
            rows.append(
                {
                    "distance_bucket": str(bucket),
                    "prediction": column,
                    "rows": int(len(group)),
                    "rmse_tvt": rmse(error),
                    "mae_tvt": float(np.mean(np.abs(error))),
                    "rmse_delta_vs_baseline": rmse(error) - baseline_rmse,
                }
            )
    return pd.DataFrame(rows)


def diversity_metrics(frame: pd.DataFrame, prediction_columns: list[str]) -> pd.DataFrame:
    y = frame["target_tvt"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for column in prediction_columns:
        if column == "baseline_pred_tvt":
            continue
        base_error = frame["baseline_pred_tvt"].to_numpy(dtype=np.float64) - y
        seq_error = frame[column].to_numpy(dtype=np.float64) - y
        pred_diff = frame[column].to_numpy(dtype=np.float64) - frame[
            "baseline_pred_tvt"
        ].to_numpy(dtype=np.float64)
        corr = float(np.corrcoef(base_error, seq_error)[0, 1]) if len(frame) > 2 else float("nan")
        rows.append(
            {
                "prediction": column,
                "baseline_error_corr": corr,
                "prediction_diff_rmse_vs_baseline": rmse(pred_diff),
                "prediction_diff_mae_vs_baseline": float(np.mean(np.abs(pred_diff))),
                "correction_mean": float(np.mean(pred_diff)),
                "correction_std": float(np.std(pred_diff)),
            }
        )
    return pd.DataFrame(rows)


def add_blends(
    frame: pd.DataFrame,
    sequence_columns: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = frame.copy()
    y = out["target_tvt"].to_numpy(dtype=np.float64)
    base = out["baseline_pred_tvt"].to_numpy(dtype=np.float64)
    blend_info: dict[str, Any] = {}
    for column in sequence_columns:
        delta = out[column].to_numpy(dtype=np.float64) - base
        denom = float(np.dot(delta, delta))
        alpha = 0.0 if denom <= 0 else float(np.dot(y - base, delta) / denom)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        blend_column = f"alpha_blend_{column}"
        out[blend_column] = (base + alpha * delta).astype(np.float32)
        blend_info[blend_column] = {"source": column, "alpha": alpha}

    matrix_columns = ["baseline_pred_tvt", *sequence_columns]
    x = out[matrix_columns].to_numpy(dtype=np.float64)
    ridge = float(get_nested(config, "model.ridge_blend_l2") or 1e-3)
    xtx = x.T @ x + ridge * np.eye(x.shape[1])
    xty = x.T @ y
    weights = np.linalg.solve(xtx, xty)
    out["ridge_blend_pred_tvt"] = (x @ weights).astype(np.float32)
    blend_info["ridge_blend_pred_tvt"] = {
        "columns": matrix_columns,
        "weights": [float(value) for value in weights],
        "l2": ridge,
    }
    return out, blend_info


def plot_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    plot_frame = metrics.sort_values("rmse_tvt")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(plot_frame))))
    ax.barh(plot_frame["prediction"], plot_frame["rmse_tvt"], color="#3572a5")
    ax.invert_yaxis()
    ax.set_xlabel("OOF RMSE")
    ax.set_ylabel("Prediction")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def select_torch_device(config: dict[str, Any]) -> torch.device:
    if not bool(get_nested(config, "runtime.kaggle.enable_gpu")):
        return torch.device("cpu")
    allow_cpu_fallback = bool(get_nested(config, "runtime.allow_cpu_fallback"))
    if not torch.cuda.is_available():
        if not allow_cpu_fallback:
            raise RuntimeError("GPU is enabled in config but torch.cuda.is_available() is false")
        return torch.device("cpu")
    major, minor = torch.cuda.get_device_capability(0)
    min_major = int(get_nested(config, "runtime.min_cuda_capability_major") or 7)
    if major < min_major:
        warning_payload = {
            "warning": "cuda_capability_too_old_for_current_torch",
            "device_name": torch.cuda.get_device_name(0),
            "cuda_capability": f"sm_{major}{minor}",
            "min_cuda_capability_major": min_major,
            "allow_cpu_fallback": allow_cpu_fallback,
        }
        print(
            json.dumps(warning_payload, sort_keys=True),
            flush=True,
        )
        if not allow_cpu_fallback:
            raise RuntimeError(
                "CUDA device capability is below runtime.min_cuda_capability_major; "
                "request a newer Kaggle accelerator such as NvidiaTeslaT4"
            )
        return torch.device("cpu")
    return torch.device("cuda")


def run_audit() -> dict[str, Any]:
    started = time.time()
    paths = ExperimentPaths()
    config = paths.config
    output_dir = paths.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(
        get_nested(config, "audit.output_prefix") or "exp088_sequence_model_residual_diversity"
    )

    predictions_path = find_first_existing(
        candidate_paths(config, "exp073_oof_predictions"),
        "exp073_oof_predictions",
    )
    feature_cache_path = find_first_existing(
        candidate_paths(config, "exp072_feature_cache"),
        "exp072_feature_cache",
    )
    print(f"anchor predictions: {predictions_path}", flush=True)
    print(f"feature cache: {feature_cache_path}", flush=True)

    predictions = load_anchor_predictions(predictions_path, config)
    feature_frame, feature_columns = load_feature_frame(feature_cache_path, config)
    frame = prepare_model_frame(predictions, feature_frame, feature_columns, config)
    n_folds = int(get_nested(config, "validation.n_folds") or 5)
    active_folds_raw = get_nested(config, "model.active_folds")
    active_folds = (
        [int(value) for value in active_folds_raw]
        if isinstance(active_folds_raw, list)
        else list(range(n_folds))
    )
    variants = get_nested(config, "model.variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("model.variants must be a non-empty list")

    device = select_torch_device(config)
    print(f"device={device} torch={torch.__version__} float32_only=True", flush=True)
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError(f"Unexpected torch default dtype: {torch.get_default_dtype()}")

    prediction_frame = frame[
        ["id", "well", "target_tvt", "baseline_pred_tvt", "distance_bucket"]
        if "baseline_pred_tvt" in frame.columns
        else ["id", "well", "target_tvt", "pred_tvt", "distance_bucket"]
    ].copy()
    if "baseline_pred_tvt" not in prediction_frame.columns:
        prediction_frame = prediction_frame.rename(columns={"pred_tvt": "baseline_pred_tvt"})

    train_history_rows: list[dict[str, Any]] = []
    sequence_columns: list[str] = []
    for variant in variants:
        variant_name = str(variant["name"])
        sequence_column = f"{variant_name}_pred_tvt"
        prediction_frame[sequence_column] = np.nan
        sequence_columns.append(sequence_column)
        print(f"training variant={variant_name}", flush=True)
        for valid_fold in active_folds:
            train_folds = set(range(n_folds)) - {valid_fold}
            mean, std = fit_normalizer(frame, feature_columns, train_folds)
            wells = build_well_arrays(frame, feature_columns, mean, std)
            ids, _target_tvt, pred_tvt, history = train_one_fold(
                variant,
                wells,
                train_folds,
                valid_fold,
                config,
                device,
            )
            pred_map = pd.Series(pred_tvt, index=ids.astype(str))
            mask = prediction_frame["id"].astype(str).isin(pred_map.index)
            prediction_frame.loc[mask, sequence_column] = (
                prediction_frame.loc[mask, "id"].astype(str).map(pred_map).astype(np.float32)
            )
            train_history_rows.extend(history)
            fold_error = (
                prediction_frame.loc[mask, sequence_column].to_numpy(dtype=np.float64)
                - prediction_frame.loc[mask, "target_tvt"].to_numpy(dtype=np.float64)
            )
            print(
                json.dumps(
                    {
                        "variant": variant_name,
                        "fold": int(valid_fold),
                        "valid_rows": int(mask.sum()),
                        "rmse_tvt": rmse(fold_error),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    complete_sequence_columns = [
        column for column in sequence_columns if prediction_frame[column].notna().all()
    ]
    incomplete = sorted(set(sequence_columns) - set(complete_sequence_columns))
    if incomplete:
        print(
            f"[warn] incomplete OOF sequence columns excluded from blends: {incomplete}",
            flush=True,
        )
    if not complete_sequence_columns:
        raise ValueError("No complete sequence OOF prediction columns were produced")

    prediction_frame, blend_info = add_blends(prediction_frame, complete_sequence_columns, config)
    prediction_columns = ["baseline_pred_tvt", *complete_sequence_columns, *blend_info.keys()]
    metrics = metric_rows(prediction_frame, prediction_columns)
    buckets = bucket_metrics(prediction_frame, prediction_columns)
    diversity = diversity_metrics(prediction_frame, prediction_columns)
    history = pd.DataFrame(train_history_rows)

    predictions_out = output_dir / f"{prefix}_oof_predictions.csv.gz"
    metrics_out = output_dir / f"{prefix}_metrics.csv"
    buckets_out = output_dir / f"{prefix}_bucket_metrics.csv"
    diversity_out = output_dir / f"{prefix}_diversity_metrics.csv"
    history_out = output_dir / f"{prefix}_train_history.csv"
    plot_out = output_dir / f"{prefix}_rmse.png"
    summary_out = output_dir / f"{prefix}_summary.json"

    prediction_frame.to_csv(predictions_out, index=False, compression="gzip")
    metrics.to_csv(metrics_out, index=False)
    buckets.to_csv(buckets_out, index=False)
    diversity.to_csv(diversity_out, index=False)
    history.to_csv(history_out, index=False)
    plot_metrics(metrics, plot_out)

    input_feature_sha = sha256_gzip_content(feature_cache_path)
    input_prediction_sha = sha256_gzip_content(predictions_path)
    predictions_content_sha = sha256_gzip_content(predictions_out)
    summary = {
        "experiment": paths.experiment_name,
        "status": "completed",
        "elapsed_seconds": round(time.time() - started, 3),
        "device": str(device),
        "torch_version": torch.__version__,
        "float32_only": True,
        "amp_enabled": False,
        "rows": int(len(prediction_frame)),
        "wells": int(prediction_frame["well"].nunique()),
        "feature_join_missing_rows": int(frame.attrs.get("feature_join_missing_rows", 0)),
        "active_folds": active_folds,
        "feature_columns": feature_columns,
        "sequence_columns": complete_sequence_columns,
        "blend_info": blend_info,
        "best_prediction": str(metrics.sort_values("rmse_tvt").iloc[0]["prediction"]),
        "best_rmse_tvt": float(metrics["rmse_tvt"].min()),
        "baseline_rmse_tvt": float(
            metrics.loc[metrics["prediction"].eq("baseline_pred_tvt"), "rmse_tvt"].iloc[0]
        ),
        "input_paths": {
            "exp073_oof_predictions": str(predictions_path),
            "exp072_feature_cache": str(feature_cache_path),
        },
        "input_sha256": {
            "exp073_oof_predictions_content": input_prediction_sha,
            "exp072_feature_cache_content": input_feature_sha,
        },
        "artifact_sha256": {
            "oof_predictions_content": predictions_content_sha,
            "metrics": sha256_file(metrics_out),
            "bucket_metrics": sha256_file(buckets_out),
            "diversity_metrics": sha256_file(diversity_out),
            "train_history": sha256_file(history_out),
            "rmse_plot": sha256_file(plot_out),
        },
        "artifacts": {
            "oof_predictions": predictions_out.name,
            "metrics": metrics_out.name,
            "bucket_metrics": buckets_out.name,
            "diversity_metrics": diversity_out.name,
            "train_history": history_out.name,
            "rmse_plot": plot_out.name,
            "summary": summary_out.name,
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    summary["artifact_sha256"]["summary"] = sha256_file(summary_out)

    metrics_json = {
        "experiment": paths.experiment_name,
        "status": "completed",
        "cv": summary["best_rmse_tvt"],
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "notes": f"Best OOF prediction: {summary['best_prediction']}",
        "summary": summary,
    }
    (paths.experiment_dir / "metrics.json").write_text(
        json.dumps(metrics_json, indent=2, sort_keys=True)
    )

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run_audit()
