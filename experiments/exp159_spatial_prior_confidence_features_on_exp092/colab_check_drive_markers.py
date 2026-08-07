from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/Kaggle/ROGII")
EXP_NAME = "exp159_spatial_prior_confidence_features_on_exp092"
EXP = ROOT / "experiments" / EXP_NAME
RUN_DIR = EXP / "colab_runs"
ARTIFACTS = EXP / "artifacts"
PREFIX = EXP_NAME


def tail(path: Path, n: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def main() -> None:
    print("root", ROOT, ROOT.exists(), flush=True)
    print("exp", EXP, EXP.exists(), flush=True)
    latest_path = RUN_DIR / "latest_run.json"
    done_path = RUN_DIR / "latest_done_summary.json"
    failed_path = RUN_DIR / "latest_failed.txt"
    print("latest_run_exists", latest_path.exists(), flush=True)
    latest = json.loads(latest_path.read_text()) if latest_path.exists() else {}
    print("latest_run", json.dumps(latest, indent=2), flush=True)
    print(
        "done_exists",
        done_path.exists(),
        done_path.stat().st_size if done_path.exists() else None,
    )
    if done_path.exists():
        done = json.loads(done_path.read_text())
        print("done_summary_keys", sorted(done), flush=True)
        print("done_best", json.dumps(done.get("best_lgb_mean_by_rmse_tvt"), indent=2), flush=True)
    print(
        "failed_exists",
        failed_path.exists(),
        failed_path.stat().st_size if failed_path.exists() else None,
        flush=True,
    )
    if failed_path.exists():
        print("failed_tail")
        print(tail(failed_path, 80), flush=True)
    log_path = Path(latest.get("log_path", "")) if latest else None
    if log_path:
        print(
            "log_exists",
            log_path.exists(),
            log_path.stat().st_size if log_path.exists() else None,
        )
        print("log_tail")
        print(tail(log_path, 120), flush=True)

    checkpoint_root = ARTIFACTS / f"{PREFIX}_fold_checkpoints"
    metric_paths = sorted(checkpoint_root.glob("**/metric.json"))
    print("checkpoint_root", checkpoint_root, checkpoint_root.exists(), flush=True)
    print("checkpoint_metrics_count", len(metric_paths), flush=True)
    for path in metric_paths:
        metric = json.loads(path.read_text())
        print(
            "checkpoint_metric",
            json.dumps(
                {
                    "model": metric.get("model"),
                    "fold": metric.get("fold"),
                    "rmse_tvt": metric.get("rmse_tvt"),
                    "best_iteration": metric.get("best_iteration"),
                    "path": str(path),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary_path = ARTIFACTS / f"{PREFIX}_summary.json"
    metrics_path = ARTIFACTS / f"{PREFIX}_metrics.csv"
    print(
        "summary_exists",
        summary_path.exists(),
        summary_path.stat().st_size if summary_path.exists() else None,
    )
    print(
        "metrics_exists",
        metrics_path.exists(),
        metrics_path.stat().st_size if metrics_path.exists() else None,
    )


if __name__ == "__main__":
    main()
