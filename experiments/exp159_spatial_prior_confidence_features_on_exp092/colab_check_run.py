from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/Kaggle/ROGII")
RUN_DIR = ROOT / "experiments/exp159_spatial_prior_confidence_features_on_exp092/colab_runs"
latest = json.loads((RUN_DIR / "latest_run.json").read_text())
print(json.dumps(latest, indent=2))

pid = str(latest["pid"])
print(
    subprocess.run(
        ["ps", "-p", pid, "-o", "pid,ppid,stat,etime,time,%cpu,%mem,rss,cmd"],
        capture_output=True,
        text=True,
    ).stdout
)

log = Path(latest["log_path"])
print("log", log.exists(), log.stat().st_size if log.exists() else None)
if log.exists():
    print("\n".join(log.read_text(errors="replace").splitlines()[-120:]))

failed = RUN_DIR / "latest_failed.txt"
done = RUN_DIR / "latest_done_summary.json"
print("failed", failed.exists(), failed.stat().st_size if failed.exists() else None)
print("done", done.exists(), done.stat().st_size if done.exists() else None)
if done.exists():
    summary = json.loads(done.read_text())
    print("done_status", summary.get("status"))
    print("active_variants", summary.get("active_variants"))
    print("active_modes", summary.get("active_modes"))
    print("best_lgb_mean_by_rmse_tvt", summary.get("best_lgb_mean_by_rmse_tvt"))
