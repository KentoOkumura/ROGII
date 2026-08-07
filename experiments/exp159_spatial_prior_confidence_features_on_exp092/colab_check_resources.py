from __future__ import annotations

import json
from pathlib import Path

import psutil

ROOT = Path("/content/drive/MyDrive/Kaggle/ROGII")
EXP = ROOT / "experiments" / "exp159_spatial_prior_confidence_features_on_exp092"
RUN_DIR = EXP / "colab_runs"


def main() -> None:
    latest = json.loads((RUN_DIR / "latest_run.json").read_text())
    parent_pid = int(latest["pid"])
    print("latest", json.dumps(latest, indent=2), flush=True)
    for pid in [parent_pid, *[p.pid for p in psutil.Process(parent_pid).children(recursive=True)]]:
        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            print("process_missing", pid, flush=True)
            continue
        info = {
            "pid": pid,
            "ppid": proc.ppid(),
            "status": proc.status(),
            "cpu_percent": proc.cpu_percent(interval=1.0),
            "cpu_times": proc.cpu_times()._asdict(),
            "memory_info": proc.memory_info()._asdict(),
            "cmdline": proc.cmdline(),
        }
        print("process", json.dumps(info, default=str), flush=True)
    vm = psutil.virtual_memory()
    print(
        "virtual_memory",
        json.dumps(
            {
                "total_gb": round(vm.total / 1024**3, 3),
                "available_gb": round(vm.available / 1024**3, 3),
                "percent": vm.percent,
            }
        ),
        flush=True,
    )
    for raw_path in [latest["local_cache"], latest["local_spatial"], latest["log_path"]]:
        path = Path(raw_path)
        print(
            "path",
            path,
            "exists",
            path.exists(),
            "size",
            path.stat().st_size if path.exists() else None,
            flush=True,
        )


if __name__ == "__main__":
    main()
