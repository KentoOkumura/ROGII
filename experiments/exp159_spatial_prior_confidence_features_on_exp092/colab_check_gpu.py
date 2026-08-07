from __future__ import annotations

import subprocess


def main() -> None:
    for cmd in (
        ["nvidia-smi"],
        [
            "nvidia-smi",
            "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
    ):
        print("$", " ".join(cmd))
        result = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(result.stdout)


if __name__ == "__main__":
    main()
