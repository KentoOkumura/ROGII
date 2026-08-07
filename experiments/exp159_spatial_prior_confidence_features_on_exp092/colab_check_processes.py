from __future__ import annotations

import subprocess


def main() -> None:
    result = subprocess.run(
        ["ps", "-ef"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for line in result.stdout.splitlines():
        lowered = line.lower()
        if (
            "exp159_spatial_prior_confidence" in line
            or "lightgbm" in lowered
            or "python" in lowered
        ):
            print(line)


if __name__ == "__main__":
    main()
