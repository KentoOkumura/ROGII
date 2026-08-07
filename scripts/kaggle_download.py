from __future__ import annotations

import subprocess

from config_utils import ROOT, get_nested, is_todo, load_project_config


def main() -> None:
    config = load_project_config()
    slug = get_nested(config, "competition.slug")
    if is_todo(slug):
        raise SystemExit("competition.slug is TODO in project.yml")

    data_dir = ROOT / str(get_nested(config, "paths.data_dir") or "data") / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "uv",
        "run",
        "kaggle",
        "competitions",
        "download",
        "-c",
        str(slug),
        "-p",
        str(data_dir),
    ]
    subprocess.run(command, check=True)
    print(f"Downloaded competition files to {data_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
