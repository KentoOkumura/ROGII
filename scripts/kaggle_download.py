from __future__ import annotations

import shutil
import stat
import subprocess
import zipfile
import zlib
from pathlib import Path

from config_utils import ROOT, get_nested, is_todo, load_project_config


def _configured_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _safe_member_path(data_dir: Path, member: zipfile.ZipInfo) -> Path:
    relative_name = member.filename.replace("\\", "/")
    target = (data_dir / relative_name).resolve()
    try:
        target.relative_to(data_dir.resolve())
    except ValueError as error:
        raise ValueError(
            f"refusing to extract {member.filename!r}: path escapes {data_dir}"
        ) from error

    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"refusing to extract symbolic link {member.filename!r}")
    return target


def _file_matches_member(path: Path, member: zipfile.ZipInfo) -> bool:
    if path.stat().st_size != member.file_size:
        return False
    checksum = 0
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF == member.CRC


def safe_extract_archive(archive_path: Path, data_dir: Path) -> tuple[int, int]:
    """Extract a Kaggle competition archive without traversal or overwrites."""
    skipped = 0
    with zipfile.ZipFile(archive_path) as archive:
        planned: list[tuple[zipfile.ZipInfo, Path]] = []
        for member in archive.infolist():
            target = _safe_member_path(data_dir, member)
            if member.is_dir():
                if target.exists() and not target.is_dir():
                    raise ValueError(f"refusing to overwrite non-directory path: {target}")
                planned.append((member, target))
                continue
            if target.exists():
                if not target.is_file():
                    raise ValueError(f"refusing to overwrite non-file path: {target}")
                if not _file_matches_member(target, member):
                    raise ValueError(
                        "existing file differs from archive member: "
                        f"{target}; move it aside before rerunning"
                    )
                skipped += 1
                continue
            planned.append((member, target))

        extracted = 0
        for member, target in planned:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            extracted += 1
    return extracted, skipped


def main() -> None:
    config = load_project_config()
    slug = get_nested(config, "competition.slug")
    if is_todo(slug):
        raise SystemExit("competition.slug is TODO in project.yml")

    raw_dir = get_nested(config, "data.raw_dir") or (
        Path(str(get_nested(config, "paths.data_dir") or "data")) / "raw"
    )
    data_dir = _configured_path(raw_dir)
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

    archive_path = data_dir / f"{slug}.zip"
    if not archive_path.is_file():
        raise SystemExit(f"downloaded archive was not found: {archive_path}")
    try:
        extracted, skipped = safe_extract_archive(archive_path, data_dir)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise SystemExit(f"failed to extract {archive_path}: {error}") from error

    sample_file = get_nested(config, "submission.sample_file")
    if is_todo(sample_file):
        raise SystemExit("submission.sample_file is TODO in project.yml")
    sample_path = _configured_path(sample_file)
    if not sample_path.is_file():
        raise SystemExit(
            f"configured sample submission was not found after extraction: {sample_path}"
        )

    print(
        f"Downloaded and safely extracted competition data to "
        f"{data_dir.relative_to(ROOT)} "
        f"({extracted} extracted, {skipped} existing files skipped)"
    )


if __name__ == "__main__":
    main()
