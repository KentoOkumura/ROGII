from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


TYPEWELL_SUFFIX = "__typewell.csv"


def typewell_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def collect_typewell_groups(raw_dir: Path, split: str) -> list[dict[str, str | int]]:
    typewell_paths = sorted((raw_dir / split).glob(f"*{TYPEWELL_SUFFIX}"))
    hash_to_wells: dict[str, list[str]] = defaultdict(list)
    rows: list[dict[str, str | int]] = []

    for path in typewell_paths:
        well_id = path.name.removesuffix(TYPEWELL_SUFFIX)
        h = typewell_hash(path)
        hash_to_wells[h].append(well_id)
        rows.append(
            {
                "split": split,
                "well_id": well_id,
                "typewell_hash": h,
                "typewell_path": str(path),
            }
        )

    group_sizes = {h: len(wells) for h, wells in hash_to_wells.items()}
    for row in rows:
        h = str(row["typewell_hash"])
        row["exact_typewell_group"] = f"typewell_{h}"
        row["exact_typewell_group_size"] = group_sizes[h]

    return rows


def write_csv(rows: list[dict[str, str | int]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "well_id",
        "typewell_hash",
        "exact_typewell_group",
        "exact_typewell_group_size",
        "typewell_path",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str | int]]) -> None:
    hash_to_wells: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        hash_to_wells[str(row["typewell_hash"])].append(str(row["well_id"]))

    size_counts = Counter(len(wells) for wells in hash_to_wells.values())
    print(f"wells: {len(rows)}")
    print(f"unique exact typewell groups: {len(hash_to_wells)}")
    print("group size distribution:")
    for size, count in sorted(size_counts.items()):
        print(f"  size {size}: {count} groups, {size * count} wells")

    print("\nlargest duplicate groups:")
    duplicate_groups = [
        (h, wells) for h, wells in hash_to_wells.items() if len(wells) > 1
    ]
    for h, wells in sorted(duplicate_groups, key=lambda item: (-len(item[1]), item[0]))[:20]:
        preview = ",".join(wells[:12])
        suffix = "..." if len(wells) > 12 else ""
        print(f"  {h}: n={len(wells)} wells={preview}{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exact typewell groups by hashing typewell CSV files."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/typewell_groups.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_typewell_groups(args.raw_dir, args.split)
    write_csv(rows, args.output)
    print_summary(rows)
    print(f"\nwrote: {args.output}")


if __name__ == "__main__":
    main()
