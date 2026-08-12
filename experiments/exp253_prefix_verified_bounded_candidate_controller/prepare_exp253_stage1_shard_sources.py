from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "exp253_prefix_verified_bounded_candidate_controller"
EXPERIMENT_DIR = ROOT / "experiments" / EXPERIMENT
SOURCE = EXPERIMENT_DIR / f"{EXPERIMENT}_train.py"
ANCHOR = "import os\n"


def main() -> None:
    source = SOURCE.read_text()
    if source.count(ANCHOR) != 1:
        raise RuntimeError(f"expected exactly one import anchor in {SOURCE}")
    for shard_index in range(4):
        prelude = (
            "import os\n\n"
            "# Generated execution prelude; all scientific code below is copied verbatim.\n"
            f'os.environ["EXP253_ACTIVE_WELL_SHARD_INDEX"] = "{shard_index}"\n'
        )
        target = EXPERIMENT_DIR / f"{EXPERIMENT}_train_variant{shard_index}.py"
        target.write_text(source.replace(ANCHOR, prelude, 1))
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
