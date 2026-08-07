from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "exp490_geometry_centered_mean_reverting_offset_hmm"
EXPERIMENT_DIR = ROOT / "experiments" / EXPERIMENT
SOURCE = EXPERIMENT_DIR / f"{EXPERIMENT}_train_aggregate.py"
MODE_ANCHOR = 'RUN_KIND_OVERRIDE = "aggregate"'


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    if source.count(MODE_ANCHOR) != 1:
        raise RuntimeError(f"expected exactly one mode anchor in {SOURCE}")
    for shard_index in range(4):
        target = EXPERIMENT_DIR / f"{EXPERIMENT}_train_variant{shard_index}.py"
        target.write_text(
            source.replace(
                MODE_ANCHOR,
                f'RUN_KIND_OVERRIDE = "shard{shard_index}"',
                1,
            ),
            encoding="utf-8",
        )
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
