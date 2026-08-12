from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "exp270_exact_hmm_posterior_mode_candidate_audit"
EXPERIMENT_DIR = ROOT / "experiments" / EXPERIMENT
SOURCE = EXPERIMENT_DIR / f"{EXPERIMENT}_train.py"
MODE_ANCHOR = 'RUN_KIND_OVERRIDE = "aggregate"'


def main() -> None:
    source = SOURCE.read_text()
    if source.count(MODE_ANCHOR) != 1:
        raise RuntimeError(f"expected exactly one mode anchor in {SOURCE}")
    for shard_index in range(2):
        mode = f"shard{shard_index}"
        target = EXPERIMENT_DIR / f"{EXPERIMENT}_train_variant{shard_index}.py"
        target.write_text(source.replace(MODE_ANCHOR, f'RUN_KIND_OVERRIDE = "{mode}"', 1))
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
