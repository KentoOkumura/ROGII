# 要件

## 依頼

`exp072_exp205_joint_exact_parity_fast_cache_generation` backlog を新しい実験番号で実装する。対象は現行バックログに含まれている full-cache exact parity 高速化であり、likPF-only/slim cache 化は扱わない。

## 制約

- Route: `pf_beam`
- 新規実験番号は `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation` とする。
- exp072 full replay feature cache は 196 features / 3,783,989 rows / 773 wells を parity 対象にする。
- exp205 exact HMM cache / direct comparison は exp205 v2 と同じ HMM grid、candidate、blend weight、metric logic を使う。
- 高速化目的で `pf_seeds=128`、`pf_particles=500`、HMM `step=0.35`、`n_rates=41`、`band_pad=100`、`numba_num_threads=4`、blend weight を変えない。
- まずは exp072 DataFrame を in-memory で direct comparison に渡し、2GB gzip 再読込を避ける。
- HMM well 外側並列は実装してよいが、既定値は parity 優先で serial outer loop とする。
- 再現性は `docs/06_reproducibility.md` に従い、gzip は raw SHA と decompressed content SHA を分けて記録する。
- 推論、提出、raw-test regeneration は実施しない。

## 受け入れ基準

- `.steering/`、`experiments/exp209.../config.yaml`、train/inference notebook source、helper、`SESSION_NOTES.md`、`result.md`、`metrics.json` が exp209 として整備されている。
- train notebook は setup、入力確認、exp072/HMM generation、direct comparison、metrics/parity 出力をセル単位で追える。
- exp072 full cache generation と exp205 HMM generation が同一 Kaggle train notebook で実行できる。
- exp205 direct comparison が exp072 baseline frame を in-memory で受け取れる。
- HMM outer parallelism は `feature_cache.hmm.outer_workers` で切り替えられる。
- 静的検証として `py_compile`、`ruff --select F821`、Jupytext 変換/テスト、`validate_exp` 相当が通る。
