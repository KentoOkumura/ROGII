# 要件

## 依頼

`strict_exp072_pf_z_multiseed_scale_cache` を実装する。exp104 は exp100 派生の `pf_z_xy_slope` seedbag 化だったため、ここでは exp072 feature cache に入っている元 `pf_z` と同一ロジックを seed 1 本で parity 確認し、その後 multi-seed / scale cache を生成できるようにする。

## 制約

- Route: `pf_beam`
- 新規実験名: `exp106_strict_exp072_pf_z_multiseed_scale_cache`
- 親: `exp072_exp063_full_replay_feature_cache`
- 比較対象: `exp072_pf_z`、`exp072_likpf_mean`、存在する場合の `exp072_likpf_scale_*`
- exp072 / exp073 の既存 schema や生成物は上書きしない。
- exp072 の `_pf_z` と同じ TVT state、transition、GR likelihood、Z velocity prior、clamp、resampling 条件、出力 transform を維持する。
- stochastic 処理、PF/Beam、Kaggle bootstrap、SHA 記録は `docs/06_reproducibility.md` に従う。
- full run 前に `max_wells` / `n_seeds` を小さくした smoke と seed 1 parity check ができる CLI を用意する。
- full output は wide cache、metrics、summary を主とし、candidate_long は optional にする。

## 受け入れ基準

- `strict_pf_z_parity_seed` が exp072 cache の `pf_z` と同じ row set で評価され、row-level diff 指標が summary に保存される。
- parity diff が許容範囲内の場合だけ multi-seed 評価へ進める設定になっている。
- `pf_z_ms_mean`、`pf_z_ms_std`、`pf_z_ms_scale_3/5/8/12`、`pf_z_ms_best_lik_seed`、`pf_z_ms_delta_vs_pf_z`、`pf_z_ms_delta_vs_likpf_mean` が wide cache に保存される。
- `candidate_metrics.csv`、`bucket_metrics.csv`、`by_well.csv`、`strict_pf_z_quality.csv`、`candidate_wide.csv.gz`、`summary.json` を生成する。
- gzip 生成物は decompressed content SHA を summary に記録する。
