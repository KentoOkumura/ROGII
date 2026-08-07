# 要件

## 依頼

`KAGGLE_DIRECTION.md` の backlog `typewell_late_range_hard_window_pct40_full_cache_replacement` を実装する。

exp192 の hard-window pct50 full replay cache replacement は `likpf_mean` と `pf_ancc` を改善したが、`pf_z` と true typewell pct `<0.50` subset を大きく壊した。pct40 へ緩めることで、後半 prior の効果を残しつつ `0.40-0.50` support を戻し、pct50 の regression が緩和するかを確認する。

## 制約

- Route: `pf_beam`
- 親実験: `exp192_typewell_late_range_hard_window_pct50_full_cache_replacement`
- 実装元: exp192 の corrected full replay hard-window 実装。
- hard-window は `0.40 <= typewell_pct <= 1.00` の 1 variant のみ。
- `0.30/0.60/0.70` grid、soft prior、LightGBM 学習、inference、submit は実行しない。
- 既存 exp072 / exp192 cache は生成 input に使わず、生成後の比較対象に限定する。
- GPU 学習はない。Kaggle CPU notebook で full replay train feature cache を生成する。
- 再現性: `docs/06_reproducibility.md` に従い、PF/Beam の stable per-well seed、Kaggle package bootstrap、gzip decompressed content SHA の記録方針を明記する。

## 受け入れ基準

- `experiments/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement/` が作成され、`config.yaml` の `experiment.route` が `pf_beam`、`model.hard_window.min_typewell_pct` が `0.40` になっている。
- train notebook が raw train horizontal/typewell file から exp072-style full replay train feature cache を生成できる。
- feature cache variant は `pixiux_likpf_hard_window_pct40_public_replay`、expected feature count は 196。
- Kaggle train push 前のコスト記録として、variant 1、LightGBM config 0、fold 0、booster 0、control / parent 再学習なしが `SESSION_NOTES.md` に記録されている。
- jupytext 変換、`py_compile`、`ruff --select F821`、`validate-exp`、Kaggle train package prepare が通る。
- 実行後は exp072 と exp192 pct50 の両方に対し、`pf_ancc`、`pf_z`、`beam_mean`、`beam_sm5`、`likpf_mean` の RMSE/MAE/within10、distance bucket、true typewell pct bucket、by-well regression、gzip/decompressed SHA、runtime を記録する。
- deterministic submission anchor としては扱わない。gzip 生成物を比較する場合は raw `.csv.gz` SHA だけでなく decompressed content SHA を主証拠として記録する。
