# exp106_strict_exp072_pf_z_multiseed_scale_cache

exp072 の元 `pf_z` 実装を strict parity で再生成し、その同一ロジックを multi-seed / scale cache 化する train-side audit。

## 概要

- Route: `pf_beam`
- 状態: `completed_train_side_audit_rejected`
- 親: `exp072_exp063_full_replay_feature_cache`
- Kaggle source: `kentookumura/exp072-exp063-full-replay-feature-cache-train`

## 仮説

exp104 は exp100 系 `pf_z_xy_slope` の seedbag 化であり、exp072 cache の `pf_z` と strict parity ではなかった。exp072 の `_pf_z` を同一ロジックのまま seed だけ増やすと、単一 seed の `exp072_pf_z` が lucky/unlucky seed だったかを検証でき、ML/selector feature として置換・追加する価値を判断できる。

## 検証方針

exp072 feature cache の同一 row set を使い、`last_known_tvt + target` を true TVT として scoring する。まず `stable_seed("pf_z", well)` で seed 1 parity を行い、exp072 cache の `pf_z` との row-level diff を保存する。parity を満たす場合だけ、`pf_z_ms_*` を候補として比較する。

実行は exp072 と同じく well-level `joblib.Parallel(... prefer="threads")` を使い、`num_workers=8` にする。seed は well / seed index から固定し、出力は id merge で揃える。

## 生成候補

- exp072 baseline: `exp072_pf_z`、`exp072_likpf_mean`、存在する場合の `exp072_likpf_scale_*`
- strict parity: `strict_pf_z_parity_seed`
- strict multi-seed: `pf_z_ms_mean`、`pf_z_ms_scale_3/5/8/12`。初回 full は 64 seeds、必要なら 128 seeds に上げる。
- wide cache feature: `pf_z_ms_std`、`pf_z_ms_best_lik_seed`、`pf_z_ms_delta_vs_pf_z`、`pf_z_ms_delta_vs_likpf_mean`

## 予定生成物

- `exp106_strict_exp072_pf_z_multiseed_scale_cache_candidate_metrics.csv`
- `exp106_strict_exp072_pf_z_multiseed_scale_cache_bucket_metrics.csv`
- `exp106_strict_exp072_pf_z_multiseed_scale_cache_by_well.csv`
- `exp106_strict_exp072_pf_z_multiseed_scale_cache_strict_pf_z_quality.csv`
- `exp106_strict_exp072_pf_z_multiseed_scale_cache_parity_diff.csv.gz`
- `exp106_strict_exp072_pf_z_multiseed_scale_cache_candidate_wide.csv.gz`
- `exp106_strict_exp072_pf_z_multiseed_scale_cache_summary.json`

## 状態

Kaggle train v3 完了。提出なし。

## 所見

seed 1 parity は `exp072_pf_z` と完全一致した。best multiseed は `pf_z_ms_scale_3` で RMSE 16.145943 / within10 0.708807。`exp072_pf_z` RMSE 17.788171 からは改善したが、`exp072_likpf_mean` RMSE 11.594898 には届かないため、direct port / submit はしない。

ただし `pf_z_ms_scale_3`、`pf_z_ms_std`、`pf_z_ms_delta_vs_pf_z`、`pf_z_ms_delta_vs_likpf_mean` は selector 候補・ML add-only feature 候補として残す。exp103 `xy_likpf_scale_12` より oracle headroom は小さいため、候補追加の優先度は exp103 が上。
