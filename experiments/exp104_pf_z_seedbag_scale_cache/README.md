# exp104_pf_z_seedbag_scale_cache

exp100 `pf_z_xy_slope` を 128 seed の pf_z seedbag cache として再生成し、exp072 の既存 PF/Beam 候補と同じ行・同じ metrics で比較する train-side audit。

## 概要

- Route: `pf_beam`
- 状態: `completed_train_side_audit_rejected`
- 親: `exp100_pf_z_unified_velocity_observation_prior`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- Kaggle source: `kentookumura/exp072-exp063-full-replay-feature-cache-train`

## 状態

Kaggle train v1 完了。提出なし。推論移植なし。

## 仮説

exp100 `pf_z_xy_slope` は単発 PF 候補だったため exp072 `likpf_mean` との比較が不公平だった。128 seed likelihood seedbag と scale ensemble に揃えれば、pf_z XY rate prior の価値を exp072 `pf_z` / `likpf_*` と同じ candidate metrics で判断できる。

## 検証方針

exp072 feature cache の同一 row set を使い、`last_known_tvt + target` を true TVT として scoring する。`pf_z_seedbag_*` は raw train horizontal/typewell から prefix-only prior で再生成し、exp072 baseline と同じ `candidate_metrics`、bucket metrics、by-well metrics で比較する。

## 何を比較するか

exp072 cache から `pf_z` / `likpf_mean` と、存在する場合は `likpf_scale_3/5/8/12` を読み、同じ行集合に対して新しく `pf_z_seedbag_mean` / `pf_z_seedbag_scale_3/5/8/12` を生成して比較する。

## 実装メモ

- `pf_z_seedbag` は 128 seed、500 particles、scale `[3, 5, 8, 12]`。
- 状態は exp072 と同じ `pos = TVT + Z`。
- pf_z XY rate prior は prefix の `d(TVT_input + Z)/dMD` を `dZ/dMD` と `dXY/dMD` で回帰する。
- 評価区間 true TVT は candidate metrics の scoring のみに使う。

## 予定生成物

- `exp104_pf_z_seedbag_scale_cache_candidate_metrics.csv`
- `exp104_pf_z_seedbag_scale_cache_bucket_metrics.csv`
- `exp104_pf_z_seedbag_scale_cache_by_well.csv`
- `exp104_pf_z_seedbag_scale_cache_pf_z_seedbag_quality.csv`
- `exp104_pf_z_seedbag_scale_cache_candidate_wide.csv.gz`
- `exp104_pf_z_seedbag_scale_cache_candidate_long.csv.gz`
- `exp104_pf_z_seedbag_scale_cache_summary.json`

## 所見

`pf_z_seedbag_scale_12` が seedbag 内 best で RMSE 14.145856 / within10 0.695260。exp072 plain `pf_z` RMSE 17.788171 は改善したが、exp072 `likpf_mean` RMSE 11.594898 には届かない。直接推論移植や提出はしない。
