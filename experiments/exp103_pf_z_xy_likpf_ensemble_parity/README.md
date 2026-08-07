# exp103_pf_z_xy_likpf_ensemble_parity

exp100 `pf_z_xy_slope` を exp072 `lik_pf` と同じ土俵に寄せる train-side parity audit。

## 概要

- Route: `pf_beam`
- 状態: `implemented_not_run`
- 親: `exp100_pf_z_unified_velocity_observation_prior`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- Kaggle source: `kentookumura/exp072-exp063-full-replay-feature-cache-train`

## 状態

Kaggle train v4 完了。best XY は `xy_likpf_scale_12` RMSE 13.916271 / within10 0.705260。

## 仮説

exp100 `pf_z_xy_slope` は単発 PF 候補だったため exp072 `likpf_mean` との比較が不公平だった。128 seed likelihood ensemble と scale ensemble に揃えれば、XY rate prior の価値を exp072 `pf_z` / `likpf_*` と同じ candidate metrics で判断できる。

## 検証方針

exp072 feature cache の同一 row set を使い、`last_known_tvt + target` を true TVT として scoring する。`xy_likpf_*` は raw train horizontal/typewell から prefix-only prior で再生成し、exp072 baseline と同じ `candidate_metrics`、bucket metrics、by-well metrics で比較する。

## 何を比較するか

exp072 cache から `pf_z` / `likpf_mean` と、存在する場合は `likpf_scale_3/5/8/12` を読み、同じ行集合に対して新しく `xy_likpf_mean` / `xy_likpf_scale_3/5/8/12` を生成して比較する。

## 実装メモ

- `xy_likpf` は 128 seed、500 particles、scale `[3, 5, 8, 12]`。
- 状態は exp072 と同じ `pos = TVT + Z`。
- XY rate prior は prefix の `d(TVT_input + Z)/dMD` を `dZ/dMD` と `dXY/dMD` で回帰する。
- 評価区間 true TVT は candidate metrics の scoring のみに使う。

## 予定生成物

- `exp103_pf_z_xy_likpf_ensemble_parity_candidate_metrics.csv`
- `exp103_pf_z_xy_likpf_ensemble_parity_bucket_metrics.csv`
- `exp103_pf_z_xy_likpf_ensemble_parity_by_well.csv`
- `exp103_pf_z_xy_likpf_ensemble_parity_xy_likpf_quality.csv`
- `exp103_pf_z_xy_likpf_ensemble_parity_candidate_wide.csv.gz`
- `exp103_pf_z_xy_likpf_ensemble_parity_candidate_long.csv.gz`
- `exp103_pf_z_xy_likpf_ensemble_parity_summary.json`

## 所見

`xy_likpf_scale_12` は exp072 `pf_z` RMSE 17.788171 を改善したが、exp072 `likpf_mean` RMSE 11.594898 には届かなかった。direct inference port / submit 候補にはしない。

ただし selector 候補・ML add-only feature 候補としては残す。`likpf_mean + exp072_pf_z + xy_likpf_scale_12` の oracle は RMSE 7.808425 / within10 0.896735 で、`likpf_mean + exp072_pf_z` oracle RMSE 9.115201 / within10 0.861225 より headroom が大きい。
