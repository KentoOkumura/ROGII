# exp146_tvt_plus_z_beam_smoothness_penalty

## 目的

Beam search の cost に `TVT + Z` と `dTVT/dMD + dZ/dMD` の平滑性 penalty を追加し、exp072 の既存 PF/Beam/likPF 候補と同じ train pseudo-tail rows で比較する。

## 状態

- `completed_train_side_beam_improved_not_adopted_no_submit`
- Kaggle train v1: https://www.kaggle.com/code/kentookumura/exp146-tvt-z-beam-smooth-train
- Output: `experiments/exp146_tvt_plus_z_beam_smoothness_penalty/kaggle/output/train_v1`
- 推論 port / 提出はしない。

## 仮説

exp083 v11 のように exp072 Beam mean が平坦すぎる Z-driven 区間では、`TVT` 単体の move penalty ではなく `U = TVT + Z - (T0 + Z0)` と `dU/dMD` を cost に入れることで、坑跡 Z の変化と整合する Beam path を残せる可能性がある。

## 方針

- raw train horizontal/typewell から Beam search を再実行する。
- `U = TVT + Z - (T0 + Z0)`、`dU/dMD`、任意の `dU/dMD` curvature を cost に入れる。
- true TVT は scoring にだけ使う。
- exp142 v2 の混在実装は無効とし、この exp146 を正しい実装先にする。

## 検証方針

- exp072 cache と同じ train pseudo-tail rows で RMSE / MAE / within10 を比較する。
- `likpf_mean`、exp072 `beam_mean`、`pf_ancc`、`pf_z`、Beam replay、TVT+Z penalty Beam variants を同じ rows で評価する。
- near-prefix、longtail、representative Z-driven wells、Beam/likPF disagreement top quartile、worst-well regression、path roughness を見る。

## 所見

- 主比較対象は従来 `beam_mean`。best generated Beam variant `tvt_plus_z_uslope_c100_uabs005` は RMSE 15.566811180 で、`beam_mean` 15.774327032 から -0.207515852 改善した。
- `near_000_050`、`longtail_1000_plus`、Beam/likPF gap top quartile でも `beam_mean` よりは改善した。
- ただし採用ガードの `likpf_mean` は RMSE 11.594897672 / within10 0.772807479 で大きく強いため、direct Beam candidate / inference port / submit はしない。
- exp142 v2 は混在実装のため、この backlog の結果としては使わない。

## 出力予定

- `exp146_tvt_plus_z_beam_smoothness_penalty_candidate_metrics.csv`
- `exp146_tvt_plus_z_beam_smoothness_penalty_bucket_metrics.csv`
- `exp146_tvt_plus_z_beam_smoothness_penalty_by_well.csv`
- `exp146_tvt_plus_z_beam_smoothness_penalty_group_metrics.csv`
- `exp146_tvt_plus_z_beam_smoothness_penalty_beam_quality.csv`
- `exp146_tvt_plus_z_beam_smoothness_penalty_candidate_wide.csv.gz`
- `exp146_tvt_plus_z_beam_smoothness_penalty_summary.json`
