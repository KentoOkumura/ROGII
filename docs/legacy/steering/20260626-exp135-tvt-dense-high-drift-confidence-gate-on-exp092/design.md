# 設計

## 方針

exp092 `lgb1` を base prediction として固定し、exp072 full replay feature cache から raw-test-compatible な PF/Beam/dense 候補を読む。`target_tvt` は評価と oracle coverage 診断だけに使い、gate 条件には使わない。

この実験は posthoc audit であり、モデル学習、inference port、submission は行わない。

## 入力

- exp092 OOF prediction: `exp092_u_projection_correction_disagreement_fullrun_predictions.csv.gz`
- exp073 OOF prediction: `exp063_full_replay_repro_guard_predictions.csv.gz`
- exp072 feature cache: `exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz`
- raw train horizontal wells: MD / Z / GR / TVT_input から `md_since`、prefix/tail context を復元する。

## 候補

- base: `pred_exp092_lgb1`
- reference: `pred_exp073_lgb_mean`
- PF/Beam: `pred_likpf_mean`、`pred_pf_ancc`、`pred_pf_z`、`pred_beam_mean`
- dense: `pred_tvt_dense`、`pred_tvt_densew`、`pred_tvt_dense50`、`pred_tvtF_ANCC`

## Gate 条件

target-free proxy のみを使う。

- `tail_rank >= min_tail_rank`
- `dense_std_abs` high quantile
- `tvt_dense_d_abs` high quantile
- `pf_dense_abs_diff` high quantile
- `exp092_dense_abs_diff` high quantile
- `pf_beam_abs_diff` high quantile

scope は `segment` と `well` を比較する。segment gate では contiguous high-run が `min_segment_rows` 未満なら捨てる。well gate では high row rate が閾値以上の well だけを対象にする。

## 出力

- variant metrics
- gate variant diagnostics
- by-well metrics
- bucket metrics
- common worst metrics
- raw-test parity checklist
- prediction sample
- summary JSON

## 採用判断

採用候補にするには、exp092 から global OOF が改善し、common PF+ML worst 26 または PF `likpf_mean` worst50 を救い、near-row / low-drift / worst-well regression / path continuity を壊さないこと。global OOF の微小改善だけでは inference port しない。
