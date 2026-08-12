# 設計

## アプローチ

`exp015` の `baseline.py` から PF/beam feature builder を import し、`data/raw/train`
の horizontal/typewell pair で candidate paths を再計算する。学習は行わず、
`exp013` の raw LightGBM OOF と row index で結合して direct RMSE を比較する。

監査対象:

- direct candidates: `pf_mean`、`pf_best`、`pf_s3/s5/s8/s12`、hold blend
- controls: `raw_lightgbm_no_gr`、`last_anchor`、`recent_linear`
- segments: distance bucket、best scale、confidence、GR missing、eval length、Z span、trajectory steepness
- well deltas: `exp015` の `control_lightgbm_no_gr` と `pf_beam_no_gr` の well-level RMSE 差

## 実験範囲

- 対象実験: `exp019_pf_beam_candidate_quality_audit`
- 親実験: `exp015_public_pf_beam_scale_selector_features`
- 変更する変数: PF/beam candidate quality の診断粒度
- 固定する変数: raw OOF、評価 mask、PF/beam path generator、GroupKFold by well

## 出力

- `artifacts/pf_beam_candidate_metrics.csv`
- `artifacts/pf_beam_well_deltas.csv`
- `artifacts/pf_beam_scale_diagnostics.csv`
- `artifacts/pf_beam_top_hurt_help.csv`
- `artifacts/pf_beam_candidate_quality_summary.json`
- `metrics.json`

## リスク

- リークリスク: train fold 全体の direct candidate 診断なので、選択・提出には使わない。
- CV/LB 不一致リスク: direct candidate が一部 group で良くても hidden distribution で再現するとは限らない。
- Kaggle input リスク: `exp013` train output と `exp015` train output を `kernel_sources` として付ける必要がある。
- ランタイム/メモリリスク: PF/beam path を全 well で再計算するため、数分かかる可能性がある。
- 解釈リスク: direct candidate が raw より悪い場合でも、confidence feature としての局所価値が完全にゼロとは限らない。
