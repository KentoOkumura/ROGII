# exp152_all_well_lightweight_multimode_beam_audit

## 目的

`exp143_multimode_pfbeam_local_correlation_audit` で 6 well scoped では改善した multimode PF/Beam 候補が、全 train well の軽量 tail slice でも従来 `exp072_beam_mean` を安定して上回るかを確認する。

## 状態

Kaggle train v1 完了。全 train well tail 500 rows の軽量監査では、multimode PF/Beam candidate は主比較の `exp072_beam_mean` に届かなかったため、不採用。inference port / submit / full-row minimal candidate cache 生成には進めない。

## 仮説

`exp143` の best multimode candidate が従来 `beam_mean` より良かった理由が 6 well の選び方だけでなければ、全 train well tail 500 rows でも `exp072_beam_mean` に対して RMSE / MAE と well-level 安定性で改善が残る。

## 方針

- Route: `pf_beam`
- 親実験: `exp143_multimode_pfbeam_local_correlation_audit`
- cache 親: `exp072_exp063_full_replay_feature_cache`
- 対象: 全 train well、各 well tail 500 rows
- 主比較: `exp072_beam_mean`
- 参考比較: `exp072_likpf_mean`, `exp072_pf_z`
- 候補: `multimode_pf_zacc_s010_a020_noise050_best_lik_seed`

## 軽量化

- exp143 の local GR correlation 診断は無効化する。
- transition variant は exp143 best の 1 種に限定する。
- multimode は 300 particles / 4 seeds で開始する。
- `candidate_long` は保存しない。
- `candidate_wide` は `id`, `well`, `row_idx`, exp072 anchors, lightweight multimode candidate を中心に最小列だけ保存する。

## 検証方針

Kaggle train notebook で全 train well の tail 500 rows を監査する。主指標は `multimode_pf_zacc_s010_a020_noise050_best_lik_seed` の `exp072_beam_mean` に対する RMSE / MAE delta、improved/worsened wells、最大悪化 well、distance / row bucket delta とする。

## 期待生成物

- `exp152_all_well_lightweight_multimode_beam_audit_candidate_metrics.csv`
- `exp152_all_well_lightweight_multimode_beam_audit_bucket_metrics.csv`
- `exp152_all_well_lightweight_multimode_beam_audit_by_well.csv`
- `exp152_all_well_lightweight_multimode_beam_audit_multimode_pf_z_quality.csv`
- `exp152_all_well_lightweight_multimode_beam_audit_candidate_wide.csv.gz`
- `exp152_all_well_lightweight_multimode_beam_audit_summary.json`

## 所見

386,407 rows / 773 wells で評価した。`multimode_pf_zacc_s010_a020_noise050_best_lik_seed` は RMSE 20.110701、`exp072_beam_mean` は 19.685742 で、Beam 比 +0.424958 悪化。by-well でも improved 364 / worsened 409 となり、exp143 の scoped positive は全 well では再現しなかった。

詳細は `result.md` と `metrics.json` を参照する。
