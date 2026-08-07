# exp143_multimode_pfbeam_local_correlation_audit

## 状態

- Route: `pf_beam`
- Status: `completed_scoped_train_side_diagnostic_no_submit`
- 親実験: `exp142_trajectory_aware_pf_transition_prior`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 提出候補: なし。train-side diagnostic only。

## 仮説

PF/Beam が局所 GR 相関の別モードを保持できず、resampling / pruning 後に早く単一モードへ潰れている可能性を確認する。直接 `likpf_mean` を超える候補を探すよりも、topK seed path に真値近傍や局所相関の別候補が残っているかを読む。

## 検証方針

exp072 train pseudo-tail cache と raw train horizontal/typewell を使い、strict exp072 PF-Z parity、strict multiseed、multimode PF variants、既存 `likpf_mean` / `pf_z` / `pf_ancc` / beam を同じ rows で比較する。

Kaggle v1 の full 773-well audit は timeout したため、v2 は代表/失敗 well 6 本、各 well 最大 2000 rows、300 particles、8 seeds、2 transition variants の scoped audit として実行する。

主要指標は RMSE / MAE / within10、effective sample size、resample count、collapse rate、seed mode count、mode entropy、seed TVT spread、local GR-correlation topK spread。true TVT は評価にだけ使い、local correlation と mode 診断は target-free 入力から計算する。

## 所見

Kaggle v3 scoped audit は完了。6 wells / 12,000 rows で 8 生成物を保存した。

主比較対象を従来 Beam の `exp072_beam_mean` と見ると、best multimode `multimode_pf_zacc_s010_a020_noise050_best_lik_seed` は RMSE 60.763085 で、`exp072_beam_mean` 70.297647 から -9.534561 改善した。

一方、best overall は `exp072_pf_ancc` RMSE 50.721842、`exp072_likpf_mean` は 52.758772、`exp072_pf_z` は 57.641691 で、best multimode はそれらに届かない。Beam 比では positive だが、direct PF/Beam 候補、inference port、submit はしない。

mode diversity は well 依存で、`1b1eba53` / `91b301ce` では z-accel variant が複数 mode と topK spread を残したが、`fb03ae90` / `86454a6f` ではほぼ単一 mode に潰れた。PF/Beam 本体の diversity 拡張より、confidence feature / segment verifier 側に下げる。

## 生成物

- `exp143_multimode_pfbeam_local_correlation_audit_candidate_metrics.csv`
- `exp143_multimode_pfbeam_local_correlation_audit_bucket_metrics.csv`
- `exp143_multimode_pfbeam_local_correlation_audit_by_well.csv`
- `exp143_multimode_pfbeam_local_correlation_audit_strict_pf_z_quality.csv`
- `exp143_multimode_pfbeam_local_correlation_audit_multimode_pf_z_quality.csv`
- `exp143_multimode_pfbeam_local_correlation_audit_parity_diff.csv.gz`
- `exp143_multimode_pfbeam_local_correlation_audit_candidate_wide.csv.gz`
- `exp143_multimode_pfbeam_local_correlation_audit_summary.json`
