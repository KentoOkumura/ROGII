# 設計

## アプローチ

保存済み OOF prediction を読む train-side posthoc audit とする。`exp102` は 773 wells 全体、`exp112` は 155 wells subset なので、全体指標と shared surface 指標を分ける。採用判断には shared surface を使い、exp102 全体スコアとは混ぜない。

## 実験範囲

- 対象実験: `confidence_gate_continuity_rawtest_parity`
- Route: `pf_beam`
- 親実験: `exp102_confidence_gated_likpf_fallback_on_exp101`
- 診断親: `exp112_learned_pf_likelihood_weight_or_feature_followup`
- 任意入力: `exp124_projection_confidence_error_map` 以降の dense/high-drift gate prediction
- 変更する変数: gate candidate の比較 surface、continuity guard、worst-well guard、raw-test parity checklist
- 固定する変数: 候補生成、exp101/112 model、PF/Beam cache、OOF prediction、gate 閾値

## 入力

- `exp102_confidence_gated_likpf_fallback_on_exp101_oof_predictions.csv.gz`
- `exp112_learned_pf_likelihood_weight_or_feature_followup_oof_predictions.csv.gz`
- raw-test parity checklist 用:
  - exp101 model manifest
  - exp101 feature schema
  - exp099 train feature cache
  - exp112 feature schema
- optional:
  - `exp124_projection_confidence_error_map_gate_predictions.csv.gz`

## 処理

1. exp102 から `likpf_mean_single`、`exp101_error_ranker_rowwise`、主要 low-switch gate を読む。
2. exp112 から `likpf_mean_single`、`learned_error_top1`、`gate_expected_error_m2p0_d20p0` を読む。
3. configured variants が全て存在する id のみを `fair_shared_surface` とする。
4. overall metrics、by-well metrics、distance / tail-rank bucket metrics を作る。
5. well 内の prediction step、candidate switch、baseline から変更された segment length を continuity metrics として作る。
6. exp102 likPF baseline の worst 26 / 50 wells で common-worst metrics を作る。
7. raw-test parity checklist と入力 SHA を記録する。

## 出力

- `exp125_confidence_gate_continuity_rawtest_parity_metrics.csv`
- `exp125_confidence_gate_continuity_rawtest_parity_by_well.csv`
- `exp125_confidence_gate_continuity_rawtest_parity_bucket_metrics.csv`
- `exp125_confidence_gate_continuity_rawtest_parity_continuity_by_well.csv`
- `exp125_confidence_gate_continuity_rawtest_parity_continuity_summary.csv`
- `exp125_confidence_gate_continuity_rawtest_parity_common_worst_metrics.csv`
- `exp125_confidence_gate_continuity_rawtest_parity_rawtest_parity_checklist.csv`
- `exp125_confidence_gate_continuity_rawtest_parity_prediction_sample.csv.gz`
- `exp125_confidence_gate_continuity_rawtest_parity_summary.json`

## 再現性設計

- seed policy: `no_new_rng_posthoc_saved_oof_audit`
- stochastic 処理の有無: exp125 自体はなし。上流 exp099/101/111 の stochastic cache/model に依存する。
- PF/Beam / likelihood-PF / seed bagging の有無: exp125 では再生成しない。raw-test regeneration は checklist で未実施として扱う。
- 並列処理と乱数の関係: なし。
- CPU/GPU runtime と deterministic flags: CPU-only。GPU 不要。
- train cache / test feature regeneration の SHA 記録方針: 入力 gzip は decompressed content SHA を記録する。hidden test feature regeneration SHA はこの実験では存在しない。
- model manifest / prediction / submission SHA 記録方針: exp101 manifest / schema の SHA、fair prediction content SHA を記録する。submission は作らない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` を実行し、generated package の config / support files を更新する。

## リスク

- リークリスク: saved OOF の評価のみなので新たな feature leakage はない。ただし gate 閾値の OOF 過適合は採用判断で割り引く。
- CV/LB 不一致リスク: shared surface は exp112 subset なので、全 hidden wells への一般化根拠ではない。
- ランタイム/メモリリスク: exp102 OOF が大きいため chunk load する。必要なら `EXPERIMENT_MAX_ROWS_PER_VARIANT` で smoke 可能。
- 再現性リスク: raw-test hidden regeneration を実施しないため、この実験単体では deterministic submission anchor にならない。
