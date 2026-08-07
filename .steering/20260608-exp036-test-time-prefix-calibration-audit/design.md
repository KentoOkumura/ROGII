# 設計

## アプローチ

exp026 と同じ `pseudo_tail_3_cutoffs_distance_balanced` recipe で fold model を学習する。validation well では元の finite `TVT_input` prefix をさらに途中で切り、cutoff 後から元の last known row までを calibration zone とする。calibration zone で pseudo prediction と true `TVT_input` の差を測り、元の hidden tail に対する exp026 control prediction を補正して score する。

比較候補:

- `exp026_bucket_shrink_control`: exp026 fixed `exp014_bucket_shrink_params`
- `prefix_bias_add`: calibration residual mean を加算
- `prefix_error_slope`: calibration residual を eval step に対して線形外挿
- `prefix_global_residual_shrink`: calibration zone で residual alpha を fit
- `prefix_distance_bucket_shrink`: calibration zone で distance bucket ごとの residual alpha を fit
- `prefix_near_continuity_decay`: prefix 末端近傍の residual を指数減衰で反映

## 実験範囲

- 対象実験: `exp036_test_time_prefix_calibration_audit`
- Route: `ml_model`
- 親実験: `exp026_pseudo_tail_bucket_shrink_inference_submit`
- 変更する変数: test-time prefix correction の候補だけ
- 固定する変数: base model、pseudo-tail augmentation、feature set、exp014 bucket shrink control

## 評価

train notebook で以下を保存する。

- candidate overall metrics
- original-fold selection audit
- stable well-hash selection audit
- row distance bucket summary
- fold metrics
- well-hash holdout metrics
- per-cutoff calibration diagnostics

## リスク

- リークリスク: 元の hidden tail 真値を補正 fit に使うと漏洩になる。実装では cutoff 後かつ original last known 以前の rows だけを calibration に使う。
- CV/LB 不一致リスク: visible prefix に過適応し、hidden tail に転移しない可能性がある。original-fold と well-hash holdout の両方で control 改善が必要。
- ランタイム/メモリリスク: 各 valid well で複数 cutoff の予測を追加するため exp025 より重い。cutoff は 3 個に制限する。
