# 要件

## 依頼

`test_time_prefix_calibration_audit` を実装する。見えない test well で見えている `TVT_input` prefix だけを使い、exp026 fixed bucket shrink anchor に足せる軽量な per-well 補正を監査する。

## 制約

- Route: `ml_model`
- 親実験は `exp026_pseudo_tail_bucket_shrink_inference_submit`。
- base model と fixed bucket shrink control は exp026 と同じにする。
- 補正 fit に使えるのは、validation well 内で元から finite な `TVT_input` prefix の疑似 hidden 区間だけ。
- 元の `TVT_input` が NaN の score tail 真値は補正 fit に使わない。
- Public LB では選ばず、original-fold holdout と well-hash holdout の両方を見る。
- Kaggle Notebook 実行を正とし、ローカル notebook 実行はしない。

## 受け入れ基準

- 新規実験 `exp036_test_time_prefix_calibration_audit` が作成されている。
- train notebook から audit 実装を実行できる。
- bias、slope、range/global residual scale、distance-bucket shrink、near-continuity 補正候補が比較される。
- overall、fold、well-hash holdout、row distance bucket、cutoff diagnostic が artifact として保存される。
- `task validate-exp EXP=exp036_test_time_prefix_calibration_audit` が通る。
