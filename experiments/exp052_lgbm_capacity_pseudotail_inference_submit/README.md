# exp052_lgbm_capacity_pseudotail_inference_submit

## 状態

- ルート: MLモデル
- 状態: submitted
- CV: 12.634392
- Public LB: 12.076
- Private LB: -
- Submit ID: 53524340
- 作成日: 2026-06-10
- 親実験: `exp051_pseudo_tail_lgbm_param_micro_tune`

## 仮説

`exp051` の LightGBM capacity pseudo-tail fixed bucket-shrink は通常 CV で `exp049` XGBoost と `exp026` LightGBM control を上回ったため、同じ構成を inference flow に移植すると pseudo-tail 自前系の Public LB 基準 `exp050` 12.083 を改善できる可能性がある。

## 検証方針

`exp050` の inference flow を維持し、final residual estimator だけを `LGBMRegressor(num_leaves=47, min_child_samples=60)` にする。Kaggle output の `submission.csv` を submit-check し、予測範囲と exp026/exp050 との差分を確認してから code submit する。

## 所見

Kaggle inference version 1 が完了し、`submission.csv` は submit-check PASS。code submit ref `53524340` の Public LB は 12.076 で、exp050 12.083 から -0.007、exp026 12.102 から -0.026 改善した。

## 参照ファイル

- 設定: `config.yaml`
- セッションノート: `SESSION_NOTES.md`
- 結果: `result.md`
- メトリクス: `metrics.json`
- 学習 notebook: `exp052_lgbm_capacity_pseudotail_inference_submit_train.ipynb`
- 推論 notebook: `exp052_lgbm_capacity_pseudotail_inference_submit_inference.ipynb`
