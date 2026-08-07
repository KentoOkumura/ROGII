# exp050_xgboost_pseudo_tail_inference_submit

## 状態

- ルート: MLモデル
- 状態: submitted
- CV: 12.779452
- Public LB: 12.083
- Private LB: -
- Submit ID: 53521999
- 作成日: 2026-06-10
- 親実験: `exp049_xgboost_pseudo_tail_residual`

## 仮説

`exp049` の XGBoost pseudo-tail fixed bucket-shrink は通常 CV で `exp026` を上回ったため、同じ構成を inference flow に移植すると ML route の Public LB 基準 `exp039` 11.740 または旧自前基準 `exp026` 12.102 に近づく可能性がある。

## 検証方針

Kaggle inference notebook で full train fit から `submission.csv` を生成し、submit-check、予測範囲、exp026 submission との差分を確認する。competition submit はユーザー確認後に限定する。

## 所見

Kaggle inference version 1 が完了し、`submission.csv` は submit-check PASS。prediction range は 11587.960181 - 12234.905349、exp026 submission との差分 RMSE は 1.431860。code submit ref `53521999` の Public LB は 12.083 で、exp026 12.102 から -0.019 改善した。

## 参照ファイル

- 設定: `config.yaml`
- セッションノート: `SESSION_NOTES.md`
- 結果: `result.md`
- メトリクス: `metrics.json`
- 学習 notebook: `exp050_xgboost_pseudo_tail_inference_submit_train.ipynb`
- 推論 notebook: `exp050_xgboost_pseudo_tail_inference_submit_inference.ipynb`
