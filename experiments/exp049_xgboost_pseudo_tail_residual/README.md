# exp049_xgboost_pseudo_tail_residual

## 状態

- ルート: MLモデル
- 状態: completed
- CV: 12.779452
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-10
- 親実験: `exp026_pseudo_tail_bucket_shrink_inference_submit`

## 仮説

`exp026` の pseudo-tail + distance-balanced LightGBM 構成で、残差モデルだけを XGBoost に差し替えると、公開上位 stack で見えているモデル多様性を ML ルート内で取り込める可能性がある。

## 検証方針

主評価は従来の well-level GroupKFold を維持する。raw XGBoost CV と、exp025-selected fixed bucket-shrink 後 CV を分けて保存し、`exp026` の 12.870780 を上回る場合だけ次段の補助検証へ進む。

## 所見

Kaggle train version 1 で full CV 完了。best は `xgboost_pseudo_tail_3_cutoffs_distance_balanced_exp014_bucket_shrink_params` 12.779452 で、`exp026` fixed bucket-shrink 12.870780 から -0.091328 改善した。

## 参照ファイル

- 設定: `config.yaml`
- セッションノート: `SESSION_NOTES.md`
- 結果: `result.md`
- メトリクス: `metrics.json`
- 学習 notebook: `exp049_xgboost_pseudo_tail_residual_train.ipynb`
- 推論 notebook: `exp049_xgboost_pseudo_tail_residual_inference.ipynb`
