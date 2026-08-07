# exp055_single_model_pseudotail_training

## 状態

- ルート: `ml_model`
- 状態: completed_no_supported_candidate
- CV: no supported candidate
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-10
- 親実験: `exp039_ravaghi_single_lgbm_inference_submit`

## 仮説

exp039 系の single LightGBM feature surface は Public LB 11.740 まで届いたが、exp051/052 系で効いた pseudo-tail training distribution はまだ正面から入れていない。同じ特徴面と estimator を固定し、学習 row policy だけを exp051 方式へ寄せることで、same-surface holdout が安定して改善するかを確認する。

## 検証方針

`exp039_same_surface_control` と `single_model_pseudotail_training` を、original-fold と well-hash holdout の両方で比較する。raw と fixed bucket-shrink は別候補として記録する。

## 所見

Kaggle train version 1 は完了。`single_model_pseudotail_training` は same-surface control を original-fold / well-hash の両方で上回らなかったため、推論 port しない。

## 参照ファイル

- 設定: `config.yaml`
- セッションノート: `SESSION_NOTES.md`
- 結果: `result.md`
- メトリクス: `metrics.json`
- 学習 notebook: `exp055_single_model_pseudotail_training_train.ipynb`
- 推論 notebook: `exp055_single_model_pseudotail_training_inference.ipynb`
