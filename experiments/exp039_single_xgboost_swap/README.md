# exp039_single_xgboost_swap

## 状態

- ルート: `ml_model`
- 状態: completed
- CV: 16.029777 original-fold / 16.028160 well-hash
- Public LB: not submitted
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-10
- 親実験: `exp038_ravaghi_public_sel15_features_single_lgbm`

## 仮説

`exp039_ravaghi_single_lgbm_inference_submit` の single LightGBM 候補は ML route Public LB 11.740 の基準を作った。`exp050` では pseudo-tail residual estimator を LightGBM から XGBoost に替えることで小幅に LB が改善したため、Ravaghi/public sel15 feature surface でも estimator swap だけを切り出して評価する。

## 検証方針

`exp038` と同じ `exp029` 疑似 test rows、feature variants、well-level split、residual target、fixed bucket shrink を使う。candidate model の estimator だけ `XGBRegressor` に変更し、original-fold と well-hash の両方で `base_geometry_bucket_shrink` を上回るかを確認する。

## 所見

Kaggle train version 1 completed。selected XGBoost candidate は `base_plus_pf_beam_diagnostics_bucket_shrink` だが、exp038 single-LGBM selected 15.850147 と public PF control `pf090_hold010` 15.089532 を下回ったため、inference submission は作らない。

## 参照ファイル

- 設定: `config.yaml`
- セッションノート: `SESSION_NOTES.md`
- 結果: `result.md`
- メトリクス: `metrics.json`
- 学習 notebook: `exp039_single_xgboost_swap_train.ipynb`
- 推論 notebook: `exp039_single_xgboost_swap_inference.ipynb`

## 読み方

この README は実験フォルダの入口です。仮説、変更点、実行コマンド、出力、失敗理由、次のアクションは `SESSION_NOTES.md` と `result.md` を正とします。

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録します。
