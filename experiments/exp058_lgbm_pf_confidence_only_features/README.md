# exp058_lgbm_pf_confidence_only_features

## 状態

- ルート: `ml_model`
- 状態: completed
- CV: 15.945678 (`lgbm_capacity_pf_confidence_only_raw`, original-fold pseudo-test surface)
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-10
- 親実験: `exp051_pseudo_tail_lgbm_param_micro_tune`
- 推論親: `exp052_lgbm_capacity_pseudotail_inference_submit`
- 実装親: `exp057_xgb_catboost_pf_confidence_only_features`

## 仮説

`exp057` で PF/Beam confidence-only features は XGBoost / CatBoost の
geometry control を大きく改善した。一方で direct PF controls には未達だった。
そこで `exp051` の LightGBM capacity 設定を固定し、PF/Beam の予測値ではなく
信頼度・不一致診断だけを追加した場合に、same-run LightGBM geometry control を
壊さず改善できるかを確認する。

## 検証方針

`lgbm_capacity_geometry_control` と
`lgbm_capacity_pf_confidence_only` を original-fold と well-hash holdout の両方で
比較する。モデルパラメータ、cutoff filtering、distance-balanced sampling、row cap、
residual shrink、fixed bucket-shrink postprocess は `exp051/052` に合わせる。

## 注意

PF raw prediction、Beam raw prediction、public PF gate、hidden branch replacement は
特徴量に入れない。現在の exp029 artifact には `PF-exp052 diff` がないため、
`abs_pf_pred_minus_exp026_oof` を利用可能な proxy として使う。

## 所見

Kaggle train version 1 は完了。`lgbm_capacity_pf_confidence_only_raw` は対応する
`lgbm_capacity_geometry_control_raw` を original-fold / well-hash の両方で上回った。
ただし `pf090_hold010` と `public_pf_selector` の direct public PF controls には
届かないため、この結果だけで推論 port / submit はしない。

## 参照ファイル

- 設定: `config.yaml`
- セッションノート: `SESSION_NOTES.md`
- 結果: `result.md`
- メトリクス: `metrics.json`
- 学習 notebook: `exp058_lgbm_pf_confidence_only_features_train.ipynb`
- 推論 notebook: `exp058_lgbm_pf_confidence_only_features_inference.ipynb`
