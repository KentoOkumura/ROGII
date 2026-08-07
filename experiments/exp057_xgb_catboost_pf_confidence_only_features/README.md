# exp057_xgb_catboost_pf_confidence_only_features

## 状態

- ルート: `ml_model`
- 状態: completed
- CV: 15.609958 (`catboost_pf_confidence_only_raw`, original-fold pseudo-test surface)
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-10
- 親実験: `exp049_xgboost_pseudo_tail_residual`
- 実装親: `exp055_single_model_pseudotail_training`

## 仮説

PF/Beam の直接置き換えや gate は不安定だったが、PF の seed spread、
likelihood、effective particles、Beam spread、PF/anchor disagreement だけなら、
XGBoost / CatBoost の非線形相互作用で hard rows の residual 推定を改善できる
可能性がある。

## 検証方針

`xgb_geometry_control` / `catboost_geometry_control` と、それぞれに
`pf_beam_confidence_only` family を足した variant を original-fold と
well-hash holdout の両方で比較する。raw と fixed bucket-shrink は別候補として
記録する。

## 注意

現在の exp029 artifact には `PF-exp052 diff` がないため、
`abs_pf_pred_minus_exp026_oof` を利用可能な proxy として使う。exp052 OOF が
供給された場合は config の `data.desired_pf_model_diff` に合わせて差し替える。

## 所見

Kaggle train version 1 は完了。`catboost_pf_confidence_only_raw` は対応する
`catboost_geometry_control_raw` を original-fold / well-hash の両方で上回った。
ただし `pf090_hold010` と `public_pf_selector` の direct public PF controls には
届かないため、この結果だけで推論 port / submit はしない。

## 参照ファイル

- 設定: `config.yaml`
- セッションノート: `SESSION_NOTES.md`
- 結果: `result.md`
- メトリクス: `metrics.json`
- 学習 notebook: `exp057_xgb_catboost_pf_confidence_only_features_train.ipynb`
- 推論 notebook: `exp057_xgb_catboost_pf_confidence_only_features_inference.ipynb`
