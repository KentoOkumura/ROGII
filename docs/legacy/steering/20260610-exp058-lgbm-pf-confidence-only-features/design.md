# 設計

## アプローチ

`exp057` の PF/Beam confidence audit 実装を再利用し、variant を LightGBM
capacity の paired control だけに絞る。入力は `exp029` の public sel15 PF/Beam
pseudo-test feature artifact を使い、original-fold と well-hash holdout の両方で
OOF 予測を作る。

Feature family は次の2つに分ける。

- `geometry_anchor_context`: cutoff、row index、eval distance、trajectory、GR availability、last anchor。
- `pf_beam_confidence_only`: PF likelihood / entropy / effective particles、PF seed/scale spread、PF/anchor disagreement、PF/Beam disagreement、beam spread、利用可能な PF/exp026 disagreement proxy。

Variant:

- `lgbm_capacity_geometry_control`
- `lgbm_capacity_pf_confidence_only`

それぞれ raw と fixed bucket-shrink を保存し、confidence-only variant は対応する
geometry control を両 holdout で上回った場合だけ supported とする。

## 実験範囲

- 対象実験: `exp058_lgbm_pf_confidence_only_features`
- Route: `ml_model`
- 親実験: `exp051_pseudo_tail_lgbm_param_micro_tune`
- 推論親: `exp052_lgbm_capacity_pseudotail_inference_submit`
- 実装親: `exp057_xgb_catboost_pf_confidence_only_features`
- 変更する変数: PF/Beam confidence-only feature family の有無
- 固定する変数: LightGBM capacity params、cutoff filtering、distance-balanced sampling、row cap、residual shrink、fixed bucket-shrink、validation audits

## リスク

- リークリスク: PF/Beam artifact は train well の pseudo-test masking 後に生成されたものだけを使う。target/error 診断列、raw PF/Beam prediction、exp026 raw prediction は特徴量から除外する。
- CV/LB 不一致リスク: exp029 pseudo-test surface の RMSE は通常 CV や Public LB と直接比較しない。paired control の差分を主判断にする。
- ランタイム/メモリリスク: 1.78M rows の2 audit x 2 variant で LightGBM を複数回 fit するため、Kaggle 上で実行し、row cap と bucket-balanced sampling を維持する。
