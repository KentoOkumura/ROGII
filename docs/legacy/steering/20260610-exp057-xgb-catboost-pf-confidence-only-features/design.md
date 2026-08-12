# 設計

## アプローチ

`exp055_single_model_pseudotail_training` の train-side audit 実装を親にする。
この実装は exp029 の public sel15 PF/Beam OOF 風 artifact を読み、well-level
original-fold / well-hash holdout で候補を比較できるため、PF/Beam confidence
feature の小さい検証に流用できる。

特徴量は 2 family に分ける。

- `geometry_anchor_context`: cutoff、row distance、XYZ/MD、GR availability、last anchor。
- `pf_beam_confidence_only`: PF likelihood / entropy / effective particles、PF seed/scale spread、PF/anchor disagreement、PF/Beam disagreement、beam spread、利用可能な PF/exp026 disagreement proxy。

model variant は次の 4 つにする。

- `xgb_geometry_control`
- `xgb_pf_confidence_only`
- `catboost_geometry_control`
- `catboost_pf_confidence_only`

それぞれ raw と fixed bucket-shrink を保存し、confidence-only variant は対応する
geometry control を original-fold と well-hash の両方で上回った場合だけ supported
candidate とする。

## 実験範囲

- 対象実験: `exp057_xgb_catboost_pf_confidence_only_features`
- Route: `ml_model`
- 親実験: `exp049_xgboost_pseudo_tail_residual`
- 実装親: `exp055_single_model_pseudotail_training`
- supporting artifact: `exp029_public_sel15_pf_oof_feature_generation`
- 変更する変数: XGBoost / CatBoost estimator と PF/Beam confidence-only feature family
- 固定する変数: exp029 artifact、last-anchor residual target、residual shrink/clip、exp014 bucket-shrink、well-level holdout 2 系統

## リスク

- リークリスク: `pf_error` など target との差分診断を特徴に入れると即リークになるため excluded feature とする。PF/Beam 予測値そのものや exp026_oof 予測値も特徴にしない。
- CV/LB 不一致リスク: exp029 pseudo-test surface は direct public PF controls が強く、実提出 LB とは一致しない例がある。supported になっても即 submit せず、別途 inference port audit を必要とする。
- ランタイム/メモリリスク: XGBoost と CatBoost を 2 holdout x 5 splits で回すため、Kaggle CPU 時間が重い。row cap と distance-balanced sampling は exp055 と同じ値で固定する。
- artifact リスク: 現在の local artifact は cutoff 0.65 のみ。0.45 / 0.82 は missing として記録されるため、multi-cutoff artifact が完成するまで pseudo-tail cutoff 比較は限定的。
