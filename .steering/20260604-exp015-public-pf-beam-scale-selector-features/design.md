# 設計

## アプローチ

`exp013` の LightGBM residual / postprocess pipeline を親にし、追加特徴だけを `pf_beam_*` として切り出す。

PF/beam feature は deterministic な lightweight snapshot として実装する。paired typewell GR に対して複数 scale の candidate TVT path を作り、GR 類似度、pseudo likelihood、beam confidence、hold weight、candidate path の mean/std/range/cost margin、eval length / Z span selector を row-level feature にする。standalone PF/beam 予測は採用せず、LightGBM residual model の入力特徴として比較する。

## 実験範囲

- 対象実験: `exp015_public_pf_beam_scale_selector_features`
- 親実験: `exp013_model_diversity_or_postprocess`
- 変更する変数: feature set に PF/beam selector/candidate/divergence features を追加する。
- 固定する変数: `lightgbm_no_gr` の model class、主要 hyperparameter、GroupKFold by well、評価 mask、last-anchor residual target。

## リスク

- リークリスク: paired typewell / GR alignment が valid well の true hidden TVT を参照しないよう、candidate path は prefix anchor と geometry/GR だけで作る。formation train-only columns は使わない。
- CV/LB 不一致リスク: public-visible branch は 見えない test で使える ではないため主軸にしない。比較は raw LightGBM control、PF/beam add-only variant、既存 postprocess anchor を分ける。
- ランタイム/メモリリスク: full public PF seed 数は使わず、scale 4 本、shift grid 小さめ、DTW downsample ありで first CV を走らせる。Kaggle train log の wall time を artifact として記録する。
