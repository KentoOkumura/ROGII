# 設計

## アプローチ

`exp017_deterministic_dtw_addonly` は `exp012/exp013` の `lightgbm_no_gr` residual pipeline を固定し、入力特徴に `dtw_dwt_*` を追加する。

DTW/DWT feature は paired typewell GR に対して known prefix anchor から作る candidate TVT path を shift grid で探索し、最良 shift の multi-scale GR texture cost を row-level feature にする。DWT は外部 wavelet ライブラリを使わず、rolling smooth と detail energy で近似する。

## 実験範囲

- 対象実験: `exp017_deterministic_dtw_addonly`
- 親実験: `exp013_model_diversity_or_postprocess`
- 変更する変数: feature set に deterministic DTW/DWT alignment quality features を追加する。
- 固定する変数: `lightgbm_no_gr` の model class、主要 hyperparameter、GroupKFold by well、評価 mask、last-anchor residual target。

## Feature Set

- `dtw_dwt_best_shift_tvt`: prefix slope prior に対する typewell GR 最良 shift。
- `dtw_dwt_best_cost` / `dtw_dwt_cost_margin`: rolling smooth/detail energy と banded DTW cost から作る alignment quality。
- `dtw_dwt_best_ncc` / `dtw_dwt_best_dtw_cost`: 最良 route の GR 類似度と DTW cost。
- `dtw_dwt_best_slope`: 最良 route の TVT delta / MD delta slope。
- `dtw_dwt_w16_*`、`w32_*`、`w64_*`、`w128_*`: scale 別の eval/typewell detail energy、smooth error、energy error。
- `dtw_dwt_best_pred_tvt` / `dtw_dwt_best_typewell_gr` / `dtw_dwt_best_gr_abs_error`: 最良 route の row-level mismatch。

## リスク

- リークリスク: valid target は alignment には使わない。paired typewell の `TVT`/`GR` は inference でも利用できる補助曲線として扱う。
- ノイズ追加リスク: exp015 PF/beam add-only は悪化済み。今回も採用可否は `control_lightgbm_no_gr` との full CV 差分で判断する。
- ランタイムリスク: shift grid と DTW downsample を小さく保ち、first snapshot は feature quality audit として扱う。
