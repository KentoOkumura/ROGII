# 設計

## アプローチ

`exp024_pseudo_tail_inference_postprocess` をコピーし、final fit / pseudo-tail augmentation / LightGBM params / feature set は固定する。`postprocess.selected_method` だけを `distance_bucket_shrink` に変え、exp025 で held-out selection された exp014 bucket alpha を設定する。

## 実験範囲

- 対象実験: `exp026_pseudo_tail_bucket_shrink_inference_submit`
- 親実験: `exp025_pseudo_tail_postprocess_cv_audit`
- 変更する変数: inference postprocess
- 固定する変数: selected training variant、pseudo cutoff recipe、LightGBM params、training row caps、feature set

## リスク

- リークリスク: postprocess alphas は exp025 held-out OOF selection に基づく。Public LB を見て調整しない。
- CV/LB 不一致リスク: exp024 raw Public LB 12.166 より悪化する可能性がある。submit 前に raw submission との差分と range を確認する。
- ランタイム/メモリリスク: exp024 と同じ final LightGBM fit のため、Kaggle inference runtime は exp024 と同程度の想定。
