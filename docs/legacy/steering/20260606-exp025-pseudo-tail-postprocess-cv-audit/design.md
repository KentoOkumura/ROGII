# 設計

## アプローチ

`exp023` は row-level OOF を保存していないため、`exp025` train notebook で `pseudo_tail_3_cutoffs_distance_balanced` を同じ GroupKFold-by-well split で再学習し、valid fold の row predictions をその場で集計する。

集計は大きな `row_oof_predictions.csv` を残さず、次を保存する。

- fixed candidate ごとの overall / fold / well-hash / bucket SSE
- raw residual と true residual の bucket 統計
- same-OOF fitted alpha
- leave-one-original-fold-out fitted alpha
- well-hash holdout fitted alpha
- fixed candidate の fold 外 selection 結果

## 実験範囲

- 対象実験: `exp025_pseudo_tail_postprocess_cv_audit`
- 親実験: `exp024_pseudo_tail_inference_postprocess`
- 変更する変数: postprocess 候補、bucket alpha の fit / selection 方法
- 固定する変数: pseudo-tail training recipe、LightGBM params、GroupKFold split、feature set、evaluation mask

## リスク

- リークリスク: alpha を同じ OOF rows に fit/evaluate すると過適合する。original-fold 外と well-hash holdout を別に保存して採用判断する。
- CV/LB 不一致リスク: exp024 raw は LB 改善済みなので、古い exp014 alphas が hidden で悪化する可能性がある。held-out 改善がない場合は raw を維持する。
- ランタイム/メモリリスク: selected pseudo-tail recipe の 5-fold 再学習が必要。row OOF CSV を保存せず aggregate のみ保存して output サイズを抑える。
