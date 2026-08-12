# 設計

## アプローチ

`exp006` の診断結果を元に、classifier ではなく rule-based hard router を実装する。router は well ごとの `condition_*` から route を決め、`all_gr`、`no_gr`、`guarded` のいずれかを選ぶ。

selected variant は以下の順で判定する。

1. `gr_weak_all` は `no_gr`
2. `short_prefix_low_gr` は `no_gr`
3. `large_gr_shift_low_gr` は `no_gr`
4. 残りの `gr_weak_any` は `guarded`
5. それ以外は `all_gr`

`guarded` は exp005 selected と同じ、prefix/eval の GR 欠損率が両方しきい値以上のときだけ no-GR に倒す strict gate とする。

## 実験範囲

- 対象実験: `exp007_hard_well_router`
- 親実験: `exp006_hard_well_router_diagnostic`
- 変更する変数: hard router の route 条件と selected variant
- 固定する変数: residual model、feature columns、sampling、GroupKFold、seed、GR/formation の利用方針

## リスク

- リークリスク: exp006 の target-derived bucket は設計参考に留め、実装 route は inference-safe condition だけで決める。
- CV/LB 不一致リスク: no-GR の Public LB は弱かったため、strong-GR wells は all-GR default に残す。
- ランタイム/メモリリスク: router variant は all-GR/no-GR の 2 モデルを学習するため、exp005/exp006 と同程度の Kaggle 実行時間を想定する。
