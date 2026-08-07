# 設計

## アプローチ

exp003 の `feature_no_gr_signal` は CV では改善したが Public LB では exp002 より悪化した。exp004 では GR signal features を全削除せず、exp002 と同じ all-GR residual model を default prediction とし、well 条件が rule に一致する場合だけ no-GR residual model の予測へ blend する。

各 gating variant は outer fold ごとに次の 2 つを train fold だけで学習する。

- base model: `model.feature_set=all`、exp002 相当。
- alternate model: `model.feature_set=no_gr_signal`、exp003 selected variant 相当。

validation well では inference-safe な condition features を作り、rule match 時に `prediction = base + weight * (alternate - base)` とする。`weight=1.0` は hard gate、`weight=0.5` は shrink/blend。

## 実験範囲

- 対象実験: `exp004_gr_gating`
- 親実験: `exp003_residual_ablation`
- 評価 control: `exp002_drift_minimal`
- 変更する変数: GR gate rule、gate weight、alternate feature set
- 固定する変数: GroupKFold、seed、HGB model params、residual target、sampling cap、residual shrink 0.85

## リスク

- リークリスク: gate condition に validation target や train-only formation columns を使わない。target は residual training/CV score のみに使う。
- CV/LB 不一致リスク: exp003 で既に CV 改善と Public LB 悪化が起きたため、control row と visible well の gate weight を必ず確認する。
- ランタイム/メモリリスク: gated variant は fold ごとに 2 モデルを fit するため、exp003 より重い。variant 数を必要最小限にする。
