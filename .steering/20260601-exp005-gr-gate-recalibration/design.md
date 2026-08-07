# 設計

## アプローチ

exp004 の実装を親としてコピーし、all-GR base model と no-GR alternate model を同一 fold 内で学習する gated model bundle は維持する。変更は `ablation.variants` の gate 条件だけに限定する。

exp004 の問題は、`prefix_gr_missing_rate >= 0.35` または `eval_gr_missing_rate >= 0.40` の any 条件で no-GR weight 1.0 を与えたため、visible `000d7d20` が eval coverage だけで no-GR へ完全に倒れた点にある。exp005 では次を比較する。

- exp004 selected gate の再実行。
- exp004 と同じ any 条件だが no-GR weight 0.5 の soft gate。
- prefix/eval の両方が低 coverage の場合だけ no-GR weight 1.0 の strict gate。
- strict 条件で no-GR weight 0.5 の soft gate。

## 実験範囲

- 対象実験: `exp005_gr_gate_recalibration`
- 親実験: `exp004_gr_gating`
- 変更する変数: `model.gr_gating.combine`、coverage thresholds、`alternate_weight`
- 固定する変数: HGB hyperparameters、sampling、feature sets、fold split、residual shrink

## リスク

- リークリスク: gate 条件は inference-safe な GR missing rate と trajectory/prefix summaries のみを使い、valid fold の target や train-only formation columns は使わない。
- CV/LB 不一致リスク: strict gate は CV が exp004 よりわずかに悪い可能性がある。visible/public 寄り well を守る目的なので、CV だけでなく visible duplicate sanity を併用する。
- ランタイム/メモリリスク: gating variant は base/alternate の 2 モデルを学習する。exp004 より variant 数が 1 つ多いため train runtime は少し増えるが、モデルと sampling は同一。
