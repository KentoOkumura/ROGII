# 設計

## アプローチ

exp005 の fold-safe CV runner を維持し、各 variant の OOF well metrics が揃った直後に router 診断を追加する。診断は `diagnostics.py` に切り出し、notebook からも保存済み artifact からも同じ処理を使う。

## 実験範囲

- 対象実験: `exp006_hard_well_router_diagnostic`
- 親実験: `exp005_gr_gate_recalibration`
- 変更する変数: OOF well-level diagnostic artifact の追加
- 固定する変数: HGB residual model、feature sets、GR gating variants、CV split、seed、row sampling

## 出力

- `router_diagnostic_well_tags.csv`
  - exp002 / exp003 / exp005 RMSE 差分
  - GR 欠損、prefix fraction、eval length、trajectory、GR shift
  - `hard_no_gr_candidate` / `public_like_keep_all_gr` / `ambiguous`
- `router_condition_summary.csv`
  - タグや condition group ごとの CV、delta、GR 欠損平均
- `router_candidate_rules.csv`
  - inference-safe rule 候補の selected wells、selected rows、OOF CV、all-GR 差分
- `router_diagnostic_metrics.json`

## リスク

- リークリスク: OOF target outcome で作ったタグをそのまま hidden test router 入力にしない。次実験で使う条件は `condition_*` のみ。
- CV/LB 不一致リスク: public-like wells を no-GR に倒しすぎると exp003 と同じ LB 悪化が再発する。
- ランタイム/メモリリスク: exp005 と同じ variants を再実行するため Kaggle train runtime は exp005 と同程度。ローカル診断 CLI は保存済み CSV のみを読む。
