# 設計

## アプローチ

`exp013` の `row_oof_predictions.csv` を primary frame として読み、同じ
`fold, well_id, row_id, row_index, eval_step` key で追加候補 OOF を結合する。
`last_anchor` は CSV 由来ではなく primary frame の `last_anchor` から仮想候補として作る。

Router は config の `audit.routers` に定義し、以下を同じ集計 API で評価する。

- `fixed`: 1 候補をそのまま使う。
- `weighted_blend`: 複数候補の固定重み blend。
- `distance_router`: row distance bucket ごとに候補または blend を切り替える。
- `disagreement_damped`: 候補間 spread が大きい遠方 row だけ raw residual を anchor 側へ縮める。
- `bucket_oracle`: bucket ごとの best candidate を同一 OOF で選ぶ上限診断。selectable から除外する。

## 実験範囲

- 対象実験: `exp018_candidate_distribution_router`
- 親実験: `exp013_model_diversity_or_postprocess`
- 変更する変数: OOF 候補の row-level routing / blend / damping
- 固定する変数: 学習済み候補 OOF、評価 mask、GroupKFold by well の original fold

## 出力

- `artifacts/candidate_router_metrics.csv`
- `artifacts/candidate_router_selection.csv`
- `artifacts/candidate_router_bucket_summary.csv`
- `artifacts/candidate_router_summary.json`
- `metrics.json`

## リスク

- リークリスク: same-OOF の best router は評価面に過適合する。fold 外 selection を別途出す。
- CV/LB 不一致リスク: OOF router が hidden distribution で再現するとは限らない。改善が fold 外で残らない場合は推論実装へ進まない。
- ランタイム/メモリリスク: row OOF は 3.8M rows で大きい。CSV は chunk 読みし、結合後の評価は pandas/numpy の配列処理に限定する。
- Artifact リスク: exp015 PF/beam row OOF はローカルから削除済み。optional として扱い、復元された場合だけ自動で含める。
