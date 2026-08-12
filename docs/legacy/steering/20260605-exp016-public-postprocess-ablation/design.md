# 設計

## アプローチ

`exp013` の `row_oof_predictions.csv` から `lightgbm_no_gr` の行だけを読み、
後処理候補ごとに予測を生成する。予測配列は候補ごとに使い捨て、
全候補の row-level 予測を同時保持しない。

各候補について以下を集計する。

- 全 OOF RMSE
- original CV fold 別 SSE / N
- stable well-hash fold 別 SSE / N
- row distance bucket 別 SSE / N

fold 外 selection audit では、各 holdout fold について残り fold 上で
RMSE が最小の候補を選び、holdout fold で評価する。

## 実験範囲

- 対象実験: `exp016_public_postprocess_ablation`
- 親実験: `exp013_model_diversity_or_postprocess`
- 変更する変数: 後処理候補のみ
- 固定する変数: OOF source、fold、raw model、評価 mask、metric

## リスク

- リークリスク: 同一 OOF の best 候補選択は楽観的なので、fold 外 selection audit を主判断にする。
- CV/LB 不一致リスク: Public LB は visible 例の構造に依存しやすいため、この実験は提出しない。
- ランタイム/メモリリスク: 1.1GB OOF を読むため、候補予測は逐次生成し、artifact には集計だけを保存する。
