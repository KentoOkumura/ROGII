# 設計

## アプローチ

`studies/trajectory_feature_error_audit.py` を追加し、既存 Kaggle full CV output を
再集計する。新しい学習や推論は行わない。

処理:

1. `exp010` の `well_metrics.csv` から variant ごとの well RMSE を読み込む。
2. `control_exp003_no_gr` を CV control とし、`trajectory_direction_no_gr`、
   `trajectory_slope_no_gr`、`trajectory_full_no_gr` の差分を計算する。
3. `trajectory_full_all` は `control_exp002_all` との差分として別途保持する。
4. `exp006` の router diagnostic tags を結合し、GR missing / hard-no-GR /
   public-like keep-all-GR / trajectory slope 条件を付与する。
5. fold、eval length bin、prefix fraction bin、GR missing bin、trajectory slope bin、
   router bucket で weighted RMSE と hurt rate を集計する。

## 実験範囲

- 対象実験: `trajectory_feature_error_audit`
- 親実験: `exp010_trajectory_drift_ablation`
- 変更する変数: なし。既存 output の診断のみ。
- 固定する変数: CV split、model、feature variants、seed、Kaggle output。

## リスク

- リークリスク: 既存 OOF well metrics と inference-safe condition tags のみ使うため低い。
- CV/LB 不一致リスク: 診断は CV well 全体が対象。public-like wells の局所判断を避ける。
- ランタイム/メモリリスク: CSV 集計のみで低い。
