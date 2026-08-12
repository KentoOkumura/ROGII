# タスクリスト

## TODO

- compact gate を継続する場合は、悪化 well を除外する narrower guard を別候補として設計する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- `docs/legacy/steering/20260618-exp078-compact-surface-longtail-gate/` を作成した。
- `experiments/exp078_compact_surface_longtail_gate/` を作成した。
- exp073/exp075 OOF align、long-tail gate audit、discussion metric guard を実装した。
- exp073/exp075 inference prediction blend による submission 生成を実装した。
- local OOF audit を実行し、baseline_exp073 が選択されたことを記録した。
- compact branch は global RMSE/SSE を改善したが、最大 well RMSE 悪化が guard を超えたため submit candidate ではないと判断した。
- `KAGGLE_DIRECTION.md` の該当 backlog を exp078 の結果に合わせて更新した。
