# タスクリスト

## TODO

- Kaggle train を push した場合は、logs / metrics / generated artifacts / SHA を `SESSION_NOTES.md` と `result.md` に記録する。
- train CV が positive の場合だけ、同じ exp190 内で current-test feature generation と inference port を検討する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- exp190 steering 作成。
- 既存 add-only train skeleton を exp190 用に移植。
- dcm OOF selector 読み込みを削除。
- target-free raw/rolling/Savitzky-Golay GR shift-scan feature generator を実装。
- control 再学習なし、単一 GPU notebook、15 boosters予定を `SESSION_NOTES.md` に記録。
- Jupytext conversion / `--test`、py_compile、ruff F821/F401、experiment validation を完了。
- `experiment_summary.md` に exp190 を反映。
- Kaggle train/inference package を exp190 kernel id/title で strict prepare。
- package 内 Python の py_compile を完了。
