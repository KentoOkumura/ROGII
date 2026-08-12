# タスクリスト

## TODO

- Kaggle push 前に variant 1、config 1、fold 5、booster 5、
  control 0、classifier 0、GPU 0 を `SESSION_NOTES.md` と config に固定する。
- canonical Notebook 採用と Kaggle 実行の承認範囲を確認する。
- Kaggle metadata と bootstrap 内 config の整合を確認する。
- Kaggle private CPU Stage B を実行し、入力 / offset / model / OOF / gate SHA を記録する。
- 科学 gate を判定し、`result.md`、`metrics.json`、`SESSION_NOTES.md` を更新する。
- 完了時に `experiment_summary.md` と `KAGGLE_DIRECTION.md` を更新する。

## 進行中

- canonical Notebook採用とKaggle実行の承認待ち。

## ブロック中

- なし

## 完了

- 親 / exp407 candidate-score OOF の row / schema / candidate order parity を確認した。
- 3,783,989 base rows、45,407,868 candidate-long rowsを確認した。
- mean-shift-only と row-local-only counterfactual を計算した。
- inverse-RMSE weight と row-local score / calibration drift の dose-response を確認した。
- 親 margin bucket 別 switch damage を確認した。
- `docs/06_reproducibility.md` に沿う再現性設計を記入した。
- exp414 の実験 scaffold を作成した。
- fold-safe candidate RMSE additive offset helper と unit test を実装した。
- root-cause counterfactual / treatment instability helper を実装した。
- compact self-contained Jupytext train 候補を別名で作成した。
- Jupytext 変換 / round-trip、py_compile、Ruff、dedicated testsを通した。
- strict `validate-exp` とstrict project validationを通した。
- 再利用化したroot-cause moduleで実OOFの9/9原因gateを再現した。
