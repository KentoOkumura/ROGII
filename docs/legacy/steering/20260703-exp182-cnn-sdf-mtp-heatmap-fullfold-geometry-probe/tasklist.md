# タスクリスト

## TODO

- Kaggle train push 前に active run specs、fold 数、CNN model 数を `SESSION_NOTES.md` に追記する。
- Kaggle output 取得後に feature content SHA、prediction SHA、model SHA を記録する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- `docs/legacy/steering/20260703-exp182-cnn-sdf-mtp-heatmap-fullfold-geometry-probe/` を作成した。
- 再現性設計を `design.md` に記入した。
- stochastic 処理の stable seed policy と DataLoader `num_workers=0` 方針を決めた。
- `experiments/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe/` を作成した。
- train / inference notebook の Jupytext percent source を exp182 用に更新した。
- `py_compile` と `ruff --select F821` を通した。
- Jupytext train / inference 変換と `--test` を通した。
- `make validate-exp EXP=exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe` を通した。
- Kaggle train package を生成し、metadata の T4 GPU 指定を確認した。
