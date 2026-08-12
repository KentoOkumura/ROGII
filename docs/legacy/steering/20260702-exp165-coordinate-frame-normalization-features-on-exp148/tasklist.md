# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成した。
- exp165 実験ディレクトリを exp163 CPU split 構成から作成した。
- exp148 を親に、coordinate-frame normalization features を add-only する設計にした。
- 学習 notebook を `lgb0` / `lgb1` / `lgb2` に分割する方針にした。
- coordinate-frame feature builder を実装した。
- `py_compile` と `ruff --select F821` を通した。
- Jupytext で正規 train/inference notebook と split train notebook を生成し、round-trip test を通した。
- `make validate-exp EXP=exp165_coordinate_frame_normalization_features_on_exp148` を通した。
- `prepare-kaggle-notebooks` で `train_lgb0` / `train_lgb1` / `train_lgb2` package を作成した。
- Kaggle metadata が CPU / internet off / run-on-push になっていることを確認した。
- `train_lgb0` / `train_lgb1` / `train_lgb2` を Kaggle に push した。
- push 後 status が 3 kernels とも `KernelWorkerStatus.RUNNING` であることを確認した。
- Kaggle 完了後に logs を取得し、fold 別 score / pooled OOF / 生成物パスを記録した。
- 大容量 prediction CSV のローカル取得が不安定だったため、Kaggle 内で3 split output を集約する `train_aggregate` notebook を追加した。
- `train_aggregate` v2 を Kaggle で完了し、3-model `lgb_mean` 8.549931602 / exp148 比 +0.048650420 を記録した。
- train-side OOF 悪化のため、推論化・提出しない判定を `result.md` / `metrics.json` / `SESSION_NOTES.md` に記録した。
