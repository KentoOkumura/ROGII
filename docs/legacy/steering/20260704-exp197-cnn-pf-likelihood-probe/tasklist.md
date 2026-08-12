# タスクリスト

## TODO

- Kaggle train v1 を push / 実行し、logs から candidate AUC、topK、negative control margin を記録する。
- output 取得が必要な場合だけ feature content SHA、prediction SHA、model SHA を記録する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- backlog `cnn_pf_likelihood_probe` の要件整理。
- 再現性設計を `design.md` に記入。
- 実験 scaffold 作成。
- train / inference Jupytext notebook 実装。
- `py_compile`、`ruff --select F821`、Jupytext `--test` 通過。
- `make validate-exp EXP=exp197_cnn_pf_likelihood_probe` 通過。
- Kaggle train package strict prepare 完了。metadata は GPU T4、internet off、exp099/exp111 kernel source 付き。
