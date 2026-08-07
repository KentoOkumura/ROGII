# タスクリスト

## TODO

- Stage 2 external weights audit の要否を CV/LB 後に判断する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- backlog から steering docs を切り出した。
- `experiments/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/` を作成した。
- K16 spline / kernel kNN / adaptive kappa / ANCC local theta / GR correction / U-projection の source-port helper を実装した。
- train notebook source を作成した。
- inference notebook source を作成した。
- `config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json` を exp226 用に更新した。
- 再現性設計を `design.md` に記入した。
- Kaggle push 前コストを `SESSION_NOTES.md` に記録した。
- `py_compile`、`ruff --select F821,F401`、Jupytext conversion/test、`validate_experiment.py` を通した。
- 8 wells の helper function smoke で kappa fit / 1 well prediction が動くことを確認した。
- Kaggle train v1 を実行し、5-fold group-safe CV 9.427109596 と OOF decompressed SHA を記録した。
- Kaggle inference v1 を実行し、14,151-row `submission.csv` を生成した。
- Kaggle output を取得し、submit-check PASS を確認した。
- Code submit の Public LB 9.837 を確認し、`submissions/SUBMISSIONS.md` v060、`metrics.json`、`result.md`、`SESSION_NOTES.md`、`KAGGLE_DIRECTION.md` に記録した。
