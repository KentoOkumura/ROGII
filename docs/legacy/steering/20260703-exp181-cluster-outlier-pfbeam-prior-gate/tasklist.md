# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 再現性設計を `design.md` に記入した。
- `docs/legacy/steering/20260703-exp181-cluster-outlier-pfbeam-prior-gate/` を作成した。
- `experiments/exp181_cluster_outlier_pfbeam_prior_gate/` を作成した。
- 補助スクリプト `cluster_outlier_pfbeam_prior_gate.py` を実装した。
- `config.yaml` を PF/Beam/likPF candidate audit 用に更新した。
- train / inference notebook の Jupytext percent source を exp181 用に更新した。
- `README.md`、`result.md`、`metrics.json`、`SESSION_NOTES.md` を初期化した。
- Kaggle train 前の variant/config/fold/booster 数を `SESSION_NOTES.md` に記録した。
- `py_compile`、`ruff --select F821,F401,E501`、Jupytext convert / `--test` を通した。
- `make validate-exp EXP=exp181_cluster_outlier_pfbeam_prior_gate` を通した。
- train / inference Kaggle notebook package を canonical kernel id で prepare した。
- `make update-summary` を実行した。
- Kaggle train v1 を実行し、`KernelWorkerStatus.COMPLETE` を確認した。
- Kaggle output を `/tmp/kaggle-output/exp181_cluster_outlier_pfbeam_prior_gate/train_v1` に取得した。
- `result.md`、`metrics.json`、`README.md`、`SESSION_NOTES.md` に train v1 の結果と no-submit 判断を記録した。
