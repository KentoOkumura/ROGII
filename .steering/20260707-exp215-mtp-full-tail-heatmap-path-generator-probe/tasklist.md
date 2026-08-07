# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- `KAGGLE_DIRECTION.md` の exp215 backlog 内容を確認した。
- `docs/06_reproducibility.md` を確認した。
- `.steering/20260707-exp215-mtp-full-tail-heatmap-path-generator-probe/` を作成した。
- `experiments/exp215_mtp_full_tail_heatmap_path_generator_probe/` を exp202 から作成した。
- config を continuous MTP full-tail generator 用に更新した。
- train source を `path_pred [K,L]` + `path_logit [K]`、closest-mode path loss、dense full-tail generation、full-grid aggregation、candidate-union readout に差し替えた。
- inference source を diagnostic-only guard に差し替えた。
- 再現性設計を `design.md` に記入した。
- `py_compile`、`ruff --select F821`、Jupytext conversion/test、`make validate-exp` を通した。
- `make prepare-kaggle-notebooks` で train notebook package を作成した。
- Kaggle train push 前に GPU cost guard を再確認し、ユーザー承認を得た。
- Kaggle train v1 を T4 GPU で実行し、`KernelWorkerStatus.COMPLETE` を確認した。
- Kaggle logs から metrics を読み、`SESSION_NOTES.md`、`result.md`、`metrics.json`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` を更新した。
