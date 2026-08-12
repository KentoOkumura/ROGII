# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- `docs/legacy/steering/20260705-exp199-typewell-hard-window-pct40-base-surface-keep-exp145-ll-on-exp148/` を作成した。
- `experiments/exp199_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148/` を exp148 から作成した。
- `config.yaml` を exp196 base surface + exp145 `ll_*` keep の mixed provenance 診断として更新した。
- train notebook source を exp196 base surface cache 参照に更新した。
- inference notebook source を train-side only の no-op contract に変更した。
- 再現性設計を `design.md` に記入した。
- Jupytext で canonical train / inference `.ipynb` を再生成した。
- `jupytext --to ipynb --test`、`py_compile`、`ruff --select F821`、`make validate-exp` を通した。
- exp196 base cache と exp145 learned-likelihood cache のローカル存在を確認した。
- コピー元由来の古い self-contained 派生ファイルを削除した。
- 削除後に `make validate-exp`、全 `.py` の `ruff --select F821`、全 `.py` の `py_compile` を再実行して通した。
- train package を `kentookumura/exp199-pct40-base-keep-ll-train` / `exp199 pct40 base keep ll train` で strict prepare し、metadata と package config を確認した。
- ユーザー依頼を受け、15 boosters / control なしの計画で Kaggle train v1 を push した。
- `kentookumura/exp199-pct40-base-keep-ll-train` kernel version 1 が `KernelWorkerStatus.RUNNING` であることを確認した。
- `kentookumura/exp199-pct40-base-keep-ll-train` kernel version 1 が `KernelWorkerStatus.COMPLETE` であることを確認した。
- Kaggle output を取得し、CV、fold metrics、生成物、SHA、解釈を `SESSION_NOTES.md` / `result.md` / `metrics.json` / `experiment_summary.md` に追記した。
