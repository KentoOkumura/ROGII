# タスクリスト

## TODO

- Kaggle push 前に metadata と bootstrap 内 config の整合を確認する。
- Kaggle push 前に 1 active spec x 5 folds = 5 CNN models、0 LightGBM boosters、control / parent 再学習なしを `SESSION_NOTES.md` に追記する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- `KAGGLE_DIRECTION.md` の `heatmap_mdn_candidate_generator_probe` を確認した。
- `docs/legacy/steering/20260705-exp202-heatmap-mdn-candidate-generator-probe/` を作成した。
- `experiments/exp202_heatmap_mdn_candidate_generator_probe/` を作成した。
- `config.yaml` を `pf_beam` route、candidate generator probe、candidate union readout 用に更新した。
- train notebook source に `id`、mode score margin / entropy、heatmap topK candidate CSV、existing-plus-heatmap union oracle readout を追加した。
- train notebook source に plot 用 local 128-row candidate path npz + sample/rank index CSV 保存を追加した。
- inference notebook source を diagnostic-only guard に更新した。
- Jupytext で train / inference `.ipynb` を再生成した。
- `py_compile` を通した。
- `ruff --select F821` を通した。
- `jupytext --to ipynb --test` を train / inference に対して通した。
- `make validate-exp EXP=exp202_heatmap_mdn_candidate_generator_probe` を通した。
- 2026-07-06 の path 保存追実装後に、train source の `py_compile`、`ruff --select F821`、`jupytext --to ipynb --test`、`make validate-exp EXP=exp202_heatmap_mdn_candidate_generator_probe` を通した。
