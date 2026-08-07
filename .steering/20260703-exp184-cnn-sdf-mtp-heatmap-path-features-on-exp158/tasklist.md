# タスクリスト

## TODO

- Kaggle train push 前に `prepare-kaggle-notebooks` を実行し、canonical kernel id / title と bootstrap support files を確認する。
- Kaggle train 実行後に `metrics.csv`、`by_well.csv`、`bucket_metrics.csv`、`subgroup_metrics.csv`、feature importance、heatmap feature summary を読んで採否を記録する。
- 実行後に `experiment_summary.md` と `KAGGLE_DIRECTION.md` を結果で更新し、backlog の扱いを決める。

## 進行中

- なし

## ブロック中

- なし

## 完了

- exp184 steering を作成した。
- exp183 の selector enrichment pattern から exp184 実験ディレクトリを作成した。
- config を heatmap path feature add-only 実験へ更新した。
- `cnn_sdf_mtp_heatmap_path_features_on_exp158.py` に exp182 validation prediction loader、sparse-to-row interpolation、heatmap row/candidate feature、heatmap diagnostics を実装した。
- train / inference percent source と `.ipynb` を exp184 用に更新した。
- `py_compile`、`ruff --select F821`、`jupytext --to ipynb --test` を実行した。
- `make validate-exp EXP=exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158` を実行した。
