# タスクリスト

## TODO

- Kaggle train を `kentookumura/exp192-typewell-hard-window-pct50-train` として push/run する。
- Kaggle completion 後に summary/schema/full train gzip を取得し、gzip integrity、row count、feature count、raw gzip SHA、decompressed SHA を記録する。
- exp072 full replay cache と row-wise に direct PF/Beam RMSE/MAE/within10 を比較する。
- 結果に基づき `result.md`、`metrics.json`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` を更新する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- `docs/legacy/steering/20260704-exp192-typewell-late-range-hard-window-pct50-full-cache-replacement/` を作成した。
- `experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/` を exp186 から作成した。
- soft-prior を無効化し、typewell 読み込み直後に `typewell_pct >= 0.50` hard-window filter を入れた。
- train / inference notebook 起点 `.py` を exp192 用に更新した。
- `py_compile`、`ruff --select F821`、Jupytext 変換/`--test`、`validate_experiment.py` を通した。
- Kaggle train package を `kentookumura/exp192-typewell-hard-window-pct50-train` / `exp192 typewell hard window pct50 train` で prepare した。
