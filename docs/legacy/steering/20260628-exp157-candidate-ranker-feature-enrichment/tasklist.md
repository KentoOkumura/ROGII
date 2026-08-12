# タスクリスト

## TODO

- Kaggle train を実行する場合は、CPU runtime、active experiment 1、LightGBM family 3、fold 5、合計 booster 15、control 再学習なしを再確認してから push する。
- output 取得後に `result.md`、`metrics.json`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` を更新する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成。
- exp157 実験フォルダを exp101 から作成。
- exp099 cache に `tvt_dense*` が無いことを確認し、exp072 auxiliary cache join 方針にした。
- exp157 config / README / result / SESSION_NOTES / metrics 初期状態を作成。
- `candidate_ranker_feature_enrichment.py` に dense candidate / feature enrichment 実装を追加。
- train / inference notebook を exp157 用に更新。
- py_compile、notebook JSON check、ruff check、ruff format --check、`validate_experiment.py` を確認。
- synthetic frame smoke で dense candidate / feature enrichment を確認。
- Kaggle train package を prepare し、metadata の CPU runtime と exp099/exp072 kernel sources を確認。
