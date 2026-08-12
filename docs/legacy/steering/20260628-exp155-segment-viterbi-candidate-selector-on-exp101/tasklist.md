# タスクリスト

## TODO

- Kaggle train を実行する場合は、CPU posthoc audit、Viterbi variants 16、新規 booster 0、control 再学習なしを確認してから push する。
- output 取得後に `result.md`、`metrics.json`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` を更新する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成。
- exp155 実験フォルダを作成。
- exp155 config / README / result / SESSION_NOTES / metrics 初期状態を作成。
- `segment_viterbi_candidate_selector_on_exp101.py` を実装。
- train / inference notebook を exp155 用に更新。
- py_compile、ruff check、ruff format --check、notebook JSON check、`make validate-exp` を確認。
- Kaggle train package を prepare し、metadata と bootstrap manifest の exp155 整合を確認。
