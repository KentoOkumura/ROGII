# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering requirements/design/tasklist を作成した。
- exp145 実験フォルダを作成した。
- exp111 saved model + exp112 schema を使う target-free generator module を追加した。
- train notebook を full-train cache generator 入口に更新した。
- inference notebook を raw-test feature generator 入口に更新した。
- `public_notebook_replay_audit.py` を同梱し、raw-test replay を実行できるようにした。
- Kaggle train/inference package を strict mode で生成した。
- `uv run python scripts/validate_experiment.py --experiment exp145_learned_likelihood_rawtest_feature_generator_parity` は通過した。
- `uv run ruff check` / `uv run ruff format --check` は generator module で通過した。
- Kaggle train v2 で full-train 3,783,989 rows / 773 wells の `ml_features` cache を生成し、schema parity pass を確認した。
- Kaggle inference v3 で raw-test 14,151 rows / 3 wells の `ml_features` cache と likelihood long cache を生成し、schema parity pass を確認した。
- result / README / metrics に Kaggle 結果と SHA を記録した。
- schema/coverage parity pass 後の次アクションを `learned_likelihood_fulltrain_addonly_on_exp092` として `KAGGLE_DIRECTION.md` に切り出した。
