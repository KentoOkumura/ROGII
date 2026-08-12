# タスクリスト

## TODO

- output 取得後に feature content SHA、prediction SHA、model SHA を記録する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- `docs/legacy/steering/20260620-exp085-u-projection-feature-ablation/` を作成した。
- `experiments/exp085_u_projection_feature_ablation/` を `exp080_u_space_target_ablation` から作成した。
- `config.yaml` を U-space projection feature ablation 用に更新した。
- `u_projection_feature_ablation.py` を実装した。
- `README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json` を exp085 用に更新した。
- `uv run python -m py_compile` を通した。
- notebook JSON validation を通した。
- `uv run ruff check` を通した。
- `uv run python scripts/validate_experiment.py --experiment exp085_u_projection_feature_ablation` を通した。
- synthetic frame で U-projection feature builder の smoke test を通した。
- `prepare_kaggle_notebooks.py --strict` で train package を生成し、metadata と config の整合を確認した。
