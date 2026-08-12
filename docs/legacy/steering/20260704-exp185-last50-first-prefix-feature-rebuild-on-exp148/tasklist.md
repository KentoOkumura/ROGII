# タスクリスト

## 進行中

- なし

## TODO

- なし

## 完了

- backlog 要件を確認。
- `docs/legacy/steering/20260704-exp185-last50-first-prefix-feature-rebuild-on-exp148/` を作成。
- `experiments/exp185_last50_first_prefix_feature_rebuild_on_exp148/` を作成。
- GPU train push 前ガードを `SESSION_NOTES.md` に記録。
- Jupytext で `.ipynb` を再生成する。
- `py_compile`、`ruff --select F821`、`make validate-exp` を実行する。
- Kaggle feature cache notebook を prepare / push する。
- feature cache 完了を確認し、feature cache 監視を停止する。
- Kaggle GPU split train 3本を prepare する。
- `train_lgb0` / `train_lgb1` を Kaggle GPU で push する。
- GPU split 実行失敗を受け、split train を CPU 実行へ切り替える。
- CPU metadata で `train_lgb0` / `train_lgb1` / `train_lgb2` を再 prepare / push する。
- CPU split の `prefix_crop_variant_join_start` 後 `DeadKernelError` を確認し、メモリ対策を実装する。
- メモリ対策版 CPU split を再 prepare / push する。
- Kaggle logs / notebook output からメモリ対策版 CPU split の CV と生成物を記録する。
- 結果に応じて `result.md`、`metrics.json`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` を更新する。

## ブロック中

- なし
