# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- [x] `make new-steering EXP=exp127_learned_likelihood_features_on_exp092` を実行した。
- [x] `make new-exp EXP=exp127_learned_likelihood_features_on_exp092 SOURCE=experiments/exp092_u_projection_correction_disagreement_fullrun` を実行した。
- [x] 分岐Bの exp092 add-only feature 評価として requirements / design を記入した。
- [x] `config.yaml` を exp127 の shared-row feature audit 用に更新した。
- [x] `learned_likelihood_features_on_exp092.py` を追加した。
- [x] train / inference notebook を exp127 用に更新した。
- [x] README / SESSION_NOTES / result / metrics を exp127 用に更新した。
- [x] `.venv/bin/python -m py_compile experiments/exp127_learned_likelihood_features_on_exp092/learned_likelihood_features_on_exp092.py experiments/exp127_learned_likelihood_features_on_exp092/settings.py` を通した。
- [x] `.venv/bin/ruff check experiments/exp127_learned_likelihood_features_on_exp092/learned_likelihood_features_on_exp092.py experiments/exp127_learned_likelihood_features_on_exp092/settings.py` を通した。
- [x] `make validate-exp EXP=exp127_learned_likelihood_features_on_exp092` を通した。
- [x] `make prepare-kaggle-notebooks EXP=exp127_learned_likelihood_features_on_exp092 EXTRA_ARGS="--notebook train --run-on-push --strict"` を通した。
- [x] `make prepare-kaggle-notebooks EXP=exp127_learned_likelihood_features_on_exp092 EXTRA_ARGS="--notebook inference --run-on-push --strict"` を通した。
- [x] GPU quota 回復後、`kentookumura/exp127-train` として Kaggle train v1 を実行した。
- [x] Kaggle output を `experiments/exp127_learned_likelihood_features_on_exp092/kaggle/output/train_v1` に取得した。
- [x] `result.md`、`metrics.json`、`SESSION_NOTES.md`、`README.md` を train v1 結果で更新した。
- [x] `experiment_summary.md`、`KAGGLE_DIRECTION.md` を更新した。
