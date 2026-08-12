# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- バックログ最上位の実験範囲確認
- `exp012` から `exp013_model_diversity_or_postprocess` を作成
- `config.yaml`、`settings.py`、notebook 名、README、SESSION_NOTES、result.md を exp013 用に更新
- `baseline.py` に inference-safe postprocess helper を追加
- train notebook に row-level OOF artifact と postprocess CV 比較 artifact を追加
- inference notebook に selected postprocess の適用を追加
- `validate_experiment.py`、ruff、py_compile、Kaggle notebook package 生成、pytest を実行して通過
- Kaggle full CV を実行し、postprocess artifacts を取得
- full CV 結果に基づいて `config.yaml` の `postprocess.selected_method` と params を固定
- `experiment_summary.md` と `KAGGLE_DIRECTION.md` を結果で更新
- `distance_bucket_shrink` 固定の inference notebook を prepare / push
- Kaggle output を取得し、submit-check を実行
- Kaggle submit を実行し、Public LB 12.271 を確認
- `SUBMISSIONS.md`、metrics、result、summary、direction を提出結果で更新
