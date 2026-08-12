# 要件

## 依頼

`exp073_oof_feature_importance_error_readout` を実装する。新たな実験として作成するため、実験 ID は既存最新 `exp085` の次である `exp086_oof_feature_importance_error_readout` とし、ディレクトリ名の接頭辞に `exp073` は使わない。

## 制約

- Route: `ml_model`
- 親 anchor: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- 入力: exp077 policy OOF predictions、exp077 fold 平均 feature importance、exp072 full replay train feature cache
- 新しい教師ありモデル学習、推論、提出は行わない。
- 結果は次実験候補を絞るための診断として扱い、anchor 更新根拠にしない。
- Kaggle 実行時は kernel sources から入力生成物を読む。ローカル smoke では `/tmp/kaggle-output` の取得済み生成物を読む。
- 再現性: stochastic 処理は追加しない。入力生成物の SHA と output summary を記録する。

## 受け入れ基準

- 実験ディレクトリが `experiments/exp086_oof_feature_importance_error_readout/` として作成されている。
- `config.yaml` の `experiment.route` が `ml_model` で、parent/cache/readout parent が明記されている。
- train notebook が OOF readout を実行し、policy metrics、feature summary、feature quantile metrics、well summary、plot、summary JSON を保存する。
- inference notebook は diagnostic only とし、`submission.csv` を生成しないことが明記されている。
- `README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json` が exp086 の内容に更新されている。
- 静的検証として py_compile、notebook JSON validation、ruff、validate_experiment が通る。
