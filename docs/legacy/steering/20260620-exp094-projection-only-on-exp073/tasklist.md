# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering requirements/design を記入した。
- `exp094_projection_only_on_exp073` の experiment scaffold を作成した。
- projection-only audit 用 `config.yaml` を作成した。
- train/inference notebook を projection audit 用セル構成に更新した。
- exp073 OOF prediction と raw well context から projection grid を評価する補助スクリプトを実装した。
- `make validate-exp EXP=exp094_projection_only_on_exp073` が strict で通った。
- `make prepare-kaggle-notebooks EXP=exp094_projection_only_on_exp073 EXTRA_ARGS="--notebook train --run-on-push --strict"` が通り、train package metadata が `enable_gpu=false`、`enable_internet=false`、source `kentookumura/exp073-full-replay-repro-guard-train` になっていることを確認した。
- `make prepare-kaggle-notebooks EXP=exp094_projection_only_on_exp073 EXTRA_ARGS="--notebook inference --run-on-push --strict"` が通り、未選択状態の inference package metadata が `enable_gpu=false`、`enable_internet=false`、source `kentookumura/exp073-full-replay-repro-guard-infer` になっていることを確認した。
- Kaggle train v1 `kentookumura/exp094-projection-only-on-exp073-train` を完了し、output を `/tmp/kaggle-output/exp094_projection_only_on_exp073/train_v1` に取得した。
- Best `degree4_beta0.75_c2` は RMSE 9.399456024 で exp073 から -0.126918725 改善した。
- Near-row guard failed、全 variant guard 通過 0 件のため inference port しない判断で `SESSION_NOTES.md`、`result.md`、`metrics.json`、`KAGGLE_DIRECTION.md` を更新した。
