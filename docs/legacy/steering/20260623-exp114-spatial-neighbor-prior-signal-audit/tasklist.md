# タスクリスト

## TODO

- `spatial_neighbor_prior_confidence_gate_on_exp092` として、spatial prior を信用してよい row/well を判定する confidence / gate follow-up を検討する。
- ML に特徴量として入れる評価は `spatial_neighbor_prior_ml_features_on_exp092` として別 backlog に分ける。

## 進行中

- なし

## ブロック中

- なし

## 完了

- 再現性設計を `design.md` に記入した。
- `config.yaml` に route、lineage、validation、neighbor variants、出力契約を記入した。
- fold-safe spatial neighbor prior audit script を追加した。
- train notebook を入力確認、audit 実行、生成物確認セルに更新した。
- inference notebook を no-submission guard に変更した。
- `py_compile`、`validate-exp`、`prepare-kaggle-notebooks --notebook train --run-on-push --strict`、対象ファイルの `ruff check` が通った。
- Kaggle train v1 が complete し、output を取得した。
- `SESSION_NOTES.md`、`result.md`、`metrics.json`、`README.md`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` を結果で更新した。
- direct correction / submit はしない判断を記録した。
