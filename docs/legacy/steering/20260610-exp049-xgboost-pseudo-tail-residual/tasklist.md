# タスクリスト

## TODO

- exp044 補助 fold / distance bucket の破壊的悪化確認へ進むか判断する。
- inference port / submit-check へ進むか判断する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成する。
- `exp023` から `exp049` を作成する。
- `config.yaml` を XGBoost pseudo-tail residual 実験に更新する。
- `baseline.py` に `XGBRegressor` 対応を追加する。
- raw XGBoost と fixed bucket-shrink 候補を同時集計できるようにする。
- README、SESSION_NOTES、result、metrics を未実行状態に更新する。
- `ruff check`、`py_compile`、strict experiment validation を通す。
- local smoke で `xgboost` local dependency 不在まで確認する。
- Kaggle train notebook package を生成する。
- Kaggle train notebook を push する。
- Kaggle logs/output を取得し、`xgboost` import と full CV 完了を確認する。
- Kaggle 上で full CV を実行し、結果を `SESSION_NOTES.md`、`result.md`、`metrics.json` に記録する。
- Kaggle output の小さい生成物を `artifacts/` に同期する。
