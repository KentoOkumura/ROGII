# タスクリスト

## 未着手

- 別承認後にprivate Kaggle CPUでknown-prefix Stage 0を実行する。
- 全gate PASS後、別承認でStage 1を実装・実行する。

## 進行中

- なし

## ブロック中

- Kaggle package / push / runは未承認。
- Stage 1はStage 0全gate PASSと別承認が必要。

## 完了

- backlog、実験scaffold、steeringを作成した。
- delta state、Stage 0/1 gate、禁止事項、再現性、実行量を確定した。
- Stage 0 compact self-contained trainとfail-closed inferenceを実装した。
- 正規Notebookへcompact候補を採用した。
- 専用9 tests、py_compile、Ruff、Jupytext round-trip、strict experiment validationを通した。
