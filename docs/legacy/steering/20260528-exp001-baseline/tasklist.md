# タスクリスト

## TODO

- `exp002_drift_minimal` で `TVT - last_anchor_tvt` target と prefix / GR / geometry features を追加する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- 現状調査メモから exp001 の範囲を決めた。
- `config.yaml` を last-known anchor baseline 方針に更新した。
- `baseline.py` を追加し、prefix-only 予測ロジックを共通化した。
- `train.py` で full CV と artifact 出力を実装した。
- `inference.py` で `id,tvt` 提出生成を実装した。
- full CV を実行し、`last_anchor` OOF RMSE 15.909853 を記録した。
- `submission/submission.csv` を生成した。
