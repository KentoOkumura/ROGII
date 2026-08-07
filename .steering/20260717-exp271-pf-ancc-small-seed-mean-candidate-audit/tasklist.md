# タスクリスト

## 目的

固定PF ANCC mean4/mean8のcandidate追加価値を0-boosterで監査できる実行契約を完成させる。

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 依頼、制約、受け入れ基準を固定した。
- 再現性設計を `design.md` に記入した。
- 1 PF dynamics × 8 seeds × 600 particles、LightGBM/fold/booster 0を固定した。
- compact self-contained Jupytext train / disabled inferenceを実装した。
- exp072 seed0 / exp266 aggregate / exp263 manifest-partition SHAのfail-closed guardを実装した。
- row/block/well oracle、distance/hidden-like/worst-well、seed disagreement readoutを実装した。
- contract unit test 5件と静的validationを通した。
- run-on-push offのKaggle CPU packageを作成し、metadataとbootstrap dependencyを確認した。
- 並行追加されたexp268/269/270との番号衝突を避け、未実行の本実験をexp271へ改番した。
- 生成物契約をcandidate path、exp266 parity、standalone/oracle/by-well、seed disagreement、
  input/artifact manifest、summaryに固定した。
- 実行依頼後に`run_on_push=true` packageを再生成し、metadata / bootstrap SHAを再確認した。
- version 1のfloat32 target復元精度差をfail-closedで検出し、raw TVT評価へ修正した。
- 修正後のcontract test 6件、Jupytext round-trip、strict validationを通した。
- canonical Kaggle CPU kernel version 2を3,783,989 rows / 773 wellsで完了した。
- seed0 / exp266 / exp263 parity、input/output content SHA、runtime、oracle/readoutを確認した。
- candidate gzipを除く小型生成物を選択取得し、manifest SHA一致を確認した。
- `result.md`、`metrics.json`、`SESSION_NOTES.md`、`README.md`、
  `experiment_summary.md`、`KAGGLE_DIRECTION.md`へ結果と次候補を記録した。
