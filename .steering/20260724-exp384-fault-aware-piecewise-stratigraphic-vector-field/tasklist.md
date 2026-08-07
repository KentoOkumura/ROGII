# タスクリスト

## TODO

- exp383 Stage 0/1の結果と生成物SHAを確認する。
- exp383 PASS後、parent manifest SHAをpinして16-well Stage 0 resource auditを実行する。

## 進行中

- なし

## ブロック中

- exp383 Stage 0/1 PASS待ち。
- exp383 artifact manifest/SHA pin前のKaggle package/push/run。
- 科学score、inference、submissionは未取得・未実施。

## 完了

- 2026-07-24: `exp384_fault_aware_piecewise_stratigraphic_vector_field`として採番した。
- 2026-07-24: steering 3文書と実験scaffoldを作成した。
- 2026-07-24: fault graph、component条件、posterior、base floor、Stage 0/1 gate、
  再現性、停止条件をdesign-only契約として固定した。
- 2026-07-24: ユーザーの実装指示をimplementation-only承認として記録した。
- 2026-07-24: compact self-contained train、fail-closed inference、正規Notebookを実装した。
- 2026-07-24: fault graph、component field、soft posterior、prefix likelihood、
  exp383-compatible path solve、target-free SHA freeze、late truth joinを実装した。
- 2026-07-24: 専用contract test `14 passed`、Ruff、py_compile、Jupytext
  train/inference round-trip、strict experiment validationをPASSした。
- 2026-07-24: ユーザーからexp384のKaggle package / push / CPU実行承認を得た。
- 2026-07-24: ローカルとKaggleを監査し、exp383 kernel・Stage 0/1成果物・
  manifest SHAが存在しないため、package前にfail-closedで停止した。
