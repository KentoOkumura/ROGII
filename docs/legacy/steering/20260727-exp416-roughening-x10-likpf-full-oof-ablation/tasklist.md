# タスクリスト

## 目的

exp072 likelihood-PFのroughening 2値だけを10倍にする全OOF単一介入を、control再実行
なしで実装・評価する。

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- exp416 experiment scaffoldを作成。
- exp416 steering requirements / design / tasklistを作成。
- route、親、単一変更、実行量、再現性、scientific AND gateを固定。
- KAGGLE_DIRECTIONのbacklogをexp416へ採番。
- ユーザーから実装の明示承認を取得。
- Jupytext compact self-contained train候補とcontract testsを実装。
- exact exp072 kernel parity、roughening-only parameter diff、stable seed、LPT shard、
  truth freeze、persistent episode AND gateをテスト。
- 初回実装時点では正規Notebook、package、Kaggle実行、inference、submissionは未着手。
- ユーザーから正規train Notebook採用、package、4 shard push / run、strict merge、
  train-side評価の明示承認を取得。probe / inference / submissionは未承認のまま。
- Kaggle CPU 4 shardを3,783,989 rows / 773 wellsで完走。
- strict merge/readout version 1をpushし、Kaggle側の入力・SHA・CPU設定を確認。
- merge version 1のexp209 schema mismatchを特定し、exp410と同じfloat32復元へ修正。
- 23 targeted tests、構文、Ruff、Jupytext、strict実験検証を通過。
- 同じstrict merge kernelへversion 2をpushし、`RUNNING`とembedded SHAを確認。
- strict merge/readout version 2の`COMPLETE`、metrics、gate、SHAを確認。
- scientific / technical gate FAILとしてroughening x10を棄却し、救済なしでbranchを閉鎖。
- result / metrics / SESSION_NOTES / summary / directionを最終更新。
