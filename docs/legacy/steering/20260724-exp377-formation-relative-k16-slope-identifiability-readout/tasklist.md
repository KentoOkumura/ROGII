# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 実験番号、route、親、変更変数、固定変数、primary、gateを確定。
- 実験ディレクトリとsteeringを作成。
- 再現性・leakage・停止条件を設計。
- 2026-07-24: ユーザー指示「exp377を実装してください」をimplementation-only承認として受領。
- outer-train限定のFormationPlaneKNN、6相対勾配、exp226固定XY kernel、role read guard、target-free SHA freezeを実装。
- Stage 0 integrityと、PASS時だけtruthをlate joinするStage 1 identifiability固定AND gateを実装。
- 別名compact self-contained train/inference候補を作成し、正規Notebookは未変更のまま保持。
- 専用test 8件、Jupytext train/inference round-trip、py_compile、Ruff、strict experiment validationをPASS。
- 2026-07-24: ユーザー指示「実行してください」により正規train Notebookを採用。
- canonical CPU packageのconfig/source/bootstrap SHA、slug/title、internet offを確認し、Kaggle v1をpush。
- kernel `kentookumura/exp377-formation-relative-k16-slope-readout-train` v1 / id_no `128452991`が`COMPLETE`。
- Stage 0はeffective donors p05 `2.59469484575288 < 10`だけFAIL。truth joinとStage 1を開かず、exp378〜380を停止してbranchを閉じた。
- 2026-07-24: ユーザー指示「1を実行してください」により、共通kernelのlow supportをreport-onlyとするv2を承認。
- K16 / nearest 50 / bandwidth 500 / ridge 1 / 6 surfaces / median primary / Stage 1 gateは変更しない。
- v2用コード・test・正規Notebook・packageを検証し、同一canonical kernelへversion 2としてpush。
- canonical kernel version 2が`COMPLETE`。Stage 0 blocking checksは全PASSし、
  effective donors p05 `2.59469484575288`だけをreport-only warningとして保存。
- truth-late Stage 1はrate `0.012301 → 0.038454`、path
  `16.100131 → 38.776238 ft`、rate/path改善fold各`0/5`、
  609/773 wells悪化で7 checksすべてFAIL。
- 個別6 formation pathも全てdirectより悪化したため、scientific negativeとして終了。
- exp378 / exp379 / exp380 / exp382を未実装・未実行のまま閉鎖し、
  surface / kernel / scope救済、inference、submissionを実行しない。
