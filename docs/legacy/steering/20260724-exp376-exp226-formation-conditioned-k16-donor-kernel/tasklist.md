# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 2026-07-24: `exp376` として採番した。
- 2026-07-24: steering 3文書と実験scaffoldを作成した。
- 2026-07-24: 単一変更、fold-safe formation契約、weight式、Stage 0/1/2 gate、
  実行量、再現性、禁止事項を設計として固定した。
- 2026-07-24: ユーザーの実装指示を受け、exp226のfull downstreamとexp287
  `FormationPlaneKNN` semanticsを展開した3,974行・9章・19セルの
  compact self-contained Jupytext候補を実装した。
- 2026-07-24: outer-validのTVT/6地層列を開かず、outer-train donorも
  leave-one-well-outで11次元signatureを作るfold-local formation契約を実装した。
- 2026-07-24: Stage 0のreference/signature/weight/ESS/SHA freezeとfail-closed境界、
  Stage 1 direct、Stage 2 fixed12 H512/whole-well add-one noveltyを実装した。
- 2026-07-24: 専用test 3件、`py_compile`、ruff F821、Jupytext round-trip、
  strict experiment validationをPASSした。
- 2026-07-24: 既存正規train/inference scaffoldは変更せず、Kaggle package/push/run、
  current-test、推論、提出を未承認のまま維持した。
- 2026-07-24: ユーザーの実行指示を、正規train notebook採用とKaggle CPU run
  1回の承認として記録した。
- 2026-07-24: push前コストを1 variant / 0 config / 5 reporting folds /
  0 trained fold / 0 booster / parent control再実行0として再確認した。
- 2026-07-24: compact版を正規train notebookへ採用し、metadata/bootstrapの
  config/source/run flags、3 kernel sources、GPU/internet無効を確認した。
- 2026-07-24: Kaggle CPU v1（id_no 128436621）を実行し、5 foldsの予測後
  1237.618秒でtruth前freezeのlist-valued manifest hashによりERRORとなった。
  Stage 0/1/2、CV、truth scoringは未評価。
- 2026-07-24: container-valued object cellだけをcanonical JSON化する局所修正と
  再現testを追加し、専用test 4件、構文、F821、Jupytext round-tripをPASSした。
- 2026-07-24: ユーザーの再実行指示を修正版Kaggle CPU v2の明示承認として記録した。
- 2026-07-24: v2 packageのconfig/source byte parity、CPU・internet off、
  3 kernel sources、run-on-push、config/source/notebook SHAを確認した。
- 2026-07-24: Kaggle CPU v2（同じid_no 128436621）を`COMPLETE`まで監視した。
  Technical / Stage 0はPASS、direct / noveltyはFAIL。
- 2026-07-24: summary、freeze/SHA manifest、guard、metrics、by-well、
  correlation、input/schema manifestを選択取得し、主要SHAを記録した。
- 2026-07-24: fixed decisionどおり救済grid、current-test、selector組み込み、
  inference、submission、version 3なしでbranchを閉じた。
