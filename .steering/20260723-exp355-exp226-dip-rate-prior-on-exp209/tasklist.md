# タスクリスト

## TODO（別途承認後）

- なし

## 進行中

- なし

## ブロック中

- Stage 1 scientific gate FAILのためinference、submissionへ進まない。

## 完了

- 2026-07-23: exp355として採番し、exp209直結の独立系譜を確定した。
- 2026-07-23: exp307--309/338 dependencyを禁止し、単一変更をrate-prior meanへ限定した。
- 2026-07-23: Stage 0/1 gate、実行量、再現性、fail-closed境界を固定した。
- 2026-07-23: Stage 0 compact self-contained train候補とfail-closed inference候補を実装した。
- 2026-07-23: exp226互換K16分割、rate schedule、fallback、truth late-join、
  segment/path/fold/stress/hidden-like/by-well gateと契約テストを実装した。
- 2026-07-23: Jupytext変換/test、構文、F821、専用pytestを通過した。
- 2026-07-23: 正規train Notebookを採用し、canonical Kaggle CPU version 1を実行した。
- 2026-07-23: 7/8 gates PASS、worst-well `+69.017669 ft`で総合FAILを確定した。
- 2026-07-23: Stage 1、parameter rescue、inference、submissionなしでbranchを閉じた。
- 2026-07-23: ユーザーが平均改善を根拠にworst-well gateをoverrideし、Stage 1
  train-side 1 candidate / 773 exact-HMM well-runsの実行を承認した。
- 2026-07-23: Stage 0 schedule/ledger SHAをhard guardする残差rate座標の
  self-contained Stage 1 trainと契約テストを実装した。
- 2026-07-24: canonical Kaggle CPU version 2で1 candidate / 773 exact-HMM
  well-runsを完了した。technical gateはPASS。
- 2026-07-24: direct RMSEは`0.646311 ft`、5/5 foldsで改善したが、
  hidden-like 2面とworst-well `+52.743754 ft`をFAILし、branchを閉じた。
