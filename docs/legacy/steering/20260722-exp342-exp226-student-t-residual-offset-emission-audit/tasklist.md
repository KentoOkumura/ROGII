# タスクリスト

## 進行中

- なし。

## 閉鎖

- inference、submission、df/scale/temperature/grid、Huber/cap、
  missing/ACF救済、Gaussian control再実行。
- flattening signalがfalseのためexp344 Huber依存も不成立。

## 完了

- 2026-07-22: exp342として採番し、steeringと実験scaffoldを作成した。
- 2026-07-22: df=4、Stage 0/1、実行量、gate、救済禁止を設計固定した。
- 2026-07-22: 実装、Notebook編集、Kaggle実行、推論、提出を行っていない。
- 2026-07-23: Stage 0 compact self-contained train/inference、正規Notebook、
  saved Gaussian control、df=4 Student-t score、truth-late gate、contract testを実装した。
- 2026-07-23: py_compile、Ruff、専用pytest 7件、Jupytext test、strict exp validationをPASSした。
- 2026-07-23: Kaggle private CPU Stage 0 version 1（id_no `128356155`）を
  `468.127417 sec`で完了した。
- 2026-07-23: pooled MRR/top3 gainとstress MRR/top3非劣化がFAIL。
  Stage 1不適格、exp344依存pattern不成立として救済なしで閉じた。
- 2026-07-23: ユーザー明示依頼により、Stage 0 FAILを保持したまま
  Stage 1実装・Kaggle CPU実行を探索的overrideとして承認した。
- 2026-07-23: Stage 1 compact self-contained exact HMMを実装し、
  exp281保存OOF・kernel parity・SHA・truth-late join guardを検証した。
- 2026-07-23: canonical Kaggle private CPU version 2へ1 variant /
  773 HMM well-runsをpushした。Gaussian control再実行、model、booster、
  inference、submissionは0。
- 2026-07-24: version 2の`COMPLETE`と最終logsを確認した。Student-tはexp281比
  `0.047648 ft`改善したが、必要`0.05 ft`、4/5 folds、hidden-like、
  by-well p95、worst-well gateをFAILした。
- 2026-07-24: `stage_1_failed_close_without_rescue`を記録し、救済・再実行・
  inference・submissionなしでterminal closeした。
