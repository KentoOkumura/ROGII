# タスクリスト

## 未着手

- なし。

## ブロック中

- Stage 0B、Stage 1、inference、submission。
- acceleration state/parameter/prior/gateのsame-OOF rescue。

## 完了

- 2026-07-29: exp444へ採番し、P4 high-risk条件付き候補とした。
- 2026-07-29: 3値state、transition、boundary、initial prior、更新順を固定した。
- 2026-07-29: Stage 0A/0B/Stage 1、runtime、truth-late、SHA契約を固定した。
- 2026-07-29: backlog、steering、実験scaffoldとdesign-only記録を作成した。
- 2026-07-30: exp441 Stage 0 FAILにより当初の条件付き先行条件が不成立と確認した。
- 2026-07-30: ユーザー判断でexp444を独立仮説へ変更し、元の実装依頼を再開した。
- 2026-07-30: acceleration/joint transition、small-state dense reference、
  identity-only fixed4 selector、truth-read guardを実装した。
- 2026-07-30: compact self-contained train/inference候補と専用14 testsを作成し、
  Jupytext、構文、Ruff、strict experiment/template validationをPASSした。
- 2026-07-30: 正規train Notebookを採用し、Kaggle private CPU Stage 0A
  version 1（id_no `129154702`）を実行した。
- 2026-07-30: 4 wells / 21,962 rowsを完走。exactness、normalization、
  leakage、RSSはPASSした。
- 2026-07-30: fixed32/full runtime投影`5,970.830 / 144,232.851 sec`が
  上限`3,600 / 30,600 sec`をFAILし、terminal closeした。

## 実装結果

- compact trainは2,537行、24セル、compact inference guardは8セル。
- 専用pytestは14/14 PASS。
- py_compile、Ruff、Jupytext round-trip、strict experiment/template validationは
  すべてPASS。
- Kaggle Stage 0A technical preflightは完了。科学score、CV、LBは存在しない。
- Stage 0B eligibleはfalse。

## 次のアクション

- exp444内のruntime/state/kernel/parameter/gate救済を行わない。
- Stage 0B/1、inference、submissionへ進まない。
