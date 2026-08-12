# タスクリスト

## 未着手

- なし。

## ブロック中

- Stage 1、inference、submission。
- weight/sigma/trigger/gateのsame-OOF rescue。

## 完了

- 2026-07-29: exp442へ採番し、P2条件付き候補とした。
- 2026-07-29: mixture weight `0.01`、broad sigma `0.02`、対称性を固定した。
- 2026-07-29: fixed32 gate、実行量、truth-late、SHA、fail actionを固定した。
- 2026-07-29: backlog、steering、実験scaffoldとdesign-only記録を作成した。
- 2026-07-29: ユーザー依頼「exp442を実装してください」によりcompact実装候補と
  contract testの作成を承認した。正規Notebook採用と実行は未承認のまま。
- 2026-07-29: local / broad / mixture kernel、smoothed branch responsibility、
  truth-late direction / episode / control gateをcompact self-contained trainへ実装した。
- 2026-07-29: inference guard、専用12 test、Jupytext、構文、Ruff、
  strict experiment / template validationを完了した。
- 2026-07-29: 正規Notebookはscaffoldのまま保持し、Stage 0をfail-closedにした。
- 2026-07-30: exp441をtechnical/control/direction prerequisiteとする条件を撤回し、
  exp442をexp209に対する独立defensive mixture仮説へ再定義した。
- 2026-07-30: ユーザー依頼により正規train Notebook採用、Kaggle private CPU
  package、1候補×fixed32 Stage 0を承認した。parent rerun / ML / PF / Beam / GPUは0。
- 2026-07-30: compact self-contained trainを正規train Notebookへ採用した。
  compact / canonicalとも24 cells、3,407 source lines、cell source SHA一致を確認した。
  inference Notebookは未変更。
- 2026-07-30: canonical ID/titleでprivate CPU packageを作成し、internet無効、
  exp209/exp408 kernel source、bootstrap 8 filesと全展開SHAを検証した。
- 2026-07-30: Kaggle private CPU Stage 0 version 1（id_no `129101211`）を
  1候補×32 wellsで完走した。technical 14/15、mechanism 4/9 PASS。
- 2026-07-30: branch responsibility `0.00976695`、non-adjacent mass
  `0.00684557`、control pooled / p95 delta
  `-0.155414 / +0.069364 ft`は成立したが、方向一致`0.529732`、
  persistent SSE削減`-4.4385%`、改善well/fold `9/16` / `2/5`、
  full runtime投影`222,019.844 sec`がgateをFAILした。
- 2026-07-30: `stage0_fail_closed`として完了。rerun、Stage 1、inference、
  submission、weight/sigma/trigger/emission/grid/gate救済は行わない。
