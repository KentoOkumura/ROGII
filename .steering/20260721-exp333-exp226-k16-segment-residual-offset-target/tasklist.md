# タスクリスト

## 未着手

- 低優先の別候補として、保存済みexp333 OOFを変更せずnear/worst悪化のoffset magnitude/sign・well寄与偏りだけを調べる0-booster failure readoutを設計するか判断する。

## ブロック中・停止

- Stage 1固定gate FAILのためexp333 inference、submission、追加config、same-OOF救済を停止。
- 32-well parity/runtime preflightの承認は消費済み。full trainへ自動移行しない。

## 完了

- 2026-07-21: `exp333_exp226_k16_segment_residual_offset_target`を採番。
- 2026-07-21: backlog、steering、experiment scaffoldを作成。
- 2026-07-21: K16境界、mean residual target、row-count weight、feature allowlist、strict nested fold、1-config CPU model、実行量、promotion/停止条件を確定。
- 2026-07-21: `docs/06_reproducibility.md`に基づくfold/feature/model/prediction SHA方針を確定。
- 2026-07-21: implementation/Kaggle/inference/submissionを無効化。
- 2026-07-21: ユーザー承認によりStage 0だけを実装。saved exp226 OOFのtarget-free freeze、exp226互換K16 assignment、late-truth join、oracle mean-offset readout、固定5/5 fold gate、SHA evidenceをcompact self-contained trainへ追加。
- 2026-07-21: fail-closed compact inference、専用pytest、別名`.ipynb`を追加。正規Notebook scaffoldは未上書き、Kaggle実行は0件。
- 2026-07-21: compact trainを正規Notebookへ採用し、`kentookumura/exp333-k16-segment-residual-stage0-train` version 1 / CPU / internet offで実行。
- 2026-07-21: Stage 0はexp226 RMSE`9.427109597`、K16 oracle RMSE`1.130602526`、改善`8.296507070 ft`、fold改善5/5でtechnical/scientific PASS。Stage 1実装許可条件を満たしたが、別承認待ち。
- 2026-07-21: ユーザー依頼「Stage1に進んでください」をStage 1実装承認として受領。実装範囲をstrict nested exp226 25 fits、許可済み3 feature群、K16 finite-mean集約、固定LightGBM `lgb1` 1 config × 5 folds、固定promotion gateとSHA証跡に限定した。Kaggle実行、推論、提出は別承認のまま。
- 2026-07-21: Stage 1 Jupytext source / 別名Notebook候補、fail-closed preflight/full-run分岐、bootstrap dependency、専用pytestを実装。`14 passed`、py_compile / Ruff / Jupytext変換PASS。実行0件。
- 2026-07-21: ユーザー依頼「実行してください」を32-well Stage 1 CPU preflightの承認として受領。実行量は1 preflight audit、LightGBM variant/config/trained fold/booster `0/0/0/0`、full-source exp226 25 donor-field/kappa fits、対象32 wells・最大160 prediction well-runs、control再学習0、GPU 0。full Stage 1 train、inference、submissionは未承認。
- 2026-07-21: `kentookumura/exp333-k16-segment-residual-stage1-preflight` version 1 / CPU / internet offを完走。32 wells・166,533 feature rows、25 fits・160 prediction well-runsを`491.885 sec`で実測し、outer-valid parent parity最大差`1.819e-12 ft`で`1e-8 ft` gateをPASS。full Stage 1外挿は`6,434.437 sec = 1.787 h`で8.5時間gateをPASS。LightGBM model/boosterは`0/0`、full trainは未実行。
- 2026-07-21: ユーザー依頼「次にすすんでください」をfull Stage 1 Kaggle CPU trainの承認として受領。実行量は1 variant × 1 config × 5 folds = 5 CPU boosters、strict nested exp226 25 donor-field/kappa fits・3,865 prediction well-runs、parent/control再学習0、GPU 0。canonical kernelは`kentookumura/exp333-k16-segment-residual-stage1-train`。inference、submissionは未承認。
- 2026-07-21: canonical full train version 1（id_no `128116592`）をCPU / internet offで`1,781.997 sec`完走。CV`9.076676661`はexp226を`0.350433 ft`改善し5/5 foldsを改善したが、固定pooled上限`8.894085501`未達、near 0--250`+0.057439 ft`、worst well`+8.099023 ft`で3 gate FAIL。decision=`FAIL_CLOSE_BRANCH`、推論・提出なし。
