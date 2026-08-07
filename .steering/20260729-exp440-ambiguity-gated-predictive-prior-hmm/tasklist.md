# タスクリスト

## 実行しない

- inference、submission、Stage 0 rerun、same-OOF rescue。

## 進行中

- なし

## ブロック中

- inference、submissionはStage 1結果とそれぞれの追加承認まで開始しない。

## 完了

- `exp440_ambiguity_gated_predictive_prior_hmm`を採番した。
- 仮説を「二峰GR行でcurrent emissionよりpredictive prior holdが安全」と固定した。
- 親exp209からの唯一の変更をambiguous rowのemission lambda `1.0 -> 0.0`に固定した。
- steering 3文書を作成し、GMM非依存のpredictive-prior holdとして科学設計を固定した。
- Stage 0/1のvariant、HMM well-run数、親control再実行数、gate、no-rescueを固定した。
- `docs/06_reproducibility.md`を確認し、RNGなし、truth-late freeze、SHA方針を記録した。
- design-only実験scaffoldを作成した。
- `KAGGLE_DIRECTION.md`のbacklogでP3として位置付ける。
- 実行時に予定する生成物をambiguity schedule、prediction、diagnostic、
  well/fold/scope metrics、summaryと定義した。
- exp209 exact-HMM、exp236 fixed bimodality判定、exp411 fixed32
  truth-late運用を抽出したcompact self-contained train候補を実装した。
- raw-GR-observed ambiguous rowだけemission lambdaを0にし、
  candidate predictive jointをfiltered jointとして保持するcausal
  forward scheduleを実装した。
- forwardで固定したrow-wise lambda scheduleをbackwardで再利用する実装と、
  smoothed posterior mean/std readoutを実装した。
- role/fold/truth/episode/causeを全32 schedule/prediction/diagnostic SHA
  freeze後にだけ読むledgerとlate loaderを実装した。
- fixed32 technical/mechanism AND gateとfail actionを実装した。
- fail-closed compact inference候補を実装した。
- Jupytext compact notebookを生成し、`--test`、`py_compile`、
  Ruff、専用pytest 13件、exp408/411/440関連pytest 39件、
  `make validate-exp` strictを通過した。
- compact train候補を正規train Notebookへ採用し、24 cells /
  2,481 source linesと実行時cell source SHA一致を確認した。
- strict Kaggle train packageを作成し、private CPU / internet disabled /
  親kernel source / competition source / 3 asset SHA / 実行量契約を検証した。
- Kaggle private CPU canonical version 1（id_no `129064462`）で
  scientific candidate 1本、32 HMM well-runs、control再実行0を完走した。
- technical gate `13/15`、mechanism gate `2/8`で
  `stage0_fail_closed`を確定した。
- outputを`/tmp`へ取得し、input、schedule、prediction、diagnostic、
  truth-late readout、metricsの行数とSHAを実ファイルで照合した。
- Stage 1、inference、submission、rerun、same-OOF rescueなしでbranchを閉じた。
- 結果記録後のlocal compact / canonical trainへrerun禁止guardだけを追加し、
  24 cells / 2,486 source linesの最終source SHA一致を確認した。
- 2026-07-30のユーザー依頼により、Stage 0 FAILの解釈を維持したまま
  full-well確認だけを明示override承認した。
- 773 wellsをsuffix row数のdeterministic LPTで
  `193 / 193 / 193 / 194 wells`、`946,128 / 946,017 / 946,112 /
  945,732 rows`へ4分割する実装とstrict mergeを追加した。
- full実行量をscientific candidate 1、candidate HMM 773 well-runs、
  parent control rerun 0、LightGBM / model / PF / Beam / GPU 0に固定した。
- 4 CPU shardsで773 wells / 3,783,989 rowsを完走し、全target-free
  SHAをstrict merge後にtruth / fold / hidden-like roleを結合した。
- Stage 1 technical gateを全PASSした。
- candidate RMSE 12.992063、parent 11.938287、positive fold 1/5、
  ambiguous-row SSE -21.3117%、by-well p95 +11.631749 ftでscientific
  FAILを確定した。
- `stage1_full_oof_failed_closed`としてrerun、inference、submission、
  same-OOF rescueなしでterminal closeした。
