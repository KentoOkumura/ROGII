# タスクリスト

exp444 scientific contractを変えず、同じposteriorをruntime上限内で計算できる
独立engine仮説の実装・検証・段階実行を管理する。

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- `docs/06_reproducibility.md`を確認し、RNGなし、well-local parallel、
  stable output order、repeat SHA方針を設計した。
- exp444 v1のfixed4 runtime、RSS、scientific contract、生成物SHAを固定した。
- exp399のexact-equivalent fusion実績とKaggle CPU varianceを設計根拠にした。
- scientific contract、許可するruntime差分、禁止する近似、段階、実行量、
  数値/runtime/RSS/leakage gateを確定した。
- steering、design-only実験scaffold、backlog、experiment summaryを作成した。
- ユーザー承認後、compact self-contained train/inference候補を実装した。
- float64 scaled probability-space forward/backward、factorized/fused transition、
  exact-bit `delta_MD` cache、4-process outer parallelを実装した。
- exp444科学contract SHA、small dense trellis、OU/position kernel、
  candidate-parent parity、repeat determinism、oversubscription guardのtestを追加した。
- 専用test 9件、Jupytext round-trip、構文/F821、
  `make validate-exp`をPASSした。
- 正規Notebookはtemplate scaffoldのまま維持し、未採用とした。
- 2026-07-30のユーザー依頼で正規train Notebook採用、Kaggle package、
  private CPU Stage 0A runを承認済みとした。
- push前実行量をvariant 1、engine 1、repeat 2、fold 0、
  candidate HMM 8 well-runs、parent/control/model/booster/PF/Beam/GPU各0と記録した。
- 58文字の初回slugはKaggle `SaveKernel 400`、直前pull 403で未作成だった。
  科学contractを変えず、`acceleration`を`accel`、`engine`を`eng`へ
  短縮した48文字の
  canonical id/titleへ再packageすることを記録した。
- version 1は2 repeatsの計算後、summary JSON保存時のNumPy `bool_`
  serializer欠落だけでERRORになった。科学計算を変えず型変換とregression
  testだけを修正し、同じcanonical slugのversion 2でtechnical retryする。
- version 2（id_no `129168013`）はCOMPLETE。runtime/RSS/repeat SHA/
  leakageはPASSしたが、保存exp444比のprediction mean/stdとacceleration
  posteriorが固定閾値をFAILした。
- `stage0a_fail_closed`、Stage 0B不適格としてexp458をterminal closeした。
  Stage 0B/1、inference、submission、favorable rerun、gate救済は行わない。
