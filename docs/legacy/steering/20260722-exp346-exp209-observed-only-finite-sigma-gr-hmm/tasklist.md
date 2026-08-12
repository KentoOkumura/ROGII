# タスクリスト

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- なし。scientific gate FAILによりraw-test inference、submissionは実行対象外としてbranchを閉じた。

## 完了

- 2026-07-22: exp307、exp308、exp337、exp339/341との重複と差分を確認した。
- 2026-07-22: `exp346_exp209_observed_only_finite_sigma_gr_hmm`としてsteeringと実験scaffoldを作成した。
- 2026-07-22: raw finiteだけを確実行とする定義、2-scale schedule、fallback、固定HMM、実行量、AND gate、禁止救済を設計固定した。
- 2026-07-22: `docs/06_reproducibility.md`に基づくSHA、RNG、Kaggle bootstrapの記録方針を固定した。
- 2026-07-22: ユーザーの実装依頼を受け、compact self-contained train候補とfail-closed inference候補を実装した。
- 2026-07-22: raw mask、2本のscale、row schedule、raw missing emission parity、truth late-join、固定AND gateのcontract testを追加した。
- 2026-07-22: Jupytext round-trip、py_compile、Ruff、専用11 test、strict experiment validation、template validationを通過した。
- 2026-07-22: 全体testはexp346を含む605件PASS・3件skip。既存exp296の完了済みconfigと古い期待値の不一致2件だけがFAILし、exp346変更とは無関係と確認した。
- 2026-07-22: ユーザーがKaggle CPU実行を承認。1 variant、773 HMM well-runs、schedule audit 1、control再実行0、booster0を再確認した。
- 2026-07-22: 53文字の初回kernel IDはKaggle SaveKernel 400となり、metadata pull 403で未作成を確認。仮説名を維持した46文字のcanonical IDへ短縮した。
- 2026-07-22: 短縮後packageを再監査し、Kaggle CPU train version 1をpush。metadata pullでid_no 128227279、private CPU、internet off、入力source一致を確認した。
- 2026-07-23: version 1 COMPLETEを確認。773/773 HMM runs、technical gate PASS、direct `11.938287→13.295027`、fixed blend `10.269693→10.531118`、改善1/5 folds、全必須scope・p95・worst FAILを記録した。
- 2026-07-23: decision `observed_only_finite_sigma_failed_close_without_rescue`に従い、inference/submission/同一prediction救済なしでbranchを閉じた。

## 次のアクション

exp346はterminal close。新しい同family救済を追加せず、既存の独立候補exp340を優先し、GR重複過信を再訪する場合もexp343 Stage 0の事前gateと別承認を必須とする。
