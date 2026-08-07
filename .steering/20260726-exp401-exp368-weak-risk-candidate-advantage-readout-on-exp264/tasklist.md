# タスクリスト

## 未着手

- なし

## 進行中

- なし

## ブロック中

- Stage 1 selector学習はStage 0 scientific gate FAILにより閉鎖。

## 完了

- [x] 2026-07-26: exp401を採番し、route、親、補助入力、特徴1列を固定。
- [x] 2026-07-26: Stage 0のlabel、legal domain、circular control、
  technical/scientific gateを固定。
- [x] 2026-07-26: 条件付きStage 1の実行量を
  1 variant / 2 objectives / outer 5 × inner 4 / 40 CPU boosters /
  control再学習0に固定。
- [x] 2026-07-26: 再現性、raw-test parity、禁止事項を固定。
- [x] 2026-07-26: implementation-only承認を受領。
- [x] 2026-07-26: Stage 0用compact self-contained train候補を別名で実装。
- [x] 2026-07-26: fail-closed inference候補を別名で実装。
- [x] 2026-07-26: overlap block row risk、truth前SHA freeze、exp264
  row-group scan、2 legal domain readout、全AND gateを実装。
- [x] 2026-07-26: 専用contract test 9件、Jupytext、py_compile、Ruff、
  strict experiment validationをPASS。
- [x] 2026-07-26: Stage 0実行承認を受領し、実行量
  1 diagnostic / 5 reporting folds / 45,407,868 candidate-long rows /
  model・LightGBM・trained fold・booster・PF・prediction各0を再提示。
- [x] 2026-07-26: compact self-contained train候補を正規train Notebookへ採用。
- [x] 2026-07-26: private・internet/GPU off・run-on-push true・入力4件で
  strict Kaggle packageを準備。
- [x] 2026-07-26: `exp401-weak-risk-readout-on-exp264-train` version 1をpush。
- [x] 2026-07-26: version 1の`pred_abs_error` guard誤判定をexact-name
  allowlistで修正し、test 9件とstrict validation後にversion 2をpush。
- [x] 2026-07-26: version 2のexp264 generation fold / exp226 reporting fold
  誤同一視を独立ledgerへ修正し、test 9件とstrict validation後にversion 3をpush。
- [x] 2026-07-26: version 3のpost-run `numpy.bool_`表示ERRORを既存JSON
  serializerで修正し、test 9件とstrict validation後にversion 4をpush。
- [x] 2026-07-26: version 4を129.300秒で完走。technical 15/15 PASS、
  scientific 4/12 PASS・総合FAILを記録し、output SHAを実ファイル照合。
- [x] 2026-07-26: `stage_0_failed_close_without_rescue`としてStage 1、
  inference、submissionなしでbranchを閉鎖。
