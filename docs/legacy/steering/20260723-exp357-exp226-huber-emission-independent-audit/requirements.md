# 要件

## 依頼

閉鎖済み`exp344_exp226_huber_residual_offset_emission_audit`をreopenせず、
exp342 Student-t patternを解禁条件にしない新番号の独立Huber監査として設計する。
今回はdesign-onlyで、実装・実行は行わない。

## 2026-07-24 追加依頼

ユーザーの「exp357を実装してください」によりStage 0実装だけを解禁する。
正規Notebookの上書き・採用、Kaggle package/push/run、Stage 1、inference、
submissionは引き続き別承認とする。

## 2026-07-24 実行依頼

ユーザーの「実行してください」によりcompact train候補の正規Notebook採用と、
Kaggle private CPU Stage 0を1回だけ解禁する。Stage 1、inference、submissionは
引き続き未承認とする。

## 2026-07-24 HMM実行依頼

ユーザーの「HMM実行に進んでください」により、Stage 0 FAILの通常停止条件を
明示的にoverrideし、固定Huber Stage 1を1回だけ実装・Kaggle CPU実行する。
実行量は1 scientific variant / 773 HMM well-runs / model config・trained fold・
booster・親Gaussian control再実行各0とする。inferenceとsubmissionは解禁しない。

## 制約

- Route: `pf_beam`
- 科学的親: `exp281_exp226_residual_offset_exact_hmm_transition_probe`
- Stage 0 saved control: `exp280_exp226_shift_likelihood_separability_readout`
- 履歴参照: `exp344_exp226_huber_residual_offset_emission_audit`
- Huber `delta=1.345`を固定し、delta/scale/cap/temperature gridを禁止する。
- Stage 0ではGaussian lossをHuber lossへ置換した13-shift block scoreだけを生成し、
  HMM/model/trained fold/boosterは0。
- exp342はnegative referenceとして記録するが、入力・解禁条件にはしない。
- Stage 1は通常Stage 0全gate PASSが必要だが、今回はユーザーの明示overrideにより
  fixed `delta=1.345` 1 variant / 773 HMM runsだけを実行する。
- exp281はpromotion FAIL済みのため、Stage 1候補はexp281比改善に加え、
  direct RMSEがexp226 `9.427109596582213`以下であることを必須とする。

## 受け入れ基準

- block/shift/fold/missing/sigma/controlの契約をexp280/281と一致させる。
- Huber/control bundleをtruth join前にcontent SHA固定する。
- Stage 0でMRR/top3各0.01以上、各4/5 folds、stress非悪化、
  real-shuffle gap、extreme-residual regret改善をすべて要求する。
- Stage 1でexp281比0.05 ft以上、4/5 folds、1000+・hidden-like・p95・worst guard、
  exp226 direct ceilingを要求する。
- Stage 1 packageだけ`run_stage_1=true`とし、push直後に重複実行防止のためoffへ戻す。
- Stage 0再実行、inference、submissionは常にoffとする。

## 受け入れ結果

- Stage 1 actual HMMの全体gain `0.090225 ft`、改善fold `4/5`、
  1000+・hidden-like 2面非悪化はPASS。
- by-well p95 `+0.003365 ft`、worst well `+1.403715 ft`、
  exp226 direct ceiling `+0.310086 ft`はFAIL。
- `decision=stage_1_failed_close_without_rescue`として、再実行、救済、
  inference、submissionなしで要件を終了する。
