# exp356 セッションノート

## 目的

旧exp324のdonor-covariance仮説をexp323から分離し、exp209 constant-rate面で単独監査する。

## 現在の状態

- Route: `pf_beam`
- 状態: donor support非退化の独立証拠待ちでblocked/demoted、未実装、未実行
- CV/LB: なし

## 2026-07-23 設計

- exp356として採番し、steeringとscaffoldを作成した。
- parentはexp209、rate meanはconstant、変更はK16 `sig_r,t`だけ。
- Stage 0はdiagnostic 1 / 5 folds / HMM・model・booster 0。
- Stage 1予約は1 variant / 773 HMM runs / control再実行0。
- 実装、Notebook採用、Kaggle実行、inference、submissionは未実施。

## 再現性メモ

- RNGなし。fold/well/segment/donor順を固定する。
- exp226/exp209 input SHAとdonor ledger、scale schedule content SHAを記録する。
- fitted modelなし。Stage 1時のみdecoder/prediction SHAを記録する。
- deterministic anchorとは扱わない。

## 次のアクション

2026-07-24のexp362 post-run監査では、同じK16 / k50 / bandwidth 500 ft前提で
`n_eff>=10`が0/12,368 segments、nearest-distance failureが772だった。
同じsupport契約のまま実装しない。truth-freeな独立support readoutで非退化coverageを
事前固定できた場合だけ、別承認後にStage 0実装を再検討する。
