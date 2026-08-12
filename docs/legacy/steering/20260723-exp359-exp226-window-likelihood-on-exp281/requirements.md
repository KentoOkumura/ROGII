# 要件

## 依頼

閉鎖済み`exp325_exp226_window_likelihood_hmm_tempering`をreopenせず、
exp323/exp338 successor chainを使わない新番号の独立window-likelihood監査として設計する。
2026-07-23時点ではbacklog、steering、scaffoldの確定までとした。
2026-07-25の最初のユーザー依頼でStage 0実装を承認し、続く「実行してください」で
正規train Notebook採用とKaggle private CPU Stage 0を1回実行することを承認した。
Stage 1、inference、submissionは引き続き別承認とする。

## 制約

- Route: `pf_beam`
- 親: `exp281_exp226_residual_offset_exact_hmm_transition_probe`
- 履歴参照: `exp325_exp226_window_likelihood_hmm_tempering`
- exp281のtransition mean/variance、row Gaussian emission、grid、momentum、prior、
  posterior outputを固定する。
- 唯一の変更はexp226 500-row GR window scoreをfixed stride中心へ
  sparse normalized potentialとして追加すること。
- window、stride、score weight、normalization、lambda、minimum coverageを固定し、
  gridやblendを禁止する。
- Stage 0はscore readout 1、saved control 1、HMM/model/trained fold/booster各0。
- Stage 1はStage 0全gate PASSと別承認時だけ1 variant / 773 HMM runs。
- exp281はpromotion FAIL済みのため、Stage 1はexp281比改善に加えて
  direct RMSEがexp226 `9.427109596582213`以下であることを要求する。

## 受け入れ基準

- window identity、relative shift bank、GR profile、score surface、eligible mask、
  lambda scheduleをtruth join前にSHA固定する。
- Stage 0でMRR/top3各0.01以上、4/5 folds、real>shuffle 5/5 folds、
  1000+・hidden-like 2面正方向、eligible window 25%以上を要求する。
- Stage 1でexp281比0.05 ft以上、4/5 folds、stress/p95/worst非悪化、
  exp226 direct ceilingを要求する。
- exp305/343はnegative referenceだけとし、入力依存や救済gridに使わない。
- 2026-07-23設計snapshotではdesign-only、未実装、実行flag全offで整合させる。

## 2026-07-25 実装受け入れ基準

- compact self-contained train候補とfail-closed inference候補を別名で作り、
  既存の正規Notebook placeholderを上書きしない。
- exp226 window scoreは500 rows / stride 125、固定13 shifts、correlation/MSE/level、
  state normalization、fixed lambdaだけを実装する。
- posterior SDはHMMを走らせず、固定13-shift normalized scoreのsoftmaxから算出する。
- 保存exp280 controlは各window centerが属する512-row blockへ対応付け、
  部分block scoreを推定・再構成しない。
- target-free bundleをSHA固定してからtruthを読む。
- Stage 0実行量はscore 1 / saved control 1 / reporting fold 5 /
  HMM・model・trained fold・booster各0を維持する。

## 次のアクション

Kaggle private CPU Stage 0は完了し、固定科学gateをFAILした。
Stage 1の1 variant / 773 HMM runs、inference、submissionへ進まず、
同じwindow familyの救済gridを行わずに閉じる。
