# exp355 exp226 dip-rate prior on exp209

## 状態

- Route: `pf_beam`
- 状態: Kaggle CPU Stage 1 version 2完了、scientific gate FAIL、branch closed
- 優先度: 完了・追加救済なし
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 履歴参照: `exp323_time_varying_exp226_dip_rate_prior`

## 仮説

fold-safe exp226 K16 geometryの絶対TVTや最終予測ではなく相対的なU-rate変化だけなら、
exp209のconstant rate-prior meanよりknown-prefix外のdip変化を説明できる可能性がある。

## 変更点

- exp307--309/338 dependencyを削除し、trusted exp209へ直接接続する。
- exp209から変更するのはrow-wise rate-prior meanだけ。
- Stage 0は0-HMM identifiability readout。Stage 1はworst-well gateをuser overrideし、
  train-sideの1候補 / 773 HMM runsだけを実行する。
- K16はexp226と同じrow-position 16分割、区間rateはfinite positive-`ΔMD` stepの
  中央値に固定する。
- 先頭geometry区間がinvalidならwell全体、後続区間だけinvalidなら当該区間を
  parent constant rateへfallbackする。

## 検証方針

- outer-fold-safe exp226 geometry ledgerとscheduleをsuffix truth前にfreezeする。
- Stage 0はsegment rate-change RMSE、cumulative path RMSE、fold、1000+、
  hidden-like 2面、worst-wellをAND gateにする。
- fold gateはsegment rate-changeとcumulative pathの双方で4/5改善を要求する。
- exp226 `tvt_pred`、GR correction、absolute unary、blend、parameter gridは禁止する。

## 実行入口

- train: `exp355_exp226_dip_rate_prior_on_exp209_train.ipynb`
- inference: `exp355_exp226_dip_rate_prior_on_exp209_inference.ipynb`
- 実装候補:
  `exp355_exp226_dip_rate_prior_on_exp209_compact_selfcontained_train.py/.ipynb`
- Stage 1実装候補:
  `exp355_exp226_dip_rate_prior_on_exp209_stage1_compact_selfcontained_train.py/.ipynb`
- fail-closed inference候補:
  `exp355_exp226_dip_rate_prior_on_exp209_compact_selfcontained_inference.py/.ipynb`
- 正規train Notebookへcompact self-contained候補を採用した。
- inferenceの正規Notebookはplaceholderを保持し、実行・submissionは禁止する。

## 所見

Stage 1 direct RMSEはexp209の`11.938287`から`11.291977`へ`0.646311 ft`
（`5.414%`）改善し、5/5 foldsで改善した。fixed LikPF 50:50もpooledでは
`10.269696 -> 10.053144`と改善した。一方、hidden-like spatial /
typewell-purgedは`+0.414943 / +0.371720 ft`悪化し、413/773 wellsが悪化、
worst `86454a6f`は`+52.743754 ft`だった。平均signalは実在するが、
hidden-test-like transferとwell単位の安全性を満たさない。

## 次

事前のfailure actionどおりparameter/blend/selector救済、inference、submissionを
行わず閉じる。独立仮説はexp355の後付け救済として扱わず、別実験で評価する。
