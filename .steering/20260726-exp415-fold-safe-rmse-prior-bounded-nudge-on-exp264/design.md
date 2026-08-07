# 設計

## アプローチ

### Phase 1: truth-free policy freeze

corrected exp264 candidate-score OOFからtarget-free列だけをstream readする。
outer fold `f`、候補 `c` のfit RMSE `r_fc`はexp407の保存tableから読む。

```text
parent_pos = argmin_c parent_pred_abs_error[c]
prior_pos  = argmin_c (parent_pred_abs_error[c] + r_fc)
parent_tvt = candidate_tvt[parent_pos]
prior_tvt  = candidate_tvt[prior_pos]
raw_nudge  = 0.5 * (prior_tvt - parent_tvt)
correction = clip(raw_nudge, -0.25, +0.25)
prediction = parent_tvt + correction
```

`id / well / fold / md_since / selected candidates / candidate TVT /
correction / prediction`をfreeze Parquetへ保存し、SHAを確定する。
このphaseでは`actual_abs_error`、TVT、oracle、errorを読まない。

### Phase 2: evaluation

freeze SHA確定後、同じparent OOFから`actual_abs_error`を別streamで読む。
各base rowについてcandidate 0の`candidate_tvt ± actual_abs_error`の2候補を作り、
12候補すべてのabsolute errorと一致する側をtrue TVTとする。全候補で
max residual 0を要求する。

親predictionとbounded nudge predictionのSSE / RMSEをoverall、fold、distance、
hidden-like、well別に集計する。row prediction artifact、各集計CSV、risk certificate、
gate JSON、reproducibility manifestを保存する。

### 数学的risk bound

scope内の親誤差vectorを`e`、correction vectorを`d`とすると、
Minkowski inequalityにより次が成り立つ。

```text
RMSE(e + d) - RMSE(e) <= RMS(d) <= max(abs(d)) <= 0.25 ft
```

したがって、集計結果を見る前から任意well / fold / bucketのRMSE悪化上限を
0.25 ftに固定できる。observed worst-wellも別途照合する。

ローカルの探索readoutで同じ固定policyが親`8.587004`から`8.563474`へ改善し、
5/5 folds、4 distance buckets、hidden-like 2面を改善、worst-well delta
`+0.171379`だった。exp415はこの既知policyをKaggle上で再現・固定する
confirmation readoutであり、未観測のprospective探索とは表記しない。

## 実験範囲

- 対象実験: `exp415_fold_safe_rmse_prior_bounded_nudge_on_exp264`
- Route: `ensemble`
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- 原因元: `exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264`
- 変更する変数:
  保存済み親hard predictionへ、fit candidate RMSEで方向を決めた
  risk-bounded TVT correctionを追加する。
- 固定する変数:
  OOF surface、candidate order / domain / values、fold、RMSE table、
  prior係数1、blend 0.5、cap 0.25、metrics / scope。
- 実行予算:
  variants 1、model/config/fold fit/booster/control/PF/HMM/Beam/GPUは0。
- 対象外:
  model training、candidate regeneration、inference、submission、LB、
  coefficient / cap / blend grid、route anchor更新。

## 再現性設計

- seed policy:
  RNGなし。row order、candidate order、stable argmin tie-breakで決定論的。
- stochastic 処理:
  なし。
- PF/Beam / likelihood-PF / seed bagging:
  保存済みcandidate TVTをload-only。新規生成0。
- 並列処理:
  single-process Parquet stream。乱数なし。
- CPU/GPU:
  Kaggle private CPU、internet off、GPU false。
- input:
  parent candidate-score OOF SHA、exp407 RMSE table SHA、hidden assignment SHAを
  起動時とmanifestで記録する。
- freeze / prediction:
  truth-free freeze SHA、row prediction SHA、metrics / gate SHAを保存する。
- model / submission:
  model 0、submission 0。対象外として明記する。
- Kaggle package:
  push前にbootstrap内config、入力source、CPU / internet、run scopeを照合する。

## リスク

- リークリスク:
  selection前にactual errorを読むとpolicy leakageになる。二相streamとtruth-read
  ledgerでfail closedにする。候補RMSEはexact fit partition保存値だけを使う。
- 過学習:
  policyはローカル探索後のconfirmationであり、完全なprospective evidenceではない。
  数値を見た後のgrid救済を禁止し、5 folds / hidden-like / risk certificateを併記する。
- CV/LB不一致:
  保存OOF diagnosticでありLBやcurrent-test改善を主張しない。
- ランタイム/メモリ:
  約398 MB Parquetを二回streamする。base-row block単位のbatchで処理し、
  全candidate-long frameをmemoryへ保持しない。
- 再現性:
  Parquet row order、candidate block、input source driftをSHAとstrict key checkで防ぐ。
- route:
  ML selectorとPF/Beamを含むcandidate predictionを直接混ぜるため`ensemble`とする。
