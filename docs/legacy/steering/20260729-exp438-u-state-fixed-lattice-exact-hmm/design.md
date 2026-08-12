# 設計

## 1. 座標変換と実験差分

exp209 の state は `(p_t, r_t)` で、`p_t=TVT_t`、
`r_t=d(TVT+Z)/dMD` である。

```text
p_t - p_(t-1) = r_t * delta_MD - delta_Z
```

exp438 は位置 state を `u_t=p_t+Z_t` に変える。

```text
u_t - u_(t-1) = r_t * delta_MD
p_t = u_t - Z_t
```

連続空間では両式は厳密に同値であり、新しい物理 prior ではない。
差分は離散格子だけである。親は同じ TVT 格子を全rowで固定するが、exp438は
最後の既知点で作った同じ個数・step・start indexの格子へ`Z_last`を足し、
その絶対 U 格子を全rowで固定する。

```text
parent_tvt_grid = arange(parent_grid_min, parent_grid_max, 0.35)
candidate_u_grid = parent_tvt_grid + Z_last
candidate_tvt_grid[t, :] = candidate_u_grid - Z_t
```

position kernel は親の到着rate方式を保つ。

```text
mu_parent_tvt = r_t * delta_MD - delta_Z
mu_candidate_u = r_t * delta_MD
```

candidate emission は
`GR_typewell(candidate_u_grid-Z_t)`で各rowごとに作り、
readout は `E[U_t]-Z_t` とする。

## 2. 固定する exp209 contract

- grid step: `0.35 ft`
- 親と同じgrid cell数、last-known anchor、band pad
- rate states: 41
- rate span: `0.10`
- `sig_r=0.002`, `momentum=0.998`
- `sig_p=0.02`, effective sigma `0.1225 ft`
- position support: mean近傍5 cells
- arrival-rate position mean
- Gaussian Type Well GR emission、prefix affine calibration、sigma、lambda
- initial U-rate prior、rate support
- exact sum-product forward/backward
- smoothed posterior mean/std

source/destination rate台形積分、acceleration covariance、
row-adaptive grid、grid interpolationは入れない。

## 3. 数値contract

- 任意のrow/stateで`TVT_state == U_state-Z_t`を`1e-12 ft`以内で満たす。
- 任意のedgeで
  `delta_TVT == delta_U-delta_Z`を`1e-12 ft`以内で満たす。
- emission lookupは
  `GR_typewell(U_state-Z_t)`の直接計算と`1e-12`以内で一致する。
- `Z_t`が全rowで`Z_last`と同じsynthetic caseでは、親とcandidateの
  emission、transition、predictionが`1e-6 ft`以内で一致する。
- small synthetic HMMをdense brute-force referenceと比較し、
  log-likelihoodとsmoothed posteriorを`1e-6`以内で一致させる。
- transition row sum最大誤差`<=1e-10`、
  posterior normalization最大誤差`<=1e-8`。
- fixed32で親格子と異なる非自明なZ-phaseを持つrowが存在し、
  candidate predictionが単なる保存control copyでないことをtruth-freeに確認する。
- well、row、position、rate、edgeの加算順を固定し、pruning / Viterbi近似を使わない。

## 4. Stage 0: fixed32 mechanism

exp411/435と同じSHA固定fixed32
（persistent 16 + matched control 16）でcandidateだけを実行する。
保存済みexp209はprediction controlとしてloadし、HMMは再実行しない。

candidate prediction、posterior/rate readout、parent/candidate position-kernel
quantization ledger、content SHAをfreezeした後だけ、
role、fold、persistent episode、suffix truthをjoinする。

technical AND gate:

- 32 wells、finite prediction coverage `1.0`。
- treatment `1 × 32 = 32 HMM well-runs`、parent rerun `0`。
- 上記coordinate / constant-Z / brute-force / normalization contractを全PASS。
- truth / fold / episode reads before freeze=`0 / 0 / 0`。
- Stage 1 runtime projection `<=30,600 sec`、peak RSS `<=25 GB`。

mechanism AND gate:

- posterior-weighted absolute position-kernel quantization biasを親式比`>=10%`削減。
- exp408 forward-cause episode SSEを保存exp209比`>=5%`削減。
- persistent episode SSEを`>=3%`削減。
- persistent 16 wells中`>=10` wells改善。
- persistent fold改善`>=4/5`。
- matched-control pooled RMSE delta`<=+0.02 ft`。
- matched-control by-well delta p95`<=+0.25 ft`。

1条件でもFAILならStage 1へ進まず、grid anchor/phase/step/band、position sigma、
rate、emission、blend、selectorで救済しない。

## 5. Stage 1: full direct OOF

Stage 0のtechnical/mechanism gateを全PASSし、かつ別承認がある場合だけ
同じ1 variantを773 wellsへ実行する。

hard gate:

- exp209 direct RMSE `11.938287234887435`から`>=0.05 ft`改善。
- 改善fold`>=4/5`。
- forward-cause / persistent episode SSEを各`>=5% / 3%`改善。
- raw observed、raw missing、high-missing、1000+、hidden-like spatial、
  hidden-like typewell-purgedの各delta`<=+0.02 ft`。
- by-well delta p95`<=+0.25 ft`、worst`<=+2.0 ft`。

exp263 fixed blend `8.238331667`はreport-onlyとし、exp438を混ぜたblend、
weight、selector、route adoption gateは作らない。

## 6. 既存実験との区別

- exp435はpositionをTVTのままrate historyを捨てた。exp438はjoint rate historyを保つ。
- exp437はexp435のTVT-only chainへexp226 geometry incrementを入れる。
  exp438はexp209のjoint rate historyとarrival-rate積分を維持する。
- exp436はouter-trainからglobal U potentialをfitする。
  exp438は学習済みpotentialや他well情報を使わない。
- exp364は第3のcurvature stateを追加した。exp438はstate次元数を増やさない。

## 7. 再現性と実行量

- RNGなし。single worker、固定well/row/state/rate/edge順。
- fixed32、input、coordinate contract、transition ledger、prediction、
  posterior/rate readout、gate reportのlogical/content SHAを保存する。
- NumPy、Numba、CPU/thread情報とstate orderをmanifestへ保存する。
- gzipはdecompressed content SHAを主証拠にする。
- 初回runをdeterministic anchorとしない。同一設定rerunのtransition /
  prediction SHA一致後だけanchor再評価を可能にする。

確定した実行量:

- Stage 0: 1 variant × 32 wells = 32 HMM well-runs。
- Stage 1最大: 1 variant × 773 wells = 773 HMM well-runs。
- reporting folds: 5。
- parent control HMM rerun: 0。
- model / LightGBM config / trained ML fold / booster=`0 / 0 / 0 / 0`。
- PF / Beam / GPU=`0 / 0 / 0`。
- 設計確定時点では実装、実行可能notebook、package/run、inference、
  submission各0。

## 8. Stage 0実装

2026-07-29の追加実装承認に基づき、Jupytext percent形式の
`exp438_u_state_fixed_lattice_exact_hmm_compact_selfcontained_train.py`
を実装した。

- generic fixed-lattice joint forward/backwardへ、candidateは
  `transition_coordinate_delta=0`を渡し、親式監査時だけ`delta_Z`を渡す。
- candidate gridは`parent_tvt_grid+last_known_Z`を一度だけ作り、
  emissionはrowごとに`U_grid-Z_t`へ変換する。
- smoothed position/rate posteriorを保持し、`E[U]-Z`、TVT std、
  rate mean/std/edge massを保存対象にする。
- parent/candidate quantization biasは同じcandidate smoothed rate posteriorで
  重み付けし、座標差だけを比較する。
- small HMMは全initial/suffix state pathを列挙する独立referenceと比較する。
- fixed32のwell/prefix/suffixだけを先に読み、role/fold/truth/episode/causeは
  全prediction/rate/transition SHA freeze後に読む。
- compact inference候補はfail-closed guardのみである。

正規notebookは既存placeholderを維持し、Kaggle package / push / run、
Stage 1、inference、submissionは未承認のままにした。
