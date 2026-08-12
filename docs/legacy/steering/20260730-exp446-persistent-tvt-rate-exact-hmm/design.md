# 設計

## 1. 検証する問い

exp209の状態は`(P_t,r^U_t)`で、

```text
r^U = d(TVT+Z)/dMD
delta_TVT_mean = r^U_t * delta_MD - delta_Z
```

である。candidateは状態を`(P_t,q_t)`へ変える。

```text
q = dTVT/dMD
delta_TVT_mean = q_t * delta_MD
```

`g_t=delta_Z_t/delta_MD_t`、
`a_t=1-(1-momentum)*delta_MD_t`とすると、親から誘導される平均は
`E[q_t]=a_t*(q_{t-1}+g_{t-1})-g_t`である。この既知forcingを含めれば
単なる再パラメータ化になる。本実験ではそれを行わず、
`E[q_t]=a_t*q_{t-1}`として`q`自身へexp209と同じzero-centered local
OU/Euler kernelを適用する。したがって、既知`Z`勾配をrate dynamicsから
除いた方がTVT軌道を追いやすい、という新しい科学仮説である。

## 2. 単一candidateの状態契約

prefix末尾50 stepの有効行から初期rateを次で固定する。

```text
parent r0_U = median((delta_TVT_input + delta_Z) / delta_MD)
candidate q0 = median(delta_TVT_input / delta_MD)
```

candidate rate gridはexp209と同じ規則をTVT-rateへ適用する。

```text
span_q = max(0.10, abs(q0) + 0.04)
q_grid = linspace(-span_q, span_q, 41)
initial q prior = Normal(q0, 0.01)
```

rate transitionはexp209の隣接3-bin kernelをそのまま`q_grid`上で使い、
`momentum=0.998`、`sig_r=0.002`、current `delta_MD`を固定する。
destination rateでposition kernelを中心化する。

```text
candidate position mean = q_destination * delta_MD
```

TVT position grid、0.35-ft step、5-cell position kernel、
`sig_p=0.02`（effective floorを含む）、band、GR emission、
emission lambda、position prior、forward/backward、posterior mean/stdは
exp209から変更しない。

## 3. Technical sentinel

実データのtruthを使わない次のcontractを実装時に必須とする。

1. prefixとsuffixの`Z`が一定のsynthetic系列では`delta_Z=0`なので、
   parentとcandidateのrate grid、rate kernel、position kernel、
   log-likelihood、position/rate posterior、TVT mean/stdが一致する。
2. 小さいposition/rate/time stateについてdense brute-forceと
   forward/backwardを比較する。
3. 各rate/position transitionのrow sum、有限性、posterior normalization、
   `delta_TVT=q_destination*delta_MD`のedge残差を監査する。
4. fixed32のwell identityだけを先に読み、全prediction、posterior、
   diagnostic SHAをfreezeするまでtruth、role、fold、episode、errorを読まない。

constant-Z sentinelは実験variantや性能controlではなく、
rate定義以外の意図しない実装差を検出するtechnical negative controlである。
exp445の座標パリティを性能改善の根拠には使わない。

## 4. Stage 0 fixed32

- scientific variant: `persistent_tvt_rate` 1本。
- candidate HMM well-runs: 32。
- 保存exp209 control rerun: 0。
- fixed32: persistent 16 + matched control 16、合計156,088 suffix rows。
- fixed32はmechanism preflightであり、CVやpromotion evidenceではない。

Technical AND gate:

- finite coverage `=1.0`。
- rate / position transition row-sum error `<=1e-6`。
- posterior normalization error `<=1e-6`。
- constant-Z sentinelのkernel差`<=1e-12`、loglik/posterior/prediction差
  `<=1e-6`。
- small-state brute-force prediction差`<=1e-6`。
- position edge残差`<=1e-12 ft`。
- prediction freeze前のtruth / role / fold / episode / error read `=0`。
- 773-well runtime projection `<=30,600 sec`、peak RSS `<=25 GB`。

Mechanism AND gateはfreeze後にだけ評価する。各current transitionについて
`g_t=delta_Z/delta_MD`、`q_true=delta_TVT_true/delta_MD`、
親のimplied TVT-rateを`E[r^U_t]-g_t`として同じ座標で比較する。

- TVT-rateのzero-directed under-response SSE shareを絶対`0.05`以上削減。
- exp408 forward-cause episode SSEを`10%`以上削減。
- persistent episode SSEを`5%`以上削減。
- persistent 16 wells中10以上、かつ4/5 foldsで改善。
- matched control pooled RMSE delta `<=+0.02 ft`。
- matched control by-well delta p95 `<=+0.25 ft`。

1項目でもFAILなら、rate定義、span、momentum、noise、grid、emission、
prior、gate、blend、selectorを救済せずbranchを閉じる。

## 5. Stage 1

Stage 0のtechnical/mechanism gateをすべてPASSし、別のユーザー承認を得た場合だけ
773 wellsのcandidateを実行する。parentは保存exp209を使い再実行しない。

Promotion AND gate:

- direct RMSEをexp209より`0.05 ft`以上改善。
- 4/5 folds以上で改善。
- forward-cause SSE `10%`以上、persistent SSE `5%`以上削減。
- near 0--250、mid 250--1000、1000+、hidden-like spatial、
  hidden-like typewell-purged、raw-GR observed/missingの各固定scopeで
  RMSE delta `<=+0.02 ft`。
- by-well delta p95 `<=+0.25 ft`、worst-well `<=+2.0 ft`。

Stage 1 PASS後もinference、既存候補との固定blend、selector、submissionは
自動承認しない。必要なら同じexp446内の後続段階として別承認を得る。

## 6. 実験範囲

- 対象実験: `exp446_persistent_tvt_rate_exact_hmm`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 参照: exp408、exp435、exp441、exp445
- 変更する変数: persistent rateの物理定義、prefix初期値、
  rate gridの数値、rate dynamicsが作用する座標、position mean式
- 固定する変数: state数、span規則、momentum、rate/position noise、
  TVT grid、position support、GR emission、prior family、readout、評価scope
- 実行量: Stage 0候補32、Stage 1候補773、parent rerun 0、
  model / booster / PF / Beam / GPU 0

## 7. 再現性設計

- seed policy: RNGなし。well、row、TVT position、TVT-rate、edge、message、
  reduction順を固定する。
- stochastic処理、PF/Beam、likelihood-PF、seed bagging、GPU学習はない。
- Kaggle private CPU、1 worker、Numba thread 1を既定とする。
- input manifest、rate grid、rate kernel、joint transition、posterior、
  prediction、diagnostic、metricsのSHAを保存する。
- gzipはraw SHAとdecompressed content SHAを分け、後者を主証拠にする。
- model / submission SHAは対象外。Kaggle packageを作る場合はbootstrap内
  configと正規configのSHA一致をpush前に確認する。
- 初回runはdeterministic anchorにしない。独立rerunでinput、
  transition、posterior、prediction SHAが一致した場合だけanchor化する。

## 8. リスク

- 物理仮説: `U=TVT+Z`の既知幾何をrate dynamicsから外すため、
  U-rateより弱いモデルになり、matched controlを大きく壊す可能性がある。
- 離散化: wellごとに`q0`が変わるため、親とrate supportの数値は一致しない。
  spanは同じ規則に固定し、support tuningは行わない。
- 解釈: exp435の失敗をpersistent TVT-rateの失敗と混同せず、
  exp445のparityを性能改善と解釈しない。
- leakage: fixed32 roleやexp408 episodeをprediction前に読むと選択リークになる。
- CV/LB: fixed32はCVではない。full OOF PASS前にinferenceやLB評価へ進めない。
- runtime/memory: state数は親と同じだが、実測projectionが8.5時間または25 GBを
  超えた場合は高速化チューニングで救済しない。

## 9. 承認境界と実行結果

2026-07-30のユーザー依頼により、compact実装、正規Notebook採用、
Kaggle package、fixed32 Stage 0を順に承認した。Stage 1、inference、
submissionは承認していない。

Kaggle private CPU version 1（id_no `129106260`）は32/32 wellsを完走した。
technicalはruntime projectionだけFAILして`17/18`、mechanismは`0/7`。
under-response、forward/persistent SSE、改善well/fold、matched-control安全性を
すべてFAILしたため、事前設計のfail actionどおりbranchを閉じる。
Stage 1、rerun、inference、submission、parameter/gate/blend/selector救済は
行わない。
