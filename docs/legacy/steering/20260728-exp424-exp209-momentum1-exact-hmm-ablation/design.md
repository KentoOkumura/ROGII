# 設計

## アプローチ

exp209 exact HMMのrate transitionは、source rate `r`、rate step `h=0.005`、
depth step `dMD`に対して、rate-cell上の意図mean moveを次で与える。

```text
mean_move_cells = -(1 - mom) * r * dMD / h
```

現行`mom=0.998`では、実データの`dMD=1 ft`に対して
`E[r_t | r_(t-1)] = 0.998 r_(t-1)`となり、rateを0方向へ縮める。
treatmentは`mom=1.0`だけへ変更し、

```text
E[r_t | r_(t-1)] = r_(t-1)
```

としてこの0方向driftだけを除く。rate varianceに対応する`sig_r=0.002`、
隣接3-state transition、rate step / span、position transition、emission、
prior、smoother、readoutは変更しない。

## 実験範囲

- 対象実験:
  `exp424_exp209_momentum1_exact_hmm_ablation`
- Route: `pf_beam`
- 科学的親:
  `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 原因証拠:
  `exp408_hmm_message_rate_basin_audit`
- 変更する変数:
  `model.treatment_hmm.mom: 0.998 -> 1.0`
- 固定する変数:
  raw入力、prefix/suffix境界、position grid `0.35`、41 rate states、
  rate span `0.10`、rate step `0.005`、`sig_r=0.002`、`sig_p=0.02`、
  Gaussian GR emission、GR補間、Type Well、start / rate prior、
  forward-backward、posterior mean
- inference / submission:
  design時点では無効

## 仮説

exp408でpersistent episode rowsのrate posteriorは、真のrateと同方向でも絶対値が
小さい0方向under-responseとなる割合が`70.9074% rows / 70.3580% SSE`だった。
`mom=1.0`で0方向mean reversionだけを除けば、globalなrate diffusionを増やさずに
rate絶対値の縮小とその積分によるposition offsetを減らせる可能性がある。

一方、exp408のtransition-only actual-prefix診断では`mom=1.0`単独効果がactual
offset方向と対応せず、盲目的介入にはnegative evidenceもある。このためP3の
小規模mechanism preflightから開始し、Stage 0 sampleのスコアをpromotionには使わない。

## Stage 0: fixed32 mechanism preflight

### 対象well

exp411の固定manifestをそのまま再利用する。

- persistent wells: 16
- matched control wells: 16
- total: 32 unique wells / 5 folds
- manifest:
  `experiments/exp411_predictive_filtered_rate_innovation_destick/assets/stage0_fixed32_manifest.csv`
- expected SHA256:
  `fbbc62b7cb79e16a7fb436f3a9d11f8975e935ad2475a17e2dec4fd7b142e4d6`

manifestはpersistent-error情報を使ったmechanism診断sampleであり、
full OOFの代替、CV、promotion evidence、current-test policy選択には使わない。
well role、truth、episode、errorはHMM predictionとrate readoutをfreezeするまで
実行処理へ渡さない。

### 実行量

- baseline variants / HMM well-runs: `1 / 32`
- treatment variants: 1
- treatment HMM well-runs: 32
- total HMM well-runs: 64
- reporting folds: 5
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- PF / Beam / GPU: `0 / 0 / 0`

sample-matchedなparent rate momentは保存されていないためbaseline passを再実行する。
baseline TVT predictionが保存済みexp209と一致しなければ、treatment評価前にfail-closeする。
parent control再実行を含むため、Stage 0 package / push / runには別の明示承認を必要とする。

### technical AND gate

- manifest SHA、32 unique wells、persistent 16 / control 16、5 foldsを確認する。
- parent input / saved prediction SHAをexp209記録と照合する。
- implementation時の`mom=0.998` untreated parityを、小trellisとfixed wellで
  親exp209へ`<=1e-5 ft`で確認する。
- treatmentのfinite coverage=`1.0`、posterior normalization error`<=1e-5`。
- prediction / filtered-rate / smoothed-rate readout freeze前のtruth / role /
  episode / error read=`0`。
- full 773-well runtime projection`<=30,600 sec`、peak RSS`<=25 GB`。
- prediction / rate readout / metricsのcontent SHAを保存する。

### mechanism AND gate

freeze後にだけtruth、persistent episode、manifest roleをjoinする。

- persistent episode SSEをsaved exp209比`>=5%`削減する。
- persistent 16 wellsのうち`>=10` wellsでRMSEを改善する。
- persistent episode rowsの0方向under-response SSE占有率を
  同じStage 0 baseline pass比`>=2 percentage points`下げる。
- persistent episode SSEが改善するfoldを`>=4/5`とする。
- matched control RMSE deltaを`<=+0.02 ft`とする。
- matched control by-well RMSE delta p95を`<=+0.25 ft`とする。
- rate-grid edge massとnonfinite rate momentをparentより悪化させない。

一つでもFAILした場合、momentum、`sig_r`、gate、sample、blendを救済せずbranchを閉じる。

## Stage 1: full 773-well OOF

Stage 0 technical / mechanism gateが全PASSし、ユーザーが別途明示承認した場合だけ
実装・実行資格を得る。

### 実行量

- treatment variants: 1
- treatment HMM well-runs: 773
- saved parent control HMM reruns: 0
- reporting folds: 5
- model / LightGBM config / trained fold / booster / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`

### promotion AND gate

- exp209 direct HMM RMSE比`>=0.05 ft`改善する。
- 改善fold`>=4/5`。
- exp408 persistent episode SSEを`>=5%`削減する。
- MD 1000+、hidden-like spatial、hidden-like typewell-purgedを各非悪化とする。
- raw-GR observed / missingを各非悪化とする。
- by-well RMSE delta p95`<=+0.25 ft`、worst delta`<=+5.0 ft`。
- fixed LikPF / HMM 50:50 blendをsaved parent blendより非悪化とする。
- finite / normalization / input / prediction / truth-late / SHA gateを全PASSする。

PASSしてもinference / submissionへ自動移行しない。別の設計確認と承認を必要とする。

## 再現性設計

- seed policy: RNGなし。well / row / state / fold順を固定する。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: なし。
- 並列処理: HMMの既存Numba計算順を親と一致させ、outer well順を固定する。
- runtime: Kaggle private CPU、GPU / internet無効。
- input SHA:
  exp209 saved prediction、raw input、fixed32 manifest、persistent episode ledgerを記録する。
- output SHA:
  treatment prediction、filtered / smoothed rate readout、metricsを記録し、
  gzipはdecompressed content SHAを主証拠にする。
- deterministic anchor:
  submissionを生成しないためfalse。train-side数値再現性はcontent SHAで監査する。
- package:
  実装後push前にloose / bootstrap config、Notebook body、input asset SHAを照合する。

## リスク

- mechanism誤帰属:
  exp408はrate under-responseを確認したが、actual-prefix transition-only診断では
  momentum単独効果の向きがoffsetと一致しなかった。
- 過剰持続:
  `mom=1.0`はrate変化への追従速度そのものを増やすのではなく、過去rateの0方向減衰を
  除くだけなので、stale initial rateを長く保持して悪化させる可能性がある。
- sample bias:
  fixed32はpersistent-errorを含む診断sampleであり、Stage 0 gainをCVと呼ばない。
- CV/LB不一致:
  PF/Beam science branchであり、既存ML / ensemble route anchorを更新しない。
- runtime:
  Stage 0は64、full treatmentは773 HMM well-runsの高コストCPU処理なので、
  Stage 0 projection gateと各段階の別承認を必須にする。
- rescue bias:
  FAIL後の`mom=0.999`、`0.9995`、`sig_r`、rate grid、position / emission変更は禁止する。

## 2026-07-28 実装追補

- 正規実装:
  `exp424_exp209_momentum1_exact_hmm_ablation_compact_selfcontained_train.py`
- 参照構成:
  exp411 / exp412の9章compact self-contained Stage 0 Notebook。
- exact-HMM kernel:
  exp209と同じforward-backwardを自己完結実装し、position posterior meanに加え
  predictive / filtered / smoothed rate mean、filtered / smoothed rate std、
  rate-grid両端1 stateのmassを保存する。
- 単一変更境界:
  parent / treatment configのleaf差分が`mom`だけであることを実行前に検証する。
- truth-late:
  32 wells全variantのpredictionとrate readoutをfreezeしてSHAを作るまで、
  suffix truthとpersistent episode ledgerを読まない。
- Stage 0:
  2026-07-28に実行承認を得てVersion 1を完走した。実行後は
  `run_hmm=false` / `create_prediction=false`へ戻し、再実行をfail-closeする。
- Stage 1:
  未実装。Stage 0全gate PASSと別承認後だけ実装候補にする。
- inference:
  fail-closed placeholderのみ。prediction / submissionは作らない。

## 2026-07-28 Stage 0実測

- canonical kernel:
  `kentookumura/exp424-exp209-momentum1-exact-hmm-ablation-train`
- version / id:
  `1 / 128924158`
- elapsed / peak RSS:
  `2,077.533832秒 / 1.030926 GB`
- technical gate:
  13 / 13 PASS
- mechanism gate:
  3 / 7 PASS
- final:
  `stage0_fail_closed`

rate under-response SSE shareは`46.601854%`から`36.751859%`へ低下したが、
persistent episode SSE削減は`0.475550%`、改善wellは`8 / 16`、
改善foldは`3 / 5`に留まった。smoothed rate edge massも
`+0.000377954`悪化した。0方向rate mean reversion除去はrate readoutへ作用しても、
persistent TVT offsetを一貫して修復しない。

設計済みfail policyに従い、momentum、`sig_r`、sample、gate、blendを
same-OOFで救済せずbranchを閉じる。Stage 1、inference、submissionは行わない。
