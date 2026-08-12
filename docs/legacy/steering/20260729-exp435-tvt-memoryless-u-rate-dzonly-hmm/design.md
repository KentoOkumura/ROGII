# 設計

## アプローチ

exp209 exact HMMの持続状態`(TVT, r_U)`を、TVTだけの持続状態へ縮約する。
ここで`U=TVT+Z`、`r_U=dU/dMD`なので、1行の期待変位は次で表す。

```text
ΔTVT = r_U * ΔMD - ΔZ
```

`memoryless_41rate`では非ゼロrate候補を残すが、rateは行間をまたぐ状態ではなく
1遷移だけのedge latent variableとする。各行で41候補を周辺化した後、
rate responsibilityを破棄し、TVT posteriorだけを次行へ渡す。

```text
q_t(T) ∝ L_t(T)
           Σ_Tprev q_(t-1)(Tprev)
           Σ_r pi(r) Kp(T - Tprev; r * ΔMD - ΔZ)
```

`dz_only_r0`は同じ式で`pi(r)=delta(r=0)`とした特殊ケースである。

```text
q_t(T) ∝ L_t(T)
           Σ_Tprev q_(t-1)(Tprev)
           Kp(T - Tprev; -ΔZ)
```

両variantともforward-backward smoothingをTVT状態上で行い、出力はTVT posterior
mean / stdとする。単一の直前TVT予測値だけを再帰する設計にはしない。

## 因果比較

| 比較 | 切り分ける効果 |
| --- | --- |
| exp209 vs `memoryless_41rate` | 非ゼロrate supportを残したままrate履歴を除く効果 |
| `memoryless_41rate` vs `dz_only_r0` | rate履歴なし条件で非ゼロU-rate supportを除く効果 |
| exp209 vs `dz_only_r0` | rate状態と非ゼロrateを同時に除く総合効果 |

`memoryless_41rate`は、条件付き`p(r_t|r_(t-1))`を固定`pi(r_t)`へ置換するため、
前行rate posteriorとprefix rate meanの両方を次行へ持ち越さない。これは本実験でいう
「rate履歴なし」の操作そのものであり、結果後に別解釈へ変更しない。

## 実験範囲

- 対象実験:
  `exp435_tvt_memoryless_u_rate_dzonly_hmm`
- Route:
  `pf_beam`
- 科学的親:
  `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 原因証拠:
  `exp408_hmm_message_rate_basin_audit`
- negative intervention evidence:
  `exp424_exp209_momentum1_exact_hmm_ablation`
- nonzero-rate evidence:
  `exp355_exp226_dip_rate_prior_on_exp209`
- 変更する変数:
  rateを持続stateからedge latentへ変更し、さらに`r_U=0`特殊ケースを比較する。
- 固定する変数:
  raw入力、prefix/suffix境界、TVT grid、initial TVT prior、Type Well、
  GR preprocessing / emission、position kernel、forward-backward、
  posterior mean / std、reporting folds
- inference / submission:
  design時点では無効

## 固定HMM contract

親exp209から次を固定する。

- TVT grid step: `0.35 ft`
- band pad: `100 ft`
- GR emission: Gaussian
- GR sigma policy: known-prefix robust residual
- missing GR: parentと同じ補間
- position noise input: `sig_p=0.02`
- effective position sigma:
  `max(sig_p, 0.35 * grid_step)=0.1225 ft`
- position transition support:
  親と同じ5 grid cells
- posterior readout:
  smoothed TVT posterior mean / std
- RNG:
  なし

親exp209のrate設定は、memoryless supportと固定重みを定義するためだけに使う。

- `n_rates=41`
- `rate_span=0.10`
- `sig_r=0.002`
- `mom=0.998`
- `rate_center=zero`
- per-well support half width:
  `span=max(0.10, abs(init_rate)+0.04)`
- rate grid:
  `linspace(-span, span, 41)`

## Memoryless rate重み

毎行の41候補重みは、親AR(1)の1 ft step定常分布をgrid上で離散化する。

```text
stationary_sd = sig_r / sqrt(1 - mom^2)
              = 0.002 / sqrt(1 - 0.998^2)
              ≈ 0.0316386

raw_weight_i = exp(-0.5 * (rate_i / stationary_sd)^2)
pi_i = raw_weight_i / Σ_j raw_weight_j
```

- `pi_i`は全行で固定する。
- `pi_i`はprefix `init_rate`、前行posterior、GR、TVT、truthで更新しない。
- `init_rate`はrate gridのsupport幅にだけ使用する。
- rowごとのrate responsibilityをreport-onlyで計算しても、次行transitionへ渡さない。
- uniform、prefix-centered、filtered-centered、temperature付き重みは本実験では扱わない。

## dz-only contract

`dz_only_r0`は独立実装にせず、position-only transitionへ次を渡す。

```text
rates = [0.0]
weights = [1.0]
```

したがって、

```text
expected ΔTVT = -ΔZ
```

となる。synthetic contract testでは、41-rate kernelへdelta-at-zero重みを与えた場合と
dz-only出力が`1e-10 ft`以内で一致することを要求する。

dz-onlyが使う「直近TVT」は前行の予測分布であり、suffix中の真のTVTではない。
最初のlast-known TVT以外のabsolute anchorを追加しないため、これはreanchor実験ではない。

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

manifestはpersistent-error情報で選ばれたmechanism sampleであり、CV、promotion evidence、
current-test policy選択には使用しない。

### 実行量

- treatment variants: 2
- variants:
  `memoryless_41rate`, `dz_only_r0`
- wells per treatment: 32
- treatment HMM well-runs: 64
- parent HMM well-runs: 0
- reporting folds: 5
- model / LightGBM config / trained fold / booster:
  `0 / 0 / 0 / 0`
- PF / Beam / GPU:
  `0 / 0 / 0`

親対照は保存済みexp209 predictionを使う。Stage 0 package / push / runは
2026-07-29のユーザー指示`実行してください。`により承認済み。Stage 1、
inference、submissionにはこの承認を拡張しない。

### technical AND gate

- manifest SHA、32 unique wells、persistent 16 / control 16、5 foldsを確認する。
- exp209保存predictionの入力SHA、row identity、fixed32 coverageを確認する。
- TVT transition row sum / posterior normalization誤差を各`<=1e-6`とする。
- candidate finite coverageを各`1.0`とする。
- delta-at-zero memorylessとdz-only predictionの最大差を`<=1e-10 ft`とする。
- rate responsibilityが次行stateに保存されないことをcontract testで確認する。
- 全candidate prediction / diagnostic freeze前のtruth、role、fold、episode、
  error readを`0`とする。
- 全Stage 1 eligible treatmentの合計runtime投影を`<=30,600 sec`、
  peak RSSを`<=25 GB`とする。
- input、config、transition contract、prediction、diagnostic、metricsの
  logical content SHAを保存する。

### variant別mechanism AND gate

freeze後にだけtruth、persistent episode、exp408 cause、manifest roleをjoinし、
2 treatmentをそれぞれ独立に判定する。

- exp408 forward-cause episode SSEをsaved exp209比`>=10%`削減する。
- 全persistent episode SSEをsaved exp209比`>=5%`削減する。
- persistent 16 wellsのうち`>=10` wellsでRMSEを改善する。
- persistent episode SSEが改善するfoldを`>=4/5`とする。
- matched control pooled RMSE deltaを`<=+0.02 ft`とする。
- matched control by-well RMSE delta p95を`<=+0.25 ft`とする。

一方がFAILしても他方の事前固定評価は完了させる。FAIL variantはStage 1へ進めない。
両方FAILならbranchを閉じ、rate重み、support、noise、emission、gateを救済しない。

## Stage 1: full 773-well OOF

Stage 0 technical gateとvariant別mechanism gateを通過し、ユーザーが別途承認した
treatmentだけを実装・実行対象とする。

### 最大実行量

- eligible treatment variants: 0--2
- HMM well-runs per treatment: 773
- maximum treatment HMM well-runs: 1,546
- saved parent HMM reruns: 0
- reporting folds: 5
- model / LightGBM config / trained fold / booster / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`

### scientific promotion AND gate

- exp209 direct HMM RMSE比`>=0.05 ft`改善する。
- direct HMM改善foldを`>=4/5`とする。
- exp408 forward-cause episode SSEを`>=10%`削減する。
- 全persistent episode SSEを`>=5%`削減する。
- MD 1000+、raw-GR observed / missing、hidden-like spatial、
  hidden-like typewell-purgedのRMSE deltaを各`<=+0.02 ft`とする。
- paired by-well RMSE delta p95を`<=+0.25 ft`、
  worst deltaを`<=+2.0 ft`とする。
- finite、normalization、truth-late、input / prediction SHA gateを全PASSする。

### route adoption gate

exp263の固定式

```text
0.50 * exp226 + 0.25 * likPF + 0.25 * exact HMM
```

のexact HMM成分だけをcandidateへ置換する。weightは変更しない。

- saved exp263 OOF `8.238331667 ft`比`>=0.03 ft`改善する。
- fixed blend改善foldを`>=4/5`とする。
- 上記scientific promotion gateを全PASSする。

scientific gate PASS、adoption gate FAILの場合はmechanism-positive /
deployment-negativeとして記録し、inference / submissionへ進めない。
両gate PASSでもinference / submissionは別の設計確認と明示承認を必要とする。

## 事前固定する診断

prediction freeze後、真の

```text
r_U_truth = (ΔTVT_truth + ΔZ) / ΔMD
```

を次の絶対値bucketでreport-only集計する。

- `abs(r_U_truth) <= 0.01`
- `0.01 < abs(r_U_truth) <= 0.03`
- `abs(r_U_truth) > 0.03`

このbucketはrateが不要な領域と必要な領域を解釈するためだけに使い、
variant選択、row gate、blend、parameter調整には使わない。

## 再現性設計

- seed policy:
  RNGなし。well / row / TVT grid / rate candidate / variant順を固定する。
- stochastic処理:
  なし。
- PF/Beam / likelihood-PF / seed bagging:
  なし。
- 並列処理:
  well単位の出力順を固定し、reduction順とdtypeをcontract SHAへ含める。
- runtime:
  Kaggle private CPU、GPU / internet無効。
- input SHA:
  raw input、exp209 saved prediction、fixed32 manifest、exp408 episode/cause ledger、
  exp226 / likPF / exp263固定式入力、hidden-like assignmentを記録する。
- output SHA:
  candidate prediction、posterior std、report-only edge-rate readout、metricsを記録し、
  gzipはdecompressed content SHAを主証拠にする。
- deterministic anchor:
  submissionを生成しないためfalse。train-side数値再現性はlogical SHAで監査する。
- package:
  実装後push前にloose / bootstrap config、Notebook body、input asset SHAを照合する。

## リスク

- absolute datum:
  rate履歴を除いても新しいabsolute TVT anchorは増えず、translation-gauge lockは残り得る。
- dz-only misspecification:
  真の非ゼロ形成傾斜を0と置くため、rate-model誤差の代わりにformation-slope誤差を
  累積する可能性がある。
- memoryless jitter / multiplicity:
  rate continuityを失うため、GR alias区間で行ごとにrate responsibilityが変動し、
  多数の遷移経路がwrong position massを増幅する可能性がある。
- prior choice:
  zero-centered stationary重みはprefix履歴を除く一意の事前登録選択であり、
  uniformやprefix-centeredとのgridを同じOOF上で探索しない。
- sample bias:
  fixed32はpersistent-error選択sampleなのでStage 0 scoreをCVと呼ばない。
- CV/LB不一致:
  exp226はOOF `9.427110`に対しPublic LB `9.837`、exp355はpooled改善と
  hidden-like / tail悪化が共存したため、fullではscope / tail gateを必須とする。
- runtime:
  position-only stateは軽量化余地があるが41 transition mixtureを含む。
  fixed32実測から全eligible variantsの合計full runtimeを投影してfail-closeする。
- rescue bias:
  FAIL後のrate weight / count / span、noise、GR sigma、position grid、
  bucket gate、blend weight、selector救済は禁止する。
