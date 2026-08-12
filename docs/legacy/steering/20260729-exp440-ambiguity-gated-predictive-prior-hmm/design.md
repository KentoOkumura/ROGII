# 設計

## 仮説

GRが複数のTVT modeを同程度に支持する行では、current emissionをneutralizeし、
前row posteriorを親transitionで進めたpredictive priorを維持すると、
通常のGR更新よりwrong-basin移行とpersistent SSEを減らせる。

## アプローチ

exp209 exact HMMの各rowで、candidate側の前row filtered posteriorを親transitionで
伝播してpredictive joint distributionを得る。親exp209と同じGaussian GR emissionを
通常どおり一度適用し、rateを周辺化したprovisional filtered TVT marginalを作る。

このTVT marginalへexp236の固定二峰判定を適用する。raw GR observedかつ
二峰判定がtrueなら、そのrowのemissionをneutralizeしてcandidate filtered
distributionをpredictive distributionへ戻す。falseなら通常のprovisional
filtered distributionを採用する。

```text
predictive_t = transition(candidate_filtered_{t-1})
provisional_t = normalize(predictive_t * exp(parent_emission_t))
ambiguous_t = exp236_bimodality(provisional_t.position_marginal)

candidate_filtered_t =
    predictive_t               if raw_gr_observed and ambiguous_t
    provisional_t              otherwise
```

全forward passで決めた`ambiguous_t` scheduleを固定し、backward passでは
ambiguous rowのemission exponentを0、その他を1として同じscheduleを使う。
最終出力は親と同じsmoothed posterior TVT meanとする。

この介入は「前rowのTVT点推定を固定」するものではない。MD/Z、rate state、
position noiseを含む親transitionで進めたpredictive priorを保持する。

GMM、Student-t、Huber、連続temperature、soft blendは使わない。

## 実験範囲

- 対象実験: `exp440_ambiguity_gated_predictive_prior_hmm`
- Route: `pf_beam`
- 科学的親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 診断根拠:
  - `exp408_hmm_message_rate_basin_audit`: persistent offset SSEの59.3978%は
    current emission前のforward transition/prior hysteresis。current GRによる
    truth優勢からwrong優勢への新規反転は9/807,710 rows。
  - `exp236_exact_hmm_posterior_bimodality_audit`: 二峰row 0.9355%、
    mean-in-valley 0.1792%。判定thresholdの固定参照元。
  - `exp133_gr_bimodal_match_ambiguity_detector`: broad ambiguity flagは
    56.6857%で、単純な高誤差gateにならなかった。
  - `exp363_sticky_gr_reliability_exact_hmm`: weak reliability signalは
    pooled AUC 0.607552だがweak mass 0.589441でscientific FAIL。
- 変更する変数:
  - exp236固定判定でambiguousとなったraw-GR-observed rowだけ、
    current emission exponentを`1.0 -> 0.0`へ変更する。
- 固定する変数:
  - exp209のstate、rate support、transition、momentum、noise、grid、band、
    prior、GR interpolation/missing処理、Gaussian emission family、GR sigma、
    observation weight、forward-backward、posterior mean readout。
  - exp236の`min_peak_height=0.02`、
    `min_top2_mass=0.10`、
    `min_top2_to_top1_mass_ratio=0.25`、
    `min_peak_separation_ft=6.0`、
    `min_valley_depth=0.30`。
  - candidate 1本。soft lambda、GMM、threshold grid、fallback candidateは作らない。

## 段階設計

### Stage 0: fixed32 mechanism gate

- manifest:
  `exp411_predictive_filtered_rate_innovation_destick/assets/stage0_fixed32_manifest.csv`
- scope: persistent 16 wells + matched control 16 wells、合計156,088 suffix rows。
- 実行量:
  - scientific candidate: 1
  - candidate HMM well-runs: 32
  - saved parent control HMM rerun: 0
  - reporting folds: 5
  - model / LightGBM config / trained fold / booster / PF / Beam / GPU:
    `0 / 0 / 0 / 0 / 0 / 0 / 0`
- technical gate:
  - 32 wells / 156,088 rows / persistent16 / control16 / 5 folds。
  - finite coverage 1.0、posterior normalization error `<=1e-6`。
  - no-ambiguity synthetic pathでexp209 parentと最大差`<=1e-6 ft`。
  - ambiguous activation fraction `[0.001, 0.10]`、active wells `>=8`。
  - truth / role / fold / episode read before schedule+prediction freeze = 0。
  - Stage 1 runtime projection `<=30,600 sec`、peak RSS `<=25 GB`。
- mechanism gate:
  - ambiguous rowでpredictive-prior holdの絶対誤差がprovisional updateより
    小さい割合`>=0.55`、改善fold`>=4/5`。
  - ambiguous-row SSE reduction `>=5%`。
  - persistent episode SSE reduction `>=5%`。
  - persistent改善well `>=10/16`、改善fold `>=4/5`。
  - matched-control pooled RMSE delta `<=+0.02 ft`、
    by-well delta p95 `<=+0.25 ft`。
- いずれかFAILなら`stage0_fail_closed_without_ambiguity_lambda_threshold_or_transition_rescue`。

fixed32はerror-selected mechanism sampleでありCVやpromotion evidenceではない。

### Stage 1: full 773-well OOF

Stage 0の全technical/mechanism gate PASSと別のユーザー承認を前提とする。

2026-07-30、ユーザーはStage 0 FAIL closedを認識した上で、念のためのfull-well
確認を明示承認した。このoverrideはcandidate、threshold、lambda、gateを変更せず、
Stage 0結果をpromotion evidenceへ読み替えない。

実行結果はcandidate RMSE `12.992063`、parent exp209 `11.938287`、
positive fold `1/5`、ambiguous-row SSE reduction `-21.3117%`だった。
technical gateは全PASSしたがscientific gateはFAILし、事前契約どおり
no-rescueでterminal closeした。

- candidate 1本、773 HMM well-runs、saved parent control再実行0。
- direct RMSE gain vs saved exp209 `>=0.02 ft`。
- 改善fold `>=4/5`。
- ambiguous-row SSE reduction `>=5%`。
- raw observed / raw missing / high missing / 1000+ /
  hidden-like spatial / hidden-like typewell-purgedをすべてnonworse。
- by-well delta p95 `<=0.0 ft`、worst-well regression `<=+0.25 ft`。
- FAIL時はblend、selector、continuous gate、threshold/weight grid、
  same-OOF rescue、inference、submissionへ進まない。
- 実行はsuffix row数のdeterministic LPTによる4 CPU shardsとstrict mergeを使う。
  合計773 candidate HMM well-runsであり、各wellは1 shardにだけ属する。

## 実装境界

2026-07-29の追加依頼で、compact self-contained train/inference候補と
専用testまでを実装範囲に変更した。train候補にはexp209互換の
forward-backward、exp236のpeak/valley判定に必要な関数、fixed32
target-free freeze、truth-late readout、Stage 0 AND gateをself-containedで
持たせる。inference候補はfail-closed guardだけを実装する。

正規Notebook採用、Kaggle package、push、Stage 0 run、Stage 1、
inference、submissionは今回の範囲外であり、別の明示承認を必須とする。

## 再現性設計

- seed policy: RNGなし。well、row、position、rate、forward/backward順を固定する。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。deterministic exact HMMのみ。
- 並列処理と乱数の関係: 外側well並列を使う場合もRNGはないが、Stage 0は
  parity優先で固定worker数とwell順を記録する。
- CPU/GPU runtime と deterministic flags: CPU-only、GPU 0、internet disabled。
  HMM message dtypeと集計順を親exp209へ合わせる。
- SHA記録方針:
  - fixed32 manifest、exp209 control、exp408 episode/cause input SHA。
  - ambiguity schedule、prediction、diagnostic、metricsのlogical/content SHA。
  - gzipはdecompressed content SHAを主証拠にする。
  - truth/role/fold/episodeを読む前にschedule、prediction、diagnosticをfreezeする。
- model manifest / submission SHA: model、booster、submissionは作らないため非該当。
- deterministic anchor: 初回runだけでは呼ばない。独立rerunでscheduleとpredictionの
  logical SHAが一致し、必要な実行証拠が揃った場合だけ再判定する。
- Kaggle package bootstrap: 将来packageする場合、正のconfigとbootstrap内configの
  `selected_stage`、execution guard、CPU/internet、input source、kernel slugを照合する。

## リスク

- リークリスク:
  - ambiguity flagにtrue TVT、error、fold、hidden-like role、persistent episodeを
    使用すると介入gateがリークする。全target-free生成物のfreeze後だけlate joinする。
  - exp236 thresholdは完了済み別実験から固定し、exp440 OOFで再選択しない。
- 科学リスク:
  - exp408ではcurrent-row emissionが新規wrong-mode反転を起こす割合が極小であり、
    本介入がroot causeを外している可能性が高い。
  - predictive priorがすでにwrong basinなら、emissionを無効化すると誤りを固定する。
  - exp133/363からambiguity/reliability判定が広すぎる可能性がある。
- CV/LB 不一致リスク:
  - Stage 0 fixed32はCVではない。Stage 1 PASS前にroute anchor、inference、
    submission候補として扱わない。
  - hidden testは約200 wellsであり、well-tail gateを平均RMSEと独立に維持する。
- ランタイム/メモリリスク:
  - full exact HMMは高コストCPU。Stage 0 32 runsとfull runtime projectionを先行gateにする。
  - full posterior tensorは保存せずwell単位でschedule/summaryへ縮約する。
- 再現性リスク:
  - float32 messageの加算順差を避けるため親exp209のposition/rate累積順を固定する。
  - 初回runをdeterministic anchorと呼ばない。
