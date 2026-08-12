# 設計

## 仮説

exp440は「posteriorが曖昧」という広い条件で653,589行をactiveにし、
predictive priorがすでにwrong basinにある行でもcurrent emissionを無効化したため、
full OOFでparent比`+1.053776 ft`悪化した。

本実験は曖昧さを使わない。過去と未来のHMM messageが同じTVT近傍を支持し、
raw GRの現在点だけが前後から孤立し、current emissionだけが予測を動かす場合を
「isolated GR shock」と定義する。この場合だけcurrent emissionを除いた
row-local posterior meanへ置換すれば、exp440の広い誤保持と後続伝播を避けながら、
単発の観測外れ値に対して前の予測を維持できる。

## アプローチ

親exp209のunchanged exact HMMから、row `t`について次を得る。

```text
predictive_t = p(x_t | y_<t)
provisional_t = p(x_t | y_<=t)
beta_t       = p(y_>t | x_t)
loo_t        = normalize(predictive_t * beta_t)
parent_t     = normalize(provisional_t * beta_t)
```

`loo_t`はcurrent observation `y_t`だけを除いたposteriorである。
trigger rowの出力だけ`mean(parent_t) -> mean(loo_t)`へ置換する。
親のforward filtered stateは書き換えず、`t+1`以降は保存済みexp209 predictionを
そのまま維持する。したがってexp440のようにholdが後続のforward/backward pathへ
伝播しない。

```text
candidate_prediction_t =
    mean(loo_t)                 if isolated_shock_trigger_t
    saved_exp209_prediction_t   otherwise
```

## 固定trigger

### 1. Raw-GR単発shock

current rowを除くraw GRの`[t-5, ..., t-1, t+1, ..., t+5]`をneighborsとする。

- current raw GRがfinite。
- 左右それぞれfinite neighborが3点以上。
- `center = median(neighbors)`。
- `scale = max(1.4826 * MAD(neighbors), 1.0)`。
- `abs(raw_gr_t - center) / scale >= 4.5`。
- 左medianと右medianの差が`2.0 * scale`以下。
- suffix先頭/末尾5行は非active。
- raw-shock row同士が`±2`行以内にあるclusterは全行非active。

window 11は既存のrolling-median observation監査で使われた固定幅、
4.5は通常の3-sigmaより保守的な高精度側Hampel cutoffとして事前固定する。
exp482内でwindow、cutoff、MAD floor、side consistencyを探索しない。

### 2. Past/future agreement

- `abs(mean(loo_t) - mean(predictive_t)) <= 1.05 ft`。
- `max(std(loo_t), std(predictive_t)) <= 6.0 ft`。

`1.05 ft`は親position grid 3 cells（`3 * 0.35 ft`）、
`6.0 ft`は既存mode-separation contractと同じ物理幅である。
future evidenceをoutput値へ追加するのではなく、current emissionを除いても
past/futureが同じ近傍を支持することの確認に使う。

### 3. Current-emission conflict

- `abs(mean(provisional_t) - mean(predictive_t)) >= 1.05 ft`。
- `abs(saved_exp209_prediction_t - mean(loo_t)) >= 0.35 ft`。

current emissionがpredictive meanを3 grid cells以上動かし、最終parent outputも
leave-one-out outputと1 grid cell以上異なる場合だけ、意味のある介入とみなす。

最終triggerは上記3群のANDである。truth、error、fold、hidden-like role、
persistent episode、exp440 active flagは含めない。

## 実験範囲

- 対象実験: `exp482_isolated_gr_shock_prior_hold`
- Route: `pf_beam`
- 科学的親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 根拠:
  - `exp408_hmm_message_rate_basin_audit`: current emissionによる新規wrong反転は
    `9 / 807,710 rows`で、広いemission holdはroot causeを外しやすい。
  - `exp440_ambiguity_gated_predictive_prior_hmm`: full OOFで
    `12.992063`、parent `11.938287`、positive fold `1/5`、全安全scope FAIL。
  - `exp358_exp209_missing_distance_emission_downweight`: missingnessだけの
    downweightはoverall`+0.074283 ft`悪化、0/5 folds。
  - `exp363_sticky_gr_reliability_exact_hmm`: weak mass`0.589441`で広すぎた。
  - `exp389_exp209_huber_exact_hmm_emission`: fixed Huberはoverall
    `0.085546 ft`、5/5 folds改善したがwell-tailをFAILし、観測外れ値耐性には
    平均signalがある一方、global適用は安全でない。
- 変更する変数:
  - 固定trigger行のfinal TVT readoutだけをparent smoothed meanから
    current-observation leave-one-out meanへ置換する。
- 固定する変数:
  - exp209のstate、transition、prior、TVT/rate grid、GR前処理、
    Gaussian emission、sigma、missing処理、forward/backward演算。
  - 非trigger行とtrigger後の全行のexp209 prediction。
  - raw-shock/message threshold、manifest生成、candidate数、gate。

## 独立性

本実験はexp440のsame-OOF threshold/lambda/row-gate rescueではない。

- exp440の二峰性、ambiguity schedule、active rowsを入力にしない。
- raw-GR単発shockという独立した観測品質仮説を必須にする。
- current emissionの無効化をHMM stateへ伝播させず、row-local outputだけを変える。
- exp440 full OOF truthをthreshold選択へ使わない。
- exp482 FAIL後にwindow/cutoff/message距離/output差を同じOOFで調整しない。

## 段階設計

### Stage A0: raw-only censusとfixed64 manifest

全773 train wells / 3,783,989 suffix rowsを対象にraw-GR条件だけを計算する。
true TVT、error、fold、hidden-like role、persistent episodeを読まない。

- technical eligibility:
  - isolated raw-shock rows `>=128`。
  - isolated raw-shock support wells `>=32`。
  - zero-shock control wells `>=32`。
- support 32 wells:
  - shock count降順、suffix row数降順、well ID昇順で一意に選ぶ。
- control 32 wells:
  - zero-shock wellsからsuffix row数とraw missing fractionの標準化L1距離で
    support wellsへdeterministic greedy one-to-one matchingする。
  - tieはwell ID昇順。
- raw censusとfixed64 manifestをSHA freezeした後だけfold/roleをattachする。
- eligibility FAIL時はHMM replayへ進まずterminal closeする。

Stage A0はtarget-free prevalence確認であり、CVまたは性能証拠ではない。

### Stage A1: fixed64 mechanism readout

- scientific candidate: 1。
- unchanged exp209 internal-message HMM replays: 64 wells。
- candidate HMM state-modifying runs: 0。
- saved exp209 control prediction rerun: 0。
- LightGBM config / trained fold / booster / fitted model / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`。
- reporting folds: late join後5。

Technical gate:

- 64 wells、support32、control32、well重複0。
- parent replay meanとsaved exp209 predictionの最大差`<=1e-5 ft`。
- posterior normalization error`<=1e-6`、finite coverage`1.0`。
- final trigger rows`>=32`、active wells`>=16`、active folds`=5`。
- final trigger fraction`<=0.005`。
- zero-shock controlsのcandidate-parent差`0.0 ft`。
- truth/fold/role/error read before census/manifest/message/trigger/prediction freezeは0。
- full runtime projection`<=30,600 sec`、peak RSS`<=25 GB`。

Scientific gate:

- trigger rowでcandidate absolute errorがparentより小さい割合`>=0.60`。
- trigger-row SSE reduction`>=0.10`。
- improving folds`>=4/5`。
- support32 pooled RMSE improvement`>=0.01 ft`。
- support32 by-well delta p95`<=0.0 ft`。
- worst-well regression`<=+0.25 ft`。

全technical/scientific gate PASSと別承認がある場合だけStage 1を許可する。
一つでもFAILなら
`stage0_failed_close_without_trigger_threshold_or_output_rescue`とする。

fixed64はshock-enriched mechanism sampleであり、CVやpromotion evidenceではない。

### Stage 1: full 773-well OOF

Stage A0/A1全PASSと別のユーザー承認が前提。

- scientific candidate: 1。
- unchanged exp209 internal-message HMM replays: 773 wells。
- candidate state-modifying HMM runs: 0。
- saved exp209 control prediction rerun: 0。
- model / booster / PF / Beam / GPU: 0。
- direct RMSE improvement vs saved exp209`>=0.01 ft`。
- improving folds`>=4/5`。
- trigger-row SSE reduction`>=10%`。
- raw observed、raw missing、高missing、MD 1000+、hidden-like spatial、
  hidden-like typewell-purgedをすべてnonworse。
- by-well delta p95`<=0.0 ft`、worst-well regression`<=+0.25 ft`。

FAIL時はthreshold/window/scale/output差、soft blend、Huber/Student-t、
well/row selector、ML feature、same-OOF rescue、rerun、inference、
submissionへ進まない。

## Assumption

見えないtest wellでもunknown suffix全体のraw GRは予測時に観測可能であり、
exp209のforward-backward smootherと同様にfuture GRを使える。
この前提がhidden inference contractで成立しない場合、本実験は実装前に
fail-closeする。causal past-only版へ暗黙に変更しない。

## 実装境界

当初の承認はdesign-onlyであった。

- generic train/inference Notebook scaffoldはexperiment templateのまま保持する。
- candidate helper、Jupytext source、専用test、正規Notebook実装は作らない。
- 実装時はcompact self-contained候補を別名で作り、正規Notebookを上書きしない。
- 実装、正規Notebook採用、Kaggle package、Stage A0/A1 run、Stage 1、
  inference、submissionはそれぞれ必要な段階で明示承認を得る。

### 2026-07-30 実装承認追記

- compact self-contained train候補、fail-closed inference候補、専用test、
  Jupytext / 構文 / Ruff / strict validationまで承認済み。
- fixed64 controlの`standardized_l1`は全773-well raw censusの
  `suffix_rows` / `raw_missing_fraction`をpopulation mean / population std
  （`ddof=0`）で標準化する。
- supportの固定順にwithout-replacement greedy matchingし、distance同値時は
  well ID昇順で一意にする。
- 正規Notebook採用、Kaggle package、Stage A0/A1 run、Stage 1、
  inference、submissionは未承認のまま維持する。

## 再現性設計

- seed policy: RNGなし。well、row、position、rate、forward/backward、
  reduction、manifest matching順を固定する。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: なし。deterministic exact-HMM
  message replayとrow-local readoutだけ。
- 並列処理: well外側並列を使う場合も各wellは独立。worker数、thread数、
  shard assignment、merge orderを記録する。
- runtime: CPU-only、GPU 0、internet disabled。親exp209と同じmessage dtype、
  position→rate reduction順を使う。
- SHA:
  - raw file identity、raw-shock census、fixed64 manifest。
  - parent input、predictive/provisional/beta/loo message summary。
  - trigger schedule、candidate prediction、technical metrics。
  - truth-late fold/scope/by-well metrics。
  - gzipはdecompressed content SHAを主証拠にする。
- freeze順:
  raw census → manifest → message → trigger → candidate prediction →
  SHA readbackを完了してからtruth/fold/role/errorを結合する。
- model/submission SHA: model、booster、submissionを作らないため非該当。
- deterministic anchor: 初回成功runだけでは呼ばない。独立rerunでmanifest、
  trigger、predictionのlogical SHAが一致した場合だけ再判定する。
- Kaggle package: 将来packageする場合、正configとbootstrap configの
  stage、threshold、input source、CPU/internet、kernel slugを照合する。

## リスク

- 科学リスク:
  - exp408ではcurrent emissionの大きなmode反転が極小で、support不足の可能性がある。
  - 単発GR変化がsensor artifactではなく薄い実地層の場合、正しい観測を捨てる。
  - past/futureが同じwrong basinで一致し、current observationだけが正しい場合がある。
- 選択バイアス:
  - fixed64はshock-enrichedでCVではない。Stage 1 full OOF PASSまで昇格しない。
- リークリスク:
  - raw census/manifest/triggerへtrue TVT、error、fold、role、episodeを入れない。
  - full OOFの結果を見てthresholdやmatchingを選び直さない。
- CV/LB不一致:
  - exp209/440はtrain-side HMM証拠で、exp482にはPublic LB根拠がない。
  - 現行ML Public-LB基準exp413 `7.201`を置き換える候補として扱わない。
- runtime/memory:
  - full message replayはexp209相当で高コスト。A0とfixed64投影で先に止める。
  - full joint tensorはwell処理後に破棄し、必要なrow summaryだけ保存する。
- 再現性:
  - float messageの加算順差、raw row order、manifest tie-breakを固定する。
  - 初回runをdeterministic anchorと呼ばない。
