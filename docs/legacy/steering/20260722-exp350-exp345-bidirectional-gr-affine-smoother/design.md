# 設計

## 仮説

exp345のcausal affine scheduleはpooled RMSEを改善した一方、未来のraw GRを使わず各rowの状態を逐次確定したため、一部wellでscale/offset stateが局所的に誤ったままsuffixへ伝播し、worst tailを生んだ可能性がある。推論時に利用可能な井戸全体のraw GRで同じforward stateを固定区間平滑化すれば、exp345の平均gainを保ちながらper-well tailを縮小できると仮定する。

## 実験範囲

- 対象実験: `exp350_exp345_bidirectional_gr_affine_smoother`
- Route: `pf_beam`
- 科学的親: terminal closedの`exp345_exp209_time_varying_gr_affine_calibration_hmm`
- root parent: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更する変数: causal affine stateを、そのforward recordに対する固定区間extended RTS stateへ置換する。
- 固定する変数: exp345のmask、base HMM mean/std、state、初期fit、Q、observation、slope bound、missing policyと、exp209のHMM全parameter・decoder。
- 現在作るもの: design-only scaffold、steering、backlog、記録。
- 現在作らないもの: 科学実装、Jupytext source、正規Notebook採用、Kaggle package/push/run、Stage 0/1 prediction、inference、submission。

## 推論契約

このコンペでは各horizontal wellの`MD/Z/GR`が全rowで与えられるため、score suffixの未来raw GRは利用可能な入力である。一方、score suffixの`TVT_input`とtrue TVTは未知であり、使用禁止とする。

したがって本candidateは次のoffline full-well contractを持つ。

```text
許可: full-well MD, Z, raw GR, Type Well table, visible-prefix TVT_input
禁止: masked/future TVT_input, true TVT, error, formation, fold/hidden role before freeze
```

streaming predictionやrow到着時点のonline predictionには使用できない。inferenceを将来実装する場合も、well全体を受け取ってから一括生成する。

## Stage 0データフロー

1. exp345 version 2のpromotion gate、prediction、causal schedule、process noise、mask manifest、paired metricsを取得し、事前SHAを照合する。
2. raw horizontalのlast-640 `TVT_input`をmaskする。raw GRは全row保持する。
3. exp345保存`parent_hmm_tvt/std`をbase path、保存process noiseをQとして、exp345 causal EKFを再生成する。
4. regenerated forward scale/intercept/update maskを保存causal scheduleへ照合する。parity FAILならsmoother/HMM前に停止する。
5. forwardで各score rowの予測state/covarianceとposterior state/covarianceを保存する。
6. well末尾から先頭へ固定区間extended RTS backward passを1回行う。
7. smoothed schedule、covariance audit、入力manifestをfreezeし、raw/decompressed logical SHAを取得する。
8. smoothed scheduleをGR observation centerに渡し、exp209 exact-HMMを1回だけ実行する。
9. predictionとruntime auditをfreezeしてSHAを取得する。
10. freeze再検証後だけtrue TVT、fold、hidden-like rolesをjoinし、saved parent、saved causal、新smootherを比較する。

## Forward filter

stateはexp345と同じである。

```text
x_t = [b_t, log(a_t)]
x_t^- = x_(t-1)
P_t^- = P_(t-1) + Q_w
GR_t ~ N(exp(log(a_t)) * GR_typewell(base_t) + b_t, R_t)
```

- visible prefixのrobust affine fit、minimum pair 40、typewell GR std 5、slope bound`[0.25,4.0]`、prefix RMSE上限60、trim 0.90、2 iterationsを固定する。
- `Q_w`はexp345のouter-train empirical-Bayes tableをSHA固定して再利用する。
- raw GR欠損rowはmeasurement updateをskipし、state/covarianceをpropagateする。
- Joseph covariance updateを維持する。
- regenerated forward outputがexp345保存scheduleへ`1e-10`で一致しなければcandidateを生成しない。

## Bidirectional extended RTS smoother

transitionはidentityなので、score suffix内で次を末尾から先頭へ適用する。

```text
J_t = P_t^f @ pinv(P_(t+1)^-, rcond=1e-12)
x_t^s = x_t^f + J_t @ (x_(t+1)^s - x_(t+1)^-)
P_t^s = P_t^f + J_t @ (P_(t+1)^s - P_(t+1)^-) @ J_t.T
```

- terminalは`x_T^s=x_T^f`、`P_T^s=P_T^f`とする。
- covarianceは`(P+P.T)/2`で対称化する。
- 最小固有値が`-1e-8`未満ならtechnical FAILとする。`[-1e-8,0)`だけを数値誤差として`1e-12` floorへ射影する。
- output scaleは`clip(exp(log(a_t^s)),0.25,4.0)`、interceptは`b_t^s`とする。clip率`>1%`はFAIL。
- measurement欠損rowもbackward informationは伝播する。
- smoother出力をforwardへ戻さず、HMM predictionをbase pathへ戻さない。forward/backward/HMMは各1回で終了する。

## Exact-HMM candidate

GR観測中心だけを次へ置換する。

```text
mu_GR(state, t) = a_t^smooth * GR_typewell(TVT_state) + b_t^smooth
```

exp209のzero-fill population std、missing weight 1、Gaussian emission、41 rate states、`sig_r=0.002`、`sig_p=0.02`、position floor、momentum、prior、band、posterior meanは変更しない。新規candidateは`one_pass_bidirectional_rts_affine_schedule_on_exp209`の1本だけである。

## Controlsと計算量

Stage 0 controlはexp345保存artifactを使う。

- masked exp209 parent: `parent_hmm_tvt/std`
- exp345 causal: `one_pass_causal_affine_schedule_on_exp209_hmm_tvt/std`
- new candidate: bidirectional scheduleによる1 HMM rerun

予定量:

- scientific variants: 1
- forward filter / smoother: 773 / 773 wells
- new exact-HMM: 773 well-runs
- parent HMM rerun / causal HMM rerun: 0 / 0
- LightGBM configs / trained folds / boosters: 0 / 0 / 0
- PF / Beam / GPU: 0 / 0 / 0

exp345は同一HMMのparent+variant 1,546 runsを1.3531時間で完了しているため、別microbenchmarkを設けない。Stage 0実測は8.5時間をhard gateとする。

## 評価とpromotion gate

### Technical AND gate

- 全input SHA、forward schedule parity、saved control metric parity。
- 494,720 rows / 773 wells / 773 HMM runs、全finite。
- terminal state parity、covariance PSD・contraction、scale clip率`<=1%`。
- runtime`<=8.5 h`。

### Scientific AND gate

- pooled gain: masked parent比`>=0.05 ft`、exp345 causal比`>=0.02 ft`。
- folds: 両baseline比で各`>=4/5`改善。
- hidden-like: spatial/typewell-purged 2 scopeを必ず作り、両baseline比で各非悪化。
- parent比well delta: median`<=0`、p95`<=0`、worst`<=+0.25 ft`。
- boundary jump p95`<=3 sigma`。

by-well p95は「wellごとの`candidate_rmse-parent_rmse`の95 percentile」と定義し、exp345で使った「candidate RMSE p95とparent RMSE p95の差」と混同しない。

GR reconstruction NLLはsmootherが同じcurrent/future GRを見ているためin-sampleとなる。identity/causal/smoother値は診断表示してよいが、promotionには使用しない。

## Stage 1境界

Stage 0全gate PASSかつ別承認時だけ、同じ固定smootherをofficial full suffixへ適用するStage 1を同じexp350内へ追加できる。saved exp209 base cacheを使い、新candidate 773 HMM、control再実行0とする。Stage 0の結果だけでinferenceやsubmissionを実装しない。

## 2026-07-23 実行結果

固定設計どおりKaggle CPU Stage 0 version 1を実行した。technical gateは全PASSしたが、candidateはmasked parentを`0.133499 ft`改善した一方でexp345 causalより`0.036006 ft`悪化し、2/5 folds、parent比by-well p95`+1.346427 ft`、worst`+20.887374 ft`でscientific gateをFAILした。したがってStage 1は不適格で、設計済みのno-rescue条件に従ってbranchを閉じる。

## 再現性設計

- seed policy: RNGなし。fold、well、row、forward、backward順を固定する。
- stochastic処理: なし。
- PF/Beam/likelihood-PF/seed bagging: なし。routeはphysical sequential solver分類として`pf_beam`。
- CPU/GPU: CPU float64、GPU/internet off、workers 2、Numba threads 2を開始契約とする。
- 並列: well単位だけを許し、出力をwell/rowでstable sortする。canonical single/parallel parity確認前はdeterministic anchorとしない。
- SHA: input、forward state/covariance、smoothed state/covariance、schedule、numerical audit、prediction、runtime、late-readout manifestを記録する。gzipはdecompressed logical content SHAを主証拠とする。
- model/submission SHA: modelとsubmissionは存在しないため非該当。
- Kaggle bootstrap: package時にembedded config/source、kernel source、CPU/GPU/internet metadataのSHA/値を照合する。design-onlyではpackageしない。

## リスクと停止条件

- future GRが誤ったbase-HMM pathへ強く整合し、誤calibrationをwell全体へ逆伝播する可能性がある。
- offline transductive gainがあってもstreaming用途には転用できない。
- exp345のpooled gainが少数wellの大改善に依存しており、smootherで平均を保ってもtailが直らない可能性がある。
- hidden-like role欠落を繰り返さないよう、2 scopeの存在とwell数をtechnical readoutで明示する。
- gate FAIL後にQ、rcond、clip、smoother回数、blend、well gateを調整しない。新しい独立根拠と別設計がない限りfamilyを閉じる。

## 優先度

低・P3・CPUとする。exp345の平均gainは根拠だがtail failureが強く、既存P1のexp349/exp340やP2のexp346を追い越さない。ユーザーが実装を別承認した場合だけ着手する。
