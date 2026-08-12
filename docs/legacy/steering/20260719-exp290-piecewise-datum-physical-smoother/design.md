# 設計

## 仮説

exp226 の OOF RMSE は `9.4271095966` だが、保存済み by-well 誤差に対する診断上の定数 bias oracle は
`5.7775908563`、補正量を `±15 ft` に制限しても `6.3880769132` である。この値は達成可能スコアではなく、
誤差の大きな部分が曲線形状ではなく persistent な datum 成分へ集中していることだけを示す。

一方で、exp169/201/285 は prefix、Type Well group、近隣 well の bias 符号をそのまま suffix へ移す方法が
不安定であることを示した。exp280 は exp226 座標系の固定 shift に raw-GR likelihood の弱い識別力がある
ことを示したが、top-1 accuracy は `0.189547` に留まる。exp281 の always-on residual-offset HMM は MAE、
within-5、復帰率を改善しながら RMSE `9.827420`、worst-well `+30.961675 ft` と悪化した。

したがって、補正値を一回決めて足すのではなく、次の三つを同じ確率モデルで同時に扱う。

1. known prefix から得る well 固有 datum は、pseudo-cut 再現性に応じて zero へ縮約する。
2. suffix の補正は自由 random walk ではなく、長い duration を持つ bounded piecewise-constant state とする。
3. Type Well / 近隣 well は補正符号を決めず、duration、scale、likelihood temperature の不確実性だけを共有する。

この構造なら、exp281 が改善した中心誤差を残しつつ、persistent wrong-offset を生む自由度だけを閉じられる、
という反証可能な仮説を検証する。

## 単一モデル契約

exp226 と同じ group-safe 手順で得る固定幾何場を `g_w(t)`、補正 datum を `delta_w(t)` とする。

```text
TVT_w(t) = g_w(t) + delta_w(t)
delta_w(t) in {-15.0, -14.5, ..., +14.5, +15.0} ft
```

`g_w(t)` は別モデルの予測を blend する anchor ではなく、同じ物理モデルの既知 geometry term である。
学習・推論時とも同じ fold-safe/raw-test-compatible 関数で再生成し、保存済み exp226 は parity と比較にだけ使う。

### 状態と遷移

- latent state: absolute correction `delta_t` と segment duration `d_t`。
- offset grid: `[-15,+15] ft`、`0.5 ft` step、61 states。
- transition checkpoint: MD順の64 rowごと。checkpoint間は同じ posterior correction をrowへ線形補間する。
- minimum segment duration: 256 rows。duration 未満の reset は確率0。
- duration representation: 64-row checkpointを4つ通過するまでreset不能なlocked phaseと、その後のeligible geometric phaseの5 phase。expanded stateは`61 x 5 = 305`とし、全duration値を展開しない。
- stay transition: 同じ absolute state を維持する。
- reset transition: target-free event gate が開いた checkpoint だけで、別の absolute stateへ移る。jump priorは zero-centered Laplace、scale `4 ft`。
- base reset hazard: row換算 `1/2048`。event gate による hazard は上限 `1/64` とし、必ず確率として正規化する。
- non-cumulative constraint: reset後のstateも常に絶対grid `[-15,+15]` 内に置く。jumpを積み上げて範囲外へ歩かせない。

event evidence は true TVT/error を使わず、`128/256/512 rows` の三つのGR窓について、固定shift likelihoodの
符号一致、posterior entropy低下、局所geometry curvatureをrobust rank化し、等重み平均する。三窓中2窓以上の
符号が一致しない場合はgateを閉じる。thresholdはofficial suffixで選ばず、outer-train pseudo-cutだけで一度固定する。

### 観測モデル

known prefix では基準残差を直接観測する。

```text
r_w(t) = TVT_input_w(t) - g_w(t)
r_w(t) ~ StudentT(df=4, location=delta_w(t), scale=sigma_prefix,w)
```

suffix では対応 Type Well group のGR template `T_g(v)`を候補TVT座標で読む。

```text
GR_w(t) ~ StudentT(
    df=4,
    location=a_w * T_group(g_w(t) + delta_w(t)) + b_w,
    scale=sigma_gr,w
)
```

`a_w/b_w/sigma_gr,w` は known prefix の Huber affine fit と MAD だけで推定し、
`sigma_gr,w` は `[10,60]` にclipする。raw pointwise likelihoodの過信を防ぐため、64-row blockごとに
log-likelihoodを平均し、pseudo-cut reliability `rho_w` を温度として掛ける。

### well / Type Well / 近隣 well の階層化

- well固有: prefix residual の robust location、GR affine calibration、prefix noise、pseudo-cut reliability `rho_w`。
- pseudo-cut: last-known rowから `512/256/128 rows` 戻した3点。各cut直後の固定128-row validation windowだけを未知として同じsmootherを解く。各windowの予測時は時間的に前のpseudo-cut結果だけを使い、評価window自身と未来cutのheld-known TVTをreliabilityへ入れない。
- reliability: 各validation windowでfixed zero-datum exp226基準に対するNLL excessをfreezeする。Stage 0の各windowは過去windowだけのmedian excess、final suffixは3 windowすべてのmedian excessを `rho_w=clip(exp(-excess),0.10,1.00)` に写し、prefix datum事前の強さとGR likelihood温度の両方へ使う。
- Type Well group: outer-train wellのmasked-prefix backtestから `sigma_prefix`、reset hazard、jump scale、GR noiseのrobust log-scale priorを作る。group well数8未満ではpooled priorへ縮約する。
- spatial neighbor: outer-trainのXY標準化空間でdeterministic k=16、self-well除外、distance/well/rowでstable tie-breakする。neighbor情報は上記log-scale priorの分散だけを狭め、datum meanやjump signには入れない。
- combine: pooled -> Type Well -> local -> current-well prefix の順にprecision-weighted empirical Bayesで一つのhyperposteriorを作る。official suffix targetで重みを変えない。

Type Well groupやneighborが欠けても予測を別方式へfallbackせず、同じモデルのpooled priorへ分散を広げる。

### Stage 0 実装時の固定化

各 pseudo-cut の exp226 geometry は、その cut の既知 `TVT_input` を anchor にして未来128 rowへ再生成する。
したがって Stage 0 の initial datum location は厳密に0とし、prefix/Type Well/neighborから補正符号を移さない。
outer-train masked-prefix backtestのheld-known残差は、base geometry content SHAをfreezeした後にだけ読み、
`prefix_noise`、`jump_scale`、`reset_hazard`、`GR noise` のlog-scale priorへ集約する。exact Type Well file SHA groupは
scale locationだけ、outer-train XY k=16 neighborはlog-scale varianceだけを更新する。

event thresholdはouter-train pseudo-cutのtarget-free `event_evidence` 75 percentileをfoldごとに一度freezeする。
128-row Stage 0 validation windowは64-row checkpoint 2個で、minimum duration 256 rowsのlocked phase内にある。
そのためStage 0でreset pathは発生せず、ここで反証する対象はGR evidenceがbounded constant datum posterior meanを
識別できるかである。duration/reset性能はStage 0通過後に別承認されるStage 1で初めて評価する。

### 推論と出力

61 offset statesとdurationをlog-space exact forward-backwardで解く。全suffix GRを使うoffline smootherだが、
true suffix TVTは使わない。canonical predictionは各rowのposterior meanだけとする。

```text
TVT_hat_w(t) = g_w(t) + E[delta_w(t) | known prefix, GR, trajectory, hyperprior]
```

posterior mode、Viterbi、top-k state、no-reset pathは監査用でもcandidate predictionとして保存しない。
entropy、reset probability、effective state countは安全性readoutとして保存できるが、同一OOFでhard gatingには使わない。

## exp289との境界

- exp289: outer-train formationから全hidden-like wellを一括で解く、fault cut付き2D geological-potential MAP surface。
- exp290: exp226 group-safe geometryを固定し、各wellのMD方向にGR evidenceでbounded datum stateを平滑化するhierarchical semi-Markov model。
- exp289は空間topology仮説、exp290はwell内persistent datum仮説であり、相互のpredictionを入力、blend、fallbackに使わない。
- 両方がdirect OOF guardを通過しても、このexp内でensembleやwell selectorへ進まない。

## 段階実験

### Stage 0: known-prefix pseudo-tail audit

1. exp226の固定fold identityを使い、outer-valid wellのformation 6列とofficial suffix true TVTをmodel tableからdropする。
2. known prefix内の`512/256/128` pseudo-cutごとに直後128-rowだけをmaskする。
3. outer-train hyperprior、cut以前のprefix、それより前にfreeze済みのpseudo-cut reliabilityだけで同じsemi-Markov smootherを解く。
4. truthを戻す前にprediction、entropy、reset probability、`rho_w`、state manifestをfreezeしcontent SHAを取る。
5. held-known TVTをjoinしてRMSE、correction sign、fold、well tailを評価する。
6. 全guard通過時だけStage 1 direct OOF実装の承認を求める。不通過ならparameter rescueなしでcloseする。

Stage 0はofficial suffixのoracle candidate品質を測るものではなく、raw testでも利用できるknown-prefix backtestが
well固有のdatum/GR reliabilityを識別できるかを先に反証する工程である。

### Stage 1: full suffix direct OOF

Stage 0と同一のstate grid、transition、likelihood、hyperprior式を凍結し、outer-valid evaluation suffixを一つの
posterior meanで予測する。比較は保存済みexp226 OOFだけでcontrolを再生成しない。overall/fold/distance、
hidden-like spatial/typewell-purged、by-well、well bias、p90/p95/p99/worst、persistent `|error|>=10 ft`
episode、correction magnitude、entropy/reset bucketsを報告する。

raw-test shadow/inference guardを通過しても自動で実装せず、結果、runtime、SHAを提示して別承認を求める。

## 実験範囲

- 対象実験: `exp290_piecewise_datum_physical_smoother`
- Route: `pf_beam`
- 親実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`。基準geometry contractだけを継承する。
- sibling comparison: `exp289_fault_aware_transductive_geological_potential`。
- negative anchors: `exp169`、`exp201`、`exp281`、`exp285`。positive diagnostic: `exp280`。
- 変更する変数: deterministic geometry-only TVTから、known-prefix-calibrated bounded piecewise datumを持つ一つのhierarchical posteriorへmodel classを変更する。
- 固定する変数: exp226 group-safe geometry、official 5-fold well split、evaluation mask、RMSE、raw coordinate/known-prefix contract、Type Well table、correction bound。
- 今回作成するもの: Stage 0 compact self-contained train source/notebook、専用tests、fail-closed inference source/notebook、更新済みconfig/README/SESSION_NOTES/result/metrics。
- 今回作成しないもの: Kaggle package/push/run、Stage 1 direct OOF、raw-test inference、submission。

## oracle禁止契約

- per-well constant bias oracle `5.7775908563` と cap `±15 ft` oracle `6.3880769132` は誤差分解の根拠に限り、モデル下限・promotion基準・parameter選択に使わない。
- row/segment/block/well best-of-N、truth-nearest shift、oracle path、oracle prediction artifactを作らない。
- true suffix TVT/error、exp226 official-suffix errorをstate、event gate、hyperprior、rho、temperature、stop conditionに使わない。
- pseudo-tailとofficial suffixのprediction/content SHAをfreezeした後にだけ対応truthをjoinする。
- Stage 0/1のguard不通過後に同じOOFでgrid、clip、group、neighbor、likelihoodを調整しない。

## 再現性設計

- seed policy: canonical exact forward-backwardはRNGなし。fold、well、row、Type Well、neighbor、stateをstable辞書順で処理する。
- stochastic処理の有無: なし。particle sampling、seed bagging、GPU booster、random pseudo-cutを使わない。
- PF/Beam / likelihood-PF / seed baggingの有無: すべてなし。route名はrepository上のphysical sequential solver分類として`pf_beam`を使う。
- 並列処理と乱数の関係: canonical runはsingle process。将来parallel化してもwell出力をstable sortし、single-process parity確認前はcanonicalにしない。
- CPU/GPU runtimeとdeterministic flags: CPU float64、GPU/internet off、BLAS/thread数固定。logsumexp library/version、dtype、underflow countをmanifestへ記録する。
- train cache / test regeneration SHA: input identity、fold map、base geometry、Type Well mapping、neighbor table、pseudo-cut table、hyperprior table、event evidence、state grid、posterior summaryのschema/content SHAをfold別に記録する。
- model manifest / prediction / submission SHA: transition matrix contract、likelihood config、hyperprior config、row predictionを別々にSHA化する。Stage 0ではofficial OOF/submissionなしを明記する。
- gzip: raw gzip SHAとdecompressed logical content SHAを分け、logical content SHAを主証拠にする。
- Kaggle package bootstrap: prepare後にembedded `config.yaml`、decoder source、project config、CPU/GPU/internet metadataのSHA parityを確認する。今回はprepareしない。
- deterministic anchor: rerun parityとraw-test regeneration parityが確認されるまでfalse。

## リスク

- リークリスク: outer-valid formation、official suffix true TVT/error、pseudo-cut後のheld-known TVTをfreeze前に残すと直接leakする。列drop、mask audit、truth-after-freezeをhard guardにする。
- CV/LB不一致リスク: train known-prefix長、Type Well groupサイズ、spatial密度がtest約200 wellsと異なる。hidden-like spatial/typewell-purged、prefix-length bucket、200-well shadow runtimeを必須にする。
- モデルリスク: exp281と同様、弱いGR evidenceが誤ったoffsetを長時間維持する可能性がある。absolute ±15 bound、minimum duration、multi-window gate、posterior meanでtailを制限する。
- 過縮約リスク: prefix reliabilityをzeroへ縮めすぎるとexp226と同じになる。pseudo-tail correction signとRMSEをStage 0で先に確認する。
- group biasリスク: Type Well/neighbor mean biasはexp201で方向が混在した。group情報をscale/hazard/noiseに限定し、mean/signを禁止する。
- 状態離散化リスク: 0.5 ft gridと64-row checkpointが細かな変化を落とす。初回run後のgrid救済は禁止し、必要なら別仮説として新expにする。
- ランタイム/メモリリスク: 61 states×duration×約378万rowのexact smoothingは重い。Stage 0でwall time、peak RSS、state countを測り、Kaggle 9時間へ1.5時間以上の余裕を要求する。
- 再現性リスク: neighbor tie、row ordering、floating reduction、logsumexp実装差で変動しうる。stable sort、single process、float64、thread固定、content SHAで管理する。
