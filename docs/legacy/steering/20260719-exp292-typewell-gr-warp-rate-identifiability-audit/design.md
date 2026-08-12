# 設計

## 仮説

exp288の全well可視化で見えたように、horizontal GRの周波数はType Well GRをcandidate TVT pathで
MD軸へwarpした形に依存する。exp268の5本は初期 `dTVT/dMD` だけが異なるため、Type Well GRの
振幅差をknown prefixだけで補正した後も、forward GRの波形と微分がhorizontal GRへ最も整合する
candidateはtruth-best rate candidateである確率がrandomized controlより高いはずである。

## アプローチ

### 1. 入力と前提条件

exp268 aggregateを実行前hard prerequisiteとする。exp292は次を読み、well / row / idでstrict alignする。

1. exp268 shard 0/1の `hmm_ir_w32 / w64 / w128 / w256` path。
2. exp209保存済み `hmm_ir_tail30` control。
3. exp268 aggregate manifest / summaryのcoverage、candidate duplicate/diversity、入力・出力SHA。
4. raw train horizontal `MD, GR, TVT_input` と対応するType Well `TVT, GR`。
5. exp115 hidden-like spatial / typewell-purged assignment。

exp268 aggregateが773 wells / 3,783,989 rows、5候補のstrict identity、decompressed SHAを確定して
いなければfail-closedで停止する。exp292ではHMM、PF、candidate pathを再生成しない。

### 2. Type Well前処理とprefix calibration

Type Wellはfinite `TVT, GR`だけを使い、TVT昇順へstable sortし、重複TVTはGR medianへ集約する。
範囲外は外挿しない。known prefixの末尾最大512行から、次をfitする。

```text
x_t = G_typewell(TVT_input_t)
y_t = GR_horizontal_t
y_t ~= a * x_t + b
```

robust affineはexp211契約を固定参照し、90% residual trimを2 iteration、minimum 40 pairs、
Type Well GR std `>= 5.0`、slope `[0.25, 4.0]`、prefix RMSE `<= 60`とする。fit後residualの
`1.4826 * MAD`をGaussian scaleとし、`sigma = clip(scale, 10, 60)`へ固定する。
derivative scaleはprefix residualのfirst differenceから同じMAD式で作り、`clip=[1, 30]`とする。

calibrationが不成立ならraw affineへのfallbackや別fitを試さず、そのwellの全GR scoreをinvalidにして
safe tail30へ戻す。これはexp170/211のaffine direct replacementがnegativeだったため、calibration
自体を採用候補にせず、識別性が測れるwellだけを事前契約で限定するためである。

### 3. 固定horizonとcandidate forward GR

unknown suffix先頭から `H in {128, 256, 512}` 行を取り、`H256`をprimaryにする。short suffixでは
`H_eff=min(H, suffix_rows)`とし、`H_eff < 64`はinvalidにする。candidate `c`ごとに次を作る。

```text
g_ref[c,t] = a * G_typewell(T_candidate[c,t]) + b
d_ref[c,t] = a * G'_typewell(T_mid[c,t]) * delta(T_candidate[c,t])
d_obs[t] = delta(GR_horizontal[t])
```

`G'_typewell`はdeduplicated Type Well curveのfinite differenceをTVT midpointへ線形補間する。
score比較は5候補すべてでfiniteなcommon paired rowsだけを使う。common coverageは
`max(32, ceil(0.5 * H_eff))` rows以上、各candidateのcalibrated forward GR stdは `>= 5.0`、
`median(abs(d_ref)) >= 0.25 GR/row`を必要とする。1つでも満たさないhorizonはGR selectionを
invalidにし、top1 policyはsafe tail30を選ぶ。

### 4. 固定score

各candidate / horizonで次の3成分を作る。

```text
gaussian[c] = mean(-0.5 * min(((g_obs - g_ref[c]) / sigma)^2, 600))
ncc[c] = corr(g_obs, g_ref[c])
derivative[c] = -median(abs(d_obs - d_ref[c])) / derivative_sigma
```

成分ごとに5候補間で `z=(value-median)/(1.4826*MAD+1e-6)` を計算し `[-5,5]`へclipする。
MADが`1e-9`未満の成分は全candidate 0とする。compositeは3成分の単純平均で、weightは
`1/3, 1/3, 1/3`に固定する。同点はcandidate順
`tail30, w32, w64, w128, w256`で解き、safe側を優先する。

score、eligibility、top1 candidate、calibration metadata、raw input SHAをtarget-free tableへ保存し、
schema/content SHAを凍結するまでtrue TVTを読み込まない。

### 5. negative controlとgeometry-only control

- real: raw unknown-suffix horizontal GRをそのまま使う。
- shuffled: wellのunknown-suffix GR全体を、
  `SHA256(experiment, seed=42, well)`由来local RNGで1回circular rotateする。shiftは
  `[max(32, ceil(N/4)), N-max(32, ceil(N/4))]`から選び、H128/H256/H512は同じrotated seriesの
  prefixを使う。範囲が空ならshuffle scopeはinvalidとする。
- geometry-only safe: exp292のunknown-suffix GR scoreを使わず、常に `hmm_ir_tail30`を選ぶ。

real/shuffledは同じcalibration、coverage、component、composite、tie関数を通す。shuffleは
observed GRだけをrotateし、candidate、eligibility threshold、truth label、fold、score分布の集計契約を変えない。

### 6. truth joinと評価

target-free tableのSHA freeze後に別関数でtrain true TVTをjoinする。candidate-best labelは各
well/horizonのcandidate RMSE最小とし、RMSE差 `<=1e-9`のtieは全てpositiveにする。
primary H256ではcandidate-long recordsに対するbinary ROC AUCをreal / shuffledで計算する。

fixed top1はprimary H256 composite最大candidateをwell全体のunknown suffixへ適用してRMSEを読む。
これは診断用であり、selected row predictionは保存しない。比較対象は常時tail30 safeである。
foldはwell単位のcanonical 5-fold GroupKFold assignmentを一度生成してmanifest/SHAを固定し、
fold別AUC/RMSEの安定性だけに使う。foldでscoreやthresholdをfitしない。

### 7. 成功条件と停止条件

次を全て満たした場合だけscientific PASSとする。

1. technical: exp268入力契約、score-before-truth、5 folds、finite/common coverage、SHA guardが全PASS。
2. coverage: primary H256 eligible well / row coverageがともに90%以上。
3. identifiability: H256 pooled `AUC(real)-AUC(shuffled) >= 0.02`、4/5 foldsでliftが正。
4. selection: real H256 top1のfull-suffix pooled RMSEがsafe tail30より `>=0.10 ft`改善し、
   4/5 foldsで改善。
5. subgroup: `md_since >= 1000`、hidden-like spatial、hidden-like typewell-purgedでsafeから非悪化。

1つでもFAILならfrequency-warp rate branchをcloseする。同じtruthを見てwindow、horizon、affine、
coverage、variance/gradient threshold、component weight、shuffle、tie、guardを変更するrescueは行わない。
PASSしてもtop1 replacementやinferenceへ直行せず、safe baseを絶対保持するcontinuous rate-mode
add-only candidate bankを別follow-upとして設計する。

## 実験範囲

- 対象実験: `exp292_typewell_gr_warp_rate_identifiability_audit`
- Route: `pf_beam`
- 科学的親実験: `exp268_multi_scale_initial_rate_candidates`
- 参照: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`、
  `exp288_known_tvt_typewell_horizontal_gr_visualization`、
  `exp170_heel_calibrated_shift_scan_pfbeam_audit`、
  `exp211_affine_calibrated_gr_observation_pfbeam`、
  `exp132_multi_scale_gr_observation_likelihood`、
  `exp115_hidden_like_spatial_holdout_from_ppt`
- 変更する変数: 5本の固定rate candidateに対するType Well forward-GR frequency/shape scoreだけ。
- 固定する変数: candidate path、initial-rate window、HMM grid/grammar/emission、horizon、
  calibration、eligibility、3成分、等重み、shuffle、tie、fold、guard。
- 実行量: audit variant 1、LightGBM config 0、trained fold 0、booster 0、HMM/PF regeneration 0。
- 除外: candidate追加/平均/softmax、HMM/PF再生成、ML selector、raw-test inference、submission。

## 将来の生成物

- scientific contract JSON
- exp268/input SHA manifest
- well-level calibration and eligibility manifest
- target-free candidate/horizon score table
- target-free real/shuffled top1 selection table
- truth-attached candidate-best AUC readout
- overall/fold/horizon/subgroup/by-well/worst-well metrics
- content/schema SHA manifest
- summary JSON

model、selected row prediction、submissionは生成しない。

## 再現性設計

- seed policy: real scoreはRNGなし。circular-shuffleだけを
  `stable_sha256(experiment, seed, well)`由来のlocal `np.random.default_rng`で固定する。
- stochastic処理の有無: stable within-well circular-shuffle negative controlのみ。
- PF/Beam / likelihood-PF / seed baggingの有無: 新規実行なし。保存済みexp268/209 pathだけを読む。
- 並列処理と乱数の関係: 初回はsingle process、well/candidate/horizon固定順。global RNGを使わない。
- CPU/GPU runtime: Kaggle CPU、GPU/TPU/AMP/internet off、worker 1、model fitなし。
- train cache / test feature regenerationのSHA: exp268 shardとexp209 gzipはraw/decompressed SHAを記録し、
  decompressed content SHAをhard guardする。score/readoutはschema/content SHAを記録する。testは生成しない。
- model manifest / prediction / submission SHA: 非生成。scientific contract、fold/input manifest、
  target-free score freeze SHA、kernel versionを代替証拠とする。
- Kaggle package bootstrap: 実装・push承認後にmetadataとloose/bootstrap内config/sourceをbyte比較する。
- deterministic anchor: submission anchorではなく、固定入力に対するdeterministic diagnosticだけを主張する。

## リスク

- リークリスク: unknown-suffix true TVTをcandidate scoreへ混入する危険が最大。target-free APIで
  truth/error/oracle列をrejectし、score/selection content SHA freeze後の別stageでだけtruthをjoinする。
- GR二重利用リスク: upstream exact HMMもType Well GRを使う。本実験はcandidate生成価値ではなく、
  保存済みcandidate間の追加future-GR識別性だけを測り、geometry-onlyという語はexp292 selectionに限定する。
- affine過信リスク: exp170/211ではdirect useがnegative。fit不成立をrawへfallbackせずsafeへ戻し、
  affine parameter自体を候補選択やgridの対象にしない。
- coverageリスク: Type Well範囲外、flat GR、missing GRで評価可能wellが減る。common maskと90% coverage
  guardを固定し、低coverage結果をpositive evidenceとして扱わない。
- multiplicityリスク: 3 horizonを出すがprimaryはH256だけ。H128/H512のbestを採用判断へ使わない。
- CV/LB不一致リスク: train-side fixed-candidate auditであり、raw-test生成可能性やLB改善を直接主張しない。
- ランタイム/メモリリスク: 3.78M rows x 5 candidateをwell単位stream処理し、row x candidate x horizon
  tensorを全well同時保持しない。
- 再現性リスク: shuffleだけstable local RNGを使い、gzip metadata差はdecompressed content SHAで分離する。

## 次

2026-07-19に実装とKaggle CPU version 1を完了した。primary H256はeligible 29/773 wells、
real-minus-shuffled AUC -0.046991、top1 RMSE gain 0で事前guardをFAILした。設計どおり
`FAIL_CLOSE_NO_RESCUE_GRID`とし、frequency-warp rate branchを閉じる。救済grid、inference、
submissionへ進めない。
