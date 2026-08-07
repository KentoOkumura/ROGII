# 設計

## アプローチ

exp189はrolling median / Savitzky-GolayをPF/Beamへ直接差し替え、PF RMSEをそれぞれ
`+6.667912 / +7.717879 ft`悪化させた。ESS上昇とresampling低下は、識別性改善ではなく
wrong modeへの安定化でも起こる。このためexp304はdecoderのRMSEを見ず、exp280で分離できた
固定vertical-shift候補に対し、平滑化がtruth-nearest候補のraw-GR log-likelihood gap/rankを
改善するかを先に測る。

exp304はscreening実験であり、rawと3種類のtarget-free denoiserだけを比較する。全scoreを
truth-freeで凍結・hashした後にtrue TVTを付与し、5 foldでrank metricsを読む。1方式だけを
predeclared gateとtie-breakで後続候補に選ぶ。合格しても同じexpでHMM/PFへ進めない。

## 実験範囲

- 対象実験: `exp304_gr_denoiser_emission_separability_readout`
- Route: `pf_beam`
- 親実験: なし（独立audit）。方法論親はexp280、emission基準はexp209、失敗参照はexp189。
- 変更する変数: horizontal/typewell GRへ適用するdenoiserだけ。
- 固定する変数: exp226 OOF geometry、exp280の13 shiftと512-row block/fold identity、
  row/scope identity、Gaussian emission式、known-prefix sigma、clip、missing policy、truth freeze順序、
  shuffled control、tie order、全HMM/PF/Beam/model設定。

## 固定入力面とscore contract

- exp226 group-safe OOFのsafe columns `well_id,row_idx,suffix_offset,fold,tvt_geop`を使う。
- shift bankはexp280と同じ
  `[-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80] ft`、順序も固定する。
- prediction suffix先頭から非重複512行blockを作り、short tailを保持する。
- raw horizontal/typewell GRはexp209/exp280と同じ補間、endpoint holdを行う。この補間済み系列を
  全denoiserの共通入力とし、original-missing maskは診断列として保存する。
- 各candidate TVTは`tvt_geop + shift`。row log-likelihoodをblock内平均し、同点はshift bank順。
- Gaussian sigmaはknown-prefix raw residual std、clip `[10,60]`。variantごとに再推定しない。
  row NLL clipは`600`、typewell extensionは`40 ft`で固定する。
- real scoreのnegative controlはexp280と同じseed 42のstable SHA256 per-well/block candidate-score
  permutationとする。全denoised series、target-free score、contract SHAを凍結してからtrue TVTを読む。

## 固定denoiser

`raw`は共通補間後の系列を変更しないcontrol。残りはobserved GRのみで決まり、truth/CVで調整しない。

1. `robust_rts`
   - stateはlevel+slope `[g,dg/du]`、座標`u`は各系列のmedian positive spacingで正規化する。
   - Gaussian constant-velocity processとStudent-t measurement（自由度`4`）のiteratively reweighted
     Rauch-Tung-Striebel smootherを使う。
   - robust scaleを`rs(v)=median(abs(v-median(v)))/0.67448975`と定義する。measurement標準偏差は
     `rs(diff(y))/sqrt(2)`、process acceleration標準偏差は正規化座標上のsecond divided
     differenceの`rs`から一度だけ決め、両方にfinite floor `1e-6`を置く。
   - `F(dt)=[[1,dt],[0,1]]`、`Q(dt)=sigma_a^2*[[dt^4/4,dt^3/2],[dt^3/2,dt^2]]`、
     `R=sigma_e^2`とする。初期meanは先頭levelと全slope中央値、初期covarianceは
     `diag(R,R+sigma_a^2)`。最大8 IRLS反復、相対mean変化`1e-6`で停止する。
     Q/R、自由度、反復数、初期値のgridは持たない。
2. `swt_db4_l3`
   - PyWavelets stationary wavelet transform、`db4`、level 3、reflection paddingを使う。
   - 各detailのnoise scaleを`MAD/0.67448975`、soft universal thresholdを
     `sigma_j*sqrt(2*log(n))`としてinverse SWT後にcropする。
   - level低下、別wavelet、rolling/Savitzky fallbackは禁止。短系列等で成立しなければtechnical FAIL。
3. `l1_trend`
   - `argmin_x 0.5||y-x||_2^2 + lambda||D2 x||_1`のsecond-order L1 trend filtering。
   - `lambda=(MAD(diff(y))/0.67448975/sqrt(2))*sqrt(2*log(n))`を系列ごとにtruth-freeで一意に決める。
   - ADMMは`rho=1`、最大500反復、absolute/relative tolerance `1e-4`。lambda/rho/orderのgridは持たない。

horizontal GRはwell内MD順、typewell GRはtypewell内TVT順に別々に処理し、well/typewell境界を跨がない。
全方式で長さ、並び、finite、missing policyを一致させる。設計値の変更が必要なら、実装前にsteeringと
予約契約を改訂し、暗黙のfallbackを入れない。

### 実装時に固定した低レベル定義

- SWTのreflection paddingは元row indexを保つため右側だけへ追加し、inverse SWT後に元長へcropする。
- sharp-edge scopeは、raw Type Well GRのabsolute gradientをexp226 `tvt_geop`位置へ補間し、block平均の
  全raw block pooled 90 percentile以上とする。denoiser別にscopeを再計算しない。
- denoised seriesはwellごとにwide CSVをdeterministic gzipへstreamし、gzip bytesではなくdecompressed CSV
  content SHAを主証拠にする。RTS meanとposterior varianceは同じfreeze bundleへ含める。
- PyWavelets/SciPy unavailable、solver non-convergence、non-finite、length mismatchはその方式のtechnical FAILとし、
  level低下、別solver、別filter、rolling/Savitzky fallbackを行わない。

## readoutとpromotion gate

- primaryはblock-level truth-nearest shiftのMRRとtop3。secondaryはtop1、mean rank、sign accuracy、
  `truth-nearest score - best-decoy score`、real-minus-shuffled liftとする。
- scopeはoverall、5 fold、`md_since >=1000 ft`、exp115 hidden-like spatial / typewell-purged、
  original GR missing有無、typewell-GR absolute gradientの上位10%をsharp-edgeとして固定する。
  Late専用scopeは作らない。
- distortion診断としてraw-smoothed MAE/correlation、detail-energy ratio、sharp-edge attenuation、
  output finite coverageを保存するが、primaryの代用にはしない。
- common technical PASSはrawのrow/block/fold identity 100%、score finite、score-before-truth SHA、
  expected input SHA一致。各denoiserは長さ/finite/solver convergenceを個別判定し、technical failure時は
  設定変更やfallbackで救済せずその方式だけFAILにする。valid denoiserが0本ならexp304全体をFAILとする。
- quality PASSは各denoiserをrawと比較し、次をすべて要求する。
  - pooled MRRとtop3がともに`>= +0.01` absolute改善し、top1は非悪化。
  - MRRとtop3がrawより4/5 folds以上で改善する。
  - 1000+、hidden-like 2 scope、sharp-edgeでMRR/top3がraw比非悪化。
  - realがshuffledよりMRR/top3で5/5 folds良い。
  - pooled truth-minus-best-decoy gapがrawより改善する。
- PASS方式が複数ならpooled MRR lift最大を後続候補とする。差が`<=1e-4`なら
  `robust_rts > swt_db4_l3 > l1_trend`の順でtie-breakする。
- 1方式もPASSしなければ案2〜4をすべて閉じる。同じOOFでfilter設定、閾値、scope、bank、sigmaを
  tuningして救済しない。PASSしても結論はexact-HMMの別実験を許可するだけである。

## 後続分岐

案2〜4のsingle source of truthは実験配下`reserved_followup_contract.md`とする。

1. exp304で選ばれた1 smootherだけを固定betaのtempered raw/smoothed exact-HMMで確認する。
2. 案2がrobust RTSでPASSした場合だけ、RTS posterior varianceを含むuncertainty-aware exact-HMMを確認する。
3. 案2または案3がHMM gateをPASSした場合だけ、canonical PFをpaired raw control付きで再実行し、
   cumulative path divergenceを監査する。
4. どの段階も別exp・別steering・実行承認が必要で、FAILを次段で救済しない。

## 再現性設計

- seed policy: real denoiser/scoreはRNGなし。shuffled controlのみexp280と同じ
  `SHA256(experiment,42,well,block)`由来local RNGを使う。
- stochastic 処理の有無: shuffled negative controlのみ。global RNGは使わない。
- PF/Beam / likelihood-PF / seed bagging の有無: exp304ではすべて0。後続案4のみpaired seedを使う。
- 並列処理と乱数の関係: scoreは単一process基準。並列化する場合もwell-local deterministic処理とし、
  sort後のcontent SHAが単一process基準と一致することを要求する。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU/TPU/internet off。BLAS thread数とsolver
  toleranceをmanifestに記録する。prediction/submission anchorではなく固定入力diagnostic anchorとする。
- train cache / test feature regeneration の SHA 記録方針: input raw/decompressed SHA、exp226 OOF
  decompressed SHA、denoised GR table content SHA、target-free score content SHA、contract SHAを記録する。
- model manifest / prediction / submission SHA 記録方針: model/prediction/submissionは生成しないため非該当。
- Kaggle package bootstrap 確認方針: push前にloose/package/bootstrapのconfig/source SHA一致、kernel source、
  CPU/internet metadataを確認する。今回の設計作業ではpackage/pushを行わない。

## リスク

- リークリスク: denoiser自体はtarget-freeだが、filter選択はOOF truth readoutに依存する。score freezeを強制し、
  3方式のpredeclared screeningに限定し、次のexact-HMMをconfirmatory stageにする。
- CV/LB 不一致リスク: shift-rank改善がdecoder RMSEやtest一般化を保証しない。direct補正・submitへ進めず、
  HMM、次にpaired PFの順でcontainment gateを置く。
- ランタイム/メモリリスク: 3,783,989 rows x 13 candidates x 4 signalsが大きい。well/block streamingと
  score artifact分割を前提にし、全likelihood matrixを同時保持しない。
- 再現性リスク: Student-t IRLS、SWT padding、ADMM停止判定の実装差が出る。library version、solver status、
  convergence count、per-series failureを保存し、silent fallbackを禁止する。
- 科学的リスク: smooth-only replacementはexp189と同じwrong-mode安定化を再現し得る。rawを捨てず、
  exp304ではseparabilityだけ、後続はbeta 0.15の弱いmulti-scale emissionだけに限定する。
