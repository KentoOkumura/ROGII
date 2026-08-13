# exp304 reserved follow-up contract

## この文書の役割

この文書は案2、案3、案4のsingle source of truthである。3案はexp304のvariantではなく、必要条件を
満たした後に別exp番号とsteeringで作る後続実験である。案2は2026-07-21に
`exp305_tempered_raw_smoothed_exact_hmm_emission`として設計・実装・Kaggle CPU実行を完了し、全promotion gate FAILで閉じた。
案3は開始条件不成立、案4は案2 FAILにより未採番のまま閉鎖済み。
別セッションで再開しても、入力、式、自由度、gate、分岐、禁止事項を暗黙に変更しない。変更が必要なら、
実装前にこの文書、exp304 steering、backlog/KAGGLE_DIRECTION.mdを明示改訂する。

## 2026-07-21 分岐状態

- exp304 Kaggle CPU version 1はtechnical/quality gateをPASSした。
- 固定tie-break後の`selected_denoiser`は`stationary db4 level-3 SWT`である。
- 案2は`exp305_tempered_raw_smoothed_exact_hmm_emission`としてKaggle CPU v3を完了した。directは`11.938287 → 13.218199`、fixed likPF 50/50は`10.269693 → 10.767674`へ悪化し、両方とも改善1/5 folds、1000+/hidden-like 2面/p95/worstを全FAILしたため救済せず閉じた。
- 案3は`selected_denoiser == robust_rts`条件を満たさないため閉じる。
- 案4は案2 FAILにより開始条件不成立として閉じる。

## 共通原則

- Lateフェーズ専用の分岐・scope・重みは作らない。
- exp189のrolling median / Savitzky-Golay direct replacementを再試行しない。
- raw emissionを捨てない。案2と案3はraw/smoothedの弱いmulti-scale emissionだけを使う。
- exp209 exact HMMのstate grid、transition、prior、known-prefix clamp、posterior-mean decoder、missing-GR
  policy、raw sigma/clipを固定する。state band、sigma、temperature、process noiseを同時に変えない。
- 各案は1 scientific variantだけを実行し、FAIL後に同じOOFでbeta、Q/R、variance、thresholdをgridしない。
- PASSは次段の別実験を設計可能にするだけで、inference/submissionを自動許可しない。

## 案2: tempered_raw_smoothed_exact_hmm_emission

### 開始条件

- exp304 technical gateを通過している。
- exp304のquality gateを1方式以上がPASSし、固定tie-breakで`selected_denoiser`が1つ確定している。
- selected denoiserのtarget-free series、solver status、content SHA、raw common-input series、known-prefix
  sigmaが保存されている。
- exp304大容量seriesはlocal CLI downloadが0 byteだったため、案2 notebookの入力preflightでkernel source上の
  nonzero size、6,659,300 rows、manifest記録raw/decompressed/content SHA一致を確認する。不一致ならdecodeを開始しない。
- いずれかが欠ければ案2を開始しない。exp304 FAILをexact HMMで救済しない。

### 固定する設計

- Route: `pf_beam`。
- selected denoiserはexp304が選んだ1方式だけ。再選択、2方式blend、filter設定変更はしない。
- row/state log emissionを
  `ell_beta = (1 - beta) * ell_raw + beta * ell_smooth`、`beta = 0.15`
  とする。これはrawを保持する固定geometric mixtureで、`beta=1`のdirect replacementではない。
- `ell_raw`と`ell_smooth`は同じexp209 known-prefix raw sigma、clip `600`、missing policyを使う。
- exp209 exact-HMM default `step=0.35`、`n_rates=41`、`rate_span=0.10`、`sig_r=0.002`、
  `sig_p=0.02`、`df=4`、Gaussian raw emission、`start_sig=0.75`、`r0_sig=0.01`、
  `band_pad=100`、`mom=0.998`、rate center zero、posterior meanを固定する。
- active variant 1、773 wells、773 HMM well-runs、model/LightGBM/PF/Beam 0。saved exp209 raw HMMを
  controlにし、raw controlは再実行しない。
- direct HMMに加え、saved exp072の`last_known_tvt + likpf_mean_d`との固定50/50 blendだけを比較する。blend weightを変えない。

### baselineと成功条件

- raw HMM baseline: exp209 compatible exact posterior mean RMSE `11.9382872349`。
- saved likPF baseline: RMSE `11.5948976722`。
- saved raw-HMM/likPF 50/50 baseline: RMSE `10.2696961466`。
- tempered HMMがraw HMMからpooled `>=0.05 ft`改善し、4/5 folds以上で改善する。
- tempered-HMM/likPF 50/50がsaved 50/50からpooled `>=0.05 ft`改善し、4/5 folds以上で改善する。
- directとblendの両方で1000+、hidden-like spatial/typewell-purgedが非悪化し、by-well p95が非悪化、
  worst-well RMSE deltaが`<=+0.25 ft`である。
- input/row/fold/finite coverage 100%、saved-control SHA、selected denoiser SHA、HMM state/grid contractが一致する。
- 全条件PASSで案2 PASS。1つでもFAILならbeta/sigma/clip/filter救済をせず、案4を開かない。

### 禁止事項

- `logp_raw + beta*logp_smooth`による総emission scale変更、beta grid、beta 1、smooth-only replacement。
- HMM state/grid/transition/prior/decoder変更、Viterbi/top-k、PF併走、rolling/Savitzky追加。
- selected denoiserの再fit設定変更、smoothed GRからのsigma再推定、truth-error gate。
- inference/submissionへの自動進行。

## 案3: uncertainty_aware_robust_rts_exact_hmm_emission

### 開始条件

- exp304の`selected_denoiser`が`robust_rts`である。RTSが単独PASSでも未選択なら開始しない。
- 案2が同じrobust RTS meanを使って全gate PASSしている。
- exp304 RTS mean/posterior variance、Q/R scale、solver convergence、content SHAと案2 artifactが保存されている。
- いずれかが欠ければ案3を開始しない。SWT/L1をKalman variance風に拡張しない。

### 固定する設計

- Route: `pf_beam`。
- raw likelihoodは案2と同一。RTS smooth likelihoodのvarianceだけを
  `V = sigma_match_raw^2 + P_horizontal + P_typewell`
  に変更する。posterior covarianceは各candidate TVTでtypewell側を線形補間し、finite floorは
  `1e-6`、上限は`60^2 + 2*60^2`とする。
- emissionは
  `ell_unc = 0.85 * ell_raw + 0.15 * ell_rts_uncertainty`
  とする。beta、raw/smooth mean、HMM、clip、missing policyは案2と同じ。
- Student-t measurement自由度4、Q/R推定、RTS反復はexp304 artifactを再利用し、再推定・gridしない。
- active variant 1、773 HMM well-runs。controlはsaved exp209 rawと案2 saved tempered HMMで、再実行0。

### 成功条件

- uncertainty-aware HMMが案2 tempered mean-only HMMからpooled `>=0.02 ft`改善し、4/5 folds以上で改善する。
- fixed likPF 50/50も案2の50/50からpooled `>=0.02 ft`改善し、4/5 folds以上で改善する。
- 案2と同じ1000+、hidden-like 2面、by-well p95、worst `<=+0.25 ft`、coverage/SHA guardを通す。
- calibration診断としてposterior standard deviation decile別のraw-vs-uncertainty gainを保存し、
  highest-uncertainty decileでmean-only比が悪化しない。
- PASSなら案4へ渡すemissionは案3、FAILなら案2 PASS emissionを維持する。案3 FAILで案2を閉じない。

### 禁止事項

- Q/R truth tuning、posterior variance multiplier、variance temperature/floor/cap grid、mean-only direct replacement。
- RTS meanの再平滑化、beta変更、HMM state/grid/transition/decoder変更。
- 案2 FAIL時の開始、SWT/L1へのvariance捏造、inference/submission。

## 案4: hmm_supported_tempered_emission_pf_transfer_containment_audit

### 開始条件とemission選択

- 案2がPASSしている。案3を実行してPASSした場合は案3 emission、未実行またはFAILなら案2 emissionを使う。
- selected emission、raw/smoothed input、sigma/variance、HMM result、全SHAが保存されている。
- canonical exp072 PFのcode/config/seed manifestを固定でき、paired raw control再生成のCPUコストについて
  ユーザーの明示承認を得ている。承認前にpushしない。

### 固定する設計

- Route: `pf_beam`。
- exp072 canonical likelihood-PFを、raw emissionとselected tempered emissionの2枝で同一run内paired実行する。
- particles `500`、seeds `128`、stable SHA256 per-well seed、initialization、transition/process noise、
  resampling threshold/noise、missing policy、output aggregationをcanonical exp072から変更しない。
- 2 variants x 773 wells = 1,546 PF well-runs、各128 seeds、合計197,888 paired well-seed trajectories。
  LightGBM/HMM再実行/Beam/model/boosterは0。
- saved raw likPFはmetric anchorとして使うが、path/ESS/resamplingをseed-paired比較するためraw PF controlは
  同一runで再生成する。これは性能baselineの探索ではなく因果比較に必要な例外である。
- row-levelでvariant-minus-raw path delta、first `0.5/1/5 ft` divergence、cumulative absolute divergence、
  suffix first64/last64、ESS、resampling rate、seed disagreement、first-loss rowを保存する。

### technical parityと成功条件

- paired raw controlはexpected rows/wells/seeds、seed manifest、finite/identity coverageを100%満たす。
- paired raw control RMSEはsaved exp209 likPF `11.5948976722`とabsolute `1e-4 ft`以内で一致する。
  不一致ならtechnical FAILとしてvariant性能を解釈しない。
- selected PFがpaired raw PFからpooled `>=0.05 ft`改善し、4/5 folds以上で改善する。
- 1000+、hidden-like spatial/typewell-purged、by-well p95が非悪化し、worst delta`<=+0.25 ft`。
- well別first64/last64 mean absolute paired-path deltaの中央値を`D_start/D_end`とし、
  `D_end <= 2.0 ft`かつ`D_end / max(D_start,0.25) <= 4.0`を満たす。
- seed disagreement中央値はraw比`<=1.10x`。ESS/resamplingは必ず報告するが、単独改善を性能PASSの代用にしない。
- 1項目でもFAILならtemperature、mixture、particle数、process noise、reinjection、resamplingを救済せず閉じる。

### 禁止事項

- unpaired saved PFだけとの比較、raw control省略、seed/particle/process/resampling条件の枝間差。
- temperature/mixture beta grid、smooth-only PF、particle reinjection、mode jump、Beam併用。
- HMM/PF blendやML selectorによるFAIL救済、raw-test inference、submission。

## 分岐順序

1. exp304を実装・実行し、raw対比のemission separabilityを判定する。
2. exp304 FAILなら案2、案3、案4をすべて閉じる。
3. exp304 PASSならselected 1方式だけで案2を別exp化する。
4. 案2 FAILなら案3と案4を閉じる。
5. 案2 PASSかつselectedがrobust RTSの場合だけ案3を別exp化できる。SWT/L1選択時は案3を閉じる。
6. 案3 PASSなら案3 emission、案3未実行/FAILなら案2 emissionを案4へ渡す。
7. 案4はpaired raw replayの計算量を再提示して承認後にだけ実行する。PASSしてもinference/submissionは別判断とする。
