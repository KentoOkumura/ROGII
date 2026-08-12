# 設計

## アプローチ

exp304がtruth-nearest shift separabilityで唯一選択した`swt_db4_l3`を、exp209 exact-HMMのraw Gaussian emissionへ固定beta 0.15で混ぜる。変更するのはrow/state log emissionだけであり、HMMのstate、transition、prior、decoderは変更しない。HMM posterior meanをdirect candidateとして評価し、同じtempered HMMと保存済み`last_known_tvt + likpf_mean_d`の算術50/50 blendを追加評価する。control predictionは保存済み生成物から読み、再生成しない。

## 仮説

弱いmulti-scale emissionはraw likelihoodのmode識別を残しながらGRノイズだけを抑え、direct HMMと保存済みlikPF blendを両方改善する。

## 実験範囲

- 対象実験: `exp305_tempered_raw_smoothed_exact_hmm_emission`
- Route: `pf_beam`
- 科学的親: `exp304_gr_denoiser_emission_separability_readout`
- HMM/control参照: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- hidden-like参照: `exp115_hidden_like_spatial_holdout_from_ppt`
- fold/score-row参照: exp304と同じexp226 OOF identity
- 変更する変数: GR log emissionを`0.85 * ell_raw + 0.15 * ell_swt`へ置換する1点だけ。
- 固定する変数: SWT、raw/SWT missing policy、known-prefix raw sigmaとclip、HMM全設定、posterior mean、likPF prediction、blend weight、fold、scope、promotion gate。
- 実装量: 1 scientific variant、773 HMM well-runs、model / LightGBM / trained fold / PF / Beam / booster `0 / 0 / 0 / 0 / 0 / 0`。control再実行0。

## 入力契約

### exp304 selected series

- Kaggle source: `kentookumura/exp304-gr-denoiser-separability-train` version 1。
- 必須file: `exp304_gr_denoiser_emission_separability_readout_denoised_gr_series.csv.gz`と対応manifest/summary/scientific contract。
- 必須条件: file size > 0、6,659,300 data rows、selected denoiser `swt_db4_l3`、silent fallback 0。
- expected denoised series decompressed/content SHA: `a4acb72d60b833b12b2560db1e5dc3a113ae6ecf4137efbccf78c278582a0988`。
- expected scientific contract content SHA: `8822df968200b74ea9969b0bc023ec127debbff01933bdc89ff3db9844d55064`。
- expected raw well-file identity content SHA: `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`。
- raw gzip SHAとdecompressed SHAはexp304 manifest記録値を読み、その場で実ファイルと照合する。manifest自体や上記expected content SHAと不一致ならHMMを開始しない。

### saved HMM / likPF control

- exp209/exp205 exact-HMM decompressed SHA: `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`。
- exp209 v5 comparisonで使用したexp072 cache decompressed SHA: `0503de0512302b06309d26e09fc06ba5095db0ef4d610b1508afe8c8d07ca536`。
- baseline RMSE: raw HMM `11.9382872349`、likPF `11.5948976722`、raw-HMM/likPF 50/50 `10.2696961466`。
- exp072 cacheはcanonical v2とのfull artifact parityが未証明であるため、canonical性を主張しない。exp305では上記exp209 v5 comparisonと同じ保存済みrow prediction/contentだけをpaired controlとして固定し、ID/order/finite/metric parityをpreflightする。
- exp226 OOF decompressed SHA: `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`。
- exp115 hidden-like SHA: `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`。

## 固定HMM契約

- `step=0.35`, `n_rates=41`, `rate_span=0.10`, `sig_r=0.002`, `sig_p=0.02`, `df=4`。
- Gaussian raw emission、`start_sig=0.75`, `r0_sig=0.01`, `band_pad=100`, `mom=0.998`, rate center zero。
- known-prefix clamp、GR missing policy、state grid、transition、prior、backward smoothing、posterior meanはexp209から変更しない。
- `ell_raw`と`ell_swt`は同一wellのknown-prefix raw GR residual sigmaを使い、sigma clip `[10, 60]`、log likelihood clip `600`を共有する。SWTからsigmaを再推定しない。
- mixtureはlog emissionに対して行う。`logp_raw + beta * logp_swt`やz-score/GR値のblendは禁止する。

## 検証方法と実行順序

1. kernel source、全入力file、row/schema/content SHA、selected SWT、control metricをhard preflightする。
2. horizontal true TVT、error、formationを読まない状態で773 wellsのtempered HMM posterior meanを生成する。
3. prediction、posterior、runtime、input manifestのcontent SHAを凍結する。
4. unknown-suffix true TVT、exp226 fold、exp115 hidden-like assignmentをlate joinする。
5. tempered directと保存済みraw HMM、tempered/likPF 50/50と保存済みraw-HMM/likPF 50/50をpaired評価する。
6. overall、fold 0--4、distance 1000+、hidden-like 2面、by-well p95/worst、coverageを読み、全gateのANDでPASS/FAILを確定する。

scope非悪化はpaired RMSE delta `<=0.0 ft`を原則とし、float comparisonの絶対許容差だけ`1e-6 ft`とする。directとblendのどちらか一方でもFAILなら実験全体をFAILとする。

## 生成物契約

- scientific contract JSON
- input/control manifest JSON
- tempered HMM predictions CSV.gz
- posterior/runtime by-well CSV
- overall/fold/scope metrics CSV
- by-well paired metrics CSV
- promotion gate JSON
- summary JSON

予測CSV.gzはraw gzip SHAとdecompressed content SHAを分けて記録する。submission、raw-test prediction、PF/Beam path、model artifactは生成しない。

## 再現性設計

- seed policy: HMMとmetricsはRNGなし。wellは文字列昇順で処理する。
- stochastic 処理の有無: 新規stochastic処理なし。保存済みlikPFは上流のstable SHA256 per-well seed生成物だがexp305で再生成しない。
- PF/Beam / likelihood-PF / seed baggingの有無: 実行なし。保存済み`likpf_mean_d`を`last_known_tvt`へ加えてblend評価に読むだけ。
- 並列処理と乱数の関係: RNGなし。exp209 selected runtimeと同じ`outer_workers=2`, `numba_num_threads=2`を固定し、実効thread数とlibrary versionを記録する。
- CPU/GPU runtime: Kaggle CPU、GPU/TPU/internet off、8.5時間hard guard。
- train cache SHA: input file SHA、schema、row/well数、decompressed content SHA、prediction content SHAを記録する。test regenerationは行わない。
- model/prediction/submission SHA: modelとsubmissionは非該当。prediction raw/decompressed/content SHAを必須にする。
- Kaggle bootstrap: push承認後に正の実験ファイルからpackageを再生成し、bootstrap内config、kernel source、beta、HMM contract、run approval flagを照合する。
- deterministic anchor: train-side scientific candidateでありsubmission anchorとは呼ばない。

## リスク

- リークリスク: SWT選択自体はtrain-side truth readout由来である。exp305内ではprediction freeze前にtrue TVT/errorを使わず、結果はtrain-side evidenceに限定する。
- controlリスク: exp209のexp072 full cacheはcanonical v2とのSHA parityが未証明。exp209 v5で実際にbaseline metricを作った保存済みrow contentへ比較対象を限定し、canonical likPF再現とは表現しない。
- CV/LB不一致リスク: OOF学習ではなくtrain-side decoder診断であり、Public LB改善を直接示さない。PASSしてもinference/submitへ進めない。
- ランタイム/メモリリスク: 773 exact-HMM well-runsと大容量series入力がある。well streamingを使い、全wellのstate likelihoodを同時保持しない。8.5時間超過はtechnical FAILとする。
- 再現性リスク: 大容量seriesのlocal CLI downloadが0 byteだった。Kaggle source上のnonzero sizeとmanifest SHAが一致しない場合はfail-closeする。
- 科学的リスク: smoothingがwrong modeを安定化する可能性がある。direct/blend、fold、long-tail、hidden-like、p95/worstの全gateを必須とし、同一OOFの救済gridを禁止する。

## 次のアクション

Kaggle CPU v3でfixed designを完走したが、direct / blendとも改善1/5 folds、全stress scope悪化でgate FAILとなった。事前登録どおりnegative resultとして閉じ、救済、案3/案4、inference、submissionを行わない。
