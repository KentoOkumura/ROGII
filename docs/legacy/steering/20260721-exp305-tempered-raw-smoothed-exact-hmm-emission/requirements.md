# 要件

## 依頼

- `tempered_raw_smoothed_exact_hmm_emission`を`exp305`として新規作成し、steeringと実験ディレクトリで設計を確定する。
- exp304が選択したstationary db4 level-3 SWTをraw GR emissionへ弱く混ぜ、exp209互換exact-HMMと保存済みlikPFとのfixed 50/50 blendにdecoder価値があるかをtrain-sideで判定する。
- 初回依頼では設計とdesign-only scaffoldまでとした。
- 2026-07-21の追加依頼「exp305を実装してください」により、固定済み設計のNotebook実装、synthetic test、静的検証までを承認範囲へ追加する。
- 続く依頼「実行してください」により、1 variant / 773 HMM well-runs / 0 boosterのKaggle CPU trainだけを承認範囲へ追加した。inference、submissionは行わない。

## 仮説

exp304でseparabilityを改善したSWT emissionをraw exact-HMM emissionへ15%だけ混ぜれば、rawを保持したままposterior meanとfixed likPF blendのRMSEを安定して改善できる。

## 制約

- Routeは`pf_beam`とする。
- exp304の`selected_denoiser=swt_db4_l3`、SWT設定、raw/SWT series、known-prefix raw sigma、missing-GR policy、artifact SHAを固定し、再選択・再fitしない。
- emissionは`ell_beta = 0.85 * ell_raw + 0.15 * ell_swt`の1 scientific variantだけとする。rawを捨てず、beta、sigma、clip、filter設定を探索しない。
- exp209 exact-HMMのstate grid、transition、prior、known-prefix clamp、posterior-mean decoderを変更しない。
- active variant 1、773 wells、773 HMM well-runs、model / LightGBM config / trained fold / PF / Beam / boosterはすべて0とする。
- 保存済みexp209 raw HMM、保存済みexp072の`last_known_tvt + likpf_mean_d`、保存済みraw-HMM/likPF 50/50をcontrolとし、raw HMMやlikPFを再実行しない。
- exp304大容量seriesはlocal CLI取得が0 byteだったため、Kaggle kernel source上のnonzero file、6,659,300 rows、manifestのraw/decompressed/content SHA一致をdecode前のhard preflightとする。
- unknown-suffix true TVTはtempered predictionとprediction content SHAを凍結した後の評価にだけ使う。
- `docs/06_reproducibility.md`に従い、入力・予測・control・fold/hidden-like assignmentのcontent SHA、runtime version、thread設定、Kaggle kernel versionを記録する。

## 受け入れ基準

- 入力preflight、ID/order、row/well/fold、finite coverage、exp304 selected SWT、exp209 HMM contract、保存済みcontrol SHAがすべて一致する。
- tempered HMMのRMSEがsaved raw HMM `11.9382872349`からpooledで`>=0.05 ft`改善し、5 folds中4 folds以上で改善する。
- tempered-HMM/likPF 50/50のRMSEがsaved raw-HMM/likPF 50/50 `10.2696961466`からpooledで`>=0.05 ft`改善し、5 folds中4 folds以上で改善する。
- directとblendの両方で1000+、hidden-like spatial、hidden-like typewell-purged、by-well RMSE p95がsaved controlから悪化せず、worst-well RMSE deltaが`<=+0.25 ft`である。
- prediction有限率100%、ID mismatch 0、silent fallback 0、773 HMM well-runs完了、CPU runtime 8.5時間以内を満たす。
- 1条件でもFAILならbeta/sigma/clip/filter/HMM/blendの救済をせずnegative resultとして閉じ、案4を開かない。
- gzip生成物はraw gzip SHAとdecompressed content SHAを分け、後者を主証拠として記録する。
- PASSしてもinference/submissionへ自動進行せず、案4はpaired raw PF controlの計算量を再提示して別途承認を得る。

## 次のアクション

Kaggle CPU v3は完了し、direct / blendとも改善1/5 folds、全stress scope悪化でgate FAILとなった。事前登録どおりnegative resultとして閉じ、救済、案3/案4、inference、submissionを行わない。
