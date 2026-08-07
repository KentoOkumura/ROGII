# 設計

## アプローチ

比較する信号は2本だけとする。

1. `control_linear`: `exp223` と同じ raw horizontal GR と既存線形補間。
2. `typewell_gapfill`: finite raw GR はそのまま保持し、known prefix の raw GR 欠損セルだけを頑健 affine 校正済み Type Well GR で埋める。補完不能セルは control の線形補間へ戻す。

Type Well 復元 GR の全面置換、target 区間の復元、state ごとの emission 追加、候補 path の選択、hard gate、blend はこの実験に含めない。変更するのは `exp223` self-GR donor descriptor が読む欠損セルの値だけである。

## 実験範囲

- 対象実験: `exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm`
- Route: `ensemble`
- 親実験: `exp223_joint_typewell_self_gr_hmm_likelihood_probe`
- 参照 baseline:
  - `exp223` best `hmm_selfgr_boost_only_a070_c100`: RMSE 11.349950650。
  - `exp072` `likpf_mean`: RMSE 11.594897668。
  - `exp209` HMM/likPF blend: RMSE 10.269696。raw-test port の追加昇格基準。
- negative evidence: `exp225` state-known curve RMSE 14.212954500、1000+ `+2.931795 ft`、worst `+49.423573 ft`。
- 変更する変数: known-prefix self-GR donor signal の raw missing cell に入れる値だけ。
- 固定する変数: raw finite GR、raw missing mask、anchor eligibility、receiver signal、base HMM emission、grid、transition、band、affine以外の descriptor、`alpha=0.07`、`clip=1.0`、`boost_only`。

## 信号生成契約

### Type Well サンプリング

- Type Well は TVT 昇順に並べ、同一 TVT の重複は GR median に集約する。
- finite `(TVT, GR)` だけを使用し、線形補間は Type Well の最小/最大 TVT 内に限定する。
- Type Well 範囲外、finite点2個未満、局所サンプル不能時は Type Well 値を作らず、既存線形補間へ fallback する。

### well ごとの頑健 affine 校正

`x = TypeWellGR(TVT_input)`、`y = raw horizontal GR` の finite known-prefix pair へ `y = a*x+b` を fit する。

- 最低 pair 数: 32。
- Type Well側 `IQR(x) >= 5 GR` を必要とする。未達時はその well の Type Well gap-fill を無効化する。
- 初期値: ordinary least squares。
- 更新: deterministic Huber IRLS、`k=1.345`、最大20反復、係数の相対変化 tolerance `1e-8`。
- residual scale: `1.4826 * MAD`、floor `1.0 GR`。
- slope/intercept の clip や候補 grid は使わない。非有限、singular、未収束時は Type Well gap-fill を無効化する。
- Stage 0 では擬似欠損にした row を fit pair から必ず除外する。Stage 1 では raw finite known-prefix pair のみで fit する。

### hybrid donor 信号

行ごとの優先順位を固定する。

1. raw GR が finite: raw GR を exact に保持する。
2. raw GR が non-finite、`TVT_input` が finite、Type Well 内挿と affine fit が有効: `a*TypeWellGR(TVT_input)+b`。
3. その他: `exp223` の既存線形補間値。

raw missing mask、anchor center、prefix anchor stride、max anchors、window missing-rate gate は補完前の raw mask で計算する。hybrid 値は eligible と判定済み donor window の descriptor 値だけに使う。receiver と base HMM emission は control 信号のままとする。

## Stage 0: 擬似欠損信号監査

### reporting fold と block 長

- `well` 文字列の UTF-8 bytes に対する stable SHA256 から `fold = uint64(first 8 bytes) % 5` を作る。Python `hash()` は使わない。
- 各 fold の block 長は、その fold を除く well の known prefix に実在する raw GR missing-run 長の q25 / q50 / q90 を四捨五入し、`[1, 64]` rowsへclipする。missing-runが存在しない場合だけ `[1, 4, 16]` を使う。
- 各 valid well で各 quantile label につき1 blockをmaskする。候補startは、block全行のraw GRがfinite、Type Well TVT範囲内、known prefix端から12行以上内側、他maskと非重複、左右にfinite control anchorがある位置に限定する。
- 候補startの選択は `sha256(experiment|fold|well|quantile_label)` の昇順indexで決定し、global RNGを使わない。候補なしは skip 理由を記録する。

### fit と比較

- affine fit、control interpolation、variant gap-fill のすべてから held-out block の raw GR を除外する。
- primary: held-out row の pooled RMSE。
- secondary: 長さ4以上かつ真値分散が正の block について、block内 mean/std を使う ZNCC の block長加重平均。
- diagnostic: derivative NCC、MAE、by-well RMSE、run-length bucket、affine `a/b`、fit RMSE、fallback reason、coverage。
- fold判定は RMSE relative improvement と ZNCC delta を well-hash 5 foldsで再集計する。

### Stage 0 hard gate

- pooled RMSE relative improvement `>=5%`。
- pooled ZNCC delta `>=+0.02`。
- RMSE改善 fold数 `>=4/5`、ZNCC正方向 fold数 `>=4/5`。
- by-well RMSE p95 delta `<=0`。
- observed known GR exact parity、raw mask parity、held-out fit exclusion、target-side Type Well fill count 0、finite output coverage 100%を全て満たす。

Stage 0 FAIL時は補完方法や閾値を調整せず終了する。

## Stage 1: 固定 self-GR HMM 比較

Stage 0 PASSと別途承認の後だけ、`exp223` の `hmm_selfgr_boost_only_a070_c100` を1本再生成する。保存済み `exp223` prediction/controlは再実行しない。

- HMM variant: 1。
- well-runs: 773。
- LightGBM config / trained fold / booster: `0 / 0 / 0`。
- GPU、inference、submission: 0。
- 予測対象: train unknown suffix 3,783,989 rows / 773 wells の既存 exp223 readout契約。
- truth join: prediction・scope・SHA freeze後のみ。

### Stage 1 hard gate

- overall RMSE `<=11.249950650`、すなわち exp223 比 `<=-0.10 ft`。
- stable well-hash reporting fold の4/5以上で delta `<=-0.10 ft`。
- 1000+、exp115 verification-like spatial、typewell-purged の各 delta `<=+0.02 ft`。
- worst-well RMSE delta `<=+0.25 ft`。
- known-prefix raw GR missing rate 0 の wells で exp223 predictionとの最大絶対差 `<=1e-6 ft`。
- known-prefix raw GR missing rate `<=1%` scope の RMSE delta `<=+0.02 ft`。
- observed known GR exact parity、raw mask / anchor eligibility parity、target-side Type Well fill 0、row identity、finite prediction coverageを全て満たす。

Stage 1 PASSは「gap-fillがexp223を改善した」という科学的判定に限定する。raw-test port は exp209 blend 10.269696 以下、安全guard全通過、別設計、ユーザー承認を追加条件とする。

## 予定artifact

### Stage 0

- `stage0_pseudo_missing_manifest.csv.gz`: well/fold/block/run-length/start/row identity。held-out GR truth は mask freeze 後の評価側にのみ付与する。
- `stage0_affine_fit_summary.csv`: pair数、`a/b`、fit RMSE、MAD scale、収束、fallback reason。
- `stage0_by_block_metrics.csv`、`stage0_by_well_metrics.csv`、`stage0_fold_metrics.csv`。
- `stage0_summary.json`、`artifact_manifest.json`。

### Stage 1

- hybrid HMM feature/prediction CSV.gz、feature schema、by-well generation summary。
- overall / fold / distance / hidden-like / missing-rate / by-well delta metrics。
- raw-mask / anchor-parity audit、summary JSON、artifact manifest。
- submission.csv は生成しない。

## 再現性設計

- seed policy: `stable_sha256_per_well_fold_runlength`; HMM、Type Well sampling、Huber IRLS 自体は RNG を使わない。
- stochastic 処理: なし。擬似mask選択も SHA256 index で決定する。
- PF/Beam / likelihood-PF / seed bagging: なし。
- 並列処理: outer well workersは2固定。処理順による微小な浮動小数差は metrics tolerance と decompressed content SHA の両方を記録する。
- runtime: CPU-only、GPU deterministic flagsは対象外。Stage 0は軽量監査、Stage 1はexp223の2 variant 10h50mを根拠に1 variant約5-6時間を見込む。
- SHA: raw input、exp223 saved control、exp115 fold assignment、schema、擬似mask、affine summary、predictionのlogical/decompressed content SHAを記録する。raw gzip SHAは補助証拠とする。
- model manifest:学習modelは存在しないため `not_applicable_no_trained_model` を明記する。
- Kaggle bootstrap: 実装・push前に `kernel-metadata.json`、dataset source、notebook内config、`config.yaml` の整合を確認する。
- deterministic anchor: false。初回は train-side no-training diagnostic であり、submission anchorにはしない。

## リスク

- リークリスク: pseudo-mask rowをaffine fitへ残す、unknown suffix true TVTでmask/fit/gateを選ぶ、target `TVT_input` 相当を復元に使う実装を禁止する。
- 変数混同: Type Well gap-fillでanchor eligibilityまで広げると「値の補完」と「anchor追加」の2変数になる。raw mask parityをhard failにする。
- wrong-depth吸着: 復元 GR がType Wellに似すぎると自己一致scoreを人工的に強める。補完はmissing donor cellだけ、alpha固定、worst-well/hidden-like/longtail guardを必須にする。
- affine不安定: GR variation不足、小標本、outlierで係数が破綻し得る。pair/IQR/収束条件を満たさなければcontrol fallbackとする。
- CV/LB不一致: exp223はtrain-side positiveでもexp209に未達、exp274/exp264ではCV/LB方向差も観測済み。Stage 1 PASSだけでsubmissionへ進めない。
- runtime: Stage 1は約5-6時間のCPU HMMを想定する。Stage 0 PASS前に実行しない。
- 再現性: Python hash、global RNG、gzip metadata、worker順序へ依存しないidentity/content SHAを正とする。
