# 設計

## アプローチ

実装を2段に分ける。第1段はCPU deterministic auditとして、raw trainのofficial
prediction startを読み、well metadata、cutoff候補、distribution-matched selection、
fold manifest、sampling前後の分布レポートを生成する。第2段は選択cutoffごとに
`TVT_input` maskを作り、anchor、prefix統計、GR calibration、candidate、PF/HMM初期状態、
learned-likelihood / GRWR confidenceを再生成するためのprefix replay requestを保存する。

静的実装時点では親exp218の15 boostersを再学習しない。まずcutoff readoutとreplay契約を
Kaggle CPUで確認し、feature regenerationのruntimeとcoverageを監査した後、exp218系の
新規variantだけを学習する。旧exp023 controlは保存済みCV 13.494554 / best 12.942938を
historical referenceとして使い、GPU control再学習は行わない。

## 実験範囲

- 対象実験: `exp239_distribution_matched_multicut_pseudotail`
- Route: `ml_model`
- 親実験: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- 比較親: `exp023_pseudo_tail_distance_augmentation`、`exp115_hidden_like_spatial_holdout_from_ppt`
- 変更する変数: early-start cutoff selectionとaugmentation row balance
- 固定する変数: official validation rows、well GroupKFold、exp218 parent features/model family、target、metric
- target distribution: all-train official startをprimary、exp115 verification-like holdoutをstress readoutとする。current-testはweight算出に使わない。
- cutoff sources:
  - feasible prefix/eval quantile
  - official startまで新たに隠す `50/100/250/500/1000/2500+` rows
  - prefix内GR change point
  - GR missing block boundary
  - trajectory curvature change point
- matching: sparseな多次元直積binを避け、prefix fraction、prefix rows、eval rows、GR missingness、trajectory phaseの各marginal bin deficitをgreedyに減らすdeterministic quota selection。tieはstable hashで解消する。
- cap: well、source、total selected cutoffs、estimated augmented rowsをconfigで固定する。

### v2 revision

v1のwell round-robin 2-cutoff固定は短prefixを過剰化したため廃止する。v2は候補全体から
global marginal deficitを直接減らし、0-3 cutoffs/wellを許可する。target totalは800、
source diversity / hidden-like / new-well coverageは小さいsoft bonusだけで維持する。採用guardは
max marginal share差0.05、well coverage 0.65、hidden-like coverage 0.90、augmentation ratio 0.45。

### v3 prefix materialization

v2 guard通過後の最小段階として、800 replay requestsから`anchor_and_prefix_statistics`だけを
materializeする。各requestはsynthetic cutoff後をevaluation zoneとし、distance bucketごとに
deterministic samplingして最大1,000 rowsへcapする。feature builderは`TVT`列を受け取らず、
targetは特徴生成完了後に別配列として付与する。PF/Beam、learned likelihood、GRWR、LightGBMは
この段階では生成・学習しない。出力row/request coverage、fold inheritance、feature/target分離、
schema/content SHA、runtime/memoryを次段階の実行可否判断に使う。

### v4 residual learnability probe

v3 cache単独で評価できないPF/Beam candidate rankerやlearned likelihoodへ直行せず、まず
`target_tvt - anchor_tvt_input`を目的変数にしたCPU LightGBM residual probeを行う。
source well GroupKFoldを維持し、1 config x 5 folds = 5 boostersだけを学習する。identifier、fold、
target、文字列列は特徴から除外し、anchor/prefix/row geometryの数値特徴だけを使用する。
比較対象はanchor holdと`anchor_tvt + delta_z`の決定的baselineとし、overall、distance bucket、
source-well別を読む。このprobeはsynthetic pseudo-tail surface上のlearnability確認であり、
official-start OOF、親exp218との採用比較、推論・提出を代替しない。

### v5 full exp218 augmentation evaluation

元の仮説を直接評価する本試験。通常のexp218 3,783,989 official-start rowsはweight 1.0、
v3の799,961 pseudo-tail rowsはweight 0.5とし、実効augmentation massを約10.6%にする。
pseudo rowsもexp218と同じ380-feature schemaになるよう、各synthetic cutoffからexp072 base、
U projection、exp111/145 learned likelihood、exp218 GRWRをtarget-free再生成する。outer foldの
validationはofficial-start rowsだけで、valid source well由来のpseudo rowsはtrainから除外する。
新規augmentation variant 1、exp218の3 LightGBM configs、5 folds、合計15 boostersを学習し、
保存済みexp218 OOF 8.475793752をcontrolとして使う。親/control再学習は行わない。

### v8-v9 two-stage cache and training

v7は800 requestsを同時にDataFrame listへ保持したため、特徴生成開始約2時間56分後に
`DeadKernelError`で終了した。LightGBMは0/15 boostersで、official-start OOFは未取得。
評価面を変えず、CPU feature-cache kernelとGPU training kernelへ分離する。CPU側はsorted
requestを25件ずつ処理し、各batchの380特徴をParquet shardへ保存してDataFrameを解放する。
32 shards合計800 requests / 799,961 rowsを、schema SHA、file SHA、row-content SHA付きmanifestで
検証する。GPU側は検証済みcacheだけを入力とし、pseudo weight 0.5、official-only validation、
valid source-well除外、3 configs x 5 folds = 15 boostersを維持する。

GPU cached-training v1はpseudo 32 shardsを全検証後、pseudo cacheを保持したままofficial
380-feature surfaceを再assemblyしてOOMとなった。推奨復旧としてofficial-start 3,783,989行も
別CPU kernelで一度だけ生成し、250,000 rows x 16 Parquet shardsへ保存する。最終GPU stageは
official/pseudo両cacheをstreamしてdisk-backed matrixへ変換し、特徴再生成を行わない三段階構成とする。

## 再現性設計

- seed policy: fold、well、cutoff source、cutoff rowのimmutable keyからSHA256 stable keyを作る。selection自体は乱数を使わない。
- stochastic 処理の有無: cutoff候補生成・matching・fold assignmentはdeterministic。将来のLightGBM GPUだけ非bitwise再現可能性を別記する。
- PF/Beam / likelihood-PF / seed bagging の有無: 第1段では再生成しない。第2段で必要な場合は既存per-well stable seed policyを継承し、raw-test regenerationとは別監査する。
- 並列処理と乱数の関係: global RNGを使わず、well処理順をsorted well idで固定する。
- CPU/GPU runtime と deterministic flags: readoutはCPU。GPU学習は未承認・未実行。将来はexp218と同じDP/deterministic/thread設定を継承する。
- train cache / test feature regeneration の SHA 記録方針: cutoff manifest、fold manifest、distribution report、replay requestのcontent SHAとschema SHAを記録する。
- model manifest / prediction / submission SHA 記録方針: 静的/readout段階はmodel/prediction/submissionなし。将来学習時にmodel、OOF prediction、submission SHAを追加する。
- Kaggle package bootstrap 確認方針: prepare後にbootstrap内configと正本configのselected mode、seed、cap、kernel sourceを照合する。

## リスク

- リークリスク: synthetic cutoff後のtrue TVTをfeature生成に使う、またはvalid wellの派生sampleをouter-trainへ入れる危険。mask contractとsource-well fold assertionで防ぐ。
- CV/LB 不一致リスク: pseudo-tail区間の改善がofficial-startやhidden testへ転移しない。採用主指標をofficial-start OOFに固定し、hidden-like、near、1000+、worst-well guardを併用する。
- ランタイム/メモリリスク: exp218全prefix依存特徴のmulti-cut再生成は大きい。well/cutoff/rows capを先に適用し、manifest audit後に生成を段階実行する。
- 再現性リスク: event detectorのlibrary/version差、同点candidate順、gzip metadata差。detector parameter固定、stable sort/content SHA、decompressed SHAで防ぐ。

### v12 trial submission design

- v11 train kernel outputの15 LightGBM modelをmetrics記載SHAで全件検証する。
- v11のordered 380-feature schemaをexp218 train manifestから復元した推論schemaと完全一致させる。
- current-test特徴はexp218の保存済み推論関数でtarget-free再生成し、modelだけv11へ差し替える。
- exp218 controlやv11を再学習しない。inferenceは15 modelの単純平均を使う。
- submission生成後にsample submission互換性、NaN/Inf、ID順、SHAを確認してから提出する。
