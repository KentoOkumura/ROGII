# 設計

## 1. 識別する仮説

exp435の失敗は「TVT-only状態への縮約」そのものではなく、縮約後のtransition centerが
毎行ゼロrate、すなわち`-ΔZ`へ固定されたことが主因かを識別する。

変更する変数は一つだけである。

```text
control:   mu(t) = -delta_Z(t)
candidate: mu(t) = delta_tvt_geop(t)
```

candidateの`delta_tvt_geop`は、保存済みfold-safe exp226 geometry-only pathの
隣接差である。exp226のfinal predictionをunaryやblendとして利用するのではなく、
周辺井戸geometryが与える局所一行増分だけをexp435の状態遷移へ直接入れる。

## 2. 系譜と既存実験との差

- 親`exp435`: persistent stateはTVT分布だけ、GR emissionとforward-backwardを継承する。
- evidence parent `exp226`: outer-train周辺井戸から作ったK16 `tvt_geop`だけを入力にする。
- `exp355`との差: exp355はjoint `(TVT, U-rate)` HMMのrate-prior meanを変更した。
  exp437はrate stateを持たず、TVT transition centerを直接変更する。
- `exp394`との差: exp394はgeometry branchとfull-grid HMM branchをsoft-stickyに
  同時周辺化した。exp437はbranch stateを持たない単一TVT chainである。
- `exp436`との差: exp436は新しいglobal sparse potentialを解くdirect predictorである。
  exp437は保存exp226 geometry incrementとexp435 HMMだけを使う。

exp355/394/435の既存FAILを再分類せず、本介入の結果だけを評価する。

## 3. Fold-safe geometry schedule

入力は保存済みexp226 OOF
`exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_train_oof_predictions.csv.gz`
とする。decompressed content SHAは
`709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`へ固定する。

candidate freeze前のallowlistは次の5列だけとする。

```text
well_id, row_idx, suffix_offset, tvt_geop, fold
```

禁止列は`TVT/tvt_true/tvt_pred/gr_delta/error/abs_error`である。列全体を読んでから
dropする実装は禁止し、`usecols`相当でread時に遮断する。

wellごとに`row_idx`をstable sortし、raw targetの最後のfinite
`TVT_input`を`g(-1)`、保存`tvt_geop`を`g(t)`として、

```text
mu_geo(0) = g(0) - g(-1)
mu_geo(t) = g(t) - g(t-1), t >= 1
```

を作る。期待row count、連続row、fold、anchor、finite、first-step parityを確認し、
schedule schema/logical content SHAをtruth前にfreezeする。

将来のhidden-test inferenceでは保存OOFを使えないため、Stage 1 promotion後に別承認で
exp226 geometry fieldをfull trainからraw testへ再生成する。今回は設計・train OOF評価だけで、
inference実装は行わない。

## 4. TVT-only HMM

exp435から次を固定する。

- TVT grid step `0.35 ft`、last-known band pad `100 ft`。
- position noise `sig_p=0.02`とfive-cell transition kernel。
- start sigma `0.75`、Gaussian GR emission、`lambda=1.0`。
- typewell GR、known-prefix calibration、missing-GR interpolation。
- TVT probability vectorだけをpersistent stateとするforward-backward。

candidateではrate supportを構築せず、各行のposition kernelを`mu_geo(t)`へ直接中心化する。
process noise、kernel幅、emission、grid、start priorは変更しない。

```text
p(TVT_t | TVT_{t-1})
  = K_5cell(TVT_t - TVT_{t-1}; center=mu_geo(t), sigma=sig_p)
```

rate posterior、rate diagnostic、rate責任の次行持越しは存在しない。geometry scheduleは
target suffix truthやGR emission outcomeによって更新しない。

## 5. Stage 0: fixed32 mechanism preflight

exp435と同じfixed32 manifest
`fbbc62b7cb79e16a7fb436f3a9d11f8975e935ad2475a17e2dec4fd7b142e4d6`
を使う。16 persistent wellsと16 matched controls、156,088 suffix rowsを対象に、
新candidate 1本×32 wells=`32 HMM well-runs`だけを実行する。

保存control:

- exp435 `dz_only_r0` / `memoryless_41rate` stage0 prediction。
- exp435 prediction logical SHA
  `aa79810f6c189dd7fbb9d53b8c172a4a051d29ac1780ee4696237e8c24e214c3`。
- exp226 `tvt_geop`を同じ32 wellsへrestrictしたgeometry path。

保存exp226 OOFの事前read-only再集計値は、candidate freeze後のtruth join時に再計算して
parity確認する。

| scope | exp226 `tvt_geop` RMSE |
| --- | ---: |
| fixed32 all | `9.267204778` |
| matched control 16 | `8.719886308` |
| persistent 16 | `9.768805034` |

technical gate:

- 32 wells / 156,088 rows、finite coverage `1.0`。
- source foldとmanifest fold一致率`1.0`、duplicate/missing row `0`。
- forbidden列、suffix truth、role/fold/episode errorのpre-freeze read `0`。
- geometry schedule first-difference parity最大絶対差`<=1e-10 ft`。
- transition row-sum / posterior normalization error各`<=1e-6`。
- schedule/prediction readback logical SHA一致。
- full 773-well候補runtime投影`<=3,600 sec`、peak RSS`<=8 GB`。

mechanism AND gate:

- candidate all32 RMSEがexp226 geometryより`>=0.10 ft`改善。
- matched-control candidate delta vs exp226 geometry `<=+0.02 ft`。
- persistent candidateがexp226 geometryより`>=0.10 ft`改善。
- exp435 dz-only matched-control RMSEより`>=1.0 ft`改善。
- exp226 geometry比で改善fold`>=4/5`。
- paired by-well delta p95`<=+0.25 ft`、worst`<=+2.0 ft`。

fixed32は誤差情報から選ばれたmechanism sampleであり、CVまたはpromotion evidenceと
呼ばない。一条件でもFAILならStage 1へ進まず、transition scale/noise、geometry
schedule、emission、grid、gate、subsetを同じfixed32で救済しない。

## 6. Stage 1: full group-safe OOF

Stage 0全gate PASSと別のユーザー承認後だけ、保存geometry scheduleを全
`773 wells / 3,783,989 rows`へ適用する。新candidate 1本×773 wells、
parent/control rerun、model、booster、PF、Beam、GPUは0とする。

primary controlは保存exp226 final OOF `9.427109596582213`、mechanism controlは
保存exp226 geometry `10.077950290784381`とする。

promotion AND gate:

- candidate pooled RMSE`<=9.377109596582213`
  （exp226 final比`>=0.05 ft`改善）。
- exp226 geometry比`>=0.20 ft`改善。
- exp226 final比で改善fold`>=4/5`。
- suffix 1000+ RMSE gain`>=0.05 ft`。
- hidden-like spatial / typewell-purged delta各`<=0.0 ft`。
- suffix 0--250 delta`<=+0.02 ft`。
- paired by-well delta p95`<=+0.25 ft`、worst`<=+2.0 ft`。

exp263 fixed physical blend CV `8.238331`はreport-only referenceとし、candidate、
weight、gateの選択には使わない。全gate PASSでもraw-test regeneration、
inference、submission、blendは別承認とする。

## 7. 再現性

- RNGなし。well/fold/row/variant順と浮動小数点reduction順を固定する。
- worker、Numba、BLAS thread数を固定し、runtime versionをmanifestへ保存する。
- raw input、exp226 OOF、exp435 control、fixed32 manifestのfile/decompressed SHAを確認する。
- allowlist schema、geometry schedule、prediction、diagnostic、well/fold/scope metrics、
  scientific contractのlogical content SHAを保存する。
- gzipはraw SHAではなくdecompressed content SHAを主証拠とする。
- Kaggle package作成時はbootstrap内configとsource SHAを正規ファイルへ照合する。
- 初回runはdeterministic anchorではない。同一設定rerunでscheduleとprediction SHAが
  一致した後に再判定する。

## 8. 確定範囲

- scientific candidateは`neighbor_geometry_direct_transition` 1本。
- Stage 0は32 HMM well-runs、Stage 1最大773 HMM well-runs。
- parent/control HMM rerun、ML model、LightGBM config、trained fold、booster、
  PF、Beam、GPUはすべて0。
- RuntimeはKaggle private CPU、internet off。
- 初回設計時点では実装・実行可能notebook cell・test・package・run・inference・
  submissionは0だった。
- 2026-07-29の明示実装指示により、compact self-contained Stage 0 train、
  正規train notebook、専用contract test、fail-closed inference guardを実装した。
- 実装完了時点では`run_hmm/create_prediction=false`、Stage 0実行承認falseを
  維持した。その後の明示実行指示でStage 0だけを一時的に許可し、Kaggle private
  CPU version 1（id_no `129056603`）を完了した。
- technical gateは全PASSしたが、candidateはfixed32 allでexp226 geometryより
  `+3.751804309 ft`、persistent 16で`+6.823650264 ft`悪化し、
  mechanism gateをFAILした。
- Stage 1、raw-test再生成、実推論、submissionへ進まず、
  `run_hmm/create_prediction=false`と承認falseへ再ロックした。
