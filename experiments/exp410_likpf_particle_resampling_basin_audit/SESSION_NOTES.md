# exp410_likpf_particle_resampling_basin_audit セッションノート

## 目的

exp072 likelihood-PF の長いvertical offsetが、HMMと同じtransition / prior
hysteresisだけで説明できるか、PF固有のinitial support、GR emission、ESS
resampling、particle extinction、within-seed平均、128 seed平均、support / clampの
どこで形成されるかを内部粒子状態から直接特定する。

## 現在の状態

- Route: `pf_beam`
- 状態: full 496-well観察監査、12-well ×12-variant counterfactual、
  strict merge、実験横断記録を完了
- 親実験: exp072
- prediction candidate / CV / LB: なし。原因診断専用
- ユーザー承認: 2026-07-26「実行してよい。重い実験はKaggleで実行」

## 事前固定asset

exp235 merged v3に保存されたexact exp072 `exp072_likpf_mean`をfloat32固定値として、
`abs(error) > 10 ft`が128行以上連続するPF-specific episodeを作成した。HMMの
episode assetは流用していない。

- source: 3,783,989 rows / 773 wells
- PF persistent target: 496 wells / 839 episodes / 819,288 episode rows
- target suffix rows: 2,500,744
- episode SSE: 453,149,095.6093302
- source raw SHA:
  `b9f6e3aab91478410dbba9f3779b3e9e90421641516892f9f451088c2c89c0bf`
- source decompressed SHA:
  `6376acff1762b438c0bf173da3fc8c3fc6feebad692d1e3b4eb2628b0c0ae0e5`
- fixed prediction subset logical SHA:
  `b3aa657aeb24be33e710098823bd52ddf95c8484eefdda7160edee9c26198c5f`
- episode asset SHA:
  `0bffbc6bd6d89fdbfa11aa86419df8dbd84b1819af717413f0ae3bfa82799804`
- target-well asset SHA:
  `ae9d4fd7429ad2b51f8d620a0413e74cb55efccefbb30cce0e2031230548f30e`

4 shardはsuffix row数でdeterministic LPT balanceした。

| shard | wells | suffix rows |
|---:|---:|---:|
| 0 | 124 | 625,454 |
| 1 | 124 | 624,728 |
| 2 | 124 | 625,302 |
| 3 | 124 | 625,260 |

## 事前登録した排他的原因

1. initial condition support miss
2. transition propagation escape
3. observed / imputed GR emission
4. resampling particle extinction
5. within-seed particle-mean multiplicity
6. across-seed arithmetic-mean multiplicity
7. support / clamp shortage
8. mixed / unresolved

primary thresholdはbasin radius 5 ft、mass floor 1%、log-ratio effect `ln(3)`、
dominant fraction 50%。3/5/10 ft、0.2/1/5%、`ln(1.1/1.5/3)`、25/50/75%は
独立sensitivityとして保存する。

## 実装と再現性

- compact self-contained candidate:
  `exp410_likpf_particle_resampling_basin_audit_compact_selfcontained_train.py`
- exp243 v3のexact exp072 kernelと同じ500 particles × 128 seeds、stable
  `sha256("likpf::train::<well>") % 2147483647 + 1 + seed_index`。
- transition、Gaussian GR likelihood、normalization、ESS 0.5 resampling、
  systematic draw、roughening、arithmetic seed meanは変更しない。
- diagnosticsはstage形成後のparticleを読み取るだけで、RNG callを追加しない。
- 合成24 rows / 32 particles / 4 seedsでexp243 exact kernelと
  `array_equal=True / max_abs=0.0`を確認した。
- truth / fixed candidate / episode maskはdiagnostic aggregationにだけ使い、PF
  dynamicsやprediction outputを分岐させない。
- CPU-only、Numba one worker、well-level stable seed。GPU / network / inference /
  submissionなし。
- Jupytext変換、`--test`、構文check、ruff F821、`make validate-exp`: PASS。

## Kaggle push前コスト確認

### preflight

- active scientific PF variant: 1（変更なしbaseline diagnostic）
- selected wells: 4（各full shardの最長suffix wellを1件）
- PF well-runs: 4
- particles × seeds: 500 × 128
- LightGBM configs / folds / boosters / models: `0 / 0 / 0 / 0`
- HMM / Beam / GPU: `0 / 0 / 0`
- parent/control replay: 4 wells。既存outputにparticle stage / ancestryがないため必要
- inference / submission: `0 / 0`

### full（preflight PASS後）

- active scientific PF variant: 1
- full PF well-runs: 496（4 shard ×124、各well 1 replay）
- preflightを含む総PF well-runs: 500
- particles × seeds: 500 × 128
- LightGBM configs / folds / boosters / models: `0 / 0 / 0 / 0`
- HMM / Beam / GPU: `0 / 0 / 0`
- inference / submission: `0 / 0`
- exp243 full 773 wellsの実測10時間18分からbaselineのみは各shard約1時間42分。
  diagnostic overheadを含む保守見積りは各4時間、hard guard 9時間。

## コマンドログ

```bash
.venv/bin/python experiments/exp410_likpf_particle_resampling_basin_audit/build_pf_persistent_assets.py
make validate-exp EXP=exp410_likpf_particle_resampling_basin_audit
kaggle kernels push -p experiments/exp410_likpf_particle_resampling_basin_audit/kaggle/train_aggregate
```

### preflight v1

- kernel:
  `kentookumura/exp410-likpf-particle-audit-preflight` version 1
- status: ERROR（PF 1 well目のparity guardで意図どおり停止）
- fixed input検証:
  2,500,744 rows / 496 wells / logical SHA一致
- selected:
  4 wells / 39,874 suffix rows / 11 episodes / 7,970 episode rows
- failure:
  `ea3a0e38` replay float64 meanとsaved float32値のmax差
  `0.00048824253644852433 ft`を、float32保存前の値で比較した。
- 原因:
  TVT約10,000 ftにおけるfloat32半ULP。RNG / state replayの不一致ではない。
  exp243 exact parityは`predictions.mean(...).astype(np.float32)`を保存してから
  fixed float32と比較している。
- 修正:
  exp072 persistence contractと同じくseed meanをfloat32化してからparity判定する。
  particle diagnosticsとRNG順は変更しない。
- asset側true TVTもexp072 score surfaceと同じfloat32へ固定し直した。この厳密化で
  episode数 / target well数は不変、episode rowsは819,320から819,288、
  SSEは453,152,306.0363から453,149,095.6093になった。
- 併せてfull用threshold sensitivityを、条件ごとの全ledger再scanから
  episode一回切り出しへ機械的に最適化した。threshold / classificationは不変。

### preflight v2

- kernel version 2
- status: ERROR（4 well目の旧episode SSE guardで停止）
- fixed input logical SHA: PASS
- persisted replay parity:
  `ea3a0e38 / d90aa14c / 374be387` の3 wells連続でmax abs `0.0 ft`
- runtime:
  `94.8 / 88.3 / 84.4 sec`（最長suffix well群）
- peak RSS: `1.981 GB`
- failure:
  旧asset `96ae5806:000` SSE `186883.5650`に対しKaggle float32 truthでは
  `186884.3261`。v2 push後に検出していたCSV truth float64再読込差と一致する。
- 対応:
  source `true_tvt`と`last_known_tvt + target`を全3,783,989 rowsでfloat32比較し、
  mismatch `0 / max_abs 0.0`を確認。float32正規化済みfinal assetでv3を準備した。

### preflight v3

- kernel version 3 / status COMPLETE
- selected: 4 wells / 39,874 suffix rows / 11 episodes / 7,970 episode rows
- fixed prediction input SHA / asset SHA / strict coverage: PASS
- persisted replay parity:
  4 / 4 wells、max abs `0.0 ft`、mean RMSE diff `0.0 ft`
- per-well runtime:
  `94.6 / 88.1 / 84.3 / 87.5 sec`、mean `88.638 sec`
- total elapsed: `619.209 sec`（全input SHA読込約261.5秒を含む）
- peak RSS: `1.980 GB`
- all-496 sequential projection: `22,236.19 sec = 6.177 h`
- 4 balanced shard projection: 約1.55時間＋各input読込、保守的に約1.7時間
- preflight限定の排他的分類:
  within-seed particle mean 7 episodes / SSE 58.14%、
  across-seed aggregation 2 / 35.66%、mixed 2 / 6.20%。
  これは最長4 wellsだけなのでfull結論には使わない。

### full shard push

- code / config / asset SHAはpreflight v3と同一。
- shard0: `kentookumura/exp410-likpf-particle-audit-shard0` version 1
- shard1: `kentookumura/exp410-likpf-particle-audit-shard1` version 1
- shard2: `kentookumura/exp410-likpf-particle-audit-shard2` version 1
- shard3: `kentookumura/exp410-likpf-particle-audit-shard3` version 1
- 各124 wells、suffix rowsは625,454 / 624,728 / 625,302 / 625,260。
- 4 CPU kernelsを並列push。GPU / inference / submissionなし。

### full shard結果とstrict merge

- 4 kernels: すべてversion 1 / COMPLETE
- strict coverage:
  496 wells / 2,500,744 suffix rows / 839 episodes /
  819,288 episode rows / episode SSE 453,149,095.6093302
- persisted replay parity:
  496 / 496 wells、max abs `0.0 ft`、failed `0`
- shard elapsed:
  `6261.156 / 5890.934 / 5764.288 / 5481.023 sec`
- peak RSS: `1.984 GB`
- primary exclusive cause SSE:
  support / clamp 36.4701%、across-seed mean 36.2441%、
  within-seed mean 10.8561%、transition 10.7177%、observed GR 3.6664%、
  mixed 1.2880%、resampling extinction 0.7577%。
- full mergeの巨大merged ledger 2本は、二重目のgzip出力中にローカルmemory
  limitでexit 137になったため正規artifactにせず削除した。4つのdownload済み
  shard row ledgerとraw/decompressed SHAをcanonical row evidenceとして保持し、
  strict集計は同じ値・閾値のまま重複wide ledgerを再保存せずPASSした。

## full結果前に固定したsentinel counterfactual

排他的causeごとにepisode SSE最大の未選択wellを1件ずつ取り、残りをglobal
episode SSE順で最大12 wellsまで埋める。全量cause比率を読む前に、以下の12
paired variantsを固定した。

1. baseline
2. momentum 1.0
3. transition process noise 0（RNG drawは消費）
4. transition process noise 3x
5. initialization spread 3x
6. GR sigma 1.3x
7. GR sigma 3x
8. GR sigma 1,000,000x（emission near-disabled）
9. resampling threshold 0.1
10. resampling disabled
11. resampling roughening 10x
12. typewell端GR一定延長によるclamp margin 2x

- 最大sentinel wells: 12
- active scientific PF variants: 12
- counterfactual preflight: shard 0先頭1 well ×12 variants = 12 PF well-runs
- counterfactual full: 12 wells ×12 variants = 144 PF well-runs（4 CPU shard）
- counterfactual最大合計: 156 PF well-runs
- particles × seeds: 500 ×128
- LightGBM configs / folds / boosters / models: `0 / 0 / 0 / 0`
- HMM / Beam / GPU / inference / submission: `0 / 0 / 0 / 0 / 0`
- baseline再生はsentinelでseed trajectory / log-likelihood readoutを得るための12 runs
  だけ。full 496-well controlをvariantごとに再生しない。
- 同一baseline bankからarithmetic mean、seed median、full-suffix likelihood
  best / weighted、連続32 seeds ×4 blocks、truth-best oracleを追加計算する。
- within-seed平均の直接対照として、各seed・rowのpost-update particleを5 ft幅・
  half-bin shiftの重み付きhistogram modeへ縮約し、128 seed mode mean / medianを
  保存する。24 rows / 32 particles / 4 seedsの合成系列で元augmented kernelと
  prediction、log-likelihood、ESS、resampling fractionがすべて
  `array_equal=True / max_abs=0.0`、mode全値finiteを確認した。
- truth-bestはoracle、full-suffix likelihood selectorはtarget-freeだがoffline診断。
  candidate CVや推論候補として扱わない。

### 固定規則を適用したcounterfactual cost確定

- sentinel asset:
  12 wells / 68,312 suffix rows / 16 episodes / 55,104 episode rows
- sentinel SHA / content SHA:
  `7e8491d4e1cde59caaed12c638451615b8f113c42811dd7f70f356afe0cf9a04`
- shard suffix rows:
  `16,762 / 17,457 / 17,305 / 16,788`
- counterfactual preflight:
  shard0先頭 `86454a6f` 1 well ×12 variants = 12 PF well-runs
- full counterfactual:
  12 wells ×12 variants =144 PF well-runs、4 CPU shards
- preflight込み:
  156 PF well-runs
- particles × seeds: 500 ×128
- LightGBM configs / folds / boosters / models: `0 / 0 / 0 / 0`
- HMM / Beam / GPU / inference / submission: `0 / 0 / 0 / 0 / 0`

### counterfactual preflight push

- 最初のkernel ID
  `kentookumura/exp410-likpf-particle-audit-counterfactual-preflight` はslug /
  titleが52文字で、Kaggle API `400 Bad Request`。kernelは作成・実行されず
  PF well-runは0。
- 短縮ID:
  `kentookumura/exp410-pf-counterfactual-preflight` version 1
- private / CPU / internet off / run-on-push
- bootstrap:
  50 files、config SHA
  `4a10a0479c42f10c9d59f35ad73c5fd4f4072091981488ab33ca38964f8b6a65`、
  runner SHA
  `ad45d435252e6b62cd384ca3be9bca11305fcb838e1b5f4285430d5e710002e4`、
  sentinel SHA `7e8491d4e1cde59caaed12c638451615b8f113c42811dd7f70f356afe0cf9a04`
  がembedded manifestと一致。
- status: `COMPLETE`
- coverage:
  1 well / 7,964 suffix rows / 2 episodes / 7,774 episode rows /
  12 variants。variant metricsは22 readoutsを両scopeで44行、episode metricsは
  2 episodes ×22 readoutsで44行。
- baseline persisted prediction parity max abs: `0.0 ft`、failed wells: `0`
- runtime:
  elapsed 1,130.971 sec、sum variant-well 887.080 sec、peak RSS 1.981 GB
- guard:
  selected well / episode coverage、truth非使用、no LightGBM / fold / booster /
  GPU / inference / submissionをすべてPASS。
- この1 wellだけでは介入効果を一般化しない。特にresampling無効化は、以後の
  resampling由来RNG消費も変えるため、機構介入と確率経路変更の複合効果である。
  固定12 wellsでpaired集計と4 seed-block安定性を確認してから解釈する。

### counterfactual full push

- 4 CPU shards、各3 wells ×12 variants =36 PF well-runs、version 1を並列push。
- 実行slug（title正規化によりshard番号前へハイフンが入った）:
  `kentookumura/exp410-pf-counterfactual-shard-0` …
  `kentookumura/exp410-pf-counterfactual-shard-3`
- private / CPU / internet off / run-on-push
- 全packageでconfig / runner / baseline kernel SHAがpreflightと一致し、
  `make validate-exp`をpush前にPASS。

### counterfactual full結果とstrict merge

- 4 kernels: すべてversion 1 / COMPLETE
- output:
  `kaggle/output/counterfactual_shard0_v1` …
  `kaggle/output/counterfactual_shard3_v1`
- strict coverage:
  12 wells / 16 episodes / 55,104 episode rows / 12 variants /
  144 PF well-runs / 22 readouts
- baseline persisted replay parity:
  12 / 12 wells、max abs `0.0 ft`、failed `0`
- implementation identity:
  baseline kernel
  `0fb334969a47cd375889590c758292f6f3f2566154174e0bb4bbd97518298050`、
  config
  `4a10a0479c42f10c9d59f35ad73c5fd4f4072091981488ab33ca38964f8b6a65`、
  runner
  `ad45d435252e6b62cd384ca3be9bca11305fcb838e1b5f4285430d5e710002e4`、
  variant contract
  `4ef9ab14aa75ba23ce0b8d1d7457b1359b76e22df45c8c667993500269d89be9`
  が4 shardで一致。
- guards:
  four-shard / twelve-well / twelve-variant / episode-row coverage、
  duplicateなし、baseline parity、implementation SHA、全shard guardをPASS。
- shard elapsed:
  `1880.797 / 2067.560 / 1356.260 / 2118.808 sec`
- sum variant-well runtime: `6506.945 sec`
- peak RSS: `1.983 GB`

baseline episode SSEは`113,223,940.6212`、RMSE `45.3291 ft`。
主要paired episode SSE比は以下。

| variant / readout | SSE比 | episodes改善 | wells改善 |
| --- | ---: | ---: | ---: |
| truth-best seed oracle | 0.281167 | 16 / 16 | 12 / 12 |
| roughening 10倍 | 0.752997 | 10 / 16 | 8 / 12 |
| suffix likelihood best seed | 0.828952 | 10 / 16 | 8 / 12 |
| likelihood weighted seed mean | 0.830963 | 10 / 16 | 8 / 12 |
| process noise 3倍 | 0.891691 | 11 / 16 | 8 / 12 |
| initial spread 3倍 | 0.927031 | 10 / 16 | 7 / 12 |
| resampling threshold 0.1 | 0.962925 | 9 / 16 | 6 / 12 |
| clamp margin 2倍 | 1.000000 | 0 / 16 | 0 / 12 |
| particle mode seed mean | 1.000591 | 5 / 16 | 4 / 12 |
| seed median | 1.002721 | 6 / 16 | 5 / 12 |
| GR sigma 3倍 | 1.023661 | 11 / 16 | 9 / 12 |
| momentum 1.0 | 1.025531 | 10 / 16 | 8 / 12 |
| GR sigma 1.3倍 | 1.140067 | 7 / 16 | 5 / 12 |
| resampling無効 | 3.480912 | 8 / 16 | 6 / 12 |
| process noise 0 | 6.265855 | 4 / 16 | 3 / 12 |
| GRほぼ無効 | 8.835072 | 2 / 16 | 1 / 12 |

- hard clamp拡張は全16 episodesで完全同値。support分類の実体はclampでなく
  finite particle support。
- GRほぼ無効は14 / 16 episodes、11 / 12 wellsを悪化させ、符号検定
  `p=0.00418 / 0.00635`。一部のalias triggerはあってもGRは全体では修正力。
- GR sigma 3倍は多数wellを改善する一方、少数のcatastrophic outlierでepisode pooled
  `1.023661倍`、全suffix `1.309279倍`。exp400/404と同じくglobal緩和は安全でない。
- roughening 10倍はpooled `0.752997倍`、全leave-one-well-out pooled比
  `0.602572–0.826203`だが、episode / well符号検定
  `p=0.45450 / 0.38770`。原因別にはacross-seed `0.083539倍`、
  support不足`0.872倍`、transition `0.842倍`を改善する一方、
  within-seed平均`1.836倍`、resampling extinction `1.920倍`へ悪化。
- process noise 3倍もpooled `0.891691倍`、全leave-one-well-outで改善するが、
  aggregate gainの約80%が最大gain wellへ集中。
- resampling無効化は以後のRNG / roughening経路まで変える複合介入で、pooled
  `3.480912倍`。即時resampling displacementがほぼ0、majority extinctionが0、
  explicit resampling episodeへの3介入も改善しないため、単純な
  resampling extinction主因説は棄却。
- particle mode / seed medianは改善せず、単純な平均からmode / medianへの置換では
  finite-support / basin mixing問題を解けない。
- fixed sentinelは原因代表＋SSE上位のtarget-late 12 wellsであり、符号検定も
  roughening / process noiseでは有意でない。介入結果は機構確認で、CV /
  prediction candidateではない。

## 最終判断

PFとHMMはoffset区間の53.8737%が重なり、重なる区間の誤差方向は90.2655%一致する。
したがって同じGR / geometry曖昧区間を踏む。ただし内部mechanism family一致は
8.4071%に過ぎない。HMMはforward transition / prior hysteresisとbackward
smoothing、PFはfinite particle supportとwithin/across-seed算術平均が主因。

PFの通常resamplingが一行で粒子をwrong basinへ固定するのではなく、有限粒子が
truth basinを十分維持できない状態と、残った複数basinをparticle / seed平均することが
長いoffsetを作る。roughening counterfactualから、resampling時の多様性とその後の
genealogyは一部大誤差の因果レバーだが、直接のresampling extinctionを主因とはしない。

本実験は原因診断として完了。model / prediction candidate / inference / submissionは
作成しない。roughening 10倍またはprocess noise 3倍を予測候補として扱う場合は、
target-late sentinelから独立した全OOF単一variant検証を別実験・別承認で行う。

## 次のアクション

1. Kaggle CPU 4 shard version 1を完了まで監視し、outputをstrict mergeする。
2. `result.md`、`metrics.json`、`experiment_summary.md`、
   `backlog/KAGGLE_DIRECTION.md`、steeringを更新して最終validateする。
