# 設計

## アプローチ

1. exp235 merged row artifactに保存された exact exp072 `likpf_mean` とtrue TVTから、
   `abs(error) > 10 ft` が128行以上連続するPF固有episodeを固定する。
2. exp243でexact parityを確立したexp072 PF kernelを基準に、乱数呼び出し順を変えない
   augmented diagnostic kernelを作る。
3. 各seed・各rowで、予測propagation後、GR重み更新後、resampling後の粒子について
   truth basin / fixed-output basin mass、support、mean、rate、ESS、resampling、
   unique ancestor fraction、max offspring shareを集計する。truthは診断比較にのみ使い、
   dynamicsへは渡さない。
4. per-seed trajectoryから、truth-close seed率、best-seed error、seed spread、
   arithmetic mean penaltyを測り、within-seed粒子平均とacross-seed平均を分離する。
5. episode onset前128行を含めてfirst truth-basin escapeを追跡し、原因stageを時間順に
   同定する。後続の長いoffset row数だけでstageを誤認しないよう、first-lossと
   dominant-effectを両方保存する。
6. raw observed / imputed GR、error符号、fold、tail距離、episode長、seed、threshold
   sensitivityを統合し、事前固定した排他的分類を出す。
7. preflightは代表well、fullは対象wellをshardしてKaggle CPUで実行し、merge後に
   exact count / SHA / coverageを検証する。
8. full結果を読む前に、限定counterfactualのsentinel選択と介入を固定する。
   排他的causeごとにepisode SSE最大のwellを1件ずつ選び、重複を除いた残りを
   global episode SSE順で最大12 wellsまで埋める。各wellは最大SSE episodeだけで
   なく、そのwellの全固定episodeと全suffixをpaired評価する。
9. sentinelではbaselineに対して以下を一変数ずつ変更する。すべて500 particles ×
   128 seeds、同じwell seed、同じraw/interpolated GR契約で実行し、truthは評価だけに
   使う。
   - momentum `0.998 -> 1.0`
   - process noise `0`（RNG draw自体は消費）
   - process noise `3x`
   - initialization spread `4.5 -> 13.5 ft`
   - GR sigma `1.3x` / `3x` / `1,000,000x`（emission near-disabled）
   - resampling threshold `0.5 -> 0.1` / `0.0`
   - resampling roughening `10x`
   - typewell端のGR値を一定延長し、position clamp margin `100 -> 200 ft`
10. 同一baseline seed bankから、arithmetic seed mean、row-wise seed median、
    full-suffix GR likelihood最大seed、likelihood-weighted seed meanも評価する。
    さらにseed indexを連続32本ずつ4 blockへ固定分割した平均を保存し、
    mechanism readoutのseed-bank依存性を確認する。
    within-seed平均は各rowのparticle stateから5 ft幅・半bin shiftの重み付き
    histogram modeをtarget-freeに読み、per-seed particle modeの128 seed平均と
    中央値も保存する。mode読出しはestimate確定後でRNG callを追加しない。
    前3者のうちseed medianはtarget-free、likelihood readoutはtarget-freeだが
    suffix全体を使うoffline診断であり、prediction candidateとは扱わない。
    truth-best seedはoracle上限としてのみ分離する。

排他的分類の優先順位:

1. `initial_condition_support_miss`: suffix開始時点でtruth basin supportが不足。
2. `transition_propagation_escape`: emission前のpropagationで初めてtruth basinを失う。
3. `gr_emission_alias_or_imputation`: predictive supportは残るがGR更新がtruth oddsを
   固定閾値以上悪化させる。
4. `resampling_particle_extinction`: filtered supportは残るがresampling後に消失または
   ancestor concentrationが固定閾値を超える。
5. `within_seed_particle_mean_multiplicity`: truth-supporting粒子が残る一方、各seedの
   posterior meanがwrong basinへ落ちる。
6. `across_seed_aggregation_multiplicity`: truth-close seedが残る一方、128 seedの
   arithmetic meanでoffsetが形成される。
7. `support_or_clamp_shortage`: truthがposition clampまたは粒子support外。
8. `mixed_or_unresolved`: 複数stageまたは上記の固定条件で説明不能。

primary thresholdは basin radius 5 ft、mass floor 1%、log-odds effect `ln(3)`、
dominant row fraction 50%。感度は basin radius 3/5/10 ft、mass 0.2/1/5%、
effect `ln(1.1)`/`ln(1.5)`/`ln(3)`、row fraction 25/50/75%を独立集計する。

## 実験範囲

- 対象実験: `exp410_likpf_particle_resampling_basin_audit`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`
- 変更する変数: predictionを変えない内部計測、diagnostic aggregation、cause attribution
- 固定する変数: exp072 PF全パラメータ、500 particles、128 seeds、stable seed、
  pseudotail split、raw/interpolated GR契約、output arithmetic mean
- variants: diagnostic baseline 1。必要なcounterfactualはprediction候補ではなく、
  保存粒子からのtarget-late readoutまたは同一baseline replayのdisable auditとして扱う。
- sentinel counterfactual variants: baselineを含む12。最大12 wellsを4 CPU shardへ
  deterministicに割り当てる。full全496 wellsでのvariant sweepは行わない。
- counterfactualはまずshard 0の先頭sentinel 1 well ×12 variantsをKaggle CPU
  preflightし、Numba compile、baseline parity、artifact schema、runtime / RSSを
  確認してから12 wells ×12 variantsの4 shardへ進む。
- LightGBM configs / folds / boosters: `0 / 0 / 0`
- PF well-runs: preflight対象well数 + full対象well数。shard再実行は別途記録する。

## 再現性設計

- seed policy: `sha256("likpf::train::<well>") % 2147483647 + 1 + seed_index`
- stochastic 処理の有無: PF transition noise、initialization、systematic resampling、
  resampling rougheningがある。乱数呼び出し順をexp243 exact parity kernelから変更しない。
- PF/Beam / likelihood-PF / seed bagging の有無: likelihood-PF 500 particles × 128 seeds。
  BeamとHMMは実行しない。
- 並列処理と乱数の関係: 1 well内はNumba single worker。well-level shardはstable
  well seedにより順序非依存。thread scheduling依存のglobal RNGは使わない。
- CPU/GPU runtime と deterministic flags: Kaggle CPU-only、internet off、GPU off。
  同一shardの再実行parityをpreflightで確認する。
- train cache / test feature regeneration の SHA 記録方針: exp072 input、
  exp209 reconstructed controlまたはexp235固定 row artifact、episode / well assetの
  raw SHAとdecompressed/content SHAをmanifestへ保存する。
- model manifest / prediction / submission SHA 記録方針: model / submissionは非該当。
  固定 `likpf_mean` とdiagnostic replay predictionのID順content SHAを保存する。
- Kaggle package bootstrap 確認方針: local candidate、generated notebook bootstrap、
  pulled Kaggle notebookでconfig、asset bytes、seed `+1`、500×128、threshold、
  shard設定が一致することを確認する。

## リスク

- リークリスク: truthをparticle diagnosticで参照するため、dynamics関数にtruth/error/
  episode/fold/hidden-role引数を持たせない。fixed prediction / asset SHAを先にfreezeし、
  truth比較はstage particle state生成後の集計だけに限定する。
- CV/LB 不一致リスク: train-side cause auditであり提出候補ではない。full trainの
  pseudotail全773 wellsからPF-specific subsetを固定する。
- ランタイム/メモリリスク: full 773-well replayはexp243で約10時間18分。PF-offset
  subsetだけに絞り、preflight計測後に複数CPU shardへ分ける。particle stateは
  rowごとの集計に縮約し、全particle trajectoryを保存しない。
- 再現性リスク: augmented計測でRNG順が変わる、shard merge欠落、float32 roundtrip、
  exp072 / exp209 source mismatch。全対象well parity、strict well/row coverage、
  float64 replay、input SHA guardで停止させる。
- 解釈リスク: offset後のsupport消失を原因と誤認する可能性がある。onset前128行と
  first escape stageを主証拠にし、episode全体の相関は補助証拠とする。
