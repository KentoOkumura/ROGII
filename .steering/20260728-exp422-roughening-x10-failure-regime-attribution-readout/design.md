# 設計

## 結論

「場所によってrougheningを変える」という仮説を、まず保存済みexp416の失敗原因
readoutとして検証する。ただしexp416が保存したPF診断はwell単位の集約値であり、
時刻ごとのESS / resampling履歴ではない。そのため本実験が定義する「場所」は
**target-freeなwell regime**である。suffix内の位置は固定scopeとして可視化するが、
row-level roughening scheduleやadaptive gateは作らない。

exp416の局所回復を説明する軸を「粒子崩壊からの回復圧力」、全体破壊を説明する軸を
「観測なしでノイズが伝播する損傷露出」として1つずつ固定する。正解を見てから都合のよい
単独指標や閾値を選ばず、2軸の事前固定AND gateだけで原因帰属を判定する。

## 実験範囲

- 対象実験: `exp422_roughening_x10_failure_regime_attribution_readout`
- Route: `pf_beam`
- 親: `exp416_roughening_x10_likpf_full_oof_ablation`
- scientific control: 保存済みexp072 `likpf_mean`
- reporting fold / truth: 保存済みexp226
- 実行内容: 保存生成物のjoin、target-free regime freeze、統計readoutだけ
- 新規prediction / PF / model / booster / HMM / Beam / GPU: すべて0

exp416は`roughening_x10_rejected_close_without_rescue`のまま保持する。本実験の
PASSはexp416の昇格ではなく、別の将来仮説を設計するための原因証拠に限る。

## 入力と読込順

### Phase 0: source contract

1. exp416 merge kernel version 2、id_no、artifact manifest SHAを照合する。
2. manifestに記録された各fileのraw SHAを照合する。
3. exp416 scientific contract、summary、gateからterminal FAILと固定結果を照合する。

### Phase 1: target-free freeze

truth-derivedなby-well / episode metricsをまだ読まず、次だけを読む。

- `merged_well_audit.csv`
- `merged_candidate_predictions.csv.gz`のID、well、row、suffix位置、
  `md_since`、raw-GR observed、保存candidate列
- exp226の安全列`well_id / row_idx / suffix_offset / fold`

well feature、fold-safe経験分布順位、2 score、4 regime cell、固定row scopeを作り、
schema / logical content SHAをfreezeする。candidate値はparity用に保持するが、
regime定義式へ入れない。

### Phase 2: outcome attachment

freeze後だけ次を読む。

- exp072保存control
- exp226 suffix truth
- exp416 by-well metrics
- exp416 persistent episode metrics

row-levelからRMSE、by-well gainを再計算し、exp416保存metricsへparity確認した後に
scientific attributionを評価する。

## 固定診断

各wellのraw診断は次のとおり。

1. `resampling_rate`
   = `resampling_count_total / (seeds * eval_rows)`
2. `ess_collapse`
   = `1 - clip(minimum_ess_mean / particles, 0, 1)`
3. `seed_prediction_dispersion`
   = `log1p(seed_prediction_std_mean)`
4. `seed_likelihood_gap`
   = `log1p(max(seed_loglik_best_per_row - seed_loglik_mean_per_row, 0))`
5. `eval_missing_fraction`
   = `eval_raw_gr_missing_rows / eval_rows`
6. `suffix_horizon`
   = `log1p(eval_rows)`

`position_clip_rate`、`prefix_missing_fraction`、`gr_scale_clipped`はsecondary mediator
として出力するが、primary scoreやPASS経路へ追加しない。

## fold-safe scoreとregime

fold `f`のwellを評価するとき、他4 foldsのwellだけをreferenceにする。
各raw診断値`x`の経験分布順位は、
`count(reference <= x) / len(reference)`を使う。reference側も同じ経験分布で
score化し、そのscore中央値をfold `f`の固定閾値にする。

### Recovery pressure

次の4 percentileを等重み平均する。

- resampling rate: 高いほど回復圧力が高い
- ESS collapse: 高いほど回復圧力が高い
- seed prediction dispersion: 高いほど回復圧力が高い
- seed likelihood gap: 高いほど回復圧力が高い

`recovery_pressure_score > outer-train median`をhighとする。

### Damage exposure

次の2 percentileを等重み平均する。

- eval missing fraction: 高いほど損傷露出が高い
- suffix horizon: 高いほど損傷露出が高い

`damage_exposure_score <= outer-train median`をlowとする。

primary target cellは
`high_recovery_pressure__low_damage_exposure`だけとする。中央値の不等号、scoreの
重み、変換、feature、cellを結果後に変更しない。

## 固定した位置readout

位置readoutは原因の見え方を説明するsecondary表であり、別PASS経路にしない。

- normalized suffix progress:
  `[0,0.25) / [0.25,0.50) / [0.50,0.75) / [0.75,1.00]`
- raw GR: `observed / missing`
- long propagation: `md_since <1000 / >=1000 ft`
- regime: 固定4 cells

suffix progressは`eval_rows=1`なら0、それ以外は
`suffix_offset / (eval_rows - 1)`とする。

## outcomeと統計

- row primary gain:
  `RMSE(exp072 control) - RMSE(exp416 roughening x10)`
- well primary gain:
  `control_rmse - candidate_rmse`
- episode gain:
  `control_sse - candidate_sse`
- 正はroughening x10改善、負は悪化

correlationはwell等重みSpearmanを使う。p値はgainをfold内だけで4096回置換する
片側検定とし、seedはfold IDからSHA256で固定する。2つの方向仮説にはBonferroni済み
`p<=0.025`を個別に要求する。

individual diagnostics、4 regime cells、fixed位置scope、by-well p95 / worstは
必ず保存するが、primary gateの代替候補として結果後に選ばない。

## scientific gate

全条件AND:

1. recovery pressureとwell gainのrho `>=0.10`、正方向`>=4/5 folds`、
   one-sided `p<=0.025`
2. damage exposureとwell gainのrho `<=-0.10`、負方向`>=4/5 folds`、
   one-sided `p<=0.025`
3. target cell row RMSE gain `>=0.05 ft`、改善`>=4/5 folds`
4. target cell minus restのwell等重みmean gain差`>=0.25 ft`、
   target cell改善well率`>=0.50`
5. target cellのpersistent episode support `>=4 episodes / >=3 wells`、
   SSE reduction `>=5%`、全improved episodeの正のSSE reduction説明率`>=50%`

PASS時のaction:
`target_free_regime_attribution_supported_separate_policy_experiment_required`

FAIL時のaction:
`no_reproducible_target_free_regime_close_attribution_branch`

## 再現性設計

- source identity:
  exp416 kernel version 2 / id_no / artifact manifest SHAを固定
- stochastic component:
  PFや学習はなし。4096回置換だけをSHA256由来PCG64 seedで固定
- parallel RNG:
  foldごとに独立seed、fold順やworker数に依存させない
- feature freeze:
  outcome読込前にwell feature、score、cell、row scopeのschema / logical SHAを保存
- gzip:
  raw gzip SHAとdecompressed CSV SHAを分け、内容証拠はdecompressed SHAを主とする
- output:
  readout table、gate、summary、manifestのSHAを記録
- deterministic anchor:
  保存OOF原因readoutであり、inference / submission anchorとは呼ばない
- package:
  実装承認後もKaggle metadata、embedded config、source kernel、stage、
  execution countをpush前に照合する

## リスク

- post-treatment diagnostics:
  resampling / ESS / seed spreadはroughening x10を走らせた後の診断であり、
  そのまま事前のadaptive gateへ使えるとは限らない。
- 時間解像度:
  exp416はwell集約診断しか保存していないため、row-level ESS triggerを原因と断定できない。
- outcome leakage:
  truth-derived metricsをfeature freeze前に読むとbucket選択が可能になるため、
  読込順をtechnical gateにする。
- selection:
  2 score、中央値、1 target cellを固定し、個別featureの最良結果へ差し替えない。
- causality:
  associationがPASSしてもrougheningをそのcellだけで有効化すれば改善する保証はない。
- small episode support:
  persistent episodeは16件しかないため、support下限と独立gateを設ける。

## 対象外

- roughening倍率、position / rate成分、process noise、ESS thresholdの探索
- target / errorを見たfeature、score、bucket、閾値、cell選択
- adaptive well / row roughening、router、selector、blend
- parent controlやcandidate PFの再実行
- same-OOF rescue、exp416の再分類
- raw-test inference、submission
