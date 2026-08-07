# 設計

## アプローチ

exp340 は exp280 の raw Gaussian likelihood family を exp264 error に接続したが、primary bad10 AUC は 0.60 に届かなかった。一方、既知 14 common-Type-Well wells の事後診断では、target TVT に対する相関最良 shift と raw absolute Gaussian likelihood の最良 shift が一致しない例があり、絶対残差型 score が「形の一致」を十分に表していない可能性がある。

そこで、同じ exp226 path、512-row block、13-shift bank、exp264 late readout を固定し、raw finite horizontal GR と shifted Type Well GR の Pearson centered normalized correlation（ZNCC）だけを新しい observation score とする。`δ=0` を基準に、非ゼロ shift の相対優位が exp264 の bad block を target-free に予告するかを測る。

これは correction 実験ではなく confidence readout である。実験内で horizontal GR、TVT prediction、candidate、selector、model、submission は変更しない。

## 実験範囲

- 対象実験: `exp360_typewell_reference_shift_zncc_confidence_readout`
- Route: `ensemble`
- 親実験: `exp340_exp226_depth_alias_block_confidence_readout_on_exp264`
- 参照実験:
  - `exp280_exp226_rawgr_likelihood_confidence_readout_on_exp264`
  - `exp264_selector_compact_addonly_on_exp226`
  - `exp226_depth_alias_ensemble`
- 変更する変数:
  - observation score を exp280 の absolute residual Gaussian likelihood から raw-finite ZNCC に置換する。
  - `δ=0` と非ゼロ shift の相対 score、rank、shift continuity を preregistered family として出力する。
- 固定する変数:
  - exp226 OOF path、row order、fold、`tvt_geop`、exp264 prediction/error readout。
  - 512-row fixed non-overlapping block。
  - 13-shift bank `[-80, -40, -20, -10, -5, -2, 0, 2, 5, 10, 20, 40, 80] ft`。
  - 773 wells、5 folds、expected 7,787 blocks。
  - exp340 と同じ pooled / fold / 1000+ / hidden-like spatial/typewell-purged scopes。
  - exp264 block RMSE、row-weighted bad10、exp226 beats exp264 by 0.25 ft 以上の downstream labels。

## shift の定義

- 各 horizontal row の exp226 `tvt_geop = T` は固定する。
- candidate `δ` の expected GR は `GR_typewell(T + δ)` とする。
- horizontal の row、MD、GR を前後へ移動しない。
- `δ=0` が通常 matching、`δ≠0` が shifted Type Well reference view である。
- shift の符号は上式を唯一の正とし、実装時に synthetic monotone fixture で確認する。

## Stage 0 score 契約

### safe inputs

- horizontal well file は `MD`, `GR`, `TVT_input` だけを safe loader で読む。水平井の true `TVT` は score freeze 前に materialize しない。
- exp226 safe OOF view は `well_id`, `row_idx`, `suffix_offset`, `fold`, `tvt_geop` だけを読む。
- Type Well は `TVT`, `GR` を読み、TVT 昇順に整列する。
- exp264 actual/prediction/error と exp226 truth は late-join 専用 loader に分離する。

### expected GR

- Type Well 内の missing GR は exp280 と同じ forward fill / backward fill を使う。
- `GR_typewell(T + δ)` は線形補間し、範囲外は endpoint hold とする。
- horizontal GR は補間、平滑化、標準化済み値への置換を行わず、raw finite pair だけを使う。

### ZNCC

各 block `b`、shift `δ` について、finite observed pair 上で Pearson correlation を計算する。

```text
r[b, δ] =
  Σ (gr_obs - mean(gr_obs)) (gr_tw_shifted - mean(gr_tw_shifted))
  -----------------------------------------------------------------
  sqrt(Σ (gr_obs - mean(gr_obs))² Σ (gr_tw_shifted - mean(gr_tw_shifted))²)
```

- finite pair 数 `>=32`。
- observed / expected の標準偏差が双方 `>1e-6`。
- 条件未達 candidate は `valid=false`, score `-1.0` とする。
- core-supported block は `δ=0` が valid かつ valid candidate が2個以上の block とする。
- pair threshold、std threshold は preregistered 固定値で、結果を見て変更しない。

### tie policy

ZNCC と historical raw Gaussian control の双方で同じ deterministic tie policy を使う。

1. score が高い候補。
2. exact tie なら `δ=0`。
3. 次に `|δ|` が小さい候補。
4. 同じ `|δ|` なら negative shift。

実装上の exact tie 順は `[0, -2, +2, -5, +5, -10, +10, -20, +20, -40, +40, -80, +80]` とする。

## preregistered feature families

全 feature は値が高いほど exp264 error risk が高い向きに揃える。

| family | 定義 | 役割 |
| --- | --- | --- |
| `best_nonzero_minus_zero_zncc` | `max(r[δ≠0]) - r[0]` | 唯一の primary |
| `low_zero_shift_zncc` | `-r[0]` | supporting |
| `zero_shift_rank` | valid candidates 内の `δ=0` の降順 rank を `[0,1]` 化 | supporting |
| `absolute_top1_shift` | tie policy 後の `abs(argmax δ)` | supporting |
| `top1_shift_jump_from_previous_block` | 同一 well の前 block からの top1 shift の絶対差 | supporting |
| `three_block_sign_inconsistency` | 直近3 supported blocks の非ゼロ top1 shift の符号不一致率 | supporting |

- `best_zncc` と top1−top2 margin は secondary diagnostic として保存してよいが、promotion gate や primary 救済には使わない。
- feature family の追加、閾値探索、family blend、selector は exp360 では禁止する。

## controls

### historical raw Gaussian matched control

- exp280 の保存済み 13-shift raw Gaussian score bank を再利用する。
- 同じ block、tie policy、feature formula で対応 family を作り、ZNCC の incremental value を比較する。
- raw score を再計算しない。

### stable shift-label permutation control

- 各 well/block の凍結済み 13 ZNCC 値を、stable SHA256 key による deterministic permutation で shift label に割り当て直す。
- score distribution は保ち、`δ=0`、符号、絶対 shift との対応だけを破壊する。
- global RNG、Python hash、thread scheduling に依存しない。
- real と control で同一 feature formula、quantile/readout を使う。

## leakage barrier と freeze 順序

1. safe inputs と lineage SHA を検証する。
2. ZNCC score bank、valid mask、support mask を生成する。
3. real / historical raw / stable control の core features を生成する。
4. fold 内 Q1/Q4 境界を feature ごとに計算する。
5. input、schema、score、mask、control、feature、quantile、manifest の content SHA を凍結する。
6. freeze manifest と `truth_access_count=0` を検証する。
7. 初めて exp264 actual/prediction/error と exp226 truth を late join する。
8. preregistered readout を1回生成する。

## 評価と合否

primary は `best_nonzero_minus_zero_zncc` だけとし、以下を全て満たす場合だけ Stage 0 pass とする。

- technical:
  - 773 wells、5 folds、7,787 expected blocks と lineage が一致。
  - core-supported block coverage `>=0.98`。
  - 全773 wells に supported block が1つ以上。
  - truth / target / error / label の freeze 前 access が0。
  - fold Q1/Q4 が非重複で、全 frozen artifact の SHA が記録済み。
- scientific:
  - Q4−Q1 平均 block RMSE 差 `>=+0.50 ft`、中央値差 `>0`。
  - RMSE 差が正方向の fold が4/5以上。
  - pooled row-weighted bad10 AUC `>=0.60`。
  - bad10 AUC `>0.50` の fold が4/5以上。
  - 1000+ scope と hidden-like spatial/typewell-purged scope の双方で正方向。
  - historical raw Gaussian analog に対して pooled AUC gain `>=+0.02`、fold 改善4/5以上。
  - stable permutation control に対して pooled AUC gain `>=+0.02`、real 優位4/5以上。

primary が1条件でも失敗した場合は branch を閉じる。supporting family、sentinel wells、secondary metric、閾値や dense shift grid の追加で救済しない。

pass した場合も exp360 は readout のまま完了とし、別番号の add-only ML feature 実験を候補化する。random-shift prediction averaging、shifted candidate 追加、hard correction、selector は別仮説・別承認とする。

## 計算予算

- 実装時の variant: real ZNCC 1、stable control 1、保存済み raw baseline 1。
- feature families: core 6、fold 5。
- LightGBM config: 0。
- folds trained: 0。
- total boosters: 0。
- PF/Beam/HMM: 0。
- 親実験 control 再学習: 0。
- CPU-only、GPU / internet 不要、目標 wall time 30分以内。

## 再現性設計

- seed policy: real path は RNG なし。negative control だけ `stable_sha256_per_well_block` を使う。
- stochastic 処理の有無: 実質なし。control permutation も SHA key による完全決定的順序。
- PF/Beam / likelihood-PF / seed bagging の有無: 全てなし。
- 並列処理と乱数の関係: 並列化しても score と control key は `(well_id, block_index, shift)` だけで決まり、global RNG と completion order を使わない。
- CPU/GPU runtime と deterministic flags: CPU-only。well、block、shift の stable sort を保存前に強制する。GPU deterministic flag は非該当。
- train cache / test feature regeneration の SHA 記録方針: OOF readout のみ。入力 SHA、decompressed content SHA、score bank、valid mask、control、feature schema/content、quantile、late readout の SHA を記録する。test feature は生成しない。
- model manifest / prediction / submission SHA 記録方針: model、prediction、submission は生成しないため `not_applicable`。readout manifest と artifact SHA を記録する。
- Kaggle package bootstrap 確認方針: 実装・実行の承認後、Kaggle push 前に offline import と package bootstrap を確認する。設計段階では package を作らない。
- deterministic anchor: `false`。既存予測 anchor を変更しない target-free diagnostic である。

## リスク

- リークリスク: exp226 OOF artifact や horizontal file に true TVT が同居し得る。safe-column loader と freeze-before-truth barrier を必須にする。
- 事後選択リスク: 000d7d20 等から shift bank や閾値を合わせると過適合する。4 wells は non-gating sentinel に固定する。
- CV/LB 不一致リスク: score は train-only Type Well relationship を使うため、test の未知 shift 分布や typewell geometry が異なる可能性がある。1000+ と hidden-like purged scope を必須ゲートにする。
- score 解釈リスク: 高い非ゼロ ZNCC が真の TVT shift ではなく、周期的 GR や低分散 block の alias を表す可能性がある。finite-pair/std gate、raw/control 比較、fold consistency を要求する。
- ランタイム/メモリリスク: 7,787×13 score は小さいが、全 row×shift materialization を避け、well/block 単位で集約する。
- 再現性リスク: tie、NaN、sort、permutation の処理差で feature が変わり得る。exact tie order、invalid sentinel、stable sort、content SHA を固定する。
- branch proliferation リスク: exp340 の family rescue に戻る可能性がある。primary-only fail-closed とし、pass 後の prediction use も別実験・別承認に分離する。
