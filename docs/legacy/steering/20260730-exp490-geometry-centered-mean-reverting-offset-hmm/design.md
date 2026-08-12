# 設計

## アプローチ

exp357の予測を

`予測TVT = exp226 geometry予測 + residual offset`

と分解し、GRが選ぶresidual offsetに「放置するとgeometryへ戻る」事前分布を
加える。GRが継続して強い証拠を出す区間では尤度がoffsetを維持できる一方、
一時的な誤選択は遷移事前分布により徐々に0へ戻す。GR confidenceによる
hard resetや明示的gateは置かない。

## 状態と遷移式

行`t`について次を定義する。

- `G_t`: group-safeなexp226 `tvt_geop`
- `delta_t = TVT_t - G_t`: geometryからのresidual offset [ft]
- `q_t`: residual offset-rate [ft / MD-ft]
- `dMD_t = MD_t - MD_(t-1)`: 正のMD差
- `s(t)`: 行`t`が属する、exp226と同一のunknown-suffix K16区間
- `L_k`: 区間`k`へ入る全遷移の`dMD`合計。`L_k > 0`を必須とする

行ごとのgeometry復帰係数は次式に固定する。

`rho_t = 2 ^ (-dMD_t / L_s(t))`

したがって、1つのK16区間を通過すると、追加した平均回帰係数の累積積は
ちょうど`0.5`になる。

exp357のrate momentum `m = 0.998`を残し、遷移中心だけを次のように変更する。

`q_t の中心 = m × rho_t × q_(t-1)`

`delta_t の中心 = rho_t × delta_(t-1) + q_t × dMD_t`

`TVT_t = G_t + delta_t`

遷移確率の分散はexp357と同じく、rate方向`sig_r=0.002`、offset方向
`sig_p=0.02`とする。区間境界ではdestination rowの区間`s(t)`と`L_s(t)`を
用いる。最初のunknown rowでは、最後のknown-prefix rowからの`dMD`を使う。
`delta=0`かつ`q=0`なら遷移中心は常にgeometryそのものになる。

## 実験範囲

- 対象実験: `exp490_geometry_centered_mean_reverting_offset_hmm`
- Route: `pf_beam`
- 科学的親: `exp357_exp226_huber_emission_independent_audit`
- 構造参照: `exp281_exp226_residual_offset_exact_hmm_transition_probe`
- geometry参照: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 失敗機構参照: `exp408_hmm_message_rate_basin_audit`
- 変更する変数: `rho_t`をresidual offsetとrateの遷移中心へ乗じる
  geometry-centered mean reversion
- 固定する変数: Huber delta、GR emission、sigma、欠損処理、state grid、
  rate momentum、process noise、initial prior、posterior mean、fold、
  scope、評価指標、control予測

## 固定パラメータ

- residual offset grid: `[-80, 80] ft`、step `0.35 ft`
- residual rate states: `41`、span `±0.10`
- Huber delta: `1.345`
- known-prefix GR residual sigma: std、clip `[10, 60]`
- likelihood weight: `1.0`
- start offset: `0 ft`、start sigma: `0.75 ft`
- start rate: `0`、start rate sigma: `0.01`
- original rate momentum: `0.998`
- output: forward-backward posterior mean
- mean-reversion half-life: destination K16 segmentのMD span 1区間
- scientific variant数: `1`

## 段階評価

### Stage 0: fixed32 mechanism preflight

既存の
`experiments/exp411_predictive_filtered_rate_innovation_destick/assets/stage0_fixed32_manifest.csv`
をSHA固定で再利用する。persistent 16 wellsとmatched-control 16 wellsの計32 wellsを
candidateだけ1回decodeする。保存exp357予測はload-onlyで、再実行しない。
これはCVではなく、実装・runtime・機構の事前確認である。

technical gateは全ANDとする。

- manifest SHA一致、32/32 wells、16 persistent + 16 matched-control
- K16 segmentが全行を重複なく覆い、すべての`dMD`と`L_k`が正
- `0 < rho_t <= 1`、有限値coverage `1.0`
- 各完全K16区間の`rho_t`累積積が`0.5`に一致
- zero-state geometry identity、posterior normalization、prediction finiteが全PASS
- candidate freeze前のtruth / error / role / episode readは0
- full OOF runtime予測`<=30,600 sec`、peak RSS`<=25 GiB`

mechanism gateは保存exp357との比較で全ANDとする。

- persistent episode SSE reduction `>=5%`
- persistent 16 wells中、改善well `>=10`
- persistent側の改善fold `>=4/5`
- matched-control pooled RMSE回帰 `<=0.02 ft`
- matched-control by-well RMSE差のp95 `<=0.25 ft`
- persistent episode数は増やさず、256/512-row recovery rateを悪化させない

1項目でもFAILならbranchを閉じ、half-life、Huber、noise、grid、gateを
same-sampleで救済しない。全PASS時だけ、ユーザーへStage 1実行承認を求める。

### Stage 1: full group-safe OOF

Stage 0全PASSと別承認の後だけ、固定1 variantを773 wellsでdecodeする。
exp357、exp281、exp226 controlは保存予測を使い、再実行しない。

promotion gateは全ANDとする。

- exp357 RMSE `9.737195157482754`から`>=0.05 ft`改善
- 改善fold `>=4/5`
- 1000+、hidden-like spatial、hidden-like typewell-purgedの各RMSEを悪化させない
- by-well RMSE差p95 `<=0 ft`
- worst-well RMSE回帰 `<=0.25 ft`
- persistent episode SSE `>=5%`削減、episode数と256/512-row recoveryを悪化させない
- exp226 final RMSE `9.427109596582222`から`>=0.02 ft`改善
  （candidate RMSE `<=9.407109596582222`）
- row identityとfinite coverageがともに`1.0`

FAIL時はinference/submissionへ進まず、同一OOF上のhalf-life、delta、noise、
grid、confidence gate、hard reset、blend、selectorによるrescueを禁止する。

## 実行量

設計凍結・実装直後の実行量はすべて0。後続の明示承認後、Stage 0予定分だけを
Kaggle private CPUで実行した。

- Stage 0実績: 1 variant × 32 wells = 32 HMM well-runs
- Stage 1予定: 1 variant × 773 wells = 773 HMM well-runs
- reporting folds: 5
- LightGBM config / trained fold / booster / PF / Beam / GPU: すべて0
- exp357 / exp281 / exp226 control再実行: 0

Stage 0はtechnical 12/13、mechanism 6/7 PASSでfail-closedとなり、Stage 1予定分は
未実行のまま閉じた。

### 2026-07-31 Stage 1 override execution

ユーザーのfull-well明示承認により、Stage 0 FAILを覆い隠さず
`explicit_user_override_after_stage0_fail`としてStage 1を実行する。科学variantは
従来どおり1件で、保存controlの再decodeは0件とする。

単一CPU notebookの投影は`51,464.889494 sec`で上限外のため、wellだけを次の
target-freeな決定規則で4分割する。

`shard(well) = little_endian_uint64(sha256("exp490::full_well_shard::" + well)[0:8]) mod 4`

固定exp226 OOFのwell/row identityから得た分割量は次のとおりである。

| shard | wells | rows | Stage 0実測比例のruntime投影 |
| ---: | ---: | ---: | ---: |
| 0 | 192 | 950,473 | 12,927.096 sec |
| 1 | 204 | 986,223 | 13,413.321 sec |
| 2 | 182 | 890,131 | 12,106.402 sec |
| 3 | 195 | 957,162 | 13,018.071 sec |

各shardはraw/typewell、known prefix、保存exp226 `tvt_geop`だけを読み、candidate、
K16 contract、well manifest、decoder/input/prediction SHAを保存する。mergeは4 shardの
raw/decompressed SHAをconfigへ固定した後にだけ許可し、773 wells / 3,783,989 rowsの
disjoint union、stable shard再計算、finite、row identityを確認してから保存exp357の
fold/truth/controlを後付けする。mergeのHMM well-runsは0である。

Stage 1 gateは元の事前登録値を維持する。full CV結果がFAILしても同じOOF上で
half-life、noise、grid、gate、blend、selectorを救済しない。inference/submissionは
Stage 1とは別承認のままとする。

## 再現性設計

- seed policy: HMM本体は乱数なし。well、row、segment、stateの順序を固定する。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: なし。
- 並列処理: 外側well workerは1、Numba thread数を固定し、乱数を使わない。
- Stage 1 operational shard: stable SHA256 well modulo 4。candidate値には影響せず、
  shardごとに外側well worker 1、Numba thread 4を固定する。
- runtime: Kaggle private CPUを正とし、GPUとinternetは無効。
- input SHA: exp226 OOF、exp357 parent prediction、fixed32 manifest、
  hidden-like assignmentのdecompressed/content SHAを検証する。
- contract SHA: K16境界、`L_k`、`rho_t`、state grid、transition順序を含む
  decoder manifestを保存する。
- prediction SHA: raw gzip SHAに加えてdecompressed content SHAを主証拠にする。
- model SHA: 学習モデルがないためdecoder manifest SHAで代替する。
- submission SHA: inference/submission未承認のため対象外。

## 2026-08-02 hidden-dynamic inference version 2設計

### 変更面

version 1の公開test固定guardだけを置き換える。`sample_submission.csv`を読み、列契約、
nonempty、unique ID、`<well>_<row_idx>`構文を確認した後、rowsとwell集合をruntimeで導出する。
公開sample SHA / 14,151 rows / 3 wellsは`public_reference`へ移し、比較結果をmanifestへ
記録するが、一致を要求しない。

### hidden-dynamic identity gate

1. sampleから導出したwell集合と、raw test loaderが発見するwell集合を完全一致させる。
2. 各wellでsample row_idxと`TVT_input`欠損行を一致させる。
3. exp226 `PredictionResult.geop`のID集合とsample ID集合を一致させる。
4. exp490 predictionの行数 / well数は固定値でなくsample由来値と比較する。
5. 最終submissionはsample順序を保持し、unique / finiteを必須とする。

### 不変面

scientific contract SHA `6398bbac...`、exp226 source/config SHA、full-train 773 wells、
mean-reversion、K16、Huber、HMM state/rate、出力列`id,tvt`は不変とする。実行時学習は
exp226 full fit 1回だけで、exp226 geometry / exp490 HMM well-runsはruntime sample well数に等しい。

### 承認境界

実装、Jupytext notebook生成、静的・契約テストまでを今回の範囲とする。Kaggle version 2の
push/runとcompetition再提出は別承認までfail-closedとする。
- Kaggle package: 実装承認後にmetadata、bootstrap config、ローカルconfigの
  scientific contract一致をpush前に検証する。

## 2026-08-01 current-test inference override

Stage 1 safety gate FAILを保持したLB監査として、固定モデルをcurrent testへ移す。
current testはsample submission上で3 wells / 14,151 unknown-suffix rowsなので、
operational shardは置かず1 CPU notebookで順にdecodeする。

1. competition raw train 773 wellsとraw test 3 wellsを読む。
2. SHA固定したexp226 K16 sourceでfull-train field / kappaを1回fitし、各test wellの
   `PredictionResult.geop`を生成する。GR correction後の`pred`は使わない。
3. raw testのknown prefix、GR、typewellと`geop`だけをexp490 HMMへ渡す。
4. full OOFと同じK16 segment / `rho_t` / Huber exact forward-backwardを実行する。
5. sample `id`とcandidate `(well,row_idx)`の完全一致後、sample順の`submission.csv`を保存する。

technical AND gateは、exp226 source/config SHA、OOF scientific contract SHA、3 wells /
14,151 rows、sample ID/order、有限値、posterior normalization、segment half-life、
zero-state identity、GPU/internet off、submission schemaを含む。FAIL時は提出候補を
無効にし、LB用fallbackや再実行を行わない。

## リスク

- リークリスク: fixed32のpersistent/control roleは過去truth由来である。
  roleはcandidate freeze後の評価にだけ使い、transitionやparameter選択へ渡さない。
- 過適合リスク: half-lifeを1点固定し、Stage 0/1で再調整しない。
- CV/LB不一致リスク: exp226 geometryとGR尤度の誤りがtestで異なる可能性がある。
  direct OOF gateを通ってもinferenceとsubmissionは別承認とする。
- tailリスク: mean reversionが本当に必要なgeology offsetを消す可能性があるため、
  matched-control、p95、worst-well、hidden-likeを全ANDで判定する。
- ランタイム/メモリリスク: exact HMMは重い。fixed32でfull換算とRSSを先に測る。
- 数値リスク: 不規則MD、区間境界、極端に小さい`L_k`で`rho_t`が不安定になり得る。
  正値・finite・segment half-life sentinelをtechnical gateにする。
- 再現性リスク: exp357 saved artifactの取得元差異をSHAで拒否し、1回の成功だけで
  deterministic anchorとは呼ばない。
