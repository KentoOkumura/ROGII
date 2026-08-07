# 設計

## アプローチ

exp368の成果は二つに分ける。

1. 固定2-state reliabilityをPF尤度へ組み込む根拠は不足した。
   known-prefix predictive NLL gainは`0.037356% < 1%`、row-weighted weak
   massは`0.009689 < 0.02`であり、exp368 Stage 1は閉鎖を維持する。
2. 保存suffix blockではbad10 AUC `0.636675`、circular差`+0.058264`、
   5/5 folds、hidden-like 2面もPASSしており、連続risk readoutとしては
   独立した情報を持つ可能性がある。

そこでexp401は、weak posteriorが単に`likpf_mean`の誤差を当てるだけでなく、
exp264の既存scoreが指名する別候補でその誤差を回復できる区間を識別するかを
先に確認する。Stage 0ではselectorを学習せず、保存済みstrict-nested
outer-valid scoreとlate truthを使う決定論的readoutだけを行う。

## 仮説

exp368のtarget-freeな連続weak riskは、exp264の既存
`pred_abs_error` / `p_within10`だけでは説明し切れない
「`likpf_mean`がbad10で、既存scoreが指名した別候補ならwithin10へ戻せる」
状態を識別する。これがfold、circular control、hidden-like、既存selector
margin条件付きでも再現すれば、後段selectorへ1列だけadd-onlyする価値がある。

## 実験範囲

- 対象実験: `exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264`
- Route: `ml_model`
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- 補助入力: `exp368_marginalized_reliability_pf`
- 変更する変数: exp264 selector contextへ将来追加する可能性がある
  `ctx__exp368_weak_risk` 1列だけ
- 固定する変数:
  - exp264 corrected Stage C v6のfold、12候補、88 feature schema、
    2 objectives、LightGBM設定、sampling、candidate order、legal domains
  - exp368のblock 512 / stride 256 / tail keep、transition、
    initial probability、normal/weak sigma倍率、saved weak posterior
  - exp226 foldとexp115 hidden-like assignment
- 現在のscope: Stage 0 implementation-only。template由来の正規notebook /
  `settings.py`は未編集placeholderのまま、別名compact self-contained
  train候補とfail-closed inference候補を実装する。

## Stage 0: zero-booster candidate-advantage readout

### 入力

- exp368 target-free block ledger:
  decompressed content SHA
  `7327ce8e6383d76f99c51cec6982c1db181e6f05257df28e7268d7a0549ba30a`
- exp368 target-free weak posterior blocks:
  decompressed content SHA
  `4ffa4fc761fc4db6b1c7de42c132b8102e33f9910bf5dc56752b20e95c2520ae`
- exp264 corrected Stage C v6
  `nested_outer_valid_candidate_score.parquet`:
  45,407,868 candidate-long rows、12 candidates × 3,783,989 rows、
  logical content SHA
  `a10b7848127f01bef522f4b17dfd1640c9784956892dc24fc1159e3869500abc`
- exp226 fold assignment:
  decompressed content SHA
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- exp115 hidden-like assignment:
  SHA
  `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`

### feature freeze

`ctx__exp368_weak_risk`は各suffix rowをcoverする全blockの
`weak_posterior_mean`の算術平均とする。512-row block / 256-row strideで
重複するblockはすべて等重み、tail blockも保持する。変換、平滑化、clip、
threshold、quantile選択は行わない。

negative control `ctx__exp368_circular_weak_risk`はexp368に保存された
stable within-well nonzero circular block shiftを同じrow集約で作る。
row feature、circular feature、candidate domain、fold、schema、
logical-content SHAをfreezeするまでsuffix truth / errorを読まない。

### legal domainとlabel

- primary: exp264 `primitive_pair_bank` 11候補
- secondary: exp264 `primitive_fixed_bank` 7候補
- anchor: 両domainとも`likpf_mean`
- nominated other: 各domainでanchor以外の`pred_abs_error`最小候補。
  tieは`candidate_contract.yaml`の宣言順で固定する。
- 評価cohort: `abs(likpf_mean - true_tvt) >= 10 ft`
- primary label `nominated_recovery10`:
  nominated otherのactual absolute errorが`<10 ft`
- secondary label `oracle_recoverable10`:
  同じlegal domain内のいずれかのother candidateが`<10 ft`
- realized advantage:
  `likpf_abs_error - nominated_other_abs_error`
- selector margin:
  `likpf_pred_abs_error - nominated_other_pred_abs_error`

`oracle_recoverable10`はheadroom診断だけに使い、candidate選択、feature、
threshold、合否救済には使わない。

### 既存selectorを超える情報の確認

各outer-valid foldについて、残り4 foldsのtarget-free selector marginから
decile境界を固定し、valid foldへ適用する。同じmargin decile内だけで
concordant positive/negative pairを数える
`margin_conditional_auc`を計算する。これによりweak riskが既存
`pred_abs_error` marginの単なる言い換えでないかを確認する。

weak-risk quartile境界も残り4 foldsのweak riskだけから固定し、
valid foldへ適用する。Q4-Q1は`realized_advantage_ft`の平均差とする。

### Technical gate（全AND）

- rows / wells / blocks / foldsが
  `3,783,989 / 773 / 15,174 / 5`
- exp264 candidate-longが45,407,868 rows、各rowで12候補が宣言順に存在
- primary / secondary domainが11 / 7候補で、12候補単一top1を生成しない
- row coverage 100%、real/circular risk finiteかつ`[0,1]`
- truth/error column read before feature freezeが0
- Stage 0のmodel config / trained fold / booster / PF / predictionがすべて0
- primary、各fold、hidden-like 2面でpositive/negativeが各1,000 rows以上
- input / schema / feature / readout logical-content SHAを保存

### Scientific gate（全AND）

primary `primitive_pair_bank`で次をすべて満たす。

- pooled `nominated_recovery10` AUC `>=0.60`
- real minus circular AUC `>=0.02`
- real AUC `>0.50`が4/5 folds以上
- hidden-like spatial / typewell-purged AUCがそれぞれ`>=0.55`
- pooled `margin_conditional_auc >=0.55`かつ`>0.50`が4/5 folds以上
- Q4-Q1 mean realized advantage `>=0.50 ft`かつ正が4/5 folds以上
- secondary `primitive_fixed_bank`でpooled real AUC `>=0.52`、
  real minus circular AUC `>=0.00`、Q4-Q1 advantage `>=0.00 ft`

一つでもFAILならbranchを閉じる。AUCの反転、threshold、block、domain、
候補subset、metric、gateの事後調整で救済しない。

## Stage 1: selector add-only（条件付き・未承認）

Stage 0全gate PASS後、別の実装・実行承認を得た場合だけ、exp264 corrected
Stage Cを同一contractで再学習する。

- treatment: `ctx__exp368_weak_risk` 1列だけを88列へ追加
- variants: 1
- objectives: 2 (`pred_abs_error`, `p_within10`)
- outer folds / inner folds: 5 / 4
- LightGBM configs: 2 objective contractを持つ1 selector config
- CPU selector boosters: 1 × 2 × 5 × 4 = 40
- saved exp264 control再学習: 0
- PF/HMM/Beam replay: 0
- downstream TVT booster、GPU、inference、submission: 0

学習前にraw current testから同じ1列を再生成し、14,151 rows / 3 wellsの
finite・100% coverageと、raw train replayが保存exp368 row featureの
logical-content SHAに一致することを必須とする。current-testの`likpf_mean`
pathは`kentookumura/exp263-last-anchor-pair-cache-inference`の
`current_test_formula_parity.parquet`を使い、静的submissionは使わない。

Stage 1 promotionは、保存exp264 v6 controlに対して次を全ANDとする。

- expected-error MAE、within10 logloss、within10 Brierがpooled改善かつ
  各4/5 folds以上で改善
- primary hard-domain RMSE gain `>=0.05 ft`かつ4/5 folds以上で改善
- 1000+、hidden-like spatial、hidden-like typewell-purgedの各RMSE回帰
  `<=0.02 ft`
- by-well p95回帰`<=0.02 ft`、worst-well回帰`<=0.25 ft`
- leakage、raw-test parity、40 model SHA、25 compact partition contractがPASS

PASSしてもdownstream TVT Stage D、inference、submissionは自動で行わず、
別実験または別承認で判断する。

## 再現性設計

- seed policy: Stage 0はRNGなし。Stage 1はexp264のseed 42とdeterministic
  LightGBM設定を完全継承する。
- stochastic 処理の有無: Stage 0なし。Stage 1も新しいstochastic featureなし。
- PF/Beam / likelihood-PF / seed baggingの有無: 保存済み値のload-only。
  replay、particle、seed baggingは0。
- 並列処理と乱数の関係: Stage 0はCPU single worker。Stage 1はexp264の
  deterministic / force_col_wise設定と固定thread数を継承する。
- CPU/GPU runtime: Stage 0 / Stage 1ともCPU、GPUなし。Stage 1は未承認。
- train cache / test regeneration: gzipはdecompressed logical content、
  Parquetはschemaとlogical contentを主証拠とし、raw train replayと
  raw current-test生成manifestを保存する。
- model / prediction / submission: Stage 0は非該当。Stage 1は40 model
  manifest SHAとcandidate-score / compact partition SHAだけを保存する。
  TVT prediction / submissionは生成しない。
- Kaggle bootstrap: 将来package化する場合だけ、kernel source、
  metadata、embedded config、internet/GPU off、run-on-push falseをpush前に照合する。

## リスク

- リークリスク: exp368 riskはtruth前にfreezeする。candidate actual error、
  oracle、bad10、hidden-like resultをfeatureやthresholdへ戻さない。
- 選択バイアス: 12候補を一つのhard domainにせず、既存2 legal domainsを維持する。
  oracleはsecondary診断に限定する。
- 冗長性リスク: exp368 riskがexp264 pred-error marginの言い換えに過ぎない可能性を
  margin-conditional AUCでfail closedする。
- CV/LB不一致: Stage 0はselector価値のpreflightであり、downstream RMSEやLB改善を
  主張しない。hidden-like 2面とraw-test parityを必須にする。
- ランタイム/メモリ: 45.4M candidate-long rowsを読むためchunked scanが必要。
  Stage 0ではwide copyを作らず、row-level riskと必要列だけを保持する。
- 再現性: gzip raw SHAはmetadataで揺れるためlogical contentを正とする。
- 解釈リスク: exp368 Stage 0 FAILを覆さず、PF尤度変更の根拠には流用しない。

## 2026-07-26 implementation-only設計追記

- 45,407,868 candidate-long rowsはParquet row group単位で読み、
  `id/well/fold/candidate order/downstream outer-valid/nested_model_count=4`
  を各groupで検証する。
- row-level surfaceはprimary / secondaryごとのnominated code、candidate TVT、
  anchor TVT、両predicted error、selector marginだけを常駐する。
  oracle headroomに必要な12候補TVTは一時float32 memmapとし、readout後に削除する。
- overlap row risk、fold、cross-fit weak quartile、domain別cross-fit margin decile、
  feature schema/content SHA、selector surface SHA、scientific contract SHAを
  `TruthAccessLedger`へfreezeした後だけ`tvt_true`を読む。
- train候補はinput preflightから生成物/SHAまでセルへ展開し、同一exp helper
  importを使わない。inference候補はsubmission非生成で常にfail closedとする。
- 正規Notebook採用、Kaggle package/push/runは別承認とする。
