# 設計

## 1. 変更する面

exp501のfixed13 selectorを構成する13番目のraw exp490 predictionだけを、固定tau=500の
fade predictionへ置換する。fixed12 bankへ候補を追加するのではなく、候補数は13のままにする。

`d = md_since`、`p0 = exp357_parent_prediction`、`p1 = geometry_mean_reverting_hmm`として、

```text
w(d) = 1 - exp(-d / 500)
p_fade(d) = p0(d) + w(d) * (p1(d) - p0(d))
```

とする。`w`のclip、hard cutoff、alpha shrink、well別係数は加えない。`md_since`はexp263側の
target-free base contextを正とし、exp490保存列との全行parityを確認する。

Stage Cはfade fixed13 selectorそのものをraw exp501と比較する。Stage C gate通過後だけ、
Stage Dとしてexp413のnested selector compact74をfade selector compact77へ置換する。
これによりstandalone exp490やfinal TVTへの直接fadeを避け、物理候補をMLが解釈する形にする。

## 2. 入力契約

### exp490 / exp357 prediction

- source kernel: `kentookumura/exp490-mean-revert-full-merge`, version 1
- rows / wells: `3,783,989 / 773`
- file: `exp490_geometry_centered_mean_reverting_offset_hmm_stage1_full_oof_predictions.csv.gz`
- raw gzip SHA256: `99030b33d493cc5f195f7d1a867f0d812a539143da9e1f59277e53779261b72c`
- decompressed SHA256: `e020e82e748a7836085657c4058070ff7853ed285639f2c2555cab721f9e9a07`

feature freeze前のallowlistは次の8列に固定する。

1. `well`
2. `row_idx`
3. `suffix_offset`
4. `md_since`
5. `geometry_mean_reverting_hmm`
6. `exp357_parent_prediction`
7. `geometry_mean_reverting_delta_mean`
8. `geometry_mean_reverting_hmm_std`

`fold`、`true_tvt_readout_only`、candidate / parent / exp226 error、role、episode、scope、
by-well outcomeはfeature freeze前に読まない。exp490 source foldはsplitにもfeatureにも使わず、
exp263 outer foldへglobal keyで再partitionする。

### selector / downstream sources

- fixed12 / fold / Stage A親: exp263 / corrected exp264。
- raw fixed13比較: exp501 version 2。
- exp501 feature / model / compact / score SHA:
  `2eb780b9...63e96e / 3adb894d...cabbc / 32317a71...9c257 / 1641b9cb...1599e`。
- downstream control: exp413 saved Stage D OOF `7.884802794`、
  prediction SHA `9bd2d177...cef4a9d`。
- raw downstream negative reference: exp502 OOF `7.882143903`、exp413比gain
  `0.002658891 ft`、fold 3 / 4 delta `+0.116027 / +0.234686 ft`、hidden-like 2面
  `+0.139587 / +0.140944 ft`、OOF SHA `97230e2e...77ae99`。
- fold manifest: exp413 / exp501共通
  `fa41084c5fcb4adffb88d44211b4cc5d2d2f46b5bd4d65828b6af941184b2a6d`。

全入力は実装時に完全SHAでconfigとmanifestへ固定する。省略SHAは文書上の可読表示だけに使う。

## 3. Candidate / feature契約

- parent score candidates: 12。
- added candidate slot: 1。
- total score candidates: 13。
- 13番目のID: `exp490_tau500_fade_mean_reverting_hmm`。
- raw `exp490_geometry_mean_reverting_hmm`はbankへ残さない。
- fixed7 fallbackは変更しない。
- candidate IDは宣言順one-hot、ordinal index featureは禁止する。
- exp501のraw-test-safe context、candidate-long feature family、shape windows
  `[32, 128, 512]`を変更しない。
- native confidence 2列はraw HMM source diagnosticとしてそのまま保持し、新しいconfidence式を
  作らない。candidate prediction由来の距離・shape特徴は`p_fade`から再計算する。
- feature freeze後のall-missing / constant / exact-duplicateの機械dropだけを許可し、truthを使う
  feature selectionは禁止する。

近prefixで`p_fade`が`p0`へ近づくため、既存bankとのexact duplicate countとcandidate-bank
correlationをtechnical readoutへ必須化する。ただし検出後にcandidateを追加・削除しない。

## 4. Stage C: strict-nested selector

exp501と同じ2 objectivesを使う。

1. candidate absolute errorのL1 regression。
2. candidate within-10のbinary classification。

outer 5 × inner 4、seed 42、sampling、LightGBM config、early stopping、chunk sizeをexp501から
固定継承する。outer-train compactはinner OOF score、outer-valid compactは4 inner-model
ensembleで作る。1 variant × 2 objectives × 5 × 4 = 40 CPU boostersで、raw exp501 selectorと
parent fixed12は保存済み予測をcontrolとして読み、再学習しない。

Stage C freeze順序:

1. exp263 / exp490 input、key、suffix、md_since、fold SHAを検証する。
2. truthを読まずfade candidateとcandidate-long featuresを生成する。
3. feature schema/content SHAを凍結する。
4. outer-train / inner splitだけで40 selector modelsを学習する。
5. outer-valid score、choice、compact77、prediction SHAを凍結する。
6. parent exp264 / raw exp501 scoreとtruthを読み、RMSE / scope / by-wellを評価する。

Stage Cの全AND gateはrequirementsのとおり。tail改善はfixed12比deltaについてraw exp501から
p95 `0.10 ft`、worst `1.0 ft`以上の縮小を要求する。これはabsolute safety証明ではない。

## 5. Stage D: exp413への受け渡し

Stage C PASS後、かつ別の実装・GPU実行承認後だけ有効にする。

- exp413 clean base 273列: retain。
- exp413 nested selector compact74: remove。
- exp505 fade fixed13 compact77: insert。
- exp413 signed selector compact23: retain。
- final feature width: `273 + 77 + 23 = 373`。

old74とnew77の併存、concat、blend、row gate、selector-output gateは禁止する。exp413と同じ3
LightGBM configs、outer 5 folds、seed、GPU mode、early stopping、target、scopeを使い、
1 treatment × 3 configs × 5 = 15 GPU boostersだけを学習する。saved exp413 OOFをcontrolとし、
control、exp413 selector、signed selector、exp505 selectorはStage Dで再学習しない。

Stage Dの全AND gateはrequirementsのとおり。PASSは同じexp505内のinference設計を相談できる
状態にするだけで、current-test生成やsubmissionを許可しない。

## 6. 評価と必須readout

### Stage C

- fade direct prediction RMSE / fold / absolute-depth readoutとexp503 parity。
- raw exp501、fade exp505、fixed12のpooled / fold hard OOF RMSE。
- exp501固定7 scope。
- fade candidate top1率、fold別利用率。
- by-well improved / worsened、delta p50 / p90 / p95 / max、worst wells。
- raw exp490非top1時と同様のincumbent choice change率、margin、entropy。
- near-prefix duplicate / correlation readout。

### Stage D

- exp413 controlとのpooled / fold / fixed5 scope RMSE。
- by-well delta p50 / p90 / p95 / max、+1 / +3 / +5 ft悪化well数。
- final373 feature order、old74 instance 0、new77 instance 1。
- 3 config別とmean OOF、model manifest、feature importance。

## 7. 再現性設計

- seed: 42固定。outer / inner fold assignmentとcandidate順をSHA固定する。
- fade生成: deterministic、乱数0、stable row sort。
- Stage C LightGBM: CPU、`deterministic=true`、`force_col_wise=true`、`n_jobs=8`。
- Stage D LightGBM: exp413のGPU設定とseedを完全継承する。GPUはbitwise deterministicと
  仮定せず、model / OOF SHAとruntimeを記録する。
- PF / HMM / Beam / seed bagging: 再実行0。保存済みexp490 / exp357だけを使う。
- gzip: raw SHAとdecompressed content SHAを分け、後者を主証拠にする。
- Stage C: fade prediction、feature schema/content、40-model manifest、score、compact、OOF SHA。
- Stage D: final373 schema/content、15-model manifest、OOF SHA。
- inference / submission SHA: 実装されるまでnot applicable。
- Kaggle packageを将来作る場合、metadataとbootstrap ZIP内config、source kernel version、
  CPU/GPU、internet off、実行stageを照合する。
- tau=500はexp503 full OOFから選んだため、exp505のOOFが良くてもclean independent CVまたは
  deterministic submission anchorとは呼ばない。

## 8. リスク

- selection bias: tau=500は29-profile結果を見た後の固定値で、同じOOFを再利用する。
- tail risk: exp503ではfixed fade後もcatastrophic whole-well biasがほぼ残り、fold 0は全depthで
  exp490が悪化した。
- reranking risk: exp501はexp490非top1行でもincumbent choiceを35.0%変更した。fade候補でも
  candidate bank全体の順位が動く可能性がある。
- representation risk: Stage C hard OOF改善がStage D TVT改善へtransferする保証はない。
- observed transfer failure: raw exp501 compact77のexp502はpooledでほぼ同点でもfold 3 / 4と
  hidden-likeを悪化させた。fadeはこの原因を直接同定した対策ではない。
- CV/LB risk: exp490 standalone Public LB 9.680で、exp413 Public LB 7.201より弱い。final prediction
  への直接置換は行わない。
- runtime: Stage C 40 CPU boosters、Stage D 15 GPU boosters。Stage Dは別承認まで無効。
- reproducibility: GPU Stage Dはbitwise一致を保証しない。

## 9. 禁止事項と停止条件

- tau、alpha、cutoff、relative depth、fade関数を変更・探索しない。
- raw exp490とfade exp490を14候補として共存させない。
- exp498 / exp499のwell feature、truth-aware archetype、worst-well IDをgateへ使わない。
- candidate subset、feature subset、weight、threshold、objective、LightGBM configをsame-OOFで
  救済しない。
- direct final blend、standalone inference、exp490 submissionを行わない。
- parent selector、control、HMM/PF/Beamを再学習・再生成しない。
- Stage C FAIL時はStage Dへ進まずterminal closeする。
- Stage D FAIL時もblend / gate / add-only / model gridで救済しない。

## 10. 現在の状態

Stage CのJupytext percent形式compact self-contained train、候補/特徴量contract、contract
testを実装し、2026-08-03にKaggle private CPU version 1で40 modelsを完走した。technical、
pooled、4/5 folds、固定7 scope、fade利用はPASSしたが、p95 / worst material tail reductionを
FAILした。判定は`FAIL_CLOSE_WITHOUT_STAGE_D_OR_SAME_OOF_RESCUE`であり、Stage D、GPU、
inference、submissionを実装・実行せず終端閉鎖する。
