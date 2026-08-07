# 設計

## アプローチ

exp264 corrected Stage C v6のselectorをhard候補選択器ではなく、後段TVT LightGBMへ候補品質を伝えるstrict nested meta-feature生成器として拡張する。既存`pred_abs_error` / `p_within10`と74列compactは保存済みのまま固定し、候補別のsigned residual予測から23列だけをadd-onlyする。

主仮説は、誤差規模とwithin10確率だけでは欠けている「候補が真値より浅いか深いか」という方向情報を加えると、後段TVTモデルが候補値をより適切に補正・比較できる、である。

## 実験範囲

- 対象実験: `exp335_signed_residual_meta_on_exp264`
- Route: `ml_model`
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- 変更する変数:
  - selector objective `signed_residual = true_tvt - candidate_tvt`を1 head追加
  - signed residual OOFから固定23列を後段へadd-only
- 固定する変数:
  - exp264 corrected Stage A v4のraw-test-safe 88 selector特徴
  - exp264の12候補、candidate ID/order、2 legal domain、availability/confidence契約
  - outer 5 × inner 4 well-disjoint split、seed 42
  - saved corrected Stage C v6 compact 74列
  - clean 273 downstream特徴、3 LightGBM configs、5 folds、target、GPU設定
  - saved exp264 Stage D v3 OOF/metrics/by-well/modelをcontrolとして再利用

## 比較設計

| 役割 | 特徴surface | 学習量 | 用途 |
| --- | --- | ---: | --- |
| 保存control | clean 273 + saved compact 74 = 347 | 再学習0 | exp264 RMSE `8.460811237612477`とfold/scope/by-well比較 |
| 新variant | clean 273 + saved compact 74 + signed compact 23 = 370 | 3 configs × 5 folds = 15 GPU boosters | signed residual add-only価値の検証 |

controlと新variantの差はsigned residual head由来23列だけにする。exp287 formation 74列は混ぜず、selector仮説をexp264上で単独帰属できる形にする。

## Selector学習設計

### 教師

`signed_residual = true_tvt - candidate_tvt`

- model family: exp264と同じcandidate-long LightGBM
- objective: `regression_l2`
- selector input feature schema: corrected Stage A v4の88列を固定
- candidate rows: exp264の12候補を宣言順で展開
- model count: outer 5 × inner 4 × 1 objective = 20 CPU boosters
- 既存`pred_abs_error` / `p_within10` 40 model、非nested Stage B、candidate generatorの再実行: 0

### Strict nested出力

- outer-trainのsigned featureはinner OOF predictionだけから生成する。
- outer-validのsigned featureは、そのouter-train内で学習した4 inner model ensembleから生成する。
- outer-valid wellをfit、early stopping、imputation、prior、label transformへ使用しない。
- downstream outer-trainへ同じ行をfitに使ったselector predictionを渡さない。
- test実装は将来のscopeであり、学習時と同じcandidate order / schema / model ensembleを使うfail-closed contractだけを設計に残す。

## Signed compact 23列

### 1. Candidate-specific score: 12列

`selector__pred_signed_residual__<candidate_id>`

12候補すべてに1列ずつ持たせる。実際のsigned residual、true TVT、actual errorは保存score監査には含めても、後段feature tableへは含めない。

### 2. Existing top-1 annotation: 8列

`primitive_pair_bank` / `primitive_fixed_bank`の各domainと、既存`pred_abs_error` / `p_within10`の各objectiveについて次を追加する。

- `selector__<domain>__<objective>__signed_residual_at_top1`
- `selector__<domain>__<objective>__signed_corrected_top1_minus_anchor`

top-1 identityはsaved exp264 compactと同じ既存objectiveから復元し、signed headでは候補を選ばない。補正後値は`existing_top1_value + predicted_signed_residual - last_known_tvt`とする。

### 3. Distribution: 3列

- `selector__pred_signed_residual_mean`
- `selector__pred_signed_residual_std`
- `selector__pred_signed_residual_range`

以上の23列以外は本実験に追加しない。all-candidate corrected TVT、rolling集約、well集約、softmax average、hard path、Viterbiは別仮説として除外する。

## 実行単位と承認境界

### Stage S: selector

- active variant: 1 (`signed_residual_l2`)
- objectives: 1
- outer folds: 5
- inner folds: 4
- planned CPU boosters: 20
- existing selector/control retraining: 0

### Stage D: downstream TVT

- active variant: 1 (`signed_residual_meta_addonly`)
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- planned GPU boosters: 15
- saved exp264 control retraining: 0

設計確定時点で承認されたのは設計文書とscaffold作成だけだった。2026-07-21以降の段階的承認によりStage S実装、Kaggle package/push、0-booster preflight、Stage S CPU trainまで完了した。Stage Sはversion 3でtechnical/score gateをPASSした。2026-07-22にStage D実装と固定15 GPU booster実行が明示承認された。固定tail guard FAIL後の2026-07-23に保存済みmodel CPU inferenceとsubmission file生成・submit-checkだけが別承認され、version 3で完了した。外部competition submissionは未承認である。

Stage S実装は`exp335_signed_residual_meta_on_exp264_compact_selfcontained_train.py`をJupytext編集元とし、検証後に正規train notebookへ採用した。Stage Dは既存notebookを上書きせず`exp335_signed_residual_meta_on_exp264_tvt_train.py` / `.ipynb`を別kernel用に追加した。上位の設定、入力、承認、実行、metrics/生成物確認はnotebookセルに展開し、候補long生成、strict nested LightGBM、Parquet streaming、SHA/manifest、downstream 15-model学習とgate処理は再利用可能な`src/signed_residual_meta.py`へ置いた。

## Gate

### Stage S technical / score gate

- 20/20 model、outer5-inner4組合せ、well-disjoint、88-feature schema、12-candidate order、全partition/model SHAが一致する。
- 全OOF predictionがfinite、row/candidate coverage完全、label formula parityが成立する。
- candidate別outer-train mean residual priorに対し、pooled residual RMSEを改善する。
- 同priorに対して5 outer folds中4 folds以上でresidual RMSEを改善する。
- FAIL時はStage Dへ進まない。

### Downstream scientific support gate

- saved exp264 `8.460811237612477`比でpooled RMSEを`>=0.03 ft`改善する。
- 4/5 folds以上でsaved exp264以下。
- near / mid / 1000+ / hidden-like spatial / typewell-purgedを悪化させない。
- by-well delta p95 `<=0`、worst-well delta `<=+0.25 ft`。
- 23列のgain/split readoutを保存する。重要度は補助証拠で、RMSE/safety gateを代替しない。

### Train-side promotion gate

scientific supportに加え、clean 273 controlに対する既存exp264のtail guardを悪化させず、固定済みworst-well / `+1/+3/+5 ft`悪化well数条件をすべて満たすことを要求する。Public LB改善だけでtrain-side guardを上書きしない。

## 本実験に含めないもの

- 既存74列の置換、削除、再計算、selector control再学習
- exp287 formation 74列、exp334 equal-well weight、別downstream親への移植
- oracle label classifier、pairwise/listwise ranker、squared-error head、tail quantile head
- signed target clip、Huber、quantile、sample weight、candidate別config、objective grid
- hard selector、softmax candidate average、Viterbi、rolling/segment/well集約
- feature pruning、LightGBM config変更、seed/fold/target変更
- gate緩和、未承認の外部competition submission

## 再現性設計

- seed policy: exp264と同じfixed seed `42`。新しいrandom samplingは追加せず、必要なstable samplingは既存exp264 immutable key policyを固定する。
- stochastic処理の有無: selector CPU LightGBMとdownstream GPU LightGBM。新しいPF/Beam/candidate生成乱数は追加しない。
- PF/Beam / likelihood-PF / seed baggingの有無: 新規生成0。saved exp264 candidate/compactをSHA固定入力として読む。
- 並列処理と乱数の関係: selectorはfold/objective/candidate schemaからstable seedを固定し、global RNGやthread scheduling依存の乱数を追加しない。
- CPU/GPU runtimeとdeterministic flags: selectorはCPU、downstreamはexp264と同じT4、internet off、`gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、threads 8を固定する。GPU runをbitwise deterministic anchorとはみなさない。
- train cache / test feature regenerationのSHA記録方針: corrected Stage A v4 schema、candidate contract、Stage C v6 saved compact、clean273 allowlist、fold assignment、23-feature schema/order/content、25 role partitionをSHA記録する。test実装時はraw-test生成schema/contentを別監査する。
- model manifest / prediction / submission SHA記録方針: Stage S 20 model manifest、signed OOF、Stage D 15 model manifest、OOF prediction SHAを記録する。別承認後のCPU inferenceではcurrent-test predictionのdecompressed SHAとsubmission SHAを記録し、外部提出は行わない。
- Kaggle package bootstrap確認方針: push承認後、embedded config、support archive、notebook、metadata、CPU/T4、internet off、planned model count、control retraining 0を照合する。

## リスク

- リークリスク: signed residualはtarget由来なので、outer-train inner OOF / outer-valid inner ensemble以外の生成方法では後段stacking leakageになる。actual label/error/oracleを後段へ渡さない。
- 目的重複リスク: signed residual headが独立のTVT regressor化し、候補差ではなく共通biasだけを学ぶ可能性がある。candidate別score、分散、重要度集中、downstream matched OOFで判定する。
- CV/LB不一致リスク: exp264はtrain-side worst guard FAILでもPublic LB 7.562だった。LBだけで採用せず、fold/scope/by-well guardを優先する。
- ランタイム/メモリリスク: 将来の予定量は20 CPU selector boosters + 15 GPU downstream boosters。candidate-long predictionとpartitionをchunk処理し、既存74列を複製再生成しない。
- 再現性リスク: GPU LightGBMのbitwise一致は保証しない。保存済みcontrol再学習を避け、入力/schema/model/OOF SHAとconfigを記録する。

## 次の判断

Stage SはPASSし、別承認後のStage D version 2も15/15 modelsを完了した。pooled RMSEはsaved exp264比`0.314703 ft`改善したが、by-well p95 `+1.728657 ft`、worst-well `+10.238752 ft`、clean273 promotion tailが固定gateをFAILした。設計どおり既存74列を維持し、signed target/objective/23列schemaを同一OOF上で救済しない。後日のCPU inference overrideは保存済みmodelの推論とsubmit-checkだけで、非promote判断を変更しない。

## 2026-07-23 CPU推論設計

- 親のcorrected exp264 hidden-safe CPU inferenceを構成参照し、raw current testから12候補、21 confidence列、88 selector特徴、clean273 surfaceを同一runで再生成する。
- corrected exp264 Stage C v6の40 modelsからouter別saved74、exp335 Stage S v3の20 modelsからouter別signed23を生成する。signed23は各outerの4 inner model平均とし、saved74の既存top-1 identity parityを再検証する。
- exp335 Stage D v2の15 modelsへ、各modelと同じouterのclean273 + saved74 + signed23を渡す。GPU学習済みLightGBM text modelはCPU predictorで読み、学習やmodel変換を行わない。
- 入力model manifest、各model、feature schema、current-test prediction、submissionのSHAを保存する。public-test保存特徴は入力に使わない。
- notebookはcompetition submit APIを呼ばない。出力取得後のsubmit-checkまでを今回scopeとし、外部提出は別承認とする。

CPU inference version 3はKaggle CPU / internet offで`387.808 sec`、14,151 rows / 3 wellsを処理して完了した。40 parent selector、20 signed selector、15 TVT modelをすべてSHA検証し、clean273 + saved74 + signed23の370特徴、formula parity、signed top-1 parityを確認した。生成した`submission.csv`はsampleとのheader・行数・ID順、重複、NaN/Infの検査をWARN/FAILなしでPASSした。その後ユーザーがcode submissionを実施し、ref `54928806`はPublic LB `7.517`でCOMPLETEになった。このLB結果はsubmitted reference anchorを更新するが、固定tail guard FAILを上書きしない。
