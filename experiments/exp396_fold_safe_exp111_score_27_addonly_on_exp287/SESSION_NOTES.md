# exp396_fold_safe_exp111_score_27_addonly_on_exp287 セッションノート

## 目的

exp264のavailability auditでnon-OOFとして除外されたexp111 score系27列を、downstream TVT outer foldに
対してstrict nestedとなる形で再生成し、exp287へadd-onlyする実験設計を確定する。

## 現在の状態

- Route: `ml_model`
- 状態: Stage B 15/15 boosters完了・固定promotion gate FAIL・branch閉鎖
- CV / LB: OOF RMSE `8.134294735`、Public / Private LBは未提出
- 実装承認: 2026-07-25のユーザー指示 `exp396を実装してください`
- Kaggle package / push / Stage A preflight承認: あり
- 40 CPU booster train承認: あり
- Stage B実装 / 15 GPU booster train承認: あり
- inference / submission承認: なし

## 2026-07-25 設計ログ

- `make new-steering EXP=exp396_fold_safe_exp111_score_27_addonly_on_exp287` でsteering scaffoldを作成した。
- `make new-exp EXP=exp396_fold_safe_exp111_score_27_addonly_on_exp287 SOURCE=templates/experiment` で
  experiment scaffoldを作成した。
- 親をexp287、clean tail controlをcorrected exp264、scorer仕様参照をexp111に固定した。
- 旧fold0 scorerの全train適用は再利用せず、outer 5 × inner 4 × 2目的のstrict nested化を選んだ。
- current-testはfull-data refitせず、downstream outer foldごとの保存済み4 inner scorer平均に固定した。
- Stage A全PASS後だけStage Bを候補化するCPU→GPU二段階設計にした。
- notebook、helper、test、package、学習、推論、提出は作成・変更・実行していない。

## 2026-07-25 実装ログ

- 後続のユーザー指示をStage A実装だけの承認として記録し、preflight/train、
  Kaggle package/push、Stage B、inference、submissionの承認とは分離した。
- `exp396_fold_safe_exp111_score_27_addonly_on_exp287_compact_selfcontained_train.py`
  に10章のJupytext percent形式候補を実装した。
- exp111の候補5、row特徴32、candidate-long特徴16、2目的、model設定を固定し、
  保存済みexp111 fold0 modelのprediction load口を作らなかった。
- outer 5 × inner 4のwell-group split、outer-train inner OOF、outer-valid 4-model平均、
  SHA256 local sample、inner-train固有48 median、40 CPU model manifestを実装した。
- targetをtarget-free feature builderへ渡す前にtrue TVT配列へ分離し、特徴量側で
  protected label列を受け取った場合はfail-closedにした。
- 10 score coreから固定順の27 float32列を導出し、model/schema/sample/median/partition/
  logical content SHAとStage A technical/quality/resource全AND gateを実装した。
- inference候補はcurrent-test、prediction、submissionを生成しないfail-closed contractだけとした。
- 既存の正規train/inference `.ipynb` は上書きせず、別名candidate `.ipynb` へ変換した。
- 専用testは10件PASS。非連番indexのinner fold、Pandas copy-on-write下のmedian補完、
  target分離、tie、schema/hash、quality gate、実行承認境界を含む。
- `py_compile`、Ruff `F821/F401/E9`、Jupytext `--test` をPASSした。
- 親exp287の正規Jupytext trainは7章・362行でhelper orchestration中心。exp396候補は
  10章でStage Aの上位ロジックとartifact/gateをセル上から追跡可能にした。
- Kaggle inputは読まず、booster 0、prediction 0、submission 0。科学的metricは未取得。

## 2026-07-25 Stage A 0-booster preflight承認・pre-push監査

- ユーザーに「exp287のTVT回帰へ27特徴を追加し、selector自体は変更しない」ことを再確認し、
  `実行してください` の指示を正規train Notebook採用、private CPU package/push、
  Stage A 0-booster preflightの承認として記録した。
- 今回の実行量はtechnical preflight 1件、outer folds 5、inner manifest 20行、
  LightGBM config 0、trained fold 0、CPU/GPU booster 0、control再学習0。
- 40 CPU scorer boosters、Stage B 15 GPU boosters、inference、submissionは未承認のまま。
- train候補を正規train Notebookへ採用した。candidate/canonical SHA256はともに
  `c89d6dce607083204791e83bfdef8db21595d95e549de67f9cd0bef591715d01`。
- Kaggle入力kernelをmetadata/filesで確認した。
  - exp099: `kentookumura/exp099-pf-multiobs-likelihood-train`、id_no `124135047`
  - exp287: `kentookumura/exp287-foldsafe-form74-addonly-exp264-train`、id_no `127856426`
  - corrected exp264: `kentookumura/exp264-exp263-confidence-dual-selector-tvt-train`、
    id_no `127577193`
- canonical kernel id/titleを
  `kentookumura/exp396-fold-safe-exp111-score-27-addonly-on-exp287-train` /
  `exp396 fold safe exp111 score 27 addonly on exp287 train`に固定した。
- metadataはprivate、CPU、internet off、run-on-pushで、上記3 kernel sourcesだけを接続する。
- strict packageのNotebook SHA256は
  `2b45f35be3578be5f47dad09e762f65b950596d018a3a8ef33cc9e2a2653c339`、
  metadata SHA256は
  `af0a73748439880c84c4229b322864bddc25d894c0fda3744674a0d3b6e1bcc3`。
- `task` commandは環境に無かったため、同等の`make validate-template`、
  `make validate-exp`、`make prepare-kaggle-notebooks`を使った。全検証PASS。
- 56文字のinitial kernel slug
  `exp396-fold-safe-exp111-score-27-addonly-on-exp287-train`はid/titleが一致していたが、
  `SaveKernel 400`となった。直後のmetadata pullも403で、kernelは作成されていない。
- repo内の既知のKaggle 50文字上限パターンと一致するため、同じexp396・同じ科学設定のまま、
  43文字のcanonical id/title
  `kentookumura/exp396-foldsafe-exp111-score27-exp287-train` /
  `exp396 foldsafe exp111 score27 exp287 train`へ同時に短縮する。initial slugへ再pushしない。

## 2026-07-25 Stage A 0-booster preflight実行結果

- 短縮したcanonical kernelをprivate CPU / internet off / run-on-pushでpushし、version 1、
  id_no `128540844`が`KernelWorkerStatus.COMPLETE`で完了した。
- Kernel URL:
  `https://www.kaggle.com/code/kentookumura/exp396-foldsafe-exp111-score27-exp287-train`
- all checksは`16/16` true、statusは`stage_a_zero_booster_preflight_passed`。
- `3,783,989` rows / `773` wells、duplicate ID 0、row/well/full coverageはいずれも1.0。
- outer 5 × inner 4のnested manifestは20行で、outer/inner well overlapはすべて0。
- 48 model input / 10 score core / 27 derived featureの固定schema SHAが一致した。
- exp099 / exp111 / exp287 / corrected exp264の全固定入力SHAが一致した。
- exp287に旧27列と依存GRWR 6列は存在せず、exp111保存model predictionは未使用だった。
- runtimeは`277.133756379 sec`、peak RSSは`5.168308258 GB`。
- booster学習0、prediction生成0、submission生成0、control再学習0。
- full Stage Aのresource gateとscorer quality gateは未評価であり、Stage A全PASSではない。
- 短縮slugでのstrict package SHAは、notebook
  `2b45f35be3578be5f47dad09e762f65b950596d018a3a8ef33cc9e2a2653c339`、
  metadata `1162f2a9754357bbc92e16669e5c1a398e52015a8b9fb93082748f01576dc5b2`、
  embedded config `2a8cb02e8084c16794b609159d03baf155f247431103d17da0e03e8591a37b4e`。
- 成果物SHA:
  - preflight manifest:
    `5ee569555de45417deac61368b73dde94232bc8377439ac4c873d97861f4f13c`
  - nested fold manifest:
    `3d42173760625993e51a76d91eb2b7d7dc9e9fbe1a16dbba71c2382c749d41c0`
  - nested manifest logical:
    `f8104336a64369cccdeddbf1c29c349872f6c91529677391cdb9946892035`
  - fold assignment:
    `e8c3dd328af6e5b295940467679fb36b07614f6d6f0b1120f1bc373bec85f92f`
  - synthetic derivation logical:
    `2c5e71073a8f70825b95e1f825789a00d79fff275b27b90271bffe589df8a61c`
  - logs:
    `2c8d681dea7e8b32456eb061cf28ca3c5ddb72bc8637368809719cb4a5f2086f`

## 2026-07-25 Stage A 40 CPU booster本学習承認・pre-push監査

- ユーザー指示 `次に進んでください。` は、直前に別承認境界として明示した固定Stage A本学習の
  承認として記録した。
- 実行量はactive scorer variant 1、outer folds 5、inner folds 4、objectives 2、
  合計40 CPU boosters。LightGBM configはbinary 1 + L1 1、control再学習0。
- Stage B 15 GPU boosters、inference、submissionは今回の承認に含めない。
- 実行先は既存canonical kernel
  `kentookumura/exp396-foldsafe-exp111-score27-exp287-train`のversion 2とし、別slugを作らない。
- pre-push pullでid_no `128540844`、private、CPU、internet off、exp099/exp264/exp287の
  kernel source 3件を再確認した。
- configは`stage_a_cpu_scorer_train`、`stage_a_cpu_run_approved=true`、
  `run_train=true`へ切り替える。40 model、40 median vectors、10 score-core partitions、
  technical / scorer-quality / runtime / memory gateを生成・判定し、結果にかかわらず
  Stage B、prediction、submissionは開始しない。
- version 2 strict packageはprivate、CPU、internet off、run-on-push、上記3 kernel sourcesでPASSした。
  package SHAはnotebook
  `3c86cea88b8259e7e5857dfdde307ee7349c6b813dd56881e70e053bec99f938`、
  metadata `1162f2a9754357bbc92e16669e5c1a398e52015a8b9fb93082748f01576dc5b2`、
  embedded config `de5729067cfdd193785778e3fbe4f643dc272f21191e995589339bdf971c704f`。

## 2026-07-25 Stage A 40 CPU booster本学習結果

- 同一canonical kernelへversion 2をpushし、id_no `128540844`、
  `KernelWorkerStatus.COMPLETE`を2026-07-25 05:12:40 UTCに確認した。
- private CPU、internet off、active variant 1、outer 5、inner 4、objectives 2、
  control再学習0の固定実行量から変更していない。
- runtime `3662.974058053 sec`、peak RSS `8.762432098 GB`で、
  上限`30,600 sec / 25 GB`をPASSした。
- 40/40 scorer models、40/40 model固有median vectors、40/40 schema records、
  10/10 score-core partitionsを保存した。
- 20 nested fold rolesのouter/inner well overlapはすべて0、sampleは全roleで350,000 rows、
  score partitionのduplicate ID合計は0だった。
- technical checksは22/22 PASS。target-free before label join、outer-valid well exclusion、
  global RNG不使用、saved exp111 model再利用0、coverage/schema/content SHAを含む。
- scorer-quality checksは6/6 PASS。
  - expected-error MAE: learned `5.908305`、prior `39.055075`、
    delta `-33.146770`、5/5 folds改善
  - within10 logloss: learned `0.345125`、prior `0.535689`、
    delta `-0.190564`、5/5 folds改善
  - within10 Brier: learned `0.112135`、prior `0.177406`、
    delta `-0.065271`、5/5 folds改善
- Stage A summaryは`stage_a_complete_all_gates_passed_stage_b_approval_pending`。
  Stage B、prediction、submissionは開始していない。
- 容量の大きい40 modelと10 score-core ParquetはKaggle version 2 outputを正とし、
  ローカルには判定・SHA・後続入力契約に必要なsmall manifests、quality、importance、logsだけを保存した。
- 主要SHA:
  - summary: `e5fa67ed62266a8cfed03121463692d1abec1e1e85616e3af0611912cf1a383e`
  - gate: `35314fb017234cd0ed3fe8a2eb7f998e7ceee9b837a20c4c207d54718391ff49`
  - model manifest: `b42757edd5dbb061ccc2903595e7c1676a8e21bfcc87837202a3f9387f4bd028`
  - fold manifest: `ad8279628a64f8afb812ef6c7ae4ade1307aeff24dd6b5f0ce0d57b232354e4b`
  - score partition manifest: `055d2c1f5583b6b44b0ea3413bfd66592b485adb764cec69522e7b33d571aba5`
  - score quality: `40abadfb5e9e385a9b15fbc53ae4d829cde948badbd0e2be66b64ea7f213b679`
  - feature importance: `9abf67ae6f3f73da18408cfb59eb397e8c85ffd502e7533c01e01a2a849eea91`
  - reproducibility manifest:
    `84cc8703868eea2ebddbbfb176e861505e1f3b6281c725b02e772a31518338f0`
  - logs: `a8e96ce402e6e44972121ea79034431225bb6345b7de64b4088cb14bded1353b`
- 結果記録後は`stage=implementation_only`、`run_train=false`、
  `kaggle_push_approved=false`へ戻し、無承認再学習をfail-closedにした。

## 2026-07-25 Stage B実装・15 GPU booster承認・pre-push監査

- ユーザー指示 `実行してください` を、直前に再提示したStage Bの実装とKaggle T4実行の
  明示承認として記録した。
- 固定実行量はactive TVT variant 1、LightGBM configs 3、folds 5、
  合計15 GPU boosters、exp287/exp264 control再学習0。推論・submission・外部提出は0。
- Stage A version 2のall-gates PASSと10 score-core partitionsを別kernel sourceとして固定し、
  自己参照を避けるため実行先を
  `kentookumura/exp396-score27-exp287-stageb-train`へ分離した。
- exp287の保存済みformation 10 partitionsはfile/logical SHAを検証して再利用し、再生成しない。
  clean 273、Stage C compact 74、formation 74、Stage A coreから再導出する27列を
  `273 + 74 + 74 + 27 = 448`の固定順で連結する。
- fit前にStage A/exp287/corrected exp264/Stage C/exp099 contextのSHA、ID/well/fold、
  parent 421 / final 448 schema、3 LightGBM configのexp287 parityを全AND検証する。
- Stage B promotion gateとして、exp287比pooled `<= -0.02 ft`、4/5 folds以上nonworse、
  5 scopes各`<= +0.02 ft`、by-well delta p95 `<= 0`、corrected exp264比worst
  `<= +0.25 ft`、+1/+3/+5 ft悪化well数`<= 135/39/14`を実装した。
- Jupytext train候補は12章・27 cellsへ更新し、正規train Notebookへ採用した。
  canonical/candidate Notebook SHAは
  `0a19d05759febf720bb2a60c38b9885b2cc8841295c00c6024a7a1944fa44cf2`。
- 専用testは11件PASS。`py_compile`、Ruff `F821/F401/E9`、
  Jupytext round-trip、Notebook code構文検証もPASSした。
- Stage B GPUはT4、internet off、`gpu_use_dp=true`、`deterministic=true`、
  `force_col_wise=true`、threads 8。bitwise deterministic anchorとはみなさない。
- strict packageはprivate、T4、internet off、run-on-push、competition source 1件、
  immutable kernel sources 7件でPASSした。package SHAはNotebook
  `b80878667f6f47b742143557e65858dcf30e69c89c1c04fb43d462d7b6e13e13`、
  metadata `59c887f9532ca3a4820eec1bf663f51cf4ede0478732dc94fe3d0828ffefc012`、
  embedded config `d741e1aa6d1a2c30ab5153d83c200c873a1548b34d9703083e27de990967c8d5`。
- 7件のkernel sourcesはpush直前にすべて`KernelWorkerStatus.COMPLETE`を確認した。

## 2026-07-26 Stage B 15 GPU booster実行結果

- 別Stage B kernel
  [`kentookumura/exp396-score27-exp287-stageb-train`](https://www.kaggle.com/code/kentookumura/exp396-score27-exp287-stageb-train)
  のversion 1、id_no `128570498`をprivate T4 / internet offで実行した。
- 2026-07-25 15:31:14 UTCに`KernelWorkerStatus.COMPLETE`を確認した。ログに
  Traceback、OOM、kernel diedはなく、固定15/15 GPU boostersを完走した。
- 実行量は1 variant × 3 configs × 5 folds = 15 boosters、control再学習0から変更していない。
  preflightは14/14 checks PASS、`570.220847 sec`、peak RSS `23.867611 GB`。
  学習・評価runtimeは`16830.255235 sec`、最終peak RSSは`25.557098 GB`。
- 保存済みformation 10 partitionsとStage A score core 10 partitionsをSHA照合して再利用し、
  final surfaceは`273 + 74 + 74 + 27 = 448`列、feature schema SHAは
  `07f6c2b51d166f210bae18720c32fae638aead255b24adda4ac598eaac517630`。
- pooled OOF RMSEはexp396 `8.134294735`、exp287 `8.136708220`、
  delta `-0.002413486 ft`。固定要件`<= -0.02 ft`を満たさなかった。
- fold別deltaは`+0.015713 / -0.033683 / -0.011692 / +0.000726 / +0.016578 ft`で、
  nonworseは2/5 foldsにとどまり、固定要件4/5を満たさなかった。
- scope別deltaはnear `+0.003766`、mid `+0.026000`、1000+ `-0.005071`、
  hidden-like spatial `+0.021101`、hidden-like typewell-purged `+0.026156 ft`。
  midとhidden-like 2面が固定上限`+0.02 ft`を超えた。
- by-well delta p95はexp287比`+0.342927 ft`で固定上限`0.0 ft`を超えた。
  corrected exp264比worstはwell `fb03ae90`の`+7.802733 ft`で固定上限`+0.25 ft`を超えた。
  一方、+1/+3/+5 ft悪化well数は`68 / 16 / 5`で上限`135 / 39 / 14`をPASSした。
- 固定promotion checksは1/6 PASS。pooled、fold、scope、by-well p95、worst-wellがFAILし、
  worsened-well countだけPASSしたため、statusは
  `stage_b_complete_promotion_gate_failed_closed`。
- 27列のfeature importance比率はgain `1.1025%`、split `2.4936%`だった。signal利用自体は
  確認できるが、pooled gainは小さくfold/scope/tailへ安定転移しなかった。
- 15 model manifestとOOF SHAは保存したが、OOF Parquetとmodel本体はKaggle outputを正とし、
  ローカルにはmetrics、fold/scope/by-well、feature importance、manifest、logsだけを取得した。
- 主要SHA:
  - preflight manifest:
    `0cd823b4e3058b547e3aa411de158a7c66c9a29b0f58cf66f9a1950d6be5052e`
  - component manifest:
    `1a228ffe8679a2087ff9f11dd17db69917bed104927407199047077800449af1`
  - metrics:
    `d80395f71279e7a8c1597f91902303c471ecfaf22e13fddca61193f7fcdc5146`
  - model manifest:
    `85059c057895365c53158e75a5f18246414d591123c5b52a1627501af63d75c1`
  - feature schema:
    `07f6c2b51d166f210bae18720c32fae638aead255b24adda4ac598eaac517630`
  - OOF prediction:
    `ebf4a12896c80435ab12f16e8bcb3297874edef81b7630f42dea7c53713a81c3`
  - reproducibility manifest:
    `fd7434ce4dd39be3996d1a10c403d1ed00d4d2a4ce67d34c266b265e1031c8a9`
  - logs:
    `0a28d6b1a83453e27f3ceeb4b5121fc0ae6cbca30564077974377de593989b46`
- 結果記録後は`stage=implementation_only`、`run_train=false`、
  `kaggle_push_approved=false`へ戻した。推論、submission、外部提出は生成・実行していない。

## 固定実行量

### Stage A

- active scorer variant: 1
- outer folds: 5
- inner folds: 4
- objectives: 2
- planned CPU boosters: 40
- saved exp111 model reuse: 0

### Stage B

- active TVT variant: 1
- final features: 448
- LightGBM configs: 3
- folds: 5
- planned GPU boosters: 15
- control retraining boosters: 0

## 再現性メモ

- fold seed: fixed `42`
- subsample seed: `SHA256("exp396|outer=<o>|inner=<i>|candidate_long")`由来のlocal RNG
- stochastic components: CPU scorer LightGBM、candidate-long row sample、承認後のGPU TVT LightGBM
- parallel policy: stable sort後にsample row IDをfreezeし、global RNG / thread順序に依存させない
- imputation: 各inner-trainで48 medianをfitし、そのmodelのvalid/test適用用に保存
- CPU runtime gate: `<= 30,600 sec`
- peak RSS gate: `<= 25 GB`
- GPU runtime: 承認後のみT4、internet off、DP、deterministic、force-col-wise、threads 8
- 入力SHA: exp287 OOF / models / metrics / formation manifest / schemaとexp264 OOFをpreflightで照合
- feature SHA: gzipはdecompressed content、Parquetはfileとid+float32 logical contentを記録
- model SHA: Stage Aは40 scorer + median/schema manifest、Stage Bは15 TVT model manifestを記録
- prediction/submission SHA: Stage B OOF prediction SHAは記録済み、inference/submission SHAはなし
- deterministic anchor: false。CPU/GPU LightGBMのbitwise rerun一致は主張しない

## 固定control

- exp287 CV: `8.136708220359452`
- exp287 OOF SHA: `8f026c5c5f6508fb142981832994c6ba9cded4940168c648a9df9f3e698c3913`
- exp287 model manifest SHA: `419dbdf83dd6bc343f0265aca56dd690ba1f231ee419e7cf0ff456ffdb797590`
- exp287 metrics SHA: `435434342494aaa62cee6e627809363ac34f16174973f4b81301d2923f780862`
- exp264 corrected OOF SHA: `b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2`
- exp111 source SHA: `0f9ec70f8080bf98566dee2122a36066ba161db0a7a3de4be203757d86ebec48`
- exp111 config SHA: `ca8b27de391e306ea6f3e7fbfa9510af63aacfeacd42ea99dbaa0a05b21d851f`
- exp111 row-feature schema SHA: `d5adf0f71d1ca44b1ddd7b0577b3637e3ea5e9affd9cb317f931bb3485537f8d`
- exp111 reference model manifest SHA: `178e8b3124b817a2b230080fc041aaaee1b06941e5a4223a68cc31bf26e68010`
  （variant名の参照だけに使い、保存modelはloadしない）

## 次のアクション

1. exp396のscore-27 add-only branchをnegative resultとして閉じ、exp287をtrain-side parent anchorに維持する。
2. exp396のsame-OOF subset/grid、gate緩和、再学習、inference、submissionへ進まない。
3. 次候補は保存済み生成物だけを使う0-boosterの転移失敗原因readoutを低・P4に置く。
   新しい独立した必要性と承認がない限り着手せず、既存P3候補を追い越さない。
