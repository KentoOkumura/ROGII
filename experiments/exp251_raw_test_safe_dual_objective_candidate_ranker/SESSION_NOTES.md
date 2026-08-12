# exp251_raw_test_safe_dual_objective_candidate_ranker セッションノート

## 目的

`raw_test_safe_dual_objective_candidate_ranker` backlogを実験化する。exp248 original-onlyの固定11候補、within10 classifier、expected-error regressor、outer well folds、固定Viterbiを保ち、trainではcross-fit、raw testではtest wellを除外したfull-train referenceから再生成できるfeatureだけへ限定して再監査する。

## 現在の状態

- Route: `ensemble`
- 状態: `raw_test_regenerated_copcf` Kaggle train version 4完了、train-side guard FAIL
- active stage: `completed_train_side_guard_failed`
- CV / LB: 8.502212005（295列版fixed Viterbi）/ なし
- inference / submission: `adoption_supported=false`のためdisabled
- 親: `exp248_candidate_perturbation_augmentation_for_likelihood_ranker`
- fixed control: exp248 original-only fixed Viterbi 8.421415097
- ML参照anchor: exp218 8.475793752

## 実行前コストガード

### 既定feature audit

- active variant: 0
- LightGBM config: 0
- folds: 0
- boosters: 0
- parent/control再学習: なし
- PF/Beam/dense/geometry model再生成: なし
- exp209/223 raw-test HMM: 固定source/configによるinference再生成のみ
- runtime: Kaggle CPU、GPU/internet disabled
- inference / submission: なし

### Optional train（新audit通過後だけ）

- active variant: 1 (`raw_test_regenerated_copcf`)
- objectives/config: 2 (within10 classifier、expected-error regressor)
- folds: 5
- total boosters: 1 × 2 × 5 = 10 CPU boosters
- fixed Viterbi: 1規則
- exp248 original-only control再学習: なし。保存済み8.421415097を参照する。
- parent再学習: なし
- 新auditで297/295/165列のhard contractが通過した後にだけ実行可否を判断する。旧`raw_test_safe` 130列版controlは再学習しない。

### 2026-07-16 corrected 295列版 train version 4実行契約

- ユーザーの「次に進んでください」を、直前に提示したoptional trainへの実行承認として記録した。
- notebook: 1 (`exp251_raw_test_safe_dual_objective_candidate_ranker_rawtest_copcf_parity.ipynb`)。同じcanonical kernelへversion 4として追加する。
- stage: `train_after_feature_audit`。same-runで297 / 295 / 165 feature auditとtest-well source overlap 0を再確認し、PASS時だけ学習する。
- active variant: 1 (`raw_test_regenerated_copcf`)。
- LightGBM objectives/config: 2（within10 classifier、expected-error regressor）。
- folds: 5。合計booster: 1 × 2 × 5 = 10。
- CPU、GPUなし。親exp248/control、候補生成器、PF/Beam/HMMは再学習しない。
- 比較対象は保存済みexp248 original-only、exp218、1000+、exp115 hidden-like 2群、worst-well guard。inference / submissionは実行しない。
- package再生成後、stage `train_after_feature_audit`、1/2/5/10、CPU、internet off、9 kernel sources、parent/control再学習なしを独立確認した。
- push前pullでcanonical id_no `127304533`とversion 3 sourceを確認し、同kernelへversion 4をpushした。Kaggle URL: `https://www.kaggle.com/code/kentookumura/exp251-rawtest-safe-dualobj-candidate-ranker-train`。
- push後pullで同じid_no、CPU、internet off、9 sourcesを再確認し、Kaggle側bootstrapのconfig SHA `a5f6197d5e215dc172ebd44eb3862456f466b6c113ca9126885d76e5d2bc44e0`がlocal packageと一致した。

### 2026-07-16 現行295列契約へのunit test更新

- 旧`feature_audit_only` / `raw_test_safe` / `copcf_*`禁止を固定していた2 assertionsを、
  version 4の`train_after_feature_audit` / `raw_test_regenerated_copcf`契約へ更新した。
- testはsame-run audit必須、1 variant / 2 objectives / 5 folds / 10 boosters、control/parent再学習なし、
  parent 297 / selected 295 / regenerated `copcf_*` 165列を明示的に確認する。
- `copcf_*`はraw testで再生成するため許可し、raw-testで安全に再現していないexp226 auxiliary
  4列はexact disallowedのまま維持する。
- focused pytest 4件PASS、Ruff PASS、repository全体pytest 50件PASS。

## 固定したfeature contract

- exp248 original-onlyのlong schema 297列完全一致を要求する。
- 20,000 train base rowsとraw test全行でcandidate-long surfaceを独立生成する。
- 各列についてprovenance、train/raw-test missing率、missing率差、q01/q50/q99、SMD、PSI、採否理由を保存する。
- selected featureはraw testで物理生成済み、許可provenance、missing率各5%以下、missing率差5%以下を要求する。
- trainの`copcf_*`は従来どおりouter-fold-safe OOFを使う。raw testの`copcf_*`はraw-test 3 wellsをsource poolから全除外し、full-train typewell/spatial referenceから再生成する。
- parent 297列中、`copcf_*` 165列を含む295列の選択、`exp226_gr_delta`と`exp226_geop_tvt`の2列の明示除外をexact hard guardにする。
- all-missing列のmedian/0 fallbackは禁止する。
- SMD absolute 4超 / PSI 0.5超はwarning。current testが3 wellsのため分布warningだけでは自動除外しない。
- selected train/raw-test sampleは各50,000 long rowsまで保存し、decompressed content SHAを記録する。

## リークガード

- feature採否にtrue TVT、candidate error、oracle label、exp115 hidden-like roleを使わない。
- raw testはID/well、raw horizontal/typewell、保存済みtarget-free prediction/sourceだけを使う。
- candidate-long labelはouter-trainのwithin10 / absolute errorだけに使い、model featureから除外する。
- cross-fit priorをraw testへmedian/0で埋めない。full-train referenceを使う場合もraw-test well IDをsource poolから全除外する。
- exp115 roleはtrain結果のsubgroup metricだけに使う。
- augmentation、candidate追加、Viterbi grid、true error/oracle gateを行わない。

## 再現性メモ

- `docs/06_reproducibility.md`を2026-07-15に確認した。
- audit/train row sampleはSHA256-derived local RNG。Python `hash()`とglobal RNGを使わない。
- exp209/223 HMMは固定source/configでraw testから再生成し、source SHAを記録する。
- gzip feature sample / OOFはdecompressed content SHAを主証拠にする。
- selected schema、feature contract、raw-test sample、model manifest、model、OOF prediction SHAを保存する。
- Kaggle CPU、GPU false、internet false。deterministic submission anchorとは扱わない。
- inference/submission SHAは対象外。

## コマンドログ

### 2026-07-15 Kaggle feature audit実行契約

- kernel: `kentookumura/exp251-rawtest-safe-dualobj-candidate-ranker-train`
- notebook: 1 (`exp251_raw_test_safe_dual_objective_candidate_ranker_train.ipynb`)
- stage: `feature_audit_only`
- active variant / LightGBM config / fold / booster: `0 / 0 / 0 / 0`
- parent/control再学習: なし
- runtime: Kaggle CPU、GPU/internet disabled、run-on-push enabled
- 297-feature auditのartifactとguardを確認するまでoptional 10-booster trainには進まない。

### 2026-07-15 Kaggle feature audit version 1結果

    make prepare-kaggle-notebooks EXP=exp251_raw_test_safe_dual_objective_candidate_ranker EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp251-rawtest-safe-dualobj-candidate-ranker-train --title 'exp251 rawtest safe dualobj candidate ranker train' --run-on-push --strict"
    make push-kaggle-train EXP=exp251_raw_test_safe_dual_objective_candidate_ranker
    kaggle kernels status kentookumura/exp251-rawtest-safe-dualobj-candidate-ranker-train
    kaggle kernels logs kentookumura/exp251-rawtest-safe-dualobj-candidate-ranker-train

- Kaggle status: `COMPLETE`、version 1、audit runtime 773.835秒。
- 実行量: 0 variant / 0 config / 0 fold / 0 booster。parent/control再学習なし。
- 297 parent featureを全件監査し、raw-test生成130、selected 130、excluded 167。
- hard guard 4件は全PASS。statusは`feature_audit_completed_ready_for_training`。
- selected schema順序、raw-test生成可、train/raw-test missing率各5%以下、missing率差5%以下、blacklist 0列を独立検算してPASS。
- selected provenance: raw-test base 5、HMM再生成15、multi-observation再生成24、candidate derivation 73、candidate-set context 13。
- distribution warningは38列。すべてselected側で、testが3 wellsのためhard除外には使わないが、train-stageの主要リスクとして記録する。
- feature audit / contract / selected schema / train sample / raw-test sampleのSHAを取得artifactから再計算し、全一致。
- artifactは`kaggle/output/train_v1/artifacts/`へ選択取得した。optional train、inference、submissionは未実行。

### 2026-07-15 optional train明示承認

- ユーザーから`train_after_feature_audit`実行の明示承認を受けた。
- active variant: 1 (`raw_test_safe`)
- LightGBM objectives/config: 2 (`within10_classifier`、`expected_error_regressor`)
- outer well folds: 5
- total boosters: 1 × 2 × 5 = 10 CPU boosters
- parent/control再学習: なし。exp248 original-only 8.421415097を固定参照する。
- 同一run内で297-feature auditを再実行し、全hard guard通過時だけ学習する。
- version 1でselected 130列中38列にdistribution warningがあることを承認前に提示済み。
- runtime: Kaggle CPU、GPU/internet disabled。inference / submissionは実行しない。

### 2026-07-15 Kaggle optional train version 2

    make prepare-kaggle-notebooks EXP=exp251_raw_test_safe_dual_objective_candidate_ranker EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp251-rawtest-safe-dualobj-candidate-ranker-train --title 'exp251 rawtest safe dualobj candidate ranker train' --run-on-push --strict"
    kaggle kernels pull kentookumura/exp251-rawtest-safe-dualobj-candidate-ranker-train -p /tmp/kaggle-pull/exp251-rawtest-safe-dualobj-candidate-ranker-train -m
    make push-kaggle-train EXP=exp251_raw_test_safe_dual_objective_candidate_ranker

- canonical kernel: `kentookumura/exp251-rawtest-safe-dualobj-candidate-ranker-train`、id_no 127304533。
- version 2 push成功、status `RUNNING`。
- package metadata: CPU、GPU false、internet false、run-on-push true。
- ユーザー指示によりCLI follow監視のみ停止した。Kaggle version 2自体は停止せず実行継続。完了連絡後にlogs/CV/guard/SHAを確認する。
- ユーザーの完了連絡後にstatus/logsを取得し、`COMPLETE`を確認した。
- same-run audit 1,098.551秒、train/evaluation 3,081.220秒、notebook最終log時刻4,223.765秒。
- 10 CPU boosters完走。fixed Viterbi CV 8.402085596、exp248 original-only比-0.019329500、exp218比-0.073708155。
- fixed Viterbi fold RMSE: fold0 7.489092656、fold1 9.518581363、fold2 8.025575213、fold3 8.602642649、fold4 8.239185104。
- probability rowwise 8.682859888、expected-error rowwise 8.464866112、fixed Viterbi 8.402085596。
- 1000+ 9.209531850、exp115 spatial 8.761820393、typewell-purged 8.732708436はPASS。
- worst `fb03ae90` 57.642884880だけFAIL。exp248同well 57.297084193から+0.345800687で、許容+0.25を0.095800687超過した。
- candidate AUC 0.925417787、logloss 0.325474977、Brier 0.102772972、expected-error MAE 4.481362878。
- worst well fixed Viterbiは11 switch。`tvt_dense50` 3,275行、K16 geometry 1,958行が支配し、switch暴走よりcandidate preferenceの系統差と解釈する。
- `kaggle/output/train_v2/artifacts/`へ必要生成物だけ選択取得した。OOF 3,783,989行 / 773 wellsからfold scoreを再構成。
- 診断生成物11点、10 models、5 imputers、feature生成物4点のSHAは全一致。feature audit SHAはversion 1と一致。
- best iterationはclassifier `522/308/265/204/225`、regressor `541/530/547/549/550`。
- 非fatal warningはexp223 sourceの`fillna(method=...)` deprecationと、ordered NumPy予測時のsklearn feature-name warning。schema順序と全SHAが一致しており、今回のscore無効化要因ではないが将来互換性課題として残す。
- status `completed_train_side_guard_failed`、`adoption_supported=false`。inference / submissionへ進めない。

### 2026-07-15 `copcf_*` raw-test再生成への契約訂正

- version 1/2で除外した167列のうち165列は`copcf_*`だった。これは当時のraw-test surfaceが生成していなかったことを示すが、test生成不能の証拠ではなかった。
- 新variantを`raw_test_regenerated_copcf`とし、train側は既存cross-fit/OOF列、raw test側はfull-train referenceを使う非対称だが本番整合なcontractへ訂正した。
- raw testのtypewell GRをnative overlapで既存train clusterへ再割当する。実データsmokeでは`000d7d20 -> cluster_0022`（eligible source 12）、`00e12e8b -> cluster_0022`（12）、`00bbac68 -> cluster_0003`（40）で、best overlapはすべて1.0だった。
- typewell priorは同clusterのtrain wells、spatial priorはtrain geometry KNNをsourceにする。raw-test 3 well IDはsource curve、cluster reference、spatial KNNの全poolから先に除外し、混入時はfail closedする。
- raw-test baseで41 `copcf_*`列を生成し、candidate-long展開後に165列のparent schemaを再現する。synthetic long contractで165列・unique 165を確認した。
- 実raw-test 14,151行のpartial-source smokeでは41 base列、source exclusion PASS、base生成列の最大missing率0.0を確認した。ローカルRAM制約のため3,783,989行full-source実行は完走証拠とせず、Kaggle CPU auditを正とする。
- 既存exp238 raw-test copcf parity v1は41列をfull-train referenceから生成済みだが、保存summaryでは`000d7d20`、`00bbac68`、`00e12e8b`自身が各spatial KNNに入り、typewell neighbor数も自己ID込みの14/41/14だった。visible test IDsがtrainにも存在するため、これはleakage-safeの証拠には使わない。
- exp251は3 test IDsをsource curve・typewell matching・cluster geometry・spatial KNNから先に全除外する。その結果、typewell eligible sourceは12/40/12となり、runtime metadataと最終source scanの両方で自己ID 0を要求する。
- feature hard contractはparent 297、selected 295、regenerated prefix 165。parent schema上の明示除外はraw-testで安全に再現していない`exp226_gr_delta`と`exp226_geop_tvt`の2列だけとする。
- static validationはpy_compile、Ruff、synthetic provenance testを通過した。Kaggle feature auditは0 variant / 0 config / 0 fold / 0 booster、parent/control再学習なしで行う。
- 旧version 2のCV 8.402085596とworst-well FAILは130列版の履歴として保持する。新295列版の性能を示す値ではなく、旧modelのinference / submissionも引き続き禁止する。

### 2026-07-15 corrected feature audit実行契約

- notebook: 1 (`exp251_raw_test_safe_dual_objective_candidate_ranker_rawtest_copcf_parity.ipynb`)。既存正規train ipynbは上書きせず、更新済みtrain Jupytext `.py`から別名生成した。
- canonical kernel: `kentookumura/exp251-rawtest-safe-dualobj-candidate-ranker-train`へversion 3として追加予定。push前pullでid_no `127304533`、CPU、internet off、既存9 kernel sourcesを確認した。
- stage: `feature_audit_only`。
- active training variant / LightGBM config / fold / booster: `0 / 0 / 0 / 0`。
- parent/control再学習: なし。selector/final model学習、inference、submissionもなし。
- audit target: parent 297 / selected 295 / regenerated `copcf_*` long 165 / regenerated base 41 / excluded parent 2。
- source guard: visible test 3 IDsをsource curve、typewell matching、cluster geometry、spatial KNNから全除外。両native-overlap thresholdのcluster割当必須、最終source overlap 0必須。
- py_compile、Ruff、Jupytext `--test`、strict experiment validation、JSON/YAML parseはPASS。Kaggle full-source audit結果が出るまでoptional 10-booster trainには進まない。
- version 3 push成功。Kaggle URL: `https://www.kaggle.com/code/kentookumura/exp251-rawtest-safe-dualobj-candidate-ranker-train`。ユーザーの既存指示に従いCLI継続監視は行わず、完了連絡後に通常logsと必要artifactだけを確認する。
- push後pullも成功し、id_no `127304533`、CPU、internet off、9 kernel sourcesを再確認した。取得したnotebook sourceに`raw_test_regenerated_copcf`とparent 297 / selected 295 / regenerated long 165のhard checkが含まれることを確認した。
- 最終静的確認はpy_compile、Ruff、strict experiment validation、JSON/YAML parseがPASS。補助`review_exp_docs.py`は大きな既存生成物の走査が長時間継続したため中断し、必須validatorの結果を採用した。

### 2026-07-16 corrected feature audit version 3結果

- Kaggle通常logsで`Feature-audit-only stage completed: 0 variants / 0 configs / 0 folds / 0 boosters`を確認した。audit runtime 694.333秒、完了表示708.719秒、parent/control再学習なし。
- train 3,783,989行 / 773 wells、raw test 14,151行 / 3 wells、11候補。parent 297 / generated 295 / selected 295 / excluded 2 / regenerated `copcf_*` 165、hard check 8件は全PASS。
- raw-test base `copcf_*`は41列。3 test wellを除外後のsourceは3,769,838行 / 770 wellsで、typewell priorと2 spatial KNNの全source list、typewell best matchを独立走査し、query ID overlap 0を確認した。runtime guardも`rawtest_well_source_exclusion=pass`。
- typewell source数は`000d7d20=12`、`00bbac68=40`、`00e12e8b=12`。両thresholdでcluster割当済み、valid rate 1.0。
- distribution warningは59列。testが3 wellsのためhard failではないが、optional train時の主要リスクとして残す。
- 必要生成物6点だけを選択取得した。feature audit SHA `c94d2de...`, contract `d68cf074...`, selected schema `7a9217d6...`、train sample decompressed SHA `545482cb...`、raw-test sample decompressed SHA `e97c1286...`はsummaryと一致した。
- statusは`feature_audit_completed_ready_for_training`。295列版の性能値ではなく、optional 1 variant / 2 config / 5 folds / 10 CPU boostersは未実行。inference / submissionは引き続きdisabled。

### 2026-07-16 corrected 295列版 train version 4結果

- ユーザーの完了連絡後、通常logsとKaggle statusを取得し、canonical kernel version 4の`COMPLETE`を確認した。
- same-run feature audit 1,096.555秒、train/evaluation 3,579.844秒、notebook最終log時刻4,718.378秒。1 variant / 2 objectives / 5 folds / 10 CPU boostersを完走し、parent/control再学習なし。
- same-run auditはparent 297 / selected 295 / regenerated `copcf_*` 165、test-well source overlap 0、hard check全PASS。feature audit / contract / selected schema / train/raw-test sample content SHAはversion 3と一致した。
- probability rowwise 8.479603434、expected-error rowwise 8.548425418、expected-error fixed Viterbi 8.502212005。
- fixed Viterbi fold RMSEは`7.653335511 / 9.472239430 / 8.075131903 / 9.017009135 / 8.163005322`。OOF 3,783,989行 / 773 wellsからGroupKFoldを再構成して確認した。
- candidate AUC 0.924458737、logloss 0.327735530、Brier 0.103746099、expected-error MAE 4.553844990。
- classifier best iterationは`422/264/246/180/254`、regressorは`550/550/231/550/542`。
- overallはexp218 8.475793752より+0.026418253、exp248 original-only 8.421415097より+0.080796908でFAIL。
- distance 1000+は9.326545505 > 9.234366423でFAIL。exp115 spatial 8.788133228、typewell-purged 8.746112958はPASS。
- worst `fb03ae90`は58.004236030 > 57.547084193でFAIL。exp248同well 57.297084193から+0.707151837、許容+0.25を0.457151837超過した。
- 旧130列版比でprobability rowwiseは-0.203256454改善した一方、expected-error rowwiseは+0.083559306、fixed Viterbiは+0.100126408悪化した。candidate AUCは-0.000959050、expected-error MAEは+0.072482112。
- statusは`completed_train_side_guard_failed`、`adoption_supported=false`。raw-test `copcf_*`再生成契約の技術的成立だけを採用し、295列rankerのinference / submissionへ進めない。
- Kaggle outputから必要生成物を選択取得した。OOF decompressed SHA `727b7f7b...`を含む診断11/11、10 models、5 imputersのSHAを一時取得物で照合し全一致した。model manifest SHAは`67b4f380...`。
- `kaggle/output/train_v4/artifacts/`にはOOF、summary、metrics、by-well、bucket/subgroup、feature contract/schema/importance、manifestだけを保存した。不採用model本体は保存していない。
- 非fatal warningはexp223 sourceの`fillna(method=...)` deprecation、ordered NumPy予測時のsklearn feature-name warning、nbconvertのescape warning。schema/SHAは一致しておりscore無効化要因ではない。
- 後続exp259 exact TVT datum augmentationはこのversion 4 fixed control比でoverall/1000+を改善したが、hidden-like 2面と最大well回帰guardがFAILした。`raw_test_safe_ranker_worst_well_attribution_readout`は、longtail-only再訪可否を判断する0-booster readoutとしてのみ候補に残す。

### 2026-07-15 steering / scaffold

    make new-steering EXP=exp251_raw_test_safe_dual_objective_candidate_ranker
    make new-exp EXP=exp251_raw_test_safe_dual_objective_candidate_ranker SOURCE=experiments/exp248_candidate_perturbation_augmentation_for_likelihood_ranker

- `docs/legacy/steering/20260715-exp251-raw-test-safe-dual-objective-candidate-ranker/`へ要件、設計、tasklistを記録した。
- exp250は別backlogで使用済みのため、次の空き番号exp251を使用した。

### 2026-07-15 実装

- `feature_audit_only` / `train_after_feature_audit`の二段stageを実装した。
- exp073 raw-test base、exp209 exact HMM、exp223 self-GR HMM、exp226 raw-test candidate、raw GR multi-observationからraw-test candidate-long surfaceを再生成した。
- 297 featureのprovenance/fallback/distribution audit、selected schema/sample/SHAを実装した。
- same-run audit passなしに10 boostersへ進めない停止guardを実装した。
- selected schemaだけを使うwithin10 / expected-error outer-fold学習、fixed Viterbi、overall/1000+/hidden-like/worst-well guardを実装した。
- inference notebookは明示停止し、submissionを生成しない。

### 2026-07-15 静的・合成契約検証

    .venv/bin/python -m py_compile experiments/exp251_raw_test_safe_dual_objective_candidate_ranker/*.py
    .venv/bin/ruff check experiments/exp251_raw_test_safe_dual_objective_candidate_ranker/*.py
    PYTHONPATH=experiments/exp251_raw_test_safe_dual_objective_candidate_ranker .venv/bin/python -c "from raw_test_safe_dual_objective_candidate_ranker import synthetic_feature_audit_contract_test; print(synthetic_feature_audit_contract_test())"
    JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp251_raw_test_safe_dual_objective_candidate_ranker/exp251_raw_test_safe_dual_objective_candidate_ranker_train.py
    JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp251_raw_test_safe_dual_objective_candidate_ranker/exp251_raw_test_safe_dual_objective_candidate_ranker_inference.py
    make validate-exp EXP=exp251_raw_test_safe_dual_objective_candidate_ranker
    make validate-template
    make test

- py_compile: PASS。
- Ruff: PASS。
- Jupytext convert/test: train / inferenceともPASS。
- strict experiment validation / project template validation: PASS。
- pytest: 19件PASS。exp251のstage、cost、provenance blacklist、same-run audit guard、explicit selected schemaを4 testで固定した。
- synthetic feature audit: PASS。同じ入力でschema採否が決定的で、`last_known_tvt` / `candidate_minus_last`だけを選択し、`copcf_oof_only` / `exp226_gr_delta` / raw-test未生成列を除外した。
- full Kaggle feature auditと10 CPU booster学習はversion 1 / 2で完了した。

## Notebook構成比較

- 親exp248 train source: 7章 / 249行。
- exp251 train source: 7章。stage/cost、入力、297 feature audit、optional 10-booster train、guard、SHAをnotebook cell上に展開した。
- heavy candidate/HMM/fold学習は補助moduleに残し、notebook上で入力、stage、対象variant/config/fold、metrics、生成物を追える。
- notebook sourceに`__file__`はない。

## 次のアクション

1. 保存済みexp248/exp251 OOFでworst-well candidate-family attribution readoutを行う候補をbacklogへ追加する。
2. 再学習、guard緩和、worst well専用rule、candidate追加、Viterbi gridは行わない。
3. train guard不通過のためinference / submissionは実装しない。
