# exp251_raw_test_safe_dual_objective_candidate_ranker 結果

## 状態

`raw_test_regenerated_copcf`のKaggle CPU version 4は完了しました。parent 297列中295列をraw testで生成・選択し、うち165列が再生成した`copcf_*`です。feature contractはPASSしましたが、expected-error fixed ViterbiはRMSE 8.502212で、overall、1000+、worst-wellの3 guardがFAILしました。statusは`completed_train_side_guard_failed`、`adoption_supported=false`です。295列rankerのinference / submissionは行いません。

## 2026-07-15訂正

- 旧auditの「除外167列」はraw-test surfaceに未生成だった列数であり、test生成不能な列数ではありませんでした。
- 167列中165列の`copcf_*`は、trainではcross-fit/OOF、raw testではtest well IDを全source poolから除外したfull-train typewell/spatial referenceとして生成できます。
- test typewell GRのnative overlapでtrain clusterを再割当し、同clusterのtrain source curvesとtrain geometry KNNからpriorを作ります。raw-test well混入時はfail closedします。
- 新hard contractはparent 297 / selected 295 / regenerated `copcf_*` 165で、parent schema上の明示除外は`exp226_gr_delta`と`exp226_geop_tvt`の2列です。
- 実raw-test 14,151行のpartial-source smokeは41 base `copcf_*`列、source exclusion PASS、生成列の最大missing率0.0でした。candidate-long synthetic contractでは165列・unique 165を確認しました。
- 既存exp238 copcf parity v1は41列を生成済みですが、summary上で各visible test well自身がspatial KNNに含まれ、typewell source数も自己ID込みの14/41/14でした。exp251では全3 test IDsをsourceから除外し、typewell sourceを12/40/12にしたため、この旧parityをleakage-safe証拠には使いません。
- full 3,783,989-row referenceはローカルRAM制約下の完走証拠がないため、Kaggle CPU feature auditを正とします。version 3とversion 4のsame-run auditでfull-source contractはPASSしました。

## 新295列版 Feature audit結果

- kernel: `kentookumura/exp251-rawtest-safe-dualobj-candidate-ranker-train` version 3、COMPLETE。
- audit runtime: 694.333秒。0 variant / 0 config / 0 fold / 0 booster、parent/control再学習なし。
- train 3,783,989行 / 773 wells、raw test 14,151行 / 3 wells、11候補。
- parent 297、raw-test生成295、selected 295、excluded 2、再生成`copcf_*` 165。hard check 8件は全PASS。
- raw-test baseで41 `copcf_*`列を生成した。3 test wellを除外したsourceは3,769,838行 / 770 wellsで、typewell matching、typewell prior、2種類のspatial KNNを横断した自己well overlapは0だった。
- typewell source数は`000d7d20=12`、`00bbac68=40`、`00e12e8b=12`。両native-overlap thresholdで全wellの割当とvalid rate 1.0を確認した。
- feature audit / contract / selected schemaのSHA、train/raw-test sampleのdecompressed-content SHAを取得物から再計算し、summary記録と全一致した。
- 分布warningは59列。testが3 wellsであるためhard failには使っていないが、295列版を学習する場合の主要リスクとして保持する。
- statusは`feature_audit_completed_ready_for_training`。これは特徴量生成契約の承認であり、性能・worst-well guard・inference採用を意味しない。

## 新295列版 Optional train結果

- kernel: `kentookumura/exp251-rawtest-safe-dualobj-candidate-ranker-train` version 4、COMPLETE。
- 1 variant × 2 objectives × 5 folds = 10 CPU boosters。親/control再学習なし。
- runtime: same-run audit 1,096.555秒、train/evaluation 3,579.844秒、notebook最終log時刻4,718.378秒。
- 3,783,989 rows / 773 wells / 11 candidates / selected 295 features。
- probability rowwise RMSE 8.479603、expected-error rowwise 8.548425、expected-error fixed Viterbi 8.502212。
- fixed Viterbi fold RMSE: `7.653336 / 9.472239 / 8.075132 / 9.017009 / 8.163005`。
- candidate AUC 0.924459、logloss 0.327736、Brier 0.103746、expected-error MAE 4.553845。
- best iterationはclassifier `422 / 264 / 246 / 180 / 254`、regressor `550 / 550 / 231 / 550 / 542`。

## 新295列版 Guard結果

- overall: 8.502212 > exp218 8.475794、FAIL（+0.026418）。exp248 original-only 8.421415比では+0.080797。
- distance 1000+: 9.326546 > 9.234366、FAIL（+0.092179）。
- exp115 spatial: 8.788133 <= 8.958871、PASS。
- exp115 typewell-purged: 8.746113 <= 8.909651、PASS。
- worst well: `fb03ae90` 58.004236 > 57.547084、FAIL。exp248同well 57.297084から+0.707152悪化し、許容+0.25を0.457152超過した。

旧130列版に対し、probability rowwiseは-0.203256改善しましたが、expected-error rowwiseは+0.083559、fixed Viterbiは+0.100126悪化しました。candidate AUCも-0.000959、expected-error MAEは+0.072482です。`copcf_nearest_other_cluster_dist`と`copcf_own_cluster_dist`はexpected-error feature importance上位に入りましたが、最終のexpected-error選択を改善する根拠にはなりませんでした。技術的なraw-test再生成契約は採用しますが、この295列rankerは不採用です。

## 仮説

exp248 original-onlyの8.421415というtrain-side gainのうち、raw testで再生成できるfeatureだけに由来する部分を分離できれば、exp237で発生したOOF-only median/0 fallbackと全行`pf_ancc`化を避けられる可能性があります。

## 固定設定

- Route: `ensemble`
- 親: exp248_candidate_perturbation_augmentation_for_likelihood_ranker
- candidate: exp237固定11候補
- audit: exp248 297 long feature、train 20,000 base-row sample、raw test全行
- optional train: 1 raw-test-safe variant / 2 objectives / 5 folds / 10 CPU boosters
- control: exp248 original-only 8.421415097（再学習なし）
- fixed Viterbi: exp237 1規則
- augmentation / candidate追加 / Viterbi grid / inference / submit: なし

## 旧130列版 Feature audit結果

- kernel: `kentookumura/exp251-rawtest-safe-dualobj-candidate-ranker-train` version 1、COMPLETE。
- audit runtime: 773.835秒。0 variant / 0 config / 0 fold / 0 booster。
- parent feature: 297、raw-test生成: 130、selected: 130、excluded: 167。
- hard guard 4件は全PASS。selected schema順序、raw-test生成、missing率、provenance blacklistを独立検算してPASS。
- selected provenanceはbase 5、HMM再生成15、multi-observation再生成24、candidate derivation 73、candidate-set context 13。
- 除外167列は当時のraw-test surfaceでは未生成かつfallback guard対象。`copcf_*`、`exp226_gr_delta`、`exp226_geop*`はselectedに0列だった。ただし`copcf_*` 165列は未実装だったため、生成不能という解釈は撤回する。
- feature audit / contract / selected schema / train sample / raw-test sampleの保存SHAを再計算し、全一致。
- 分布warningはselected 130列中38列。hard failではないが、testが3 wellsしかないこととtrain/testのPSI差が大きいことをoptional trainの主要リスクとして残す。

## 旧130列版の判定

raw-test feature contract自体は採用です。ただしtrain-side statusは`completed_train_side_guard_failed`、`adoption_supported=false`です。130列版modelをraw-test inferenceやsubmissionへ進めません。

## 旧130列版 Optional train結果

- kernel: `kentookumura/exp251-rawtest-safe-dualobj-candidate-ranker-train` version 2、COMPLETE。
- 1 variant × 2 objectives × 5 folds = 10 CPU boosters。親/control再学習なし。
- runtime: same-run audit 1,098.551秒、train/evaluation 3,081.220秒、notebook全体の最終log時刻4,223.765秒。
- 3,783,989 rows / 773 wells / 11 candidates / selected 130 features。
- probability rowwise RMSE 8.682860、expected-error rowwise 8.464866、expected-error fixed Viterbi 8.402086。
- fixed Viterbi fold RMSE: `7.489093 / 9.518581 / 8.025575 / 8.602643 / 8.239185`。
- candidate AUC 0.925418、logloss 0.325475、Brier 0.102773、expected-error MAE 4.481363。
- exp248 original-only比でlogloss -0.001269、expected-error MAE -0.056138。167列を除外してもcandidate ranking全体は悪化しなかった。

## Guard結果

- overall: 8.402086 <= exp218 8.475794、PASS。
- distance 1000+: 9.209532 <= 9.234366、PASS。
- exp115 spatial: 8.761820 <= 8.958871、PASS。
- exp115 typewell-purged: 8.732708 <= 8.909651、PASS。
- worst well: `fb03ae90` 57.642885 > 57.547084、FAIL。
- `fb03ae90`はexp248 original-only 57.297084から+0.345801悪化し、許容+0.25を0.095801超過した。

worst wellの固定Viterbiはswitch 11回まで抑えられているため、高頻度switch暴走ではありません。`tvt_dense50`を3,275 / 6,431行、K16 geometryを1,958行選ぶ系統的な候補偏りが残っています。candidate追加やViterbi gridを事後調整せず、保存済みexp248/exp251 OOFでcandidate-family attributionを先に行うべき結果です。

## 旧130列版 再現性確認

- OOF 3,783,989行 / 773 wellsからfold RMSEを再構成した。
- 診断生成物11点、10 models、5 imputers、feature生成物4点のSHAは全一致。
- OOF decompressed SHAは`d1b1349dc0954fbf38f1fba025067c1ed5d95dd450657f570dbdc0491a9284b3`。
- model manifest SHAは`3c6d66c740537af783019436866b0cd4b4a0194fc045f92d0848c9e3af1529d0`。
- version 1とversion 2のfeature audit / contract / schema / sample content SHAは一致した。

## 新295列版 再現性確認

- OOF 3,783,989行 / 773 wellsから5 fold RMSEを再構成した。
- 診断生成物11/11、10 models、5 imputersのSHAは全一致した。
- OOF decompressed SHAは`727b7f7b50084ea9597d894b2df8b4324d6b34b4191b57b4c32bd10e93b61829`。
- model manifest SHAは`67b4f3805aa6cb1d592c73cb75344f1bd8bcca02797214ca22b70c88f62a2af8`。
- feature audit `c94d2de...`、contract `d68cf074...`、selected schema `7a9217d6...`、train/raw-test sample content SHAはversion 3と一致した。
- 必要な診断生成物とOOFだけを`kaggle/output/train_v4/artifacts/`へ選択保存した。不採用model本体は実験配下へ保存していない。

## 次

exp259のexact TVT datum augmentationは、このversion 4を固定clean controlとして比較し、fixed Viterbiを-0.075086改善しましたがhidden-like 2面と最大well回帰guardがFAILしました。exp251単体からはinference、submission、guard緩和、worst-well専用ruleへ進みません。保存OOFによるcandidate-family attributionは、exp259のlongtail-only再訪可否を判断する0-booster readoutとしてのみ候補に残します。
