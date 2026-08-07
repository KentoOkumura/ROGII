# exp251_raw_test_safe_dual_objective_candidate_ranker

## 状態

旧130列版はKaggle CPU version 2で完走し、fixed Viterbi CV 8.402086でしたがworst-well guardが1件FAILしたため不採用です。その後、除外167列中165列の`copcf_*`はtest生成不能ではなく未実装だったと訂正し、`raw_test_regenerated_copcf`を実装しました。Kaggle CPU feature audit version 3は297 parent / 295 selected / 165 regenerated、test-well source overlap 0でPASSしました。295列版はversion 4で10 CPU boostersを実行中で、inference / submissionは無効です。

## 仮説

exp248 original-onlyは固定Viterbi RMSE 8.421415でexp237とexp218を上回りましたが、297 long featureにexp109/114等のOOF-only contextが含まれます。exp237 raw-test inferenceでは不在列をmedian/0 fallbackし、全行`pf_ancc`選択となりました。

固定11候補、2目的、outer folds、LightGBM設定、固定Viterbiを変えず、raw testから実際に再生成できる特徴だけへschemaを限定すれば、train-side gainとraw-test feature contractを原因分離して再評価できます。

## 実験範囲

- Route: `ensemble`
- 親: `exp248_candidate_perturbation_augmentation_for_likelihood_ranker`
- candidate親: `exp237_hmm_exp226_candidate_selector_on_exp183`
- candidate bank: exp237固定11候補
- feature audit: exp248 297 long featureのprovenance / fallback / distribution / SHA
- feature variant: train OOF / raw-test full-train-referenceの`raw_test_regenerated_copcf`
- optional train: 新audit通過後だけ1 variant、within10 + expected-error、5 folds、10 CPU boosters
- continuity: exp237固定Viterbi 1規則
- augmentation、candidate追加、Viterbi grid、inference、submission: 無効

## Stage

- `feature_audit_only`（既定）: 0 variant / 0 config / 0 fold / 0 booster。
- `train_after_feature_audit`: same-run audit pass後だけ1 variant × 2 objectives × 5 folds = 10 CPU boostersを実行する。

## 検証方針

まず297 long featureをtrain/raw-testで独立生成し、provenance、missing率、missing率差、quantile、SMD、PSIを監査します。trainの`copcf_*`はcross-fit/OOF、raw testはraw-test wellをsource poolから全除外したfull-train typewell/spatial referenceから再生成します。hard contractはparent 297 / selected 295 / regenerated `copcf_*` 165で、parent schema上の除外は`exp226_gr_delta`と`exp226_geop_tvt`です。optional trainではselected schemaだけを同じouter well 5-foldへ渡し、within10校正、expected-error校正、fixed Viterbi RMSE、1000+、hidden-like、worst-wellをexp248 original-onlyと比較します。

## 所見

旧version 1/2はraw-test surfaceに`copcf_*`生成を実装しておらず、130列を選択、167列を除外しました。その130列版ではfixed Viterbiがexp248 original-onlyより0.019330改善した一方、`fb03ae90`がexp248比+0.345801悪化しました。この結果とSHAは履歴として有効ですが、165 `copcf_*`列が生成不能という結論は無効です。既存exp238 parityも41列生成には成功していますが、visible test well自身がfull-train sourceに戻る問題がありました。新実装はraw-test 3 well IDsを全source poolから除外したうえでbase 41列からcandidate-long 165列を再現し、Kaggle full-source auditで297→295列、source overlap 0、hard check全PASSを確認しました。分布warningは59列で、optional train時の主要リスクです。

## 主要ファイル

- `config.yaml`: stage、feature provenance/fallback contract、10-booster cost、guard、再現性。
- `raw_test_safe_dual_objective_candidate_ranker.py`: train/raw-test surface、297列audit、optional fold学習、評価、SHA。
- `candidate_ranker_engine.py`: exp248由来のcandidate-long生成、dual-objective fold学習、fixed Viterbi評価。
- `rawtest_feature_builder.py`: exp237由来のraw-test base/HMM/exp226再生成。
- `copcf_rawtest_regeneration.py`: test well除外付きfull-train typewell/spatial priorと`copcf_*`再生成。
- `hmm_exp226_candidate_selector_on_exp183.py`: exp237固定候補/context/Viterbi engine。
- `exp251_raw_test_safe_dual_objective_candidate_ranker_train.py/.ipynb`: 正のtrain notebook。
- `exp251_raw_test_safe_dual_objective_candidate_ranker_inference.py/.ipynb`: inference停止guard。

## 採否条件

feature auditの全guardを通過し、optional trainでoverallがexp218以下、distance 1000+、exp115 hidden-like 2群、worst-wellがexp248 original-only基準を通ること。通過前はinference/submitへ進みません。
