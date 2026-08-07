# exp248_candidate_perturbation_augmentation_for_likelihood_ranker 結果

## 状態

Kaggle CPU train version 1を完了しました。candidate perturbation augmentationは事前guardを5項目すべて通過せず、不採用です。inference / submissionには進みません。

## 仮説

exp237固定候補に対する正解非依存perturbationをcandidate-long教師へ追加すると、clean original候補のwithin10 likelihood、expected-error calibration、候補選択RMSE、marginの信頼性が改善する。

## 比較

- control: `original_only`
- candidate: `perturbation_augmented`
- candidate bank / context / outer folds / model params / clean validation / fixed Viterbiは共通
- active changes: outer-train candidate-set augmentationのみ
- 3,783,989 rows / 773 wells / 11 candidates / 5 folds
- 148 base features / 297 candidate-long features / 20 LightGBM models

## 実行結果

Kaggle kernelは`kentookumura/exp248-candidate-perturbation-ranker-train` version 1、`id_no=127067118`です。runtimeは9,687.118秒（約2時間41分27秒）でした。

| 評価方法 | original-only RMSE | augmented RMSE | delta |
| --- | ---: | ---: | ---: |
| probability row-wise | 8.500238 | 8.855197 | +0.354959 |
| expected-error row-wise | 8.493973 | 8.778270 | +0.284297 |
| expected-error fixed Viterbi | 8.421415 | 8.728086 | +0.306671 |

fixed Viterbiのfold別deltaもfold 0から4まで`+0.301214 / +0.336758 / +0.293061 / +0.482689 / +0.103018`で、5 foldsすべて悪化しました。

candidate-level指標も、AUC `0.925007 -> 0.923062`、logloss `0.326744 -> 0.331255`、Brier `0.103220 -> 0.104912`、expected-error MAE `4.537501 -> 4.610538`とすべて悪化しました。fixed Viterbiの`1000_plus` RMSEは`+0.352222`、exp115 spatial / typewell-purged hidden-likeはそれぞれ`+0.328867 / +0.302609`、worst-well `389ae58f`は`+15.575246`悪化しました。well単位では360改善 / 413悪化です。

original-only controlのfixed Viterbi RMSE `8.421415`自体は、exp237 fixed Viterbi `8.545093`から`-0.123678`、exp218 `8.475794`から`-0.054379`です。ただしexp237由来のOOF-only contextを含むtrain-side結果であり、raw-test parityを満たした採用候補とは扱いません。

## 解釈

augmented variantは各foldのclean 660,000 candidate-long rowsに対し約619,000 perturbed rowsを追加しており、synthetic viewがclean viewとほぼ同じ学習量になりました。最大80 ftのshift/drift、dropout、spreadを一括で混ぜた結果、clean候補上のlikelihoodとexpected-error calibrationを広く崩した可能性が高いです。これは学習量と全指標悪化からの推論であり、7 transform個別の原因帰属は本実験からはできません。

## 採否

- `adoption_supported=false`
- selected RMSE、candidate logloss、`1000_plus`、hidden-like、worst-wellの5 guardはすべてfail
- augmentation mix / amplitude / transform gridの事後探索は行わない
- augmented modelのraw-test inference、direct候補選択、submitは行わない
- original-only dual-objective controlはtrain-side positiveとして記録するが、raw-test-safe featureだけに限定した別監査を通すまで推論化しない

## 次のアクション

augmentation branchは終了します。後続候補は`raw_test_safe_dual_objective_candidate_ranker`で、まず学習なしのfeature provenance / fallback監査を行い、raw-test parityが成立した場合だけaugmentationなしの10 CPU boostersを検討します。
