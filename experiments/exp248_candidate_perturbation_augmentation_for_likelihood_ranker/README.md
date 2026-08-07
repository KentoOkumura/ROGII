# exp248_candidate_perturbation_augmentation_for_likelihood_ranker

## 状態

Kaggle CPU train version 1完了。candidate perturbation augmentationは全5 guardを失敗し、不採用です。inference / submissionは行いません。

## 仮説

exp237の固定11候補はoracle headroomを持つ一方、候補誤差rankerはnear・worst-well・raw-test parityに課題を残した。
正解非依存のshift、drift、dropout、spread viewをouter-trainのcandidate-long教師へ追加すれば、clean original候補に対するwithin10 likelihoodとexpected errorの校正が改善する可能性がある。

## 実験範囲

- Route: `ensemble`
- 親: `exp237_hmm_exp226_candidate_selector_on_exp183`
- candidate bank: exp237固定11候補
- active variants: `original_only`, `perturbation_augmented`
- objectives: within10 classifier、expected absolute-error regressor
- validation: well単位5-fold GroupKFold、validはclean original候補のみ
- continuity: exp237固定Viterbi 1規則
- runtime: Kaggle CPU、GPU/internet/inference/submission無効

## 検証方針

well単位5-fold GroupKFoldで`original_only`と`perturbation_augmented`を同一fold比較します。augmentationはouter-trainだけへ適用し、outer-validではclean original候補に対するcandidate AUC/logloss/Brier、expected-error calibration、topK coverage、selected RMSE、固定Viterbi、distance/hidden-like/worst-wellを評価します。

## 所見

全7transformの決定性とclean-validation契約を維持して完走しましたが、fixed Viterbi RMSEは`8.421415 -> 8.728086`（`+0.306671`）と悪化しました。candidate logloss、`1000_plus`、hidden-like、worst-wellもすべてguard不通過で、5 foldsすべて悪化しました。original-only controlはtrain-sideではexp237を`-0.123678`上回りましたが、raw-test-safeとは扱いません。

## 主要ファイル

- `config.yaml`: 摂動grid、sampling cap、学習コスト、guard、生成物契約
- `candidate_perturbation_augmentation_for_likelihood_ranker.py`: heavy augmentation、multi-observation再計算、fold学習、評価、SHA
- `hmm_exp226_candidate_selector_on_exp183.py`: exp237の固定候補・context・Viterbi参照実装
- `exp248_candidate_perturbation_augmentation_for_likelihood_ranker_train.py/.ipynb`: 正のtrain notebook
- `exp248_candidate_perturbation_augmentation_for_likelihood_ranker_inference.py/.ipynb`: train-side-only停止guard

## 採否条件

augmentationがoriginal-onlyに対して、clean selected RMSE、candidate logloss、`1000_plus`、hidden-likeを悪化させず、worst-well回帰を事前上限内に保つこと。通過前はraw-test port、final exp218/238再学習、直接候補選択、submitを行いません。
