# 設計

## アプローチ

各outer foldのtrain wellsだけからexp311型のgroup support、residual sigma、fit RMSE、|bias@GR50|を作り、log support 2列とavailabilityを加えた6列を全rowへbroadcastする。outer-validにはType Well contentでのみjoinする。exp148 base featuresへadd-onlyし、lgb0/1/2を新規学習、saved exp148 OOFをcontrolとして比較する。

## 実験範囲

- 対象: `exp314_label_derived_typewell_gr_quality_addonly`
- Route: `ml_model`
- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 依存: exp311、exp313。
- 変更: 6つのgroup reliability featureのみ。
- 固定: exp148 fold/features/configs、control OOF、fallback、gate。
- 計算量: 1 variant × 3 configs × 5 folds = 15 boosters、control 0。

## 再現性設計

- exp148 fold/model seeds、固定num_threads、deterministic feature generationを使う。
- prior table/schema/content SHA、feature matrix SHA、15 model SHA、OOF SHAを保存する。
- inference/submissionを行う段階まではsubmission SHA対象外。

## リスクと停止条件

- label-derived group priorはfold外構築を誤ると強いleakになるため、outer-valid well IDがfit tableに0件であることをassertする。
- 既存target-free `same_typewell_gr_quality_features_on_exp092`とは別枝として扱う。
- global改善だけでworst-well guardを緩和しない。
