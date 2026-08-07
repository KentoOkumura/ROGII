# 要件

## 依頼

exp312の群別GR emissionをexp293 deployable12候補へ適用し、候補値を変えずrank/margin/entropyだけをexp264 selectorへadd-onlyする。設計のみでrankerやselectorは実装しない。

## 制約

- Route: `ml_model`。最終予測はML selectorが生成する。
- exp312/313 PASS、candidate manifest固定、outer/inner fold-safeが必須。
- hard top1、candidate value変更、新candidate、oracle rankは禁止する。
- Stage A 0-model rank readoutがPASSするまでStage B 40 selector modelsを開始しない。

## 受け入れ基準

- 追加列はrank percentile、top1 margin、entropy、availabilityの4列。
- Stage Aはtruth-nearest MRR +0.02、4/5 foldsを要求する。
- Stage Bはcorrected exp264比 +0.03 ft、hidden-like非悪化、worst +0.25 ft以下を要求する。
- saved corrected exp264 controlは再学習しない。
