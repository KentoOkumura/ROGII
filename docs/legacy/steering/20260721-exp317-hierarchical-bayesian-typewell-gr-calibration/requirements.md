# 要件

## 依頼

exp311のplug-in校正統計をglobal→Type Well群→wellの階層Bayesへ置き換え、少数peerでもnoise/reliabilityを安定推定できるか監査する。設計のみで実装しない。

## 制約

- Route: `pf_beam`。exp311/313 PASSが先行条件。
- outer-trainだけでhyperparameterをfitし、outer-valid scoring前にposteriorを凍結する。
- primaryはsigma-only hierarchy + identity affine。full affine hierarchyはdiagnosticだけ。
- MCMC、decoder統合、inference、submissionは禁止する。

## 受け入れ基準

- Student-t df=5、global/group/wellの3階層、deterministic MAP/Laplaceを固定する。
- global-only、full affine+sigma、unpooled wellをdiagnostic ablationにする。
- predictive NLL +0.01、suffix RMSE +0.05 ft、4/5 folds、leave-group-out非悪化、worst +0.25 ft以下を要求する。
- posterior/model manifestとcontent SHAを保存可能な設計にする。
