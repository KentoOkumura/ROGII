# 要件

## 依頼

Trainのouter-train truthを使い、Type Well群ごとにどの物理candidate familyが得意かをsoft error priorとして保持し、nested selectorの補助特徴にできるか評価する。設計のみで実装しない。

## 制約

- Route: `ml_model`。exp313 transfer guardがPASSし、exp293 family manifestが固定されていること。
- priorはouter-train wellだけで推定し、outer-valid error結合前に凍結する。
- well ID prior、hard family router、candidate固有threshold、候補値変更は禁止する。
- Stage A 0-model readoutがPASSするまでStage B 40 selector modelsを開始しない。

## 受け入れ基準

- well等重みのgroup×family MAE/RMSE/best-rateをsupport 10 wellsでglobal familyへ縮約する。
- Stage Aはfold別family rank Spearman 0.15以上、4/5 folds正方向を要求する。
- Stage Bは親比0.03 ft以上、hidden-like非悪化、worst +0.25 ft以下を要求する。
- 未seen groupはglobal family prior、supportなしはneutralへ落とす。
