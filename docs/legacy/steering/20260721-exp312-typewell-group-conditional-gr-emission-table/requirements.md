# 要件

## 依頼

exp311が平均指標で示した共通性を、単一 affine ではなく Type Well GR値・局所勾配・欠損状態に条件づけた群別GR residual分布として保持できるか監査する。2026-07-21のユーザー判断によりexp311全gate条件を上書きして実装へ進むが、fit-RMSE R²とworst-well FAILは既知リスクとして保持する。

## 制約

- Route: `pf_beam`。exp311全gate PASSという旧先行条件はユーザー判断で上書きする。
- 評価候補bankはexp293の固定deployable12とし、候補値・順序・formula・IDを変更しない。
- emission tableはouter-train wellだけで構築し、outer-valid truth結合前に凍結する。
- Student-t df=5、固定bin、support shrinkage、fallback順を変更しない。
- exact-HMM/PF/Beam decode、selector学習、inference、submissionは禁止する。

## 受け入れ基準

- 条件はType Well GR decile、|gradient| tertile、horizontal欠損flagに限定する。
- truth-nearest候補rankのMRR/top3、shuffled/shifted control差、fallback率を5 foldsで評価する。
- MRR +0.02、top3 +0.03、4/5 folds、fallback≤25%をすべて要求する。
- 未seen cellは固定 fallback chainで処理し、posthoc bin/grid調整をしない。
- baselineはouter-train global-unconditional Student-tとし、real tableとの差だけを評価する。
