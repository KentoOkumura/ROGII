# 設計

## アプローチ

Stage Aではouter-train emission tableでdeployable12各候補のGR log-likelihoodを計算し、scoreを候補内rank percentile、top1 margin、softmax entropyへ圧縮する。truthは特徴凍結後のrank quality評価にだけ結合する。Stage BはStage A PASS時のみ、4列をcorrected exp264 nested dual selectorへadd-onlyし、候補path/valueを固定して再学習する。

## 実験範囲

- 対象: `exp315_typewell_group_candidate_likelihood_rank_features`
- Route: `ml_model`
- 親: `exp264_exp263_candidate_confidence_dual_selector`
- 依存: exp312、exp313、exp293 manifest。
- 変更: selector auxiliary 4列のみ。
- 固定: deployable12、exp264 outer5/inner4、2 objectives、saved control。
- 計算量: Stage A 0 model。Stage Bは1 variant × 2 objectives × 5 outer × 4 inner = 40 models。

## 再現性設計

- candidate ID/order、fold manifest、emission table SHAをhard preflightする。
- Stage A feature/schema/content/rank SHA、Stage B model manifest/model/OOF SHAを保存する。
- LightGBM seedとthread数を固定し、feature生成に乱数を使わない。

## リスクと停止条件

- GR rankがexp297同様に候補errorを識別できない場合、Stage Aで安価に停止する。
- rank feature availabilityとtest regeneration parityを満たさなければStage Bへ進まない。
- gainがあってもworst/hidden-like guard FAILならinferenceしない。
