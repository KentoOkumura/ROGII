# 要件

## 依頼

- exp280の固定shift-bank likelihoodからdepth aliasのtarget-free block confidenceを作り、exp264の誤差/tailとの関係だけをreadoutする。
- 直接補正、decoder変更、selector学習は行わない。2026-07-23に0-booster Stage 0の
  実装を承認し、その後の明示依頼でKaggle CPU Stage 0だけを実行した。

## 制約

- Route: `ensemble`。物理scoreをML候補bankのconfidence readoutへ使うが、予測生成は行わない。
- 科学的親は`exp264_exp263_candidate_confidence_dual_selector`、score sourceは`exp280_exp226_shift_likelihood_separability_readout`、negative decoder referencesはexp236/281/286とする。
- exp280の非重複512-row block、固定13 shifts `[-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80]`、Gaussian score、foldを固定する。
- target-free familyは`top1_top2_margin`、`softmax_entropy`、`weighted_shift_std`、`zero_shift_rank`、`abs_top1_shift`、`top1_jump_from_previous_block`、`three_block_sign_inconsistency`の7件に固定する。
- 各familyのQ1/Q4境界はtruth/error join前にfold別でfreezeする。
- late targetはfixed exp264 block RMSE、`abs_error>=10 ft`率、exp226がexp264よりblock RMSEを0.25 ft以上上回るalias-like failureとする。
- model/config/trained fold/booster/HMM/control再学習は0。

## 受け入れ基準

- 全7 familyのfinite coverageが99%以上、対象block/well identityがexp280/exp264契約と一致する。
- 少なくとも1つの事前登録familyがQ4-Q1 block RMSE差`>=+0.50 ft`、median差正、4/5 folds正、1000+・hidden-like 2面正を全て満たす。
- 同familyの`abs_error>=10 ft` AUCがpooled`>=0.60`、4/5 foldsで`>0.50`。
- stable circular block-order controlよりreal sequence familyがpooledと4/5 foldsで良い。
- PASSしてもadd-only特徴化は新しい別exp・別承認とし、本expでselectorを学習しない。
- 全family FAILならdepth-alias confidence枝を閉じ、feature/quantile/threshold救済を行わない。

## 次のアクション

7/7 familyが固定gateをFAILしたためterminal close。同familyの再実行、threshold探索、
family blend、補正、selector、推論、提出へ進まない。
