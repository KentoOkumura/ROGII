# exp265_target_free_pairwise_candidate_divergence_soft_experts_on_exp264 結果

> **結果無効:** regime入力にtraining-onlyの`ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA`を直接含み、
> score readoutも無効なexp264 Stage Bに依存する。occupancy/stability/separability/calibrationを含む
> 全readoutは実行再現用の履歴であり、hidden-safeな診断・negative resultには使わない。

## 仮説

候補パス間の時間的な差分形状と非TVT contextだけで安定したregimeが作れ、そのregimeごとに
best primitive familyまたはexp264 expected-error calibrationが異なるなら、global selectorを
fallbackに残したsoft expertは検証する価値がある。

## 設定

- 親: `exp264_exp263_candidate_confidence_dual_selector`
- 候補source: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- 検証: outer well 5-fold、512-row block、outer-train-only RobustScaler + KMeans K=3
- Stage 0メトリック: occupancy、centroid-matched assignment stability、regime別best family、calibration bias差
- シード: 42、fold offset固定
- 実行量: 0 variant / 0 config / 0 fold training / 0 booster、親/control再学習0

## 結果（無効実行の再現用）

| メトリック | 値 |
| --- | --- |
| 実装 | 完了 |
| targeted unit test | 6件PASS、repository全79件PASS |
| Stage 0 Kaggle guard | FAIL |
| occupancy guard | FAIL（regime 0/1/2 = 2.20% / 5.28% / 92.53% blocks） |
| stability guard | PASS（5/5 foldsで1.000） |
| separability numerical guard | PASS（2 family、calibration bias range 1.013964 ft） |
| soft membership | mean max probability 0.997257（実質hard） |
| rows / wells / blocks / features | 3,783,989 / 773 / 7,787 / 295 |
| Stage 1 / inference / submission | 未実行 |
| CV / Public LB / Private LB | Stage 0では対象外 |

## 再現性

- deterministic anchor: false
- seed policy: fixed explicit seed + outer-fold offset
- kernel version: canonical version 2、`COMPLETE`
- feature schema SHA: `f703ce4e...de4c`
- block fingerprint logical SHA: `be391fec...6cf`
- block assignment SHA: `0e548140...730`
- row assignment SHA: `b1fb5271...7eec`
- model / prediction / submission SHA: Stage 0では対象外
- rerun result: 未実行

## 解釈

version 1のschema guard不備を修正したversion 2は正常完走した。FAILの原因は実装エラーではなく、
K=3 regimeの著しいoccupancy偏りである。regime 2が各foldの91.1--93.5% blocksを占め、
regime 1相当は5/5 foldsで中央値151--162行の末尾partial blockだけを分離した。scaled centroid距離は
`raw__md__range`、`raw__md__end_minus_start`、`raw__md__std`がほぼ100%を占め、地質・候補傾向ではなく
固定512-row windowの端数を検出している。

残るregime 0相当は0--2.2%で、存在するfoldでは主に`selfgr_hmm_a070`対`exact_hmm`のgap外れ値が
距離を支配した。terminal clusterのbest candidateは5/5 foldsで`exp226_k16`、通常clusterも4/5 foldsで
`exp226_k16`であり、fold 3だけ`selfgr_hmm_a070`がわずかにbestだった。pooled label集計の2-family PASSは、
fold 2でterminal clusterのlabelが0へ入れ替わることと極小outlier clusterを混ぜた結果なので、
expert学習の根拠にはしない。soft probabilityも平均最大0.997257で、狙ったsoft routingになっていない。

## 次

Stage 0 guard未通過のため、30 CPU boosterのconditional Stage 1、inference、submissionは実行しない。
exp264 global selectorは無効なのでfallbackにも使わない。このbranchは不採用として閉じる。再訪は
raw-test availabilityを満たす特徴だけで、block-length proxy除外とouter-train-only clipを固定した
0-booster監査に限定する。
