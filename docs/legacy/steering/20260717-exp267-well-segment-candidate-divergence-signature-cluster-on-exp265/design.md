# 設計

## アプローチ

1. exp263 Stage 0 cacheからouter fold別に6 primitive candidateを読み、well単位の評価row順を固定する。
2. exp264と同じ`evaluation_progress = eval_position / eval_len`を使い、early/middle/lateの3区間へ分ける。
3. 各区間で次の6指標をraw rowに同じ重みを与えて集約する。
   - candidate bank row range mean / p90
   - centered candidate bank SVD energy entropyから求めるeffective rank
   - adjacent valid row間で候補順位が変わるrank switch rate
   - 6候補の全15 pair absolute gap mean / p90
4. 区間が空または有限値不足なら値をNaNのまま保持し、coverage/fallback flagを保存する。
   cluster入力だけouter-train medianで補完し、outer-valid統計を前処理fitへ使わない。
5. outer foldごとにmedian imputation、RobustScaler `[25,75]`、clip `[-10,10]`、KMeans K=3をfitする。
6. cluster centroidの18 scaled feature平均をdivergence indexとしてlow/middle/highへ並べる。
7. soft membershipはexp265と同じ`softmax(-centroid_distance / temperature)`とし、temperatureは
   outer-trainの最近傍centroid距離の正の値の中央値へ固定する。
8. 主seed`42 + outer_fold`と監査seed`10042 + outer_fold`を同じouter-trainへfitし、Hungarian
   centroid matching後のouter-valid agreementを保存する。global RNGは使わない。
9. OOF assignmentと全structure guardを確定した後だけexp264 Stage B candidate-long scoreをstreaming joinする。
10. Stage A全guard通過時だけ、18署名+3 membershipをcandidate-independent featureとしてexp264の
    candidate-long rowへadd-onlyするconditional Stage Bを候補にする。本実験では学習しない。

post-assignment separability guardは次のように固定する。

- candidate winner: semantic cluster別のactual absolute error最小primitiveをfoldごとに求め、
  low/middle/high winner patternの最頻値が4/5 folds以上かつpattern内に2候補以上ある。
- calibration direction: 6 primitive全体の`pred_abs_error - actual_abs_error`についてhigh-low差の
  非zero符号が4/5 folds以上で一致する。
- 上のどちらかを満たし、worst clusterは最高actual error wellを1本除いてもpooledで同じclusterのままとする。

## 実験範囲

- 対象実験: `exp267_well_segment_candidate_divergence_signature_cluster_on_exp265`
- Route: `ensemble`
- 親実験: `exp265_target_free_pairwise_candidate_divergence_soft_experts_on_exp264`
- candidate source: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- post-assignment score: `exp264_exp263_candidate_confidence_dual_selector` Stage B
- 変更する変数: block-level 295特徴からwell-segment 18 target-free divergence特徴への置換。
- 固定する変数: 6 primitives、15 pairs、outer well 5-fold、candidate score artifact、seed 42、K=3、
  segment境界、RobustScaler、clip、score対象row、parent/control再学習0。
- Stage A実行量: 0 variant / 0 config / 0 trained fold / 0 booster。
- Conditional Stage B: disabled、1 variant × 2 objectives × 5 folds = 10 CPU boosters、別承認。

## 再現性設計

- seed policy: fixed explicit seed `42 + outer_fold`、監査seed `10042 + outer_fold`。
- stochastic 処理の有無: KMeans initializationのみ。
- PF/Beam / likelihood-PF / seed baggingの新規実行: なし。保存済みcandidate cacheだけを読む。
- 並列処理と乱数の関係: sklearn `random_state`を明示し、global RNGやthread順序へ依存しない。
- CPU/GPU runtime: Kaggle CPU、GPU/TPU/internet off。
- train cache SHA: exp263 manifest/catalog、exp264 score/model manifest/metricsの期待SHAをfail-closed照合する。
- feature SHA: 18列schema、well signature logical content、OOF assignment、centroid/preprocessorを記録する。
- model / prediction / submission SHA: Stage Aでは対象外。conditional Stage Bもdisabled。
- Kaggle package bootstrap: prepare後にcanonical source、loose config/module、bootstrap ZIP内のSHAを照合する。
- deterministic anchor: Kaggle rerun前はfalse。単発assignment SHAだけでanchorと呼ばない。

## リスク

- リークリスク: exp264 scoreやactual errorをcluster fitへ混ぜるとregime leakageになる。score pathは
  assignment保存後にだけ開き、feature schema forbidden guardを通す。
- CV/LB不一致リスク: train-side well clusterが3 current-test wellsへ一般化しない可能性が高い。
  Stage Aはselector学習可否だけを判断し、inference/submissionを実装しない。
- ランタイム/メモリリスク: exp263 3.78M rowsとexp264 candidate-long約22.7M rowsを扱う。
  signatureはfoldごとに縮約し、scoreはParquet batch streamingでwell×cluster×candidate集計だけを保持する。
- 再現性リスク: KMeans label permutationとlocal optimum。semantic sorting、別seed centroid matching、
  assignment agreement、centroid/assignment SHAで監査する。
- guard overfitリスク: feature/segment/K/clip/thresholdを結果後にgridしない。
