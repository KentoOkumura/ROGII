# exp278 formation gradient prefix stability risk readout on exp273

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU readout完了、primary guard FAIL、branch closed
- CV / Public LB / Private LB: 対象外
- Submit ID: 対象外
- 作成日: 2026-07-18
- 親実験: `exp273_two_dimensional_formation_gradient_transition`

## 仮説

exp273のfull-prefix formation planeが一見validでも、last-512 / last-256でgradient方向、
大きさ、fit RMSE、rank、conditionが崩れるwellでは、5 gradient candidateのdirect RMSE回帰が大きい。

## 変更点

- exp273 shard candidate、aggregate plane diagnostics、by-well metricsをSHA固定入力にする。
- raw trainのknown `TVT_input`と`X/Y/Z`だけからfull / last-512 / last-256 planeを再計算する。
- exp273 generation guardは変更せず、diagnostic-only Huber fitを別列に保存する。
- 6 stability componentを事前固定`[0,1]`変換し、等重み平均riskを作る。
- full-valid 111 wellsのbank-mean delta RMSEとの方向をstable SHA256 5 foldsで読む。
- HMM、gate、selector、raw-test inference、submissionは作らない。

## 検証方針

- Fold: `sha256("exp278::outer_fold::<well>") % 5` のreport-only 5 folds
- Group: well
- Primary cohort: exp273 full-gradient-valid 111 wells
- Primary outcome: 5 gradient candidatesのwell-level `delta_rmse_vs_scalar`平均
- Guard: fold別Spearman正方向5/5、pooled正方向、highest risk quintile > lowest
- Leakage check: outcome-like列をfeature freeze前に拒否し、logical SHA固定後だけoutcomeをjoinする

## 実行入口

- train notebook: `exp278_formation_gradient_prefix_stability_risk_readout_on_exp273_train.ipynb`
- inference notebook: disabled contractのみ
- package: `kentookumura/exp278-gradient-prefix-stability-readout-train`
- notebook実行はKaggle CPUを正とし、完了後は`run_approved=false`へ戻している。

## 利用可否

version 2は3,783,989 rows / 773 wells、full-valid 111 wellsを完走した。technical/parityは全PASS。
primary pooled Spearmanは`0.074245`、q4平均回帰`+2.157694 ft`はq0`-2.195778 ft`を上回ったが、
fold別は`+ / + / + / - / -`の3/5で、必須5/5を満たさない。gate候補にはしない。

## 所見

exp273 generation guardとdiagnostic-only fitを分離し、HMM候補生成条件を変えずにreadoutできた。
ただし信号はfold安定でなく、candidate別やbank-maxでも同じfold符号反転を示した。

## 次

救済grid、別gate、HMM再実行、raw-test inference、submissionは行わない。新規backlogは追加せず、
exp273 formation-gradient branchを完了negativeとして保持する。
