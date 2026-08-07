# exp399_soft_sticky_fused_exact_runtime_audit

## 状態

- ルート: `pf_beam`
- 状態: full OOF完了 / promotion rejected / branch closed
- 親実験: `exp394_soft_sticky_exp226_k16_branch_hmm`
- CV: `11.395645678`（exp263比`+3.157314012 ft`）
- LB / Submit: なし

## 目的

exp394 の soft-sticky E/H exact HMMについて、科学条件と全状態を変えずに
実行時間だけを短縮する。保存済みexp394 fixed16出力を再学習せず参照し、
予測・branch posterior・診断値の数値一致とstate-time正規化速度を同時に監査する。

## 仮説

exp394の遅さは状態数そのものより、P×R×5 tensor生成、可変pairwise
logsumexp、直列well処理の実装コストが大きい。同じ遷移確率を固定幅・factorized
kernelで評価すれば、全状態と予測精度を保ったまま実行可能時間へ短縮できる。

## 変更点

- TVT gridと41 residual-rate statesは全て保持する。
- P×R×5のposition destination / log-probability tensorを作らない。
- exp209型の固定幅3-rate / 5-position max+sum logsumexpを使う。
- source側の境界正規化は境界stateでだけon demandに計算する。
- docking、H→E split、rate propagationを同じforward kernelへ融合する。
- backwardでもrate reduction、docking、switch、beta更新を融合する。
- independent wellsを2並列、各workerのNumba thread maskを2に固定する。
- well_id / row_idxのstable sortとtruth-before-freeze禁止を維持する。

## 実行範囲

fixed16 technical preflight後に別承認を得て、scientific variant 1、
773 HMM well runs、5 reporting foldsのfull OOFまで完了した。
LightGBM config / trained fold / booster / parent rerun / GPUは全て0。

## 判定

- exp394との差: prediction 1e-5 ft以下、branch probability 2e-6以下、
  diagnostics 2e-5以下
- fixed16 state-time正規化speedup: 3.684212倍以上
- projected full runtime: 30,600秒以下
- full-grid coverage / finite coverage: 1.0
- normalization 1e-8以下、transition row sum error 1e-10以下

## 検証方針

小規模trellisではexp394関数との全出力一致を確認する。本番判定は保存済みexp394
fixed16の同一keys / schedule / state-time unitsに対し、Kaggle CPUの壁時計時間と
prediction / posterior / diagnosticsを比較する。truthとRMSEはtechnical preflightで
読まない。

## 所見

Kaggle version 2はruntimeを5.367倍へ短縮した一方、同一kernelのversion 3は
CPU実行差で3.107倍だった。固定幅reductionの演算順差は
prediction RMSE `5.65e-7 ft`、p99 `3.63e-6 ft`、最大`7.88e-6 ft`だったため、
全状態・確率・scheduleを変えず実用的数値同値を`1e-5 ft`と定義した。
遅いruntimeでもgateを満たすためbackwardの中間配列とparallel launchを削減した。

Kaggle version 4で全gateをPASSした。decodeは`589.600 sec`、親比
`6.168x`、full 773-well投影は`18,277.265 sec`（`5.077 h`）。
full OOFは別承認後にversion 5で実行した。773/773 wellsを`28,107.311 sec`で
decode・prediction freezeまで完了したが、late readoutがexp226 foldとexp263の独立
outer-foldを同一と誤って要求しERRORになった。科学計算の失敗ではない。両fold ledgerを
独立に検証し、late readout前にfrozen predictionをcheckpointするtechnical-only修正を
version 6へ入れた。

version 6は3,783,989 rows / 773 wellsを`25,118.127 sec`で完走し、
全technical gateをPASSした。候補RMSEは`11.395646`でexp209
`11.938288`より改善したが、promotion baselineのexp263 `8.238332`より
`3.157314 ft`悪化し、改善foldは`0 / 5`だった。1000+、hidden-like 2面、
by-well p95、worst-wellもすべて固定guardをFAILしたため、
`full_oof_rejected_no_rescue`で閉じる。parameter救済、blend、selector、
inference、submissionは行わない。runtime kernel自体は全状態・実用的数値同値を
保った再利用可能な実装として残す。

## 次

exp399は追加実行なしで閉じる。runtime kernelは、将来の独立した事前固定候補で
全状態exact HMMの計算量が障害になった場合だけ再利用する。
