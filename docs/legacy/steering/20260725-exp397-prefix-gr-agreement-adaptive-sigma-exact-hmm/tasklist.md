# タスクリスト

## 完了

- [x] `exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm` を採番した。
- [x] 親をexp209、Routeを `pf_beam`、変更をwell-level `sigma_gr` 係数だけに固定した。
- [x] raw finite prefix pair、Pearson相関、minimum 64、threshold `0.50`、係数 `1.0 / 1.3`、
  fallback `1.0` を固定した。
- [x] base clip後に係数を1回だけ掛け、再clipしないHMM emission契約を固定した。
- [x] Stage 0の0-HMM coverage / non-degeneracy / full-tail stability gateを固定した。
- [x] Stage 1の1 variant / 5 reporting folds / 最大773 HMM runs / control再実行0と
  global/fold/scope/by-well/fixed-blend promotion gateを固定した。
- [x] stochastic componentなし、固定順、logical/decompressed SHA方針を固定した。
- [x] backlog、experiment summary、steering、experiment scaffoldをdesign-onlyで記録した。

## Stage 0実装完了

- [x] Stage 0実装を承認する。
- [x] Jupytext percent形式のcompact self-contained train候補と専用testを実装する。
- [x] 既存の正規`.ipynb`を上書きせずcandidate notebookを変換・静的検証する。
- [x] full/tail raw-row window、support、境界、fallback、再clipなし、truth-free freeze、
  7条件AND gate、fail-closed実行を専用11 testsで固定する。
- [x] Jupytext conversion test、py_compile、ruff、strict experiment validationを通す。

## Stage 0実行完了

- [x] candidateを正規train notebookへ採用する。
- [x] 0-HMM preflightでinput SHA、raw identity、truth-read 0、agreement schemaを確認する。
- [x] Stage 0のKaggle private CPU package/push/run承認を得る。
- [x] diagnostic 1 / reporting folds 5 / HMM・model config・trained fold・PF・Beam・booster・
  parent control再実行各0をpush前に記録する。
- [x] 773 wellsのcoverage、係数非退化、full/tail安定性gateとartifact SHAを記録する。
- [x] 固定7条件中4 PASS / 3 FAILを確認し、
  `stage_0_failed_close_without_rescue`でterminal-closeする。
- [x] Stage 1、inference、submission、version 2、事後調整、同family rescueを実行しない。

## 未実施（Stage 0 FAILにより対象外）

- Stage 1実装・最大773 exact-HMM well-runs。
- Stage 1 technical / global / fold / scope / changed-group / fixed-blend評価。
- inference / submission。
