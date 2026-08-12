# 要件

## 依頼

保存済みの `exp293` deployable 12物理pathだけを使い、区間単位の
semi-Markov posteriorでTVTを融合できるかをtrain-side OOFで検証する。
`exp399`で問題になったHMM誤modeからの復帰不能を避けるため、
`exp226_k16` geometry stateへの再注入確率はdocking・GR尤度・現在modeと
独立したfloorとして持つ。

初回依頼ではbacklog、実験scaffold、steering、configと設計記録だけを作成した。
2026-07-26の追加依頼「exp405を実装してください」により、別名compact
self-contained train候補とsynthetic contract testまでをimplementation-onlyで
実装する。正規Notebook採用、Kaggle package / preflight / full実行、
current-test生成、inference、submissionは行わない。
train-sideの全scientific gateを通過した場合だけ、同じexp405内で
current-test実装を設計・実装できる状態へ昇格させる。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- candidate bank、順序、float値、row identity、fold、H256/H512 blockは
  `exp293` version 2の保存生成物とSHAへ固定し、pathを再生成・追加・削除しない。
- primary base blockはnon-overlap H256、candidate最小durationは2 blocks
  （H512相当）。最終short segmentだけ右打切りを許す。
- 観測はcandidate TVTの周囲`±55 ft`、5 ft刻みの局所Type Well GRだけを使う。
  registration offsetは周辺化し、最終TVTへ加えない。
- `exp226_k16`への区間開始prior floorは0.10とし、docking、GR score、
  HMM/PF uncertainty、現在candidateで変えない。
- hard top1、Viterbi、row-wise switch、ML selector、same-OOF parameter救済は禁止する。
- candidate score、posterior、predictionとlogical content SHAをfreezeするまで、
  suffix truth、error、oracle、hidden-like roleを読まない。
- `exp297`のprefix-affine Student-t/NCC/derivative evidence、
  `exp399`のdocking依存branch transition、`exp370`のtrigger resetを再利用しない。
- train-side PASS前はcurrent-test candidate生成器、正規Notebook、Kaggle packageを実装しない。
- implementation-only完了時も`run_fixed16_preflight=false`、
  `run_full_saved_oof=false`を維持し、各実行は別承認を要求する。

## 受け入れ基準

- 保存入力のrow / well / foldは`3,783,989 / 773 / 5`、candidate数は12で、
  exp293 candidate content SHAとblock-assignment decompressed SHAが一致する。
- 同じduration・boundary制約下のtruth-late constrained oracle RMSEが
  `<=5.50 ft`、全5 foldでexp263を改善する。
- primary posterior-mean OOF RMSEが`<=6.90 ft`で、exp263
  `8.2383315465 ft`を5/5 foldsで改善する。
- 1000+、hidden-like spatial、hidden-like typewell-purgedの全scopeで
  exp263を改善し、by-well RMSE delta p95は`<=0`、worst regressionは
  `<=+5.0 ft`とする。
- real GRが各negative controlよりpooledで`>=0.05 ft`良く、
  5/5 foldsで良い。
- `exp226_k16` posterior massはpooled mean`>=0.05`、
  per-well meanのmedian`>=0.02`、mean mass `<0.005`のwell率
  `<=0.10`とし、geometry branch collapseを許さない。
- prediction finite coverage 1.0、candidate convex-hull内1.0、
  block-center weight interpolationとphysical continuity guardがPASSする。
- fixed16 preflightからfull saved-OOF runtime `<=7,200 sec`、
  peak RSS `<=25 GB`と見積もれる。
- 上記を全ANDで満たした時だけ`current_test_implementation_eligible=true`。
  1条件でもFAILならexp405を救済せず閉じ、exp406 Stage 0を解禁する。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
