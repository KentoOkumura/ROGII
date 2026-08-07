# exp494 結果

## 状態

train-side実装とstatic validationは完了し、2026-07-31に実行承認、
canonical train採用後、Kaggle private T4 version 1を実行した。
Stage 0の物理候補semantics guardで学習0本のまま停止した後、
exp413 scale5-overlayへ契約を一意化したversion 2はkernel deathで停止した。
version 3もStage 0後処理のkernel deathで停止した。version 4はStage 0を越えたが、
CatBoost fold 0 fit開始時のhost RAM peakで停止した。version 5は10 modelsを完走し、
平均RMSEは改善したがwell-tail guardをFAILしたためterminal closeした。

## 仮説

exp413 LightGBMを最低60%維持し、同一final370面のCatBoost / XGBoostと
固定物理候補を小さく混ぜることで、scopeとwell-tailを悪化させず
pooled RMSEを0.03 ft以上改善できる。

## 固定設定

- 親: `exp413_scale5_likpf_full_replacement_on_exp335`
- 検証: exp413 outer 5 foldsのOOF-level cross-fit bounded stacking
- メトリック: suffix-row unweighted RMSE
- family: LGB / Cat / XGB / Physics
- 新規学習: 2 variants / 2 configs / 5 folds / 10 GPU models
- control再学習: 0
- 物理候補: exp413 scale5-overlay版`exp226_w500_50_50`
- confidence gate: constant stack PASS後だけ評価

## 結果

| メトリック | 値 |
| --- | --- |
| exp413保存CV | 7.884802794404715 |
| exp494 bounded stack CV | 7.827450885176479 |
| Public LB | 7.228 |
| Private LB | - |

exp413比gainは`0.057351909 ft`、5/5 foldsと全固定scopeで改善した。
一方、by-well p95は`+0.634420635 ft`、worst wellは`+3.843640672 ft`悪化し、
固定AND gateの2条件をFAILした。採用予測はexp413 LGBのままとする。

## 実装検証

- Jupytext percent source / 別名Notebookを作成した。
- exp413 Stage 0/C/S/D、final370 row/fold/schema/matrix content SHAを
  学習0本のStage 0でfail closedする。
- 学習量はCatBoost 5本 + XGBoost 5本の合計10 GPU modelsに固定した。
- family監査、bounded SLSQP cross-fit、deployment weight projection、
  条件付き0.25 ft cap gateを実装した。
- unit test: 13 passed
- Jupytext conversion/test、`py_compile`、`ruff --select F821`: passed
- `make validate-exp EXP=exp494_exp413_cat_xgb_physics_bounded_stack`: passed
- 全体`make test`はexp494実行前のcollectionで既存5実験の設定契約エラーにより
  停止した。exp494専用testは独立して全件PASSしている。

## Train version 1

- kernel: `kentookumura/exp494-exp413-cat-xgb-physics-bounded-stack-train`
- version / id_no: `1` / `129213293`
- runtime: `873.980775812 sec`
- failure stage: Stage 0 physical candidate contract
- observed candidate RMSE: `8.070218793924594`
- frozen original exp263 candidate RMSE: `8.238331`
- trained CatBoost / XGBoost models: `0 / 0`
- CV、stack、gateの科学結果: なし

原因は`ReplacementCandidateCache`がexp413のscale5 `likpf_mean` overlayを
適用して同名combinationを再構成した一方、凍結RMSEは元のexp263
`exp226_w500_50_50`を指していたcandidate semantic source mismatchである。

## Train version 2

- kernel version / id_no: `2` / `129213293`
- kernel death / 最終log: `1395.195736 / 1406.978336 sec`
- failure: `nbclient.exceptions.DeadKernelError`
- completed fold log / reusable output: `0 / 0`
- CV、stack、gateの科学結果: なし

final370の巨大生行列に対する`tobytes()` SHAコピーと、CatBoost内部Pool、
compact/signed DataFrameを同時保持したhost RAM peakが原因である。
version 3ではSHAをzero-copy化し、CatBoost Pool作成後に生行列を解放する。
XGBoost用行列はCatBoost終了後に同じfoldを再読込し、凍結SHAと再照合する。
feature/fold/model parameter/候補/stack条件は変更していない。

## Train version 3

- kernel version / id_no: `3` / `129213293`
- kernel death / 最終log: `1317.984938 / 1328.763580 sec`
- Stage 0 matrix preflight: `5 / 5 folds`完了
- family train開始log / 完了models: `0 / 0`
- CV、stack、gateの科学結果: なし

version 3の追加logにより、直接停止点は学習前のStage 0後処理と確定した。
解放済みfold memoryがresidentのまま、378万行物理OOF Parquet用DataFrameを
全行copyしたことによるhost RAM peakである。version 4では`malloc_trim(0)`、
25万行単位ParquetWriter、列先行matrix assembly、chunk finite検証へ変更した。
出力行・列・dtype・feature contentと科学条件は変えない。

## Train version 4

- kernel version / id_no: `4` / `129213293`
- kernel death / 最終log: `2372.486 / 2383.616 sec`
- Stage 0 matrix preflight / Stage 0 complete: `5 / 5` / 完了
- family train: fold 0 CatBoost Pool生成完了、fit開始後に停止
- RSS: Stage 0 complete `15.277 GiB`、train matrix ready `22.715 GiB`
- high-water mark: `27.526 GiB`
- 完了CatBoost / XGBoost models: `0 / 0`
- CV、stack、gateの科学結果: なし

version 4はchunk ParquetによりStage 0後処理を越えた。残る停止点は、
常駐clean273 DataFrameとCatBoost Poolに内部量子化work memoryが重なる
family学習時host RAM peakである。version 5はclean273特徴を一時float32
memmapへ退避し、学習前に273列DataFrameを解放する。fold matrix content SHAで
従来経路との完全一致をfail closedし、CatBoost train / valid Poolもraw matrixを
1本ずつ解放して構築する。科学契約と10-model数は不変。

## Train version 5

- kernel version / id_no: `5` / `129213293`
- status / runtime: `COMPLETE` / `5187.904674 sec`
- CatBoost / XGBoost models: `5 / 5`
- parent / selector / physics再学習: `0 / 0 / 0`
- CatBoost RMSE: `8.108026060`
- XGBoost RMSE: `8.052470087`
- fixed physics RMSE: `8.070218794`
- bounded stack RMSE: `7.827450885`
- exp413比gain / nonworse fold: `0.057351909 ft` / `5 / 5`
- by-well p95 / worst delta: `+0.634420635 / +3.843640672 ft`
- guard: `FAIL`（tail 2条件）
- selected prediction: `exp413_lgb`
- conditional gate / inference / submission（train closure時点）: 未評価 / 未実行 / 未生成

全auxiliary familyは単体でparentより悪く、by-well p95もCat
`+1.889396 ft`、XGB `+1.622275 ft`、Physics `+3.558947 ft`だった。
bounded stackは平均、全fold、全固定scopeを改善したが、pooled optimizationが
tail-riskを相殺できなかった。Physics weightは5 meta-fitすべて上限`0.20`に達し、
平均改善とwell-tail悪化が併存した。固定gateに従いweight/candidate/parameter/
bound/thresholdのsame-OOF救済は行わない。

## 再現性

- deterministic anchor: false
- seed policy: CatBoost 7 / XGBoost 42 / exp413 fold固定
- kernel version: 1（Stage 0 ERROR）/ 2・3・4（DeadKernelError）/ 5（COMPLETE）、
  id_no `129213293`
- feature schema / matrix manifest SHA:
  `45fbe8...94f3` / `1d0e2c...67e7`
- model manifest SHA: `206175...ab2c`
- OOF prediction SHA: `9cc8eb...6a38`
- blend weight SHA: `8630a4...ee9f`
- reproducibility manifest SHA: `f086f0...9cac`
- submission SHA: `29fc30575fb0bc528f6550e7f3e2158c764641e3988ffb0e5174119d643c510e`
- deterministic anchor: false（GPU bitwise一致は主張しない）
- manifest内の`exp494_train_metrics.json` SHAだけは最終rewrite前の値で、
  最終file SHAは`97c80d...5dc7`。その他のselective readbackは一致した。

## 2026-07-31 参考提出override

train-side判定はterminal FAIL、selected predictionは`exp413_lgb`のまま維持する。
その後のユーザー明示指示により、version 5のconstant stackを追加調整なしで
hidden-safe参考提出することだけを承認した。conditional gate、routing、
trajectory後処理、same-OOF weight/candidate/parameter救済は行わない。

inference v1はprivate T4 / internet offで`COMPLETE`、約393.9秒。
root `submission.csv`は14,151行、sample ID/order exact、duplicate / NaN / Inf 0で、
skill checkerとrepo checkerの両方をPASSした。SHA256は
`29fc30575fb0bc528f6550e7f3e2158c764641e3988ffb0e5174119d643c510e`。
competition code submission `ref=55134873`を
`2026-07-31 10:38:28.727000 UTC`に作成し、268分後に`COMPLETE`、Public LB
`7.228`を確認した。exp413 `7.201`より`+0.027`悪く、OOF上の`-0.057352`
改善はLBで再現しなかった。train-sideのtail guard `FAIL`と整合するnegative evidence
としてexp494は不採用、selected predictionとoverall / ML submitted anchorは
`exp413_lgb`のままとする。route別の純粋なPublic-LB記録ではexp082 `7.601`を
上回るためensemble-route referenceだけをexp494へ更新するが、robust scientific
promotionとは扱わない。
