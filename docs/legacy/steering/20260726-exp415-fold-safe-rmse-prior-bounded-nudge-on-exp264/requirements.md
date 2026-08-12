# 要件

## 依頼

exp407の原因分析で得た「候補別RMSEはtask weightではなくglobal priorとして使う」
という知見を、保存済みcorrected exp264 OOFへ適用する。候補別RMSEで親selectorを
置き換えず、各行のTVT補正量を数学的に制限した再現可能な方法として有効性を確認する。

## 制約

- Route: `ensemble`
  - corrected exp264のML selector scoreと、PF/Beamを含む候補TVTの両方が
    最終予測に本質的に寄与するため。
- 入力はcorrected exp264 Stage B v5 candidate-score OOFと、exp407が保存した
  exact fit-partition候補RMSE tableに固定する。
- 親model、候補生成、PF/HMM/Beam、LightGBMを再実行しない。
- candidate order 12、hard domain 11、fold 5、3,783,989 rows、773 wellsを固定する。
- 選択・補正量はtruth-free phaseでfreezeし、`actual_abs_error`はfreeze SHA確定後の
  evaluation phaseでだけ読む。
- 固定policy:
  - 親候補: `argmin(parent_pred_abs_error)`
  - RMSE-prior候補:
    `argmin(parent_pred_abs_error + fit_candidate_rmse)`
  - raw nudge: 親候補TVTから両候補50/50 blendへの差
  - bounded nudge: raw nudgeを`[-0.25, +0.25] ft`へclip
  - 最終予測: 親候補TVT + bounded nudge
- 0.25 ftはworst-well RMSE regressionを最大0.25 ftに上から制限するための
  risk budgetであり、score探索値ではない。
- RMSE係数1、blend 0.5、correction cap 0.25をgrid探索しない。
- 新規variant 1、model/config/fold fit/booster/GPU/control再学習はすべて0。
- inference、submission、LBは対象外。
- 初回full readoutはKaggle private CPU、internet offで実行する。
- 再現性は`docs/06_reproducibility.md`に従い、入力、freeze、row prediction、
  metrics、gate、Notebook package、kernel versionのSHAを記録する。

## 受け入れ基準

### Technical gate

- parent OOFと候補RMSE tableの実読込SHAが固定値と一致する。
- 12候補のblock order、11候補domain、5 folds、3,783,989 base rows、
  773 wellsが一致する。
- truth-free phaseのtruth readが0で、freeze file SHA確定後だけevaluationを開始する。
- RMSE tableは60 candidate × fold rowsで、fit row count / row-ID SHAが存在する。
- model / booster / candidate generation / inference / submissionがすべて0である。
- correctionがfiniteで、`max(abs(correction)) <= 0.25 ft`である。
- candidate値と保存actual errorから再構成したtruthが全候補でexact parityを満たす。

### Scientific gate

- pooled diagnostic OOF RMSEが親より`0.01 ft`以上改善する。
- fold別hard RMSEが5/5で親以下である。
- near 0--250、250--500、500--1000、1000+の4 bucketすべて親以下である。
- hidden-like spatial / typewell-purgedの両方が親以下である。
- observed worst-well RMSE regressionが`+0.25 ft`以下である。
- mathematical risk certificateとして、任意scopeのRMSE増分が
  correctionのL2ノルム以下、かつ0.25 ft以下であることをartifactへ記録する。

全ANDの場合だけ「候補別RMSEをfold-safe additive priorとして、risk-bounded
TVT nudgeへ利用する方法」を保存OOF診断上で確立したと判定する。
このreadoutだけではroute anchor更新、current-test採用、submissionを承認しない。

- prediction row artifactとKaggle kernel versionを記録する。
- submissionは生成しないためsubmission SHAは対象外。
