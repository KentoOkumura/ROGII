# exp494_exp413_cat_xgb_physics_bounded_stack

## 状態

- ルート: `ensemble`
- 状態: reference submission COMPLETE / Public LB 7.228、不採用
- CV: bounded stack `7.827450885`、selected exp413 `7.884802794`
- Public LB: `7.228`（ref `55134873`、exp413 `7.201`比`+0.027`）
- Private LB: まだなし
- Submit ID: `55134873`
- 作成日: 2026-07-30
- 親実験: `exp413_scale5_likpf_full_replacement_on_exp335`

## 仮説

exp413 LightGBMを60%以上維持し、同じ370特徴を使うCatBoost / XGBoostと
固定物理候補を小さな非負weightで混ぜれば、scopeとwell-tailを悪化させず
exp413から0.03 ft以上改善できる。

## 設計

1. exp413のOOF、outer fold、final370 schema、Stage C/S、15 LGBをSHA固定する。
2. CatBoost `cb0`とXGBoost Cdeotte v3を各1 config x 5 foldsだけ学習する。
3. family別OOF、相関、残差相関、誤差共分散、by-wellを監査する。
4. 物理候補を`exp226_w500_50_50` 1本へ固定する。
5. LGB >= 0.60のleave-one-fold-out bounded stackingを行う。
6. 定数stackが全gateをPASSした場合だけ0.25 ft capのdisagreement gateを評価する。
7. train gate PASSと別承認後だけdynamic cardinality対応のhidden inferenceを
   同じ実験内に実装する。

## 実行量

- active variants: 2
- CatBoost configs: 1
- XGBoost configs: 1
- outer folds: 5
- 新規GPU models: 10
- exp413 control再学習: 0
- selector再学習: 0
- 新規PF/HMM/Beam: 0

## 検証方針

- Fold: exp413保存outer 5 folds
- Group: well
- Primary: suffix-row unweighted RMSE
- Meta readout: leave-one-outer-fold-out OOF-level cross-fit
- 固定scope: 0--250、1000+、hidden-like spatial / typewell-purged
- Tail: by-well p95非悪化、worst-well悪化+0.25 ft以内
- Leakage: final370 / fold / target固定後にtruthをjoinし、Public LBはweight選択に使わない

## 所見

- exp413は現ML submitted anchorだが、by-well tailは弱い。
- exp274 / exp275のnegative evidenceから、CatBoost / XGBoost単独改善は前提にしない。
- 物理候補は親exp413と同じ`likpf_scale_5_x1p0` overlayを持つ
  `exp226_w500_50_50`へ固定する。実測OOFは`8.070218793924594`。
  overlay前のexp263同名候補のPublic LB `7.800`は根拠へ転用しない。
- 10-model上限のためstacking readoutはstrict nestedではない。この制約を結果にも残す。

## 実装

- Jupytext source:
  `exp494_exp413_cat_xgb_physics_bounded_stack_compact_selfcontained_train.py`
- 変換Notebook:
  `exp494_exp413_cat_xgb_physics_bounded_stack_compact_selfcontained_train.ipynb`
- Stage 0: exp413 Stage 0/C/S/D SHA、row/fold、final370 schema、
  fold別float32 matrix content SHAを学習前に検証する。
- Stage 1--3: CatBoost 5本、XGBoost 5本だけを学習し、保存LGBと固定物理候補を
  同じ行順で監査する。
- Stage 4--5: fixed-bound SLSQPの5-fold OOF-level cross-fitと、
  constant stack PASS後だけの0.25 ft cap disagreement gateを評価する。
- version 2はfinal370生行列のSHAコピーとCatBoost内部Poolの同時保持による
  host RAM peakでkernel deathした。科学結果・再利用可能modelはない。
- version 3はzero-copy SHA、CatBoost Pool後の生行列解放、CatBoost/XGBoost
  行列の直列化を実装したが、学習前のStage 0後処理で停止した。
- version 4はallocator trim、物理OOFの25万行chunk ParquetWriter、
  列先行matrix assembly、chunk finite検証を追加した。設計と10-model契約は不変。
- version 4はStage 0を完了したが、fold 0 CatBoost Pool生成後のfit開始時に
  RSS high-water `27.526 GiB`でkernel deathした。完了model / CVは0。
- version 5はclean273特徴をfloat32 memmapへ一時退避して273列DataFrameを
  学習前に解放し、CatBoost train / valid Poolもraw matrixを直列解放する。
  fold matrix SHAでfinal370内容の不変を再検証する。
- version 5はCatBoost 5 + XGBoost 5を完走。bounded stackはexp413比
  `0.057352 ft`改善、5/5 foldsと全固定scopeを改善した。
- by-well p95 `+0.634421 ft`、worst `+3.843641 ft`で固定tail gateをFAIL。
  scientific selectionはexp413を維持する。
- その後のユーザー明示overrideにより、追加調整なしのconstant stackを
  hidden-safe参考提出する。conditional gate / routing / trajectory後処理は行わない。
- unit test 14件、Jupytext test、構文/F821を通過済み。

## 現在の承認範囲

2026-07-31のユーザー指示により、正規train Notebook採用、Stage 0--5 train、
およびscientific FAILを保持したconstant-stackのhidden inference・参考提出まで
承認済み。参考提出はPublic LB `7.228`でexp413 `7.201`より`0.027`悪く、
exp494は不採用、exp413をscientific / overall submitted anchorとして維持する。
ただしroute別の数値記録では、exp494が従来exp082 `7.601`を上回る
ensemble-route Public-LB referenceとなる。robust scientific promotionとは分ける。

詳細は
`.steering/20260730-exp494-exp413-cat-xgb-physics-bounded-stack/`、
`frozen_input_contract.yaml`、`ensemble_contract.yaml`を参照する。
