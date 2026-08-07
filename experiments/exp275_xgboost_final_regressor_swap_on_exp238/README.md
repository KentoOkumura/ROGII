# exp275 XGBoost final regressor swap on exp238

## 状態

- ルート: `ml_model`
- 状態: `reference_inference_completed_submission_scored_train_guard_failed`
- CV: 8.302528（parent 7.936690、delta +0.365838）
- Public LB: 7.760
- Private LB: -
- Submit ID: `54798185`
- 作成日: 2026-07-17
- 親実験: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`

## 仮説

`exp238` の380 base + 35 nested selector rank-slot特徴を固定したまま、最終 TVT 回帰器だけをLightGBMからXGBoostへ差し替えると、木モデル間の分岐差を利用して単体性能または固定blendの多様性が得られる可能性がある。

## 変更点

- 最終回帰器だけをXGBoostに変更する。
- パラメータは公開 notebook `cdeotte/xgb-starter-cv-15` version 3の`FAST_DEBUG=False`設定をそのまま使う。
- XGBoost 1 config × outer 5 folds = 5 boostersだけを学習する。
- `exp238` LightGBM OOFを保存済み比較基準として使い、親/control LightGBMとselectorは再学習しない。
- 0.25 XGBoost固定blendは多様性readoutとして1回だけ計算し、重み探索は行わない。

## 検証方針

- Fold: `exp238` nested score生成物に保存されたouter 5 fold roleを正とする。
- Group: well単位。train/valid well overlap 0をfail-closedで確認する。
- Primary metric: evaluation zone全行のRMSE。
- Stress: fold、距離bucket、1000+、hidden-like spatial/typewell-purged、by-well、worst-well。
- Leakage check: outer-valid selector scoreとouter-train inner-OOF scoreをrole別に読み、行coverage・id/well整列・保存SHAを確認する。

## 公開XGBoost設定

`n_estimators=450`, `learning_rate=0.035`, `max_depth=5`, `min_child_weight=20`, `subsample=0.85`, `colsample_bytree=0.85`, `reg_lambda=4.0`, `reg_alpha=0.05`, `objective=reg:squarederror`, `eval_metric=rmse`, `tree_method=hist`, `max_bin=256`, `random_state=42`, `n_jobs=-1`, `device=cuda`。

## 実行入口

- 正の編集対象: `exp275_xgboost_final_regressor_swap_on_exp238_compact_selfcontained_train.py` / `.ipynb`
- 推論: adoption用途はdisabled。2026-07-18にreference-only推論とraw XGBoost 1件のscoringをユーザー承認済み。
- Kaggle T4 train version 2で、承認済みの1 variant / 1 config / 5 folds / 5 boosters / control再学習0を完走した。

## リスク / 注意

- 公開設定は約35特徴のstarterで使われており、415特徴・3.78M行への転用ではRAMとruntimeが増える。
- 公開 notebook はearly stoppingを使わないため、各foldを固定450 treesで学習する。
- GPU XGBoostをbitwise deterministicとは扱わない。
- train guard不通過の採用推論は禁止。ユーザー明示承認のreference-only raw XGBoost scoringだけを例外とする。

## 所見

Kaggle T4 train version 2は`COMPLETE`。raw XGBoost RMSEは8.302528でparent 7.936690より`+0.365838`悪化し、5/5 foldsすべてで悪化した。1000+は`+0.400383`、hidden-like spatial / typewell-purgedは`+0.668466 / +0.661976`、worst wellは`+13.880009`で、全raw guardがFAILした。予測相関は0.999996と高く、固定0.25 blendも7.990747（`+0.054057`）へ悪化したため採用しない。

## 次

reference-only推論 version 2は415.815秒、14,151行、fallback 0で完了した。raw XGBoost `submission.csv`はsubmit-checkをPASSし、ref `54798185`のPublic LBは7.760。exp238 7.775より0.015良いが、現ML submitted anchor exp274 7.715より0.045悪い。parameter grid、early stopping追加、blend weight探索、anchor更新は行わない。
