# exp274_catboost_final_regressor_swap_on_exp238

## 状態

- ルート: `ml_model`
- 状態: `reference_inference_completed_submission_scored_train_guard_failed`
- CV: 8.183504（parent 7.936690、delta +0.246814）
- Public LB: 7.715
- Private LB: -
- Submit ID: `54793316`
- 作成日: 2026-07-17
- 親実験: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`

## 仮説

exp238 の強い nested rank-slot feature surface 上では、公開 notebook 由来 CatBoost が
LightGBM と異なる誤差を持ち、単体または固定小量 blend で OOF を改善できる可能性がある。

## 変更点

- exp238 の outer fold、380 base features、35 nested selector rank-slot features、residual target を固定。
- final LightGBM のみ公開 `pixiux/rogii-dual-pipeline-blend` の CatBoost `cb0` に差し替え。
- 新規学習は 1 config x 5 folds = 5 CatBoost models。親/control、selector の再学習はなし。

## 検証方針

- Fold: exp238 保存済み outer 5 fold role
- Group: well 単位
- Metric: evaluation rows の pooled RMSE、MAE、within 10 ft
- 比較: 保存済み exp238 `lgb_mean` OOF
- 診断: fold、distance bucket、1000+、hidden-like spatial / typewell-purged、worst-well、固定0.25 blend
- Leakage check: outer fold role と fold-specific nested score artifact の row / id / well を fail-closed 照合

## 実行入口

- 学習 notebook: `exp274_catboost_final_regressor_swap_on_exp238_train.ipynb`
- 推論 notebook: `exp274_catboost_final_regressor_swap_on_exp238_inference.ipynb` 〈reference-onlyで完了、採用不可〉
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は行わない。

## 結果

| メトリック | 値 |
| --- | --- |
| CatBoost raw CV | 8.183504 |
| exp238 parent CV | 7.936690 |
| CatBoost delta | +0.246814 |
| fixed 0.25 blend CV | 7.950394（delta +0.013704） |
| reference raw CatBoost vs parent test RMSE | 1.270216 |
| reference fixed blend vs parent test RMSE | 0.317559 |
| Public LB | 7.715 |
| Private LB | - |

## 所見

- Kaggle T4 train version 1 は `COMPLETE`。5 CatBoost models を学習した。
- raw CatBoost は fold 2 のみ改善し、overall / 1000+ / hidden-like 2面 / worst-well / 3-of-5 folds の全 guard が FAIL。
- worst well regression は +12.293692、固定 0.25 blend も RMSE +0.013704 悪化。
- CatBoost GPU は bitwise deterministic とは扱わず、model / matrix / prediction SHA を保存する。
- 過去の exp012 では CatBoost が同一 no-GR feature の LightGBM より +0.301664 悪かったため、guard 不通過時の parameter rescue は行わない。
- reference inference version 1はT4で425.779秒、14,151行、fallback 0。raw CatBoost / parent / fixed0.25 blendの3出力はsubmit-check PASS。
- raw CatBoost code submission `ref=54793316` はKaggle API `COMPLETE`、Public LB 7.715。
- exp257のML Public-LB submitted anchor 7.718を-0.003更新するが、CV guard FAILのためtrain-side採用とは扱わない。ensemble anchor exp082 7.601は維持。

## 次

- reference inferenceとraw CatBoost提出は完了。LB submitted anchorだけを更新し、train-side採用、公開 `cb1`、parameter / blend-weight rescueは行わず閉じる。
- 次の既存 model-family audit は `exp275_xgboost_final_regressor_swap_on_exp238` とする。
