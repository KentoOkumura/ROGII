# exp274_catboost_final_regressor_swap_on_exp238 結果

## 仮説

exp238 の final estimator だけを公開 CatBoost `cb0` に変えると、単体または固定小量 blend で
LightGBM と補完的な OOF を得られる。

## 設定

- 親: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`
- 検証: exp238 保存済み outer 5 folds / well group
- 特徴: exp218 base 380列 + exp238 nested rank-slot 35列 = 415列
- メトリック: pooled RMSE、MAE、within 10 ft
- CatBoost: 公開 `pixiux/rogii-dual-pipeline-blend` の先頭 config `cb0`
- シード: `random_seed=7`
- 新規 model: CatBoost 1 config x 5 folds = 5
- parent/control retraining: 0
- Kaggle kernel: `kentookumura/exp274-catboost-final-regressor-exp238-train` version 1 / T4

## 変更点

特徴、fold、residual target、selector score artifact を固定し、final estimator family だけを
LightGBM から公開 CatBoost `cb0` へ変更した。公開 `cb1`、parameter grid、parent control の
再学習は行っていない。

## 結果

Kaggle train version 1 は `COMPLETE`。raw CatBoost は parent より悪化し、全 raw guard が
不通過となった。

| model | RMSE | delta vs parent | MAE | within 10 ft |
| --- | ---: | ---: | ---: | ---: |
| exp238 saved `lgb_mean` | 7.936690 | 0.000000 | 4.929302 | 0.866401 |
| CatBoost public `cb0` | 8.183504 | +0.246814 | 5.036107 | 0.864123 |
| fixed CatBoost 0.25 blend | 7.950394 | +0.013704 | 4.918042 | 0.867004 |

固定 0.25 blend は MAE と within 10 ft をわずかに改善したが、主指標 RMSE は悪化したため
採用しない。

### Fold 別

| fold | parent RMSE | CatBoost RMSE | delta | fixed blend RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 8.019499 | 8.277125 | +0.257626 | 8.048541 |
| 1 | 8.757457 | 8.795045 | +0.037588 | 8.730566 |
| 2 | 7.123627 | 7.106430 | -0.017197 | 7.076314 |
| 3 | 7.878675 | 8.116635 | +0.237960 | 7.887253 |
| 4 | 7.818025 | 8.520375 | +0.702350 | 7.921508 |

改善は 1/5 folds のみ。CatBoost の best iteration は fold 0-4 で
`1037 / 3329 / 2007 / 257 / 259` と大きくばらついた。

### Stress surface

| surface | parent RMSE | CatBoost RMSE | delta |
| --- | ---: | ---: | ---: |
| distance 000-050 | 0.821728 | 1.079968 | +0.258239 |
| distance 500-1000 | 4.528790 | 4.652935 | +0.124144 |
| distance 1000+ | 8.703449 | 8.974516 | +0.271067 |
| hidden-like spatial | 8.622845 | 8.897100 | +0.274255 |
| hidden-like typewell-purged | 8.599617 | 8.874603 | +0.274986 |

- worst well `2fd68f7b`: parent 11.479815、CatBoost 23.773506、delta `+12.293692`
- raw overall / 1000+ / hidden-like 2面 / worst-well / 3-of-5 folds の全 guard: FAIL
- `all_raw_guards_pass=false`
- `inference_allowed=false`

## Reference inference

train-side guard不通過と不採用判断を維持したまま、ユーザー明示承認による参考推論を
Kaggle T4 kernel version 1で実行した。新規学習はなく、保存済みCatBoost 5本、parent
LightGBM 15本、selector LightGBM 20本を読み込んだ。

- kernel: `kentookumura/exp274-catboost-final-regressor-exp238-inference` version 1
- Kaggle id_no / runtime: `127707471` / 425.779秒
- rows / wells / features: 14,151 / 3 / 415
- fallback rows: 0
- root `submission.csv`: raw CatBoost 5-model平均
- comparison: parent LightGBM 15-model平均、固定`0.75 parent + 0.25 CatBoost`

| test prediction | min | max | mean | std |
| --- | ---: | ---: | ---: | ---: |
| raw CatBoost | 11594.281 | 12242.556 | 11905.497 | 278.480 |
| parent LightGBM | 11590.633 | 12240.474 | 11904.924 | 278.719 |
| fixed 0.25 blend | 11591.553 | 12240.987 | 11905.067 | 278.659 |

raw CatBoostとparentのtest予測差はRMSE 1.270216、平均+0.572795、平均絶対差
0.966147、最大絶対差4.244141 ftだった。固定0.25 blendとparentの差はRMSE
0.317559、平均絶対差0.241541、最大絶対差1.060547 ftである。

3出力はいずれも公式sample submissionに対して14,151行、`id,tvt`、重複・欠損・NaN・Inf
なし、ID順完全一致でsubmit-check PASS。固定blendはfloat32で式と完全一致した。

raw CatBoostはcode submission `ref=54793316` として提出され、Kaggle APIで`COMPLETE`、
Public LB 7.715を確認した。exp257のML Public-LB submitted anchor 7.718を-0.003、exp238
hidden-safe 7.775を-0.060、exp218 7.843を-0.128改善するため、ML routeのsubmitted anchor
だけをexp274へ更新する。全体のensemble anchor exp082 7.601には+0.114届かない。

一方、同一fold OOFではCatBoost 8.183504がparent 7.936690より+0.246814悪化しており、
CVとLBの方向は反転した。この提出はreference-onlyで、train-side guard FAILとモデル採用
判断は変更しない。

## 再現性

- deterministic anchor: いいえ。CatBoost GPU の bitwise 一致は未保証。
- seed policy: `random_seed=7`
- kernel version: 1、`id_no=127597836`、T4、elapsed 3,256.205秒
- public notebook SHA: `9f80687b9582b9b47a464613433afabe74274565252a2e235c152456a0d828e8`
- parent OOF decompressed SHA: `0e7390ac3b3a432b1d432e432cb374cbf38da393a9b95f8f0d6c22732030010c`
- feature schema SHA: `f0c11f34137de7ad011c7a8317ce24c3fafb4e09f9aaf5d064efd8c7ea2494a0`
- OOF decompressed SHA: `56a7f1bbeef0e703af74650d41e546343aa6f499a71b584f1a16992a5209aa55`
- model manifest SHA: `cba180df02928d66698a67970f774278a12f6c536a7e80e23546784e82614028`
- summary SHA: `181f1564014d81bee484a064d54601fc1e727c67a6b7682d9515fb5a87d28939`
- raw CatBoost submission SHA: `565c82f8d2a0118fde4741be9f1c510198189e662224db40b03eddc2da074dc5`
- parent reference submission SHA: `829709d6a4a27c7440412ae1b24aeab51734b30b19f59a78e9d0178dadcf6e0e`
- fixed 0.25 blend submission SHA: `a2d64a5f9053a847f2ea365c8b32262992b542254d861dce0fcb8c8f63b5b6aa`
- inference predictions decompressed SHA: `67b4f9fa3d402dd962f495d0218347ba0d677d75d19575b78974ac022395dfaf`
- inference summary SHA: `3dc6aa51b54666be90c4959d025040600878925e0b052d9e195ac56eac9f6a27`

モデル5本、fold別 train / valid float32 matrix、selector score 5本についても Kaggle output の
model manifest / summary に SHA を保存した。詳細値は `metrics.json` に記録する。

## 解釈

公開 `cb0` は、この 415列 feature surface の final estimator として exp238 LightGBM を
代替できない。fold 2 だけの小改善に対し fold 4 と worst-well の悪化が大きく、1000+ と
hidden-like 2面でも同程度に悪化した。固定 0.25 blend も RMSE を救済しないため、
model-family diversity の実用的な補完性は支持されなかった。

入力した exp238 selector summary 自体は
`selector_guard_failed_final_train_forbidden` / `selector_guard_pass=false` である。この結果は
「同じ保存済み exp238 surface 上の final estimator 比較」としては有効だが、exp238 の
selector feature の因果的採用根拠を強めるものではない。

過去 exp012 でも CatBoost は対応する LightGBM より悪かったため、公開 `cb1`、parameter
grid、blend weight grid による rescue は追加しない。モデル family の次の既存監査候補は
実装済み `exp275_xgboost_final_regressor_swap_on_exp238` とし、新しい CatBoost backlog は
追加しない。

## 次

- reference inference / raw CatBoost提出はPublic LB 7.715まで完了。ML submitted anchorだけを更新する。
- CatBoost branchはtrain-side negative resultのまま閉じ、公開`cb1`やparameter / blend-weight探索へ広げない。
- exp275もnegative完了済みのため、この415列surface上のfinal-estimator family差し替え枝を閉じる。
