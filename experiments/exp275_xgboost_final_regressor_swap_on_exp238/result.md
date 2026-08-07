# exp275 XGBoost final regressor swap on exp238 結果

## 結論

Kaggle T4 train version 2は完走したが、XGBoost単体と固定0.25 blendはいずれも親LightGBMよりRMSEが悪化した。全raw guardがFAILしたため不採用とする。2026-07-18のユーザー明示承認により参考推論とraw XGBoost 1件のスコアリングを完了し、submission ref `54798185`のPublic LBは`7.760`。exp238 7.775は`-0.015`上回ったが、現ML submitted anchor exp274 7.715より`+0.045`悪いため、採用判断とanchorは変更しない。

## 仮説

`exp238`の415特徴面を固定し、最終LightGBMだけを公開設定XGBoostへ差し替えることで、単体RMSEまたは固定0.25 blendの多様性が改善する可能性がある。

## 設定

- 親: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`
- 特徴: exp218 base 380 + exp238 nested selector rank-slot 35 = 415
- 検証: 保存済みexp238 outer 5 fold role、well分離
- 比較基準: 保存済み`exp238/lgb_mean` OOF
- 新規学習: XGBoost 1 config × 5 folds = 5 boosters、合計2,250 trees
- 親/control/selector再学習: 0
- パラメータ: `cdeotte/xgb-starter-cv-15` version 3、450 trees、early stoppingなし
- シード: `random_state=42`
- Kaggle kernel: `kentookumura/exp275-xgb-final-regressor-exp238-train` version 2 / T4

## 結果

| model | RMSE | delta vs parent | MAE | within 10 ft | parentとの予測相関 |
| --- | ---: | ---: | ---: | ---: | ---: |
| exp238 saved `lgb_mean` | 7.936690 | 0.000000 | 4.929302 | 0.866401 | 1.000000 |
| XGBoost public Cdeotte v3 | 8.302528 | +0.365838 | 5.121350 | 0.863203 | 0.999996 |
| fixed XGBoost 0.25 blend | 7.990747 | +0.054057 | 4.946717 | 0.866600 | 1.000000 |

固定blendはwithin 10 ftだけを`+0.000199`改善したが、主指標RMSEとMAEはともに悪化した。XGBoostとparentの予測相関が`0.999995765`と非常に高く、固定blendに使える実用的な誤差多様性は得られなかった。

### Fold別

| fold | parent RMSE | XGBoost RMSE | delta |
| ---: | ---: | ---: | ---: |
| 0 | 8.019499 | 8.264982 | +0.245483 |
| 1 | 8.757457 | 9.070212 | +0.312755 |
| 2 | 7.123627 | 7.209015 | +0.085388 |
| 3 | 7.878675 | 7.941296 | +0.062621 |
| 4 | 7.818025 | 8.890900 | +1.072875 |

改善foldは`0/5`。fold 4の悪化が特に大きいが、ほかの4 foldsもすべて悪化しており、単一foldだけの問題ではない。

### Stress surface

| surface | parent RMSE | XGBoost RMSE | delta | fixed 0.25 blend delta |
| --- | ---: | ---: | ---: | ---: |
| distance 1000+ | 8.703449 | 9.103832 | +0.400383 | +0.059918 |
| hidden-like spatial | 8.622845 | 9.291311 | +0.668466 | +0.117840 |
| hidden-like typewell-purged | 8.599617 | 9.261592 | +0.661976 | +0.116694 |

- worst well `86454a6f`: parent 30.924660、XGBoost 44.804670、delta `+13.880009`
- raw overall / 1000+ / hidden-like 2面 / worst-well / 3-of-5 folds の全guard: FAIL
- `all_raw_guards_pass=false`
- `inference_allowed=false`

## 実行履歴

- version 1はapproval status文字列不一致でデータ読込前に停止。約21.19秒、booster 0本。
- version 2は科学条件を変えずcontractだけを修正し、5/5 boostersを完走。elapsed 2,984.807秒（約49分45秒）。
- XGBoost prediction時のCPU `DMatrix` / GPU booster mismatch警告と、DataFrame fragmentation警告は性能上の警告であり、成果物・行coverage・SHA監査は完了した。

## 参考推論・提出

- inference kernel: `kentookumura/exp275-xgb-final-regressor-exp238-inference` version 2 / T4
- runtime: 415.815秒
- output: 14,151行 / 3 wells / fallback 0
- loaded models: XGBoost 5 / parent LightGBM 15 / selector 20、推論時学習0
- raw prediction range / mean: `11590.978516 - 12242.695312` / `11904.817982`
- raw XGBoost vs parent test予測差: RMSE `0.917322348`、mean `-0.106029457`、max abs `2.934570312`
- fixed 0.25 blend vs parent test予測差: RMSE `0.229326880`
- submit-check: 14,151行、`id,tvt`、sample ID順、重複0、finite、SHAすべてPASS、FAIL/WARN 0
- raw submission SHA: `79452e652e75c3e7f60cb3b77c39dd4f4e175f853f4b1d49accc28b67c70a01c`
- submission ref: `54798185`、submitted `2026-07-18T03:43:53Z`
- Public LB / Private LB: `7.760` / 未公開
- duplicate submission: ref `54798337`もPublic LB `7.760`。monitorが追従したが、正規記録はSHA追跡済みref `54798185`

### LB比較

| reference | Public LB | exp275との差 |
| --- | ---: | ---: |
| exp082 ensemble anchor | 7.601 | +0.159 |
| exp274 ML submitted anchor | 7.715 | +0.045 |
| exp257 | 7.718 | +0.042 |
| exp275 raw XGBoost | 7.760 | 0.000 |
| exp238 hidden-safe | 7.775 | -0.015 |
| exp218 | 7.843 | -0.083 |

## 再現性

- deterministic anchor: いいえ。XGBoost GPU学習のbitwise一致は未保証。
- XGBoost version: 3.2.0
- public notebook SHA: `348323bd9f449b566301051ca1842692f4ba54bdf05e7cfcc8faa7fc72617f70`
- parent OOF decompressed SHA: `0e7390ac3b3a432b1d432e432cb374cbf38da393a9b95f8f0d6c22732030010c`
- base feature content SHA: `f6ff78f6a95e47b0ed8e76a22c31d3403d0a9e78471b7d64f37eef7a2a398e29`
- feature schema SHA: `85d57f2fce115f54d861c61bf47ba37eba3723d55ab71a4b074161460856805c`
- OOF decompressed SHA: `285614e2c510e3250012b832f8b84d91e6d83a23ac153a7fee4ecdfa31554744`
- model manifest SHA: `0ecffa597108cfa86471bf7019c92d672a191bef56d1f844e0441018a5690d5a`
- summary SHA: `12fbac5b3418d80e2e911623f1b30c92aa75f72dc7d92016d87f23d3e09bb143`
- inference summary SHA: `7da364539030e039b31313bb5f0c108aa82cc395060dc5a12ca9721b6e39658e`
- raw submission SHA: `79452e652e75c3e7f60cb3b77c39dd4f4e175f853f4b1d49accc28b67c70a01c`

一時取得したKaggle outputで、3,783,989行のOOF decompressed SHA、5モデルのSHA、主要artifactのSHAがsummary / manifestとすべて一致することを確認した。fold別train / valid matrix SHAの詳細は`metrics.json`に記録した。

## 解釈

公開XGBoost設定は、この415列feature surfaceのfinal estimatorとしてexp238 LightGBMを代替できない。全5 folds、1000+、hidden-like 2面で一貫して悪化し、worst-well regressionも大きい。予測相関がほぼ1で固定0.25 blendも悪化したため、model-family diversityの補完性も支持されない。

入力したexp238 selector summary自体は`selector_guard_failed_final_train_forbidden`である。この結果は同じ保存済みexp238 surface上のfinal estimator比較として解釈し、exp238 selector featureの因果的採用根拠にはしない。

公開parameter grid、early stopping追加、selector XGBoost化、blend weight探索による救済は、事前の単一設定監査を超えるため追加しない。新しいXGBoost救済backlogも作らない。

## 次

- reference-only current-test推論、raw XGBoost 1件のsubmit、Public LB記録は完了。
- `KAGGLE_DIRECTION.md`のtrain待ちbacklogからexp275を削除済み。
- parameter / early-stopping / blend rescueやanchor更新は行わず、scoring記録後もnegative resultを維持する。
