# exp044_stratified_groupkfold_cv_audit 結果

## 仮説

Well metadata で層化した diagnostic split を追加すると、既存 OOF の弱い条件が fold variance、trajectory direction、TVT level、spatial location、eval length、GR coverage のどこに集中するかを確認できる。

## 設定

- 親: `exp026_pseudo_tail_bucket_shrink_inference_submit`
- 検証: `StratifiedGroupKFold` stress report by `well_id`
- メトリック: RMSE for configured OOF sources; fold balance has no scalar CV
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV | なし |
| Public LB | - |
| Private LB | - |

## 解釈

ローカル full audit は完了。773 wells から 49 strat labels を作り、fold balance と OOF stress metrics を保存した。

exp013 `lightgbm_no_gr` は raw 13.549257、fixed `exp014_bucket_shrink_params` 13.501824、last 基準 15.909853 を再現した。exp017 `dtw_dwt_no_gr` は raw 13.949718、bucket shrink 13.911474。StratifiedGroupKFold fold 別 exp013 raw は 14.127027 / 13.337070 / 13.058390 / 14.126412 / 13.072675。

この split は仮説に基づく診断 split なので、primary CV 置換、候補採用、ハイパラ調整、postprocess fit には使わない。primary GroupKFold / clean CV で改善した候補だけ、exp044 で特定 bucket を壊していないか確認する。

## 次

次の XGBoost / LGBM micro tune では、primary CV 改善を先に確認し、その後に exp044 の stratified fold / metadata bucket / distance bucket を red-flag check として見る。
