# exp229_lgb_quantile_band_emission_hmm_on_exp148

## 状態

- Route: `ensemble`
- 状態: train-side 完了、不採用。inference / submit は行わない
- CV: quantile-band HMM `8.684401099`
- Public LB: 未提出
- 親実験: `exp148`, `exp193`, `exp209`, `exp221`

## 仮説

exp221 の fixed-sigma LGB emission HMM に対し、LightGBM の `q16/q50/q84` band から row-wise sigma を作れば、HMM が row ごとの予測信頼度を反映できる。

## 実装と実行

- quantile train は exp148 feature surface で 15 boosters を学習し、q16/q50/q84 OOF と saved booster を生成した。
- HMM audit は q50 を center、band sigma を observation uncertainty とした。
- 3 lambda の初回 audit は Kaggle 12-hour timeout となり、652/773 well の partial readout で最良だった `lambda=0.25` のみを再実行した。
- v3 audit は 3,783,989 rows / 773 wells で完走し、runtime は `11,320.273` sec。

## 検証方針

- LightGBM quantile は `well` GroupKFold 5 folds の OOF だけを HMM audit に渡す。
- HMM は exp148 / exp193 `lgb_mean`、exp072 `likpf_mean` と同一 train rows で比較する。
- central band coverage、crossing、sigma floor/cap を記録し、弱い quantile center や過小分散の band を隠さない。
- overall RMSE が既存 point-prediction anchor を上回れない場合、raw-test inference と submit には進まない。

## 結果

| 候補 | RMSE |
| --- | ---: |
| exp193 `lgb_mean` | 8.456676 |
| exp148 `lgb_mean` | 8.501291 |
| exp229 q50 | 8.685006 |
| exp229 quantile-band HMM | 8.684401 |

HMM は q50 からほぼ変化せず、exp148 比 `+0.183110`、exp193 比 `+0.227725` の悪化だった。central band coverage は `0.525238`、sigma floor rate は `0.767265` で、quantile band はこの emission sigma として十分に較正されていない。

## 所見

- 良かった点: timeout recovery で partial readout の最良 `lambda=0.25` に絞り、全 773 well の audit を完走した。HMM 自体は q50 をほぼ悪化させなかった。
- 悪かった点: q50 が既存 exp148 / exp193 point prediction より弱く、sigma も大半が floor に張り付いたため、row-wise uncertainty の利点を出せなかった。
- リスク: exp221 fixed-sigma HMM は train-side 改善があっても LB への転移が小さかった。弱い q50 center のまま lambda grid を増やしても、このリスクを解消しない。

## 判断

q50 が既存 point-prediction anchor より弱く、row-wise sigma もほとんど floor に張り付いたため不採用。exp221 fixed-sigma HMM よりも train-side で悪く、inference / submit は実装しない。

exp218 point prediction を center にした cross-fitted residual-scale uncertainty は、必要になった時だけ低優先の別仮説として扱う。
