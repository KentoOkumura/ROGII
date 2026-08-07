# exp190_denoised_calibrated_matching_features_on_exp148 結果

## 状態

Kaggle train v1 完了。train-side CV では exp148 を超えなかったため、inference / submit には進めない。

## 仮説

raw / rolling median / Savitzky-Golay の GR shift-scan surface sharpness、posterior ambiguity、candidate disagreement、known-prefix backtest quality を exp148 に add-only すれば、exp145 learned likelihood confidence とは別系統の不確実性信号として効く可能性がある。

## 評価設計

- `denoised_calibrated_matching_addonly`: exp148 の feature surface に `denoised_calibrated_matching` feature group を追加する。
- `exp148_fulltrain_control`: 再学習しない。保存済み exp148 metrics を historical baseline として参照する。
- GroupKFold 5 folds、well group、metric は RMSE。
- GPU runtime、3 LightGBM configs、5 folds、15 boosters。

## 結果

Kaggle kernel: `kentookumura/exp190-denoised-calibrated-matching-exp148-train` v1

| model | pooled RMSE TVT | pooled RMSE target | exp148 同 config 比 |
| --- | ---: | ---: | ---: |
| `lgb0` | 8.601678275458363 | 8.601678160524132 | +0.001892416079473 |
| `lgb1` | 8.539624480134293 | 8.539624596538978 | -0.024346641095376 |
| `lgb2` | 8.540073561506953 | 8.540073587431210 | +0.030253842712878 |
| `lgb_mean` | 8.503596159484825 | 8.503596252227380 | +0.002314977589005 |

追加 feature join は 3,783,989 rows / 773 wells で full coverage pass。base row / well の drop は 0。最終 feature 数は 431。実行時間は 15,570.458 秒。

## 解釈

`lgb1` 単体は exp148 の同 config より改善したが、採用基準の `lgb_mean` は exp148 `lgb_mean` 8.50128118189582 から +0.002314977589005 悪化した。exp160 の train-side positive reference 8.463718773783008 からも +0.039877385701817 悪い。

したがって denoised calibrated matching feature block は、このままでは exp148 ML route anchor の追加特徴として採用しない。current-test feature parity の実装、inference、submit は行わない。

補足として、Kaggle logs では DCM feature generation 中に pandas の DataFrame fragmentation warning が多数出た。実行は完了しているため今回の結果は有効だが、この特徴生成を再利用する場合は列追加を `pd.concat(axis=1)` へ寄せる runtime 改善余地がある。
