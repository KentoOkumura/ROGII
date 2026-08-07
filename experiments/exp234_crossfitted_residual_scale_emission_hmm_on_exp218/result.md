# exp234_crossfitted_residual_scale_emission_hmm_on_exp218 結果

## 状態

Kaggle train v1 では HMM cache 生成後に exp115 hidden-like assignment の誤った basename で停止した。修正済み comparison-only readout を同一 kernel の v2 として実行し、v1 cache を再利用して完走した。

**train-side 不採用。inference / submit は行わない。**

## 仮説

exp218 の強い保存済み point OOF を emission center に固定し、well-cross-fitted residual scale だけを row-wise sigma に使えば、同じ well の真値 residual を使わずに HMM posterior を改善できる。

## 実行設定

- Route: `ensemble`
- Point center: 保存済み exp218 `lgb_mean` OOF（LightGBM 再学習なし）
- Residual scale: well GroupKFold 5-fold の cross-fitted `HistGradientBoostingRegressor`
- HMM: `lambda=0.50`、sigma floor `2.5`、cap `40.0` の 1 variant
- v2 readout: residual-scale fit 0、HMM 再計算 0、LightGBM booster 0
- v1 HMM cache: 3,783,989 rows / 773 wells、content SHA `45c3b4a60a1f83e55c0b2aa965a4971adeb79bb637688b31de78fa88cfa6a911`

## 結果

v1 の residual-scale guard は通過した。Spearman `0.326486`、top/bottom scale-decile RMSE ratio `3.578534`、floor rate `0.180690`、cap rate `0.0`、fold well overlap `0` である。

| candidate | RMSE | MAE | delta RMSE vs exp218 | delta RMSE vs exp148 | delta RMSE vs exp193 |
| --- | ---: | ---: | ---: | ---: | ---: |
| exp234 HMM + exp218 `lgb_mean` | 8.427231402 | 5.155675839 | -0.048573356 | -0.074059583 | -0.029444651 |
| exp218 `lgb_mean` | 8.475804758 | 5.322180439 | 0.000000000 | -0.025486226 | +0.019128705 |
| exp193 `lgb_mean` | 8.456676053 | 5.318167527 | -0.019128705 | -0.044614931 | 0.000000000 |
| exp148 `lgb_mean` | 8.501290984 | 5.335657607 | +0.025486226 | 0.000000000 | +0.044614931 |

- hidden-like spatial: RMSE `9.578578`、exp218 比 `-0.083018`。
- hidden-like typewell-purged: RMSE `9.547965`、exp218 比 `-0.088034`。
- distance bucket は `000_050` から `1000_plus` まで exp218 より RMSE が改善した。long-tail `1000_plus` は `9.243798`、exp218 比 `-0.050661`。
- by-well は exp218 比で 526 / 773 wells が改善、247 wells が悪化。最大悪化は `f88ddb26` の `+1.257773` RMSE で、全体の小さい改善だけで推論化するには不十分である。
- step-delta は mean `0.013124`、p99 `0.078`、`>5/10/25` の各 rate は 0 で、HMM による不連続なスパイクはない。

## 判定

exp218 の point OOF を HMM center に保つ方針は、overall、全 distance bucket、両 hidden-like subgroup で正方向だった。しかし比較可能な既存 HMM anchor である exp221 fixed-sigma HMM の RMSE `8.327736951` より `+0.099494451` 悪い。residual-scale guard が有効でも、row-wise sigma をそのまま emission に入れることは scalar sigma より良い HMM posterior を作れなかった。

したがって raw-test residual-scale 再生成、inference、submission は実施しない。次候補は、同じ exp218 center で scalar sigma を対照に置いたうえで、cross-fitted scale を固定 sigma へ縮小する有限 ablation に限定する。
