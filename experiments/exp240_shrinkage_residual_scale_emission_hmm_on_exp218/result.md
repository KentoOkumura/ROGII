# exp240_shrinkage_residual_scale_emission_hmm_on_exp218 結果

## 状態

Kaggle scalar-control v2、alpha `0.25` v3、alpha `0.50` v4が完了。
ユーザー判断により本方向性をclosedとし、再実行・追加grid・inference・submissionは無効。

## 仮説

exp218-center HMM の cross-fitted row-wise sigma を fixed `sigma=20` へ分散縮小すると、
row-wise error rankingを一部保ちながら emission の過信を抑えられる可能性がある。

## 実行設定

- Kernel: `kentookumura/exp240-shrinkage-residual-scale-hmm-exp218-train` v2 / v3 / v4
- Route: `ensemble`
- center: 保存済み exp218 `lgb_mean` OOF
- scalar sigma: `20.0`
- lambda: `0.50`
- rows / wells: 3,783,989 / 773
- runtime: scalar 30,618.584 sec、alpha 0.25 29,341.119 sec、alpha 0.50 28,718.496 sec
- alpha 0.25: scale fit 5、HMM 1、LightGBM booster 0、well overlap全fold 0

## Overall

| candidate | RMSE | MAE | within10 | delta RMSE vs exp218 |
| --- | ---: | ---: | ---: | ---: |
| exp240 exp218-center scalar sigma HMM | 8.361307776 | 4.805670614 | 0.861589714 | -0.114496982 |
| exp240 variance-shrinkage alpha 0.25 HMM | 8.351122273 | 4.832954463 | 0.860046369 | -0.124682485 |
| exp240 variance-shrinkage alpha 0.50 HMM | 8.336863897 | 4.874378048 | 0.858952814 | -0.138940861 |
| exp218 `lgb_mean` OOF | 8.475804758 | 5.322180439 | 0.858116131 | 0.000000000 |
| exp234 exp218-center row-wise sigma HMM | 8.427231402 | 5.155675839 | - | -0.048573356 |

scalar sigma は exp234 row-wise sigma より RMSE `-0.065923625` 良い。異なるcenterの参考値である
exp221 exp148-center fixed-sigma HMM `8.327736951` には `+0.033570825` 届かない。

alpha 0.25はscalarよりRMSE `-0.010185503`改善したが、MAEは`+0.027283850`、within10は
`-0.001543345`悪化した。exp234 alpha 1.0相当よりRMSE `-0.076109129`良い。

alpha 0.50はalpha 0.25比RMSE `-0.014258376`、scalar比`-0.024443880`で有限grid最良。
一方、alpha 0.25比でMAE `+0.041423585`、within10 `-0.001093555`と悪化した。

## Guard readout

### alpha 0.50 vs alpha 0.25

- distance bucketは4 / 6改善。`250_500`は`+0.009600`、`500_1000`は`+0.001890`悪化。
- hidden-likeはspatial `+0.005456`、typewell-purged `+0.003032` RMSEと両方悪化。
- by-wellは352改善 / 421悪化、median `+0.008681`、mean `+0.032399`。
- 最大悪化は`6a8fa194`の`+2.898434`、最大改善は`efe96181`の`-6.760600`。
- scale guard pass、全fold well overlap 0。step delta `>5/10/25` rateはすべて0。

### alpha 0.25 vs scalar

- distance bucketは6個すべて改善。最大は`100_250`の`-0.017685`、`1000_plus`は`-0.010047`。
- hidden-likeはspatial `+0.005420`、typewell-purged `+0.004342` RMSEと両方悪化。
- by-wellは352改善 / 421悪化。median delta `+0.005307`、mean delta `+0.009298`。
- 最大悪化は`b3388334`の`+1.216115`、最大改善はscalarで最大悪化だった`2e63d9de`の`-4.456674`。
- scale guardはpass。sigma-error Spearman `0.326486`、top/bottom decile RMSE比`3.578534`、fold overlap 0。
- step deltaはmean `0.011154`、p99 `0.066`、`>5/10/25` rateはすべて0。

### scalar vs exp218

- distance bucket は6個中5個でexp218を改善。`500_1000`だけ `+0.022970` 悪化。
- `1000_plus`: RMSE `9.163031`、exp218比 `-0.131428`。
- hidden-like spatial: RMSE `9.550417`、exp218比 `-0.111178`。
- hidden-like typewell-purged: RMSE `9.519053`、exp218比 `-0.116946`。
- hidden-like within10はspatial `-0.004834`、typewell-purged `-0.003853`と小幅悪化。
- by-well: 501 / 773改善、272悪化。median delta `-0.352208`。
- 最大悪化: `2e63d9de`、exp218比 `+4.940864` RMSE。
- step delta: mean `0.011025`、p99 `0.065`、`>5/10/25` rate はすべて0。
- HMM std と誤差は単調でない。最低std bin RMSE `9.146311`、中央binは約`7.53-8.20`、最高bin `9.272461`。

## 再現性

- HMM feature decompressed SHA: `7967be9425b4d81e38b96fd6ee7943282edc0f7e74c3a1ad81b28af6f5f88a9f`
- overall metrics SHA: `53b4c7773755ef4bf6f9478ff3016c5cacce25460c1705999c5e62a4e6619916`
- audit summary SHA: `76ae9c8cdd02b0feab33b71462244e7beb3a49bea842353c903400582a90124e`
- alpha 0.25 sidecar decompressed SHA: `de45fa58dbfe48a97595706b887fe351fdd22ecbd54bd330a7db07942b643ebe`
- alpha 0.25 HMM feature decompressed SHA: `99ddbc681187c68c09f6f90889060a54a9822258b14609ce15cbb87eff196261`
- alpha 0.25 audit summary SHA: `dc6ffed1939d171482b82cc0ac86570cbe81a165d179e333091b71626703decb`
- alpha 0.50 sidecar decompressed SHA: `1057fa12fc9c4595c67ecbdd4f03307d01eae347c238ad9e4906ffdf0b967b67`
- alpha 0.50 HMM feature decompressed SHA: `8e2fcdcf3edf24f456ffb7aaba49dd5de85a9748291d2dcbb4e5a93b7bc0d4ff`

## 判定

alpha 0.50は有限gridの主指標RMSE最良だが、secondary guardはalpha増加とともに悪化した。
追加alpha探索は事前範囲外で過適合リスクもあるため終了する。alpha 0.50をtrain-side最良として記録するが、
推論化・提出は行わず、同じ仮説のgrid拡張も行わない。2026-07-14のユーザー判断で正式にclosed。
