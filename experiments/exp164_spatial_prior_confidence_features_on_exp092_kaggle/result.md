# exp164_spatial_prior_confidence_features_on_exp092_kaggle 結果

## 状態

Kaggle CPU train v2 完了。`train_lgb0` / `train_lgb1` / `train_lgb2` に分割して実行した。

## 仮説

exp114 spatial neighbor prior は direct correction では改善幅が小さい一方、selector candidate としては oracle headroom がある。したがって hard switch ではなく、exp092 LightGBM が「spatial prior を信用できる場面 / 疑うべき場面」を判断するための add-only confidence feature として使う。

## 比較基準

- exp092 `lgb1` CV RMSE: 9.322479896
- exp092 Public LB: 8.350
- exp118 best spatial tiny gate CV RMSE: 9.321625436
- exp129 spatial candidate expanded oracle RMSE: 6.709127

## 結果

Kaggle logs / notebook cell output に基づく train-side OOF RMSE:

| config | pooled RMSE | exp092 lgb1 比 |
| --- | ---: | ---: |
| lgb0 | 9.660879008 | +0.338399112 |
| lgb1 | 9.429441976 | +0.106962080 |
| lgb2 | 9.415444308 | +0.092964412 |

最良は `lgb2` の 9.415444308 だが、exp092 baseline `lgb1` CV 9.322479896 を上回れなかった。spatial prior confidence features の add-only は、この条件では negative。

## 確認事項

- 各 split notebook は 1 config のみを学習するため、logs 上の `lgb_mean` は単一 config と同一。
- 3 config の ensemble OOF を再計算するには、各 notebook output の prediction を取得して結合する必要がある。
- 現時点では CV が全 config で baseline より悪いため、inference / submit へは進めない。

## 次アクション

採用しない。追加で調べるなら output を取得し、by-well / bucket / feature importance で spatial prior feature がどの regime で悪化したかだけ確認する。
