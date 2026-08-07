# exp187_cluster_outlier_alt_typewell_pfbeam_audit

## 状態

- ルート: pf_beam
- 状態: completed_train_side_rejected_no_submit
- CV: own PF RMSE 17.011319 / own Beam RMSE 16.287400
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-04
- 親実験: `cluster_outlier_alt_typewell_pfbeam_audit` backlog

## 仮説

cluster 中心から外れた well では、自身の typewell よりも、近傍 well が参照している別 typewell cluster の composite typewell の方が、PF/Beam の GR observation model として良い場合がある。

## 変更点

- exp065 の native-overlap typewell cluster と exp114 の well geometry から cluster 外れ well を target-free に抽出する。
- 対象 well で `own_typewell`、`nearest_other_cluster_composite`、`nearby_majority_cluster_composite_k8` を同じ PF/Beam config で再生成する。
- alt cluster の typewell は representative well 1本ではなく、source cluster の available member typewell を TVT bin ごとに結合した 1本の composite typewell として参照する。
- direct switch / inference port / submit は行わず、alt candidate と diagnostic として比較する。

## 検証方針

- Fold: なし。exp072/099 と同じ `TVT_input_missing_equivalent_exp063_rows` を score rows とする train-side audit。
- Group: well。
- Stratification: cluster outlier gate、distance bucket、nearest-other / nearby-majority signal。
- Leakage Check: true TVT は exp072 cache の `last_known_tvt + target` として scoring のみに使う。strategy 選択、PF likelihood、Beam cost には使わない。

## 結果

| candidate | rows | RMSE | MAE | within10 | 備考 |
| --- | ---: | ---: | ---: | ---: | --- |
| `beam_own_typewell_top1` | 306,490 | 16.287400 | 11.798330 | 0.548997 | best non-oracle |
| `pf_own_typewell_lik_mean` | 306,490 | 17.011319 | 11.503088 | 0.586701 | primary baseline |
| `pf_nearby_majority_cluster_composite_k8_lik_mean` | 183,355 | 189.864085 | 112.125988 | 0.302059 | alt, 大幅悪化 |
| `pf_nearest_other_cluster_composite_lik_mean` | 306,490 | 191.623692 | 106.358104 | 0.295429 | alt, 大幅悪化 |
| `beam_nearby_majority_cluster_composite_k8_top1` | 183,355 | 195.101081 | 114.264089 | 0.249461 | alt, 大幅悪化 |
| `beam_nearest_other_cluster_composite_top1` | 306,490 | 205.606487 | 115.216130 | 0.273210 | alt, 大幅悪化 |

## 所見

### 良かった点

- v2 で validation surface を exp072/099 と同じ `TVT_input_missing_equivalent_exp063_rows` に修正できた。
- alt typewell は representative 1本ではなく cluster member composite に修正できた。
- 対象 64 wells / 306,490 rows で Kaggle train v2 が完了した。

### 悪かった点

- cluster-composite alt typewell は global で大幅悪化した。
- nearest composite PF は improved wells 11/64 だが最大 regression +572.507ft、nearby-majority composite PF は improved wells 9/64 だが最大 regression +615.042ft。
- Beam も同様に最大 regression が +600ft 級で、hard switch / direct replacement には使えない。

### リスク / 注意

- 本実験は train-side audit であり、inference port / submission は作らない。
- PF/Beam implementation は exp187 audit generator で、validation surface は exp072/099 と合わせたが、exp072 full replay generator の完全再実行ではない。

## 次

- `cluster_outlier_alt_typewell_pfbeam_audit` は完了/不採用として閉じる。
- 続ける場合も alt typewell 候補値そのものではなく、target-free alignment diagnostic / selector feature の材料に限定する。

## v1 からの修正

- v1 は artificial 192-row prefix holdout だったため、従来 PF/Beam audit の validation surface と一致していなかった。
- v1 は別 cluster の representative typewell 1本を参照していた。
- v2 は exp072-style eval rows と cluster-composite typewell に修正した。結論は v2 を正とする。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
