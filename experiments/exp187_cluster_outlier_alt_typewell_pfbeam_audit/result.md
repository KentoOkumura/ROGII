# exp187_cluster_outlier_alt_typewell_pfbeam_audit 結果

## 仮説

cluster 外れ well では、query well 自身の typewell ではなく、近傍 well が参照している別 typewell cluster の composite typewell を使うことで PF/Beam 候補の GR matching が改善する可能性がある。

## 設定

- 親: `cluster_outlier_alt_typewell_pfbeam_audit` backlog
- 参照: exp065 cluster assignment、exp114 well geometry、exp072 PF/Beam train cache
- 検証: exp072/099 と同じ `TVT_input_missing_equivalent_exp063_rows`
- 評価対象: cluster-outlier 64 wells / 306,490 rows
- alt typewell: source cluster の available member typewell を TVT bin ごとに結合した composite typewell
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV | own PF RMSE 17.011319 / own Beam RMSE 16.287400 |
| Public LB | - |
| Private LB | - |

主要 candidate:

| candidate | rows | RMSE | MAE | within10 | 備考 |
| --- | ---: | ---: | ---: | ---: | --- |
| `beam_own_typewell_top1` | 306,490 | 16.287400 | 11.798330 | 0.548997 | best non-oracle |
| `pf_own_typewell_lik_mean` | 306,490 | 17.011319 | 11.503088 | 0.586701 | primary baseline |
| `pf_nearby_majority_cluster_composite_k8_lik_mean` | 183,355 | 189.864085 | 112.125988 | 0.302059 | alt, 大幅悪化 |
| `pf_nearest_other_cluster_composite_lik_mean` | 306,490 | 191.623692 | 106.358104 | 0.295429 | alt, 大幅悪化 |
| `beam_nearby_majority_cluster_composite_k8_top1` | 183,355 | 195.101081 | 114.264089 | 0.249461 | alt, 大幅悪化 |
| `beam_nearest_other_cluster_composite_top1` | 306,490 | 205.606487 | 115.216130 | 0.273210 | alt, 大幅悪化 |

## 再現性

- deterministic anchor: false
- seed policy: `stable_sha256_per_query_well_seed_index_shared_across_typewell_strategies`
- kernel version: `kentookumura/exp187-alt-typewell-pfbeam-audit-train` v2
- kernel URL: https://www.kaggle.com/code/kentookumura/exp187-alt-typewell-pfbeam-audit-train
- kernel id_no: 125890196
- runtime: summary 1,049.754 sec / logs last time 1,081.803 sec
- exp072 cache decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- row candidates decompressed SHA: `7823897d6042d41c67bf1a74e2e3d9e0a1dcfed8304c79ef689c5e1478ec86ea`
- row candidates raw gzip SHA: `c5d6fed79f5a9d3bfdac6cdd1cecbd7a227031af090fab0a5b06137c8c666a5b`
- model SHA / manifest SHA: model なし
- submission SHA: submission なし

## 解釈

validation surface は v2 で意図通り exp072/099 と同じ `TVT_input_missing_equivalent_exp063_rows` に修正できた。また、alt typewell も representative 1本ではなく cluster member を TVT bin で結合した composite typewell として参照している。

そのうえで、cluster-composite alt typewell は global candidate として明確に不採用。nearest composite PF は RMSE 191.623692、nearby-majority composite PF は RMSE 189.864085 で、own PF baseline 17.011319 から +170ft 以上悪化した。Beam も nearest composite 205.606487、nearby-majority composite 195.101081 で同様に崩壊した。

一部 well では alt が改善する。nearest composite PF は 11/64 wells、nearby-majority composite PF は 9/64 wells で改善したが、最大 regression はそれぞれ +572.507ft / +615.042ft。Beam も改善 wells は 11/64 / 9/64 に留まり、最大 regression は +614.988ft / +653.803ft。target-free hard switch や direct replacement に使える安定性ではない。

主因は v1 と同じく、別 cluster composite typewell の絶対 TVT range / GR depth alignment が query well と合わず、PF/Beam observation model が wrong depth に吸い込まれること。複数 typewell を composite にしても、この alignment 問題は解消しなかった。

## 次

`cluster_outlier_alt_typewell_pfbeam_audit` は完了/不採用として閉じる。hard switch、direct candidate replacement、inference port、submit は行わない。続ける場合は alt typewell そのものを候補値として使うのではなく、query prefix で target-free alignment した confidence diagnostic / selector feature に限定する。

## v1 の扱い

v1 は 64 wells / 12,288 artificial masked-prefix rows で実行したが、validation surface が従来 PF/Beam audit と一致していなかった。また alt typewell は representative well 1本を直接参照していた。v2 でこの 2 点を修正したため、結論は v2 を正とする。
