# exp065_typewell_supertype_cluster_cv_audit 結果

## 仮説

Exact duplicate CSV だけでは説明できない共通 typewell が、GR 曲線の shifted NCC または constrained DTW 類似グループとして見つかる可能性がある。

## 設定

- 親: `studies/typewell_group_audit.py`
- 検証: なし。common typewell discovery のみ。
- メトリック: group count / pair similarity diagnostics
- シード: deterministic

## 結果

| メトリック | 値 |
| --- | --- |
| CV | なし |
| Public LB | - |
| Private LB | - |

## 解釈

Kaggle train v1 完了。

Exact hash では 752 unique groups / 34 duplicate wells で、既存 audit と一致した。

native row-lag overlap を追加した結果、`typewell.csv` の GR 列が shift / trim で一致する candidate pair は 10,713、exact containment pair は 10,697 になった。exact native overlap cluster は 54 groups / 41 multi-well groups / 760 wells / max group 71。`028d7b28` と `0dd99dc5` は lag 218 rows = 109 ft、1774 rows が GR 完全一致し、`028d7b28` が `0dd99dc5` に含まれる。

高相似の共通 typewell 候補としては、`shifted_ncc >= 0.98` が 103 multi-well groups / 314 wells、`dtw_similarity >= 0.94` が 63 multi-well groups / 167 wells を返した。より厳しい `dtw_similarity >= 0.96` では 33 multi-well groups / 75 wells、`shifted_ncc >= 0.99` では 58 multi-well groups / 133 wells まで絞られる。

Discussion の「57 unique Typewell numbers」は、resampled NCC/DTW ではなく native row-lag overlap で近く再現できる。後続では exact hash、native overlap、high-NCC、high-DTW を別々の候補 group 定義として扱う。

## 次

`typewell_neighbor_prior_features` では、exact hash、native row-lag overlap、`shifted_ncc >= 0.98`、`dtw_similarity >= 0.94` を別々の fold-safe neighbor pool として比較する。
