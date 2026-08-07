# exp181_cluster_outlier_pfbeam_prior_gate 結果

## 仮説

exp109 / exp114 の global prior correction は `likpf_mean` を大きく改善したが、worst-well regression が +6ft 級で残った。cluster-outlier well だけに弱い correction を限定すれば、global 改善は小さくても regression guard として使える可能性がある。

## 設定

- 親: `exp109_typewell_neighbor_prior_features`
- 比較親: `exp114_spatial_neighbor_prior_signal_audit`
- gate 親: `exp175_cluster_outlier_typewell_prior_gate`
- 検証: 固定 PF/Beam/likPF OOF 候補への cluster-outlier gated posthoc audit
- base candidates: `likpf_mean`, `pf_ancc`, `beam_mean`
- 提出: なし

## Kaggle train v1 結果

Kaggle train v1 は完了した。

| policy | RMSE | delta vs `likpf_mean` | MAE | within10 | gate rows / wells | max well regression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `likpf_mean` baseline | 11.594897672 | 0.000000000 | 7.067632584 | 0.772807479 | 0 / 0 | - |
| exp109 reference global best | 11.143359414 | -0.451538258 | 7.025321608 | 0.779871453 | 3,682,366 / 758 | +6.594183 |
| exp114 reference global best | 11.151818277 | -0.443079395 | 7.062013170 | 0.779272878 | 3,778,407 / 773 | - |
| best gated `any_outlier_signal_k8/std_le20/a0.2/c40` | 11.479140438 | -0.115757234 | 7.041763474 | 0.775467371 | 908,309 / 215 | +4.359666 |
| guarded `any_outlier_signal_k8/std_le20/a0.2/c20` | 11.497560716 | -0.097336956 | 7.038521400 | 0.776045861 | 908,309 / 215 | +3.032388 |

best gated は `likpf_mean` から RMSE -0.115757 改善し、exp109 global reference の worst-well regression +6.594183 を +4.359666 まで下げた。ただし global 改善は exp109 / exp114 reference よりかなり小さく、+4ft 級の worst-well regression は direct correction としてはまだ大きい。clip20 に絞ると worst-well regression は +3.032388 まで下がるが、RMSE 改善も -0.097337 に縮む。

distance bucket は best gated c40 で全 bucket 改善した。`000_050` は RMSE 1.188878 -> 1.152570、`1000_plus` は 12.704015 -> 12.580753。exp115 hidden-like stress も `spatial_valid` 13.643808 -> 13.604243、`typewell_purged_valid` 13.506801 -> 13.450540 と壊れていない。

path continuity では best gated c40 の max prediction step は 8.081055 で、10ft 以上の step は 0。clip20 は correction step max 4.0 まで抑えられる。

## 判断

prior signal は PF/Beam/likPF 候補上では有効で、cluster-outlier gate は reference global correction より worst-well regression を下げる。ただし direct posthoc correction として inference port / submit へ進めるには worst-well regression がまだ大きい。exp181 は train-side no-training audit として完了し、提出はしない。

今後この系統を使う場合は、直接補正ではなく selector / confidence feature / candidate scoring の材料に限定する。
