# exp129_spatial_prior_as_selector_candidate 結果

## 状態

Kaggle train v1 完了。不採用。

## 実行

- Kernel: `kentookumura/exp129-spatial-selector-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp129-spatial-selector-train`
- version: 1
- runtime: 3,797.765 秒
- rows: 3,783,989
- wells: 773
- CPU / internet off

## 評価

| variant | mode | RMSE | MAE | within10 | oracle acc |
| --- | --- | ---: | ---: | ---: | ---: |
| `oracle_expanded` | oracle | 6.709127 | 3.138080 | 0.929853 | 1.000000 |
| `oracle_base_only` | oracle | 7.434030 | 3.745228 | 0.906525 | 0.784658 |
| `likpf_mean_single` | baseline | 11.594898 | 7.067633 | 0.772807 | 0.321356 |
| `lgb_error_ranker_rowwise` | oof | 13.793157 | 7.187177 | 0.769536 | 0.312807 |
| `lgb_error_ranker_viterbi_p0p25` | oof_viterbi | 13.793777 | 7.185637 | 0.769577 | 0.313877 |
| `oracle_spatial_only` | oracle | 14.353528 | 9.629028 | 0.651661 | 0.215755 |

expanded oracle は base oracle から RMSE -0.724903 改善した。spatial 候補の oracle top1 rate は `xy_plus_trajectory_shape_k8_prior_tvt` 10.8478%、`xy_only_k8_prior_tvt` 10.6863%。true-error topK に spatial が入る割合は top1 21.53%、top2 41.26%、top3 67.78%、top5 95.74%。

一方、学習 selector は崩れた。best OOF は `lgb_error_ranker_rowwise` で、`likpf_mean_single` に対して RMSE +2.198259 悪化した。spatial selection rate は 10.97% あるが、正しく信用できていない。Viterbi は switch mean を 72.04 から p10 で 2.42 / 1000 rows まで落とせるが、RMSE は 13.796004 で改善しない。

## 生成物

- `kaggle/output/train_v1/artifacts/exp129_spatial_prior_as_selector_candidate_summary.json`
- `kaggle/output/train_v1/artifacts/exp129_spatial_prior_as_selector_candidate_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp129_spatial_prior_as_selector_candidate_candidate_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp129_spatial_prior_as_selector_candidate_selection_distribution.csv`
- `kaggle/output/train_v1/artifacts/exp129_spatial_prior_as_selector_candidate_by_well.csv`
- `kaggle/output/train_v1/artifacts/exp129_spatial_prior_as_selector_candidate_bucket_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp129_spatial_prior_as_selector_candidate_oof_selected_predictions.csv.gz`
- `kaggle/output/train_v1/artifacts/exp129_spatial_prior_as_selector_candidate_model_manifest.json`

## SHA

- exp099 decompressed SHA: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- exp114 decompressed SHA: `9ffa9f9a026d43d3c0721a549fdff0aaf0acbd73d6c8209218ad9a45a314fe29`
- metrics SHA: `788796940937180127a8d17ec520a5e5d97ace96877bc1a72e8aa7104694d17f`
- OOF predictions decompressed SHA: `5c1a0b66154ed7e2d58e38b804c3953614547e87293dded9d980a93e96495617`
- model manifest SHA: `65030ac43f78cb5e82d9190753a5cc9d8c1b28c15d20929cfed6e501660c778a`

## 解釈

spatial prior candidate には oracle headroom があるが、exp099/101 系の predicted-error selector では信用判定に失敗する。`spatial_prior_as_selector_candidate` は direct selector / inference port / submit に進めない。

今後は spatial path を離散候補として選ぶ方向ではなく、exp092 系 ML の add-only confidence feature / uncertainty diagnostic として扱う。特に spatial prior value、std、neighbor count、distance、same-typewell share、candidate disagreement は `spatial_neighbor_prior_ml_features_on_exp092` に統合する。
