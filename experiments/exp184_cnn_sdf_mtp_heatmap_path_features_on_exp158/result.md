# exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158 結果

## 結論

Kaggle train v2 は完了。exp182 CNN/SDF/MTP heatmap path features を exp157/158 selector に add-only で入れる方針は、train-side では支持される。

best non-oracle は `viterbi_sw200_bias000_jw000_jf025_d0150_std999999_md0000_seg012` で、RMSE は `10.560650324533297`。親の exp158 continuity RMSE `10.789163253` から `-0.2285129284667029` 改善した。exp157 row-wise RMSE `10.79579983712686` からは `-0.23514951259356387`、`likpf_mean` RMSE `11.594897672217703` からは `-1.0342473476844063`。

ただし、これは train-side audit の結果であり、inference port / submit はまだ行わない。raw-test heatmap coverage、sparse sample interpolation、exp115 hidden-like subgroup の悪化確認が残っている。

## 実行

- Kaggle kernel: `kentookumura/exp184-hmpf-train`
- version: 2
- status: `KernelWorkerStatus.COMPLETE`
- runtime: `33550.90019130707` sec
- rows / wells: `3,783,989` / `773`
- feature count: `223`
- route: `pf_beam`
- GPU: disabled

v1 は fold0 multiclass 完了後に `DeadKernelError` で失敗した。v2 では exp183 と同じ memory guard を入れ、long-model train/eval sample を `120000` rows/fold に制限し、full valid OOF は `50000` row chunk prediction で生成した。

## 主要 metrics

| variant | mode | RMSE | MAE | within10 | oracle acc |
|---|---:|---:|---:|---:|---:|
| oracle | oracle | 4.564605 | 2.317166 | 0.960054 | 1.000000 |
| best Viterbi | viterbi | 10.560650 | 6.329188 | 0.797056 | 0.271564 |
| lgb_candidate_error_ranker | oof | 10.569847 | 6.349849 | 0.796829 | 0.265474 |
| lgb_candidate_binary | oof | 11.057520 | 6.686468 | 0.778195 | 0.313763 |
| lgb_multiclass | oof | 11.490098 | 6.827812 | 0.775349 | 0.305819 |
| likpf_mean_single | baseline | 11.594898 | 7.067633 | 0.772807 | 0.263997 |

best Viterbi の path switch は `5713`、`1.509782` / 1000 rows。selection は `likpf_mean` 42.25%、`pf_ancc` 38.26%、dense family 15.13%、`beam_mean` 4.33%。

## Diagnostics

- worst well は `86454a6f`: RMSE `57.960134`、within10 `0.046208`。
- heatmap sparse distance bucket は近い q1 が RMSE `7.042431`、遠い q4 が RMSE `14.058409`。sparse heatmap sample から遠い領域で明確に弱い。
- exp115 spatial valid subgroup は RMSE `12.696140`、typewell purged valid は `12.629861`。hidden-like stress では全体 RMSE より悪い。
- `hmpf_far_from_sparse_sample_gt512` は RMSE `13.029168`。
- heatmap feature importance は上位に入っている。例: `hmpf_real_top10_mean_minus_candidate_abs`、`hmpf_real_top1_minus_candidate_abs`、`hmpf_real_top5_mean_minus_candidate_abs`、`hmpf_real_top3_mean_minus_prior_center`。

## 生成物

取得先: `experiments/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158/kaggle/output/train_v2`

- `artifacts/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158_metrics.csv`
- `artifacts/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158_oof_predictions.csv.gz`
- `artifacts/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158_summary.json`
- `artifacts/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158_model_manifest.json`
- `artifacts/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158_feature_importance_mean.csv`

主な SHA:

- metrics: `54d5e937ece5cd5c3980907416d75af5544f41e4ba2c87f11c716b378f3050bd`
- OOF decompressed: `e09fbc48d8a6e2d97efdf214e0e0fa71e6bde97ac16a33062b76de07f46d0346`
- best variant prediction: `4e5d6e1dc69183d3d42edc503553ecbb7594300888cf9e876838c1b5571863cf`
- model manifest: `a41d119bb2de30c12cf828e1fc93b0ed64d64867088a5b5fc9b8df4486e07f77`

## 次アクション

同じ exp184 内で inference port を検討する。ただし submit 前に、raw-test heatmap feature generation / sparse interpolation coverage / feature schema parity / fallback behavior を確認する。特に sparse sample から遠い領域と exp115 hidden-like subgroup の悪化を gate できるかを見る。
