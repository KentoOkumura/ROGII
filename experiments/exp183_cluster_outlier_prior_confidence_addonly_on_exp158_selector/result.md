# exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector 結果

## 目的

exp181 の cluster-outlier prior signal を direct correction ではなく、exp157/158 candidate selector の add-only confidence feature として使う。

## 実行結果

- Kaggle kernel: `kentookumura/exp183-copcf-train`
- Version: 2
- Status: `completed_train_side_audit`
- Runtime: 28,882.350 sec
- Rows / wells: 3,783,989 / 773
- LightGBM boosters: 3 configs x 5 folds = 15
- Viterbi variants: 180

v1 は fold 0 の candidate-long feature 生成中に `DeadKernelError` で落ちた。v2 では candidate-long feature frame の一括構築、train/eval long rows cap、full-valid chunk prediction を入れて完了した。

## CV

best train-side Viterbi:

- variant: `viterbi_sw200_bias000_jw100_jf025_d0075_std999999_md0000_seg001`
- RMSE: 10.601481774
- MAE: 6.386571251
- within10: 0.792418794
- oracle label accuracy: 0.266536716
- path switches: 5,650
- path switches / 1000 rows: 1.493133305
- default candidate rate: 0.459316346

baseline delta:

- vs `likpf_mean`: -0.993415899 RMSE
- vs `multiobs_score_top1`: -0.993415899 RMSE
- vs exp157 row-wise selector: -0.194318063 RMSE
- vs exp158 continuity selector: -0.187681479 RMSE

Oracle headroom は RMSE 4.564605115 / within10 0.960053531。

## 判断

exp181 の cluster-outlier prior signal を direct correction せず selector confidence feature として使う方針は train-side で支持された。exp158 continuity から RMSE を約 0.188 改善し、path switch も 1.493 / 1000 rows まで低い。

一方、v2 は OOM 対策として long-model train/eval rows を 120k/fold に cap している。full-row OOF score は chunk prediction で作っているが、model fit 条件は exp157/158 の 650k/fold long training とは異なるため、次に進める場合は同じ exp183 内で raw-test parity、worst-well / bucket / exp115 subgroup の詳細確認、必要なら高メモリ実行または split train を検討する。現時点では inference port / submit はまだ行わない。

## 詳細解釈

exp183 の改善は、row-wise ranker の改善と Viterbi 連続化の両方から来ている。新しい `lgb_candidate_error_ranker` は RMSE 10.640892 で、`likpf_mean` 11.594898 から -0.954006 改善した。そこに Viterbi をかけると RMSE 10.601482 まで下がり、row-wise からさらに -0.039410 改善する。

候補選択は exp158 より default 寄りになった。best Viterbi は `likpf_mean` 45.93%、`pf_ancc` 36.10%、dense family 14.02% を選ぶ。exp158 best は `likpf_mean` 38.37%、`pf_ancc` 39.00%、dense family 17.87% だったため、prior confidence feature は「無理に dense / PF に寄せる」より、危ない候補選択を抑えて `likpf_mean` に戻す方向でも効いている。

bucket では、`likpf_mean` 比では全距離帯で改善している。distance `000_050` は 1.188878 -> 0.508182、`1000_plus` は 12.704015 -> 11.639068。row-wise error ranker 比では near bucket はわずかに悪化し、`000_050` は +0.018146、`050_100` は +0.007944 だが、longtail 側は改善し、`1000_plus` は -0.044711。つまり Viterbi は近傍の細かい当てはまりを少し犠牲にして、長距離・不安定領域の形を整えている。

cluster-outlier subgroup では、`copcf_gate_any_outlier_signal_k8` が 11.889295 (`likpf_mean`) -> 11.285943、`copcf_nearest_other_closer` が 11.335708 -> 10.855033、`copcf_nearby_majority_diff_k8` が 12.687996 -> 12.195280。狙った cluster-outlier 条件で改善しており、仮説には合っている。exp115 hidden-like でも `spatial_valid` は 13.643808 -> 12.593127、`typewell_purged_valid` は 13.506801 -> 12.479252 と、`likpf_mean` 比では大きく改善した。ただし row-wise ranker との差はほぼ横ばいなので、Viterbi ではなく add-only feature ranker 側の改善が主因。

feature importance も仮説を支持している。`lgb_candidate_error_ranker` の top40 中 15 個が `copcf_` features で、`copcf_nearest_other_cluster_dist`、`copcf_own_cluster_dist_z`、`copcf_own_cluster_dist`、spatial/typewell prior minus candidate が上位に入った。モデルは prior signal を無視しておらず、候補品質判断に実際に使っている。

リスクは worst well と train 条件。worst well `86454a6f` は RMSE 57.581365 でまだ非常に重い。`fb03ae90`、`efe96181`、`7850c72e` などは `likpf_mean` より悪く、selector が壊す well も残る。row-wise 比の最大 regression は `7987f2f2` の +1.545085 RMSE で、exp158 の最大 regression +1.906477 よりは小さいがゼロではない。提出判断には、これらの well 型が hidden test に出る前提で raw-test guard が必要。

## 生成物

Kaggle output に以下を生成した。

- `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_metrics.csv`
- `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_oof_predictions.csv.gz`
- `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_selection_distribution.csv`
- `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_by_well.csv`
- `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_bucket_metrics.csv`
- `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_subgroup_metrics.csv`
- `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_viterbi_params.csv`
- `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_score_summary.csv`
- `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_feature_importance.csv`
- `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_feature_importance_mean.csv`
- `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_feature_schema.csv`
- `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_model_manifest.json`

主要 SHA:

- metrics: `c5fc041bcfa55b52580712d25762efdfd2439d7655eaffa46763923102456341`
- oof predictions: `d2a98e8212ff9fc06f46c5505f3dc870310453af989150921bc54ef42cedbf5d`
- oof predictions decompressed: `beddc97c04cdbddcd5d5756e90b66ff51dfc525c998f668a147bac540d0180a0`
- feature schema: `3b4c44e750e640066298542b70946b9a2d3733c71ba24ecc0a46d0e0f5b03ec4`
- score summary: `669b6fbb066d541442cbf91006bfd3fb578a0d6057defbee8376b087480ae515`
- subgroup metrics: `42aecfb0506e74c9b0a1376b74c02cf1635c928ffb21dc2da8abc53b11db587e`
- viterbi params: `002f5d2bf4be169842ec1e911ec545223b4d87a1ff4a586a388c18ffd801112b`
