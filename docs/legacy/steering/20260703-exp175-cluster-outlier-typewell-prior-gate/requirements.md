# 要件

## 依頼

`cluster_outlier_typewell_prior_gate` を実装する。exp109 / exp114 の typewell / spatial neighbor prior は direct correction では改善する一方で worst-well regression が大きいため、exp065 の native typewell cluster 内で空間的に外れている well だけに弱い posthoc correction を適用できるか train-side OOF で診断する。

## 制約

- Route: `ml_model`
- 新規モデル学習、PF/Beam 再生成、inference port、提出はしない。
- exp109 / exp114 の固定 OOF prior と exp092 / exp148 の固定 OOF prediction を入力にする。
- cluster / gate 条件には raw X/Y 由来の well geometry と exp065 cluster assignment だけを使い、valid/test true TVT、oracle best、同 fold valid wells の true TVT を使わない。
- 補正は `alpha=0.05/0.10/0.20`、clip `5/10/20/40ft` の弱い clipped correction に限定する。
- exp115 hidden-like split は診断 subgroup として見る。 exact hidden split とは扱わない。
- 再現性: `docs/06_reproducibility.md` に従い、固定 upstream output の SHA と gzip decompressed SHA を記録する。

## 受け入れ基準

- `experiments/exp175_cluster_outlier_typewell_prior_gate/` に config、train/inference notebook、補助スクリプト、記録ファイルがある。
- train notebook は入力 source、cluster gate、prior variant、policy grid、生成物をセル上で確認できる。
- `gate_metrics`、`by_well_delta`、`bucket_metrics`、`subgroup_metrics`、`cluster_outlier_well_features`、`summary.json` を保存する。
- global RMSE だけでなく、gate 対象 rows/wells、cluster-outlier subset、near `000_050`、`1000_plus`、max well regression、exp115 stress subset を確認できる。
- deterministic anchor としては扱わず、採用判断に必要な raw-test/full-train parity と hidden-like stress の未完了状態を記録する。
- gzip 生成物を比較する場合は raw `.csv.gz` SHA と decompressed content SHA を分けて記録する。
