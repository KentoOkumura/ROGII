# exp509 output contract

候補notebookはKaggle Notebook outputに次を保存する。現時点では未実行・未生成。

- `exp509_component_predictions.csv.gz`: `id`、`well`、`md_since`、`exp413_tvt`、
  `strict_public_core_tvt`、`final_tvt`。
- `exp509_prediction_difference_summary.json`: pooled、well別、MD horizon別のtruth-free差分量。
- `exp509_input_manifest.json`: source kernel/version、file、raw/decompressed SHA、ID-order SHA。
- `exp509_reproducibility_manifest.json`: config/source/prediction/submission SHAとtechnical checks。
- `submission.csv`: technical gate通過時にKaggle outputとして生成する。competitionへの外部提出は
  別承認とする。

補助境界として、dynamic exp413が生成する一時`submission.csv`は
`artifacts/exp413_intermediate_submission.csv`へ移動し、exp497 coreの診断blendは
`artifacts/strict_public_core_runtime/exp497_intermediate_submission.csv`へ隔離する。どちらも
exp509 final submissionとして扱わない。

`submission.csv`以外のprediction列を提出対象へ混入させない。gzip比較の主証拠はdecompressed
content SHAとする。
