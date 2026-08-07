# 要件

## 依頼

`cluster_outlier_pfbeam_prior_gate` backlog を実装する。exp109 / exp114 の PF/Beam/likPF 候補への prior correction を、exp175 と同じ cluster-outlier gate で限定し、train-side OOF で no-training 診断する。

## 制約

- Route: `pf_beam`
- 補正対象を exp148 / exp092 ML output にしない。
- PF/Beam 再生成、alternate typewell 差し替え、ML 学習、提出は行わない。
- true TVT、oracle best、true-error rank を gate / prior source に使わない。
- baseline は `likpf_mean`、`pf_ancc`、`beam_mean`。reference として exp109 global best と exp114 global best を同じ出力表に入れる。
- 再現性: `docs/06_reproducibility.md` に従い、固定 upstream output の SHA と gzip decompressed content SHA を記録する。

## 受け入れ基準

- `experiments/exp181_cluster_outlier_pfbeam_prior_gate/` に config、train/inference notebook、補助スクリプト、記録ファイルがある。
- exp109 OOF の PF/Beam/likPF base candidates と exp109/114 prior を join して、cluster-outlier gated correction grid を評価できる。
- `gate_metrics`、`by_well_delta`、`bucket_metrics`、`subgroup_metrics`、`path_continuity`、`cluster_outlier_well_features`、`summary.json` を保存する。
- Kaggle train push 前に、variant/config/fold/booster 数を `SESSION_NOTES.md` に記録する。
- deterministic anchor として扱わない。submission SHA は不要だが、生成物 SHA と input SHA を summary に残す。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
