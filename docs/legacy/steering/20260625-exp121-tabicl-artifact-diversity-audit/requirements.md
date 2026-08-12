# 要件

## 依頼

`tabicl_artifact_diversity_audit` backlog を実装する。TabICL / 保存済み artifact-stack 予測を単体本命にせず、後続のアンサンブル候補として exp027 / exp063 / exp073 / exp082 系 anchor とどれだけ異なるかを CPU-only で監査する。

## 制約

- Route: `ensemble`
- TabICL 本体の再推論、GPU 実行、モデル学習、提出候補生成はしない。
- 公開 notebook replay の採用判断とは分け、保存済み submission-like CSV を後続アンサンブル候補として使えるかの target-free diversity 診断に限定する。
- candidate / anchor CSV の SHA、gzip decompressed SHA、行数、id 互換性、予測範囲を記録する。
- 候補 source が mount されていない場合は失敗ではなく missing inventory として記録する。
- OOF error correlation は fold-safe OOF prediction と真値が明示的に揃う場合だけ記録する。test submission だけから誤差相関を推定しない。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理なし、GPU なし、SHA 記録ありとして設計に明記する。

## 受け入れ基準

- `experiments/exp121_tabicl_artifact_diversity_audit/` に CPU-only audit script、config、train/inference notebook、記録ファイルがある。
- `config.yaml` の `experiment.route` は `ensemble`、`runtime.kaggle.enable_gpu` は `false`。
- 監査実行時に `inventory.csv`、`pairwise.csv`、`by_well_distance.csv`、`summary.json`、README が保存される。
- exp027 / exp063 / exp073 / exp082 anchor が存在する場合、candidate-vs-anchor pairwise RMSE / MAE / p95 abs / max abs が出る。
- source が存在しない場合でも missing status を記録して正常終了する。
- deterministic anchor として扱わない。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
