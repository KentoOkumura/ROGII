# 要件

## 依頼

`pfbeam_normalized_diagnostic_score` を実装する。

## 制約

- Route: `ml_model`
- PF/Beam 候補を直接置換しない。
- hard selector / hard switch ではなく、exp092 系 LightGBM への add-only confidence feature として実装する。
- valid/test true TVT、oracle candidate、absolute error label を diagnostic feature に使わない。
- 再現性: `docs/06_reproducibility.md` に従い、PF/Beam cache、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `experiments/exp130_pfbeam_normalized_diagnostic_score/` に train/inference notebook、config、helper、README、result、SESSION_NOTES がある。
- exp072 full replay cache の PF/Beam/likPF 候補から target-free normalized diagnostic features を生成できる。
- `exp092_full_row_control` と `pfbeam_normalized_diagnostic_addonly` を同じ rows / GroupKFold by well で比較する設定になっている。
- gzip 生成物を比較する場合は raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
