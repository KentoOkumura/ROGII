# 要件

## 依頼

`KAGGLE_DIRECTION.md` のバックログ `typewell_late_range_pfbeam_generation_soft_prior` を実装する。
既存 full replay cache は入力にせず、raw well/typewell から exp072-style full replay train feature cache を作り直す。

PF/Beam 生成時には、typewell TVT 前半 range へ戻る候補を hard 禁止せず、soft prior penalty として扱う。

## 制約

- Route: `pf_beam`
- 入力: raw competition train horizontal/typewell files。
- 既存 full replay cache: generation input として読まない。比較対象 / downstream baseline としてのみ扱う。
- 出力: exp072-style full replay train feature cache 互換の train feature frame、schema、summary。
- penalty strength は true TVT、oracle candidate、true-error rank を使って選ばない。
- LightGBM 学習、inference、submission は実施しない。
- test features は downstream inference notebook で同じ generation code から raw test files を再生成する。
- gzip 生成物は raw `.csv.gz` SHA に加え、decompressed content SHA を主証拠として記録する。

## 受け入れ基準

- `experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/` に config、実装 module、train / inference notebook、記録ファイルがある。
- train notebook で raw train input、selected soft prior、PF seeds/particles、出力生成物を確認できる。
- `PF_ANCC`、`PF_Z`、Beam path cost、128-seed likelihood-PF に selected soft-prior cost が入る。
- Kaggle push 前のコスト確認として GPU 学習なし、feature cache variant 1、LightGBM config 数 0、fold 数 0、booster 数 0、control 再学習なしを `SESSION_NOTES.md` に記録する。
- Kaggle train が complete し、3,783,989 rows / 773 wells / 196 features の full replay train cache が生成される。
- feature schema、summary、train feature gzip をローカル artifacts に保存し、raw gzip SHA、decompressed SHA、gzip integrity、row count を記録する。
- v1/v2 の prefix-holdout audit は superseded として記録し、正式結果にしない。
