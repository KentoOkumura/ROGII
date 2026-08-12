# 要件

## 依頼

exp038 の inference と提出を実行するため、提出可能な inference port を作る。

## 制約

- Route: `ml_model`
- 親実験は `exp038_ravaghi_public_sel15_features_single_lgbm`
- exp038 自体は audit-only inference なので、別 experiment として port する。
- public visible wells の physical branch は変更しない。
- 見えない test の target は読まない。

## 受け入れ基準

- Kaggle inference notebook が `submission.csv` を生成できる。
- `kernel-metadata.json` は offline / CPU / required kernel source を満たす。
- output の `submission.csv` が sample submission 互換チェックに通る。
- `kaggle competitions submit` 後に submission monitor を開始する。
