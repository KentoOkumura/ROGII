# 要件

## 依頼

`learned_likelihood_fulltrain_addonly_on_exp092` を実装する。

## 制約

- Route: `ml_model`
- 親実験は `exp092_u_projection_correction_disagreement_fullrun`。
- exp127 の learned likelihood add-only feature family を引き継ぐ。
- exp112 subset cache ではなく、exp145 の full-train/raw-test target-free learned likelihood feature cache を使う。
- control は再学習しない。保存済み exp092 metrics を historical baseline として使う。
- 再現性は `docs/06_reproducibility.md` に従い、gzip は decompressed content SHA を主証拠とする。

## 受け入れ基準

- `experiments/exp148_learned_likelihood_fulltrain_addonly_on_exp092/` に config、train notebook、inference notebook、補助 `.py`、記録ファイルが揃っている。
- train notebook で exp072/exp092 train surface と exp145 full-train `ml_features` の coverage を確認できる。
- train 実行時の active variants / disabled variants / LightGBM configs / folds / booster 数が `SESSION_NOTES.md` に記録されている。
- inference notebook で exp145 raw-test `ml_features` と exp148 saved booster manifest を使う flow が実装されている。
- 実装後に静的検証を通す。
