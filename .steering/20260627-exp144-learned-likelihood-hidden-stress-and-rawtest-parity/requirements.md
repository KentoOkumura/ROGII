# 要件

## 依頼

`exp127_learned_likelihood_hidden_stress_and_rawtest_parity` を実装する。既存の最新番号は `exp143` なので、実験番号は `exp144_learned_likelihood_hidden_stress_and_rawtest_parity` とする。

## 制約

- Route: `ml_model`
- 親実験: `exp127_learned_likelihood_features_on_exp092`
- 新規学習や提出は行わない。exp127 の保存済み OOF 予測を readout する。
- exp115 の `verification_like_spatial` / `verification_like_typewell_purged` を stress split として使う。
- exp112 learned likelihood feature cache は target-free diagnostic としてだけ読む。
- raw-test/full-train parity はチェックリストとして明示し、未充足条件を隠さない。
- 再現性: `docs/06_reproducibility.md` に従い、gzip は decompressed content SHA を主証拠にする。

## 受け入れ基準

- `experiments/exp144_learned_likelihood_hidden_stress_and_rawtest_parity/` に config、train/inference notebook、補助スクリプト、記録ファイルがある。
- train notebook は目的、設定、入力 preview、監査実行、生成物確認をセル単位で追える。
- 監査スクリプトは次を生成する。
  - overall metrics
  - bucket metrics
  - by-well metrics
  - add-only minus control delta
  - worst-well delta
  - raw-test parity checklist
  - summary JSON
- `py_compile`、`ruff check`、notebook JSON validation、`make validate-exp` が通る。
- Kaggle package は CPU / internet off / no submission で prepare できる。
