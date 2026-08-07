# 要件

## 依頼と手法契約

- 依頼原文: TODO
- 期待する成果: TODO
- 一次資料 / 参照実装: TODO
- input: TODO
- target / objective: TODO
- output: TODO
- loss: TODO
- decode: TODO
- context unit: TODO (`row`、`window`、`whole-well`、`set`、`field` など)
- 実装区分: TODO (`faithful`、`staged-faithful`、`proxy`)
- 省略する機構と理由: TODO
- proxyで検証できない主張: TODO
- proxyの場合のユーザー承認: TODO (`N/A` または承認日時 / 依頼メッセージ)

## 制約

- Route: TODO (`ml_model`、`pf_beam`、`ensemble`)
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- TODO

## 受け入れ基準

- 手法契約の `input / target / output / loss / decode / context unit` がコードと一致する。
- 実装区分と実験名が実装した機構を正確に表す。
- `proxy` の場合は、省略点、検証不能な主張、ユーザー承認が記録されている。
- TODO
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
