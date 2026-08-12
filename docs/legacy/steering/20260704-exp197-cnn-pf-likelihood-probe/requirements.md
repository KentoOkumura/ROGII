# 要件

## 依頼

`KAGGLE_DIRECTION.md` の backlog `cnn_pf_likelihood_probe` を実験化する。PF の point-GR likelihood をいきなり live PF weight に入れず、固定 PF/Beam/likPF 候補に対する candidate-level learned local CNN/SDF likelihood scorer として train-side で sanity check する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 入力候補は exp099 の固定 PF/Beam/likPF train pseudo-tail cache を使い、PF/Beam を再実行しない。
- validation well の true TVT は label / metrics にだけ使い、window center、particle weight、typewell sampling の入力には使わない。
- point-GR likelihood、exp099 multi-observation score、exp111 learned likelihood、likPF baseline と比較する。
- shuffled-GR / no-GR negative control を同じ split / row sample schedule で評価する。
- 改善しても raw-test parity と worst-well guard なしで submit しない。

## 受け入れ基準

- `experiments/exp197_cnn_pf_likelihood_probe/` に config、train notebook、推論なし notebook、README、SESSION_NOTES、result、metrics の初期記録がある。
- train notebook は薄い entrypoint ではなく、入力確認、candidate index 生成、CNN training、negative control、metrics / SHA 保存をセルで追える。
- Kaggle train 前の GPU cost guard と、実行 variant 数 / fold 数 / model 数が `SESSION_NOTES.md` に記録されている。
- deterministic anchor とは扱わず、GPU training / upstream PF cache / row subsample の再現性制約を記録している。
- 出力 gzip は decompressed content SHA を主証拠として summary に記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
