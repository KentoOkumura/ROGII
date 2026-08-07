# 要件

## 依頼

同 typewell / native overlap group の情報を、生成済み `pf_z` / `likpf_mean` などへどう活かすかを検証する。まず PF 内部の likelihood を直接変更せず、exp065 の native typewell overlap group から fold-safe な neighbor TVT drift prior を作り、exp099 の既存 PF/Beam/likPF 候補へ弱い補正として掛けた場合の train pseudo-tail OOF 改善を確認する。

## 制約

- Route: `ensemble`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- validation well 自身および同 fold valid wells の true TVT を neighbor prior pool に入れない。
- validation well の typewell cluster assignment と visible prefix context は query 情報としてのみ使う。
- 既存 `likpf_mean` / `pf_ancc` / `beam_mean` 候補を再生成しない。exp099 v2 cache を固定入力にする。
- 改善しても即 inference port / submit しない。raw-test parity audit を別途要求する。

## 受け入れ基準

- exp109 の `.steering`、`config.yaml`、補助 `.py`、train notebook、README / SESSION_NOTES / result scaffold が揃っている。
- exp065 cluster assignments と exp099 train feature cache を入力にする fold-safe OOF prior 生成が実装されている。
- `likpf_mean` baseline と neighbor prior correction 候補を同じ rows / wells / metrics で比較できる。
- 出力として candidate metrics、bucket metrics、by-well metrics、OOF prediction gzip、feature schema、summary JSON が保存される。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
