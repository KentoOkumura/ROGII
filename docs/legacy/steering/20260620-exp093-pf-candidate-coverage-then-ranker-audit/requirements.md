# 要件

## 依頼

`pf_candidate_coverage_then_ranker_audit` を実装する。PF/Beam/likelihood-PF 候補集合に真値近傍候補がどの程度含まれるかを先に測り、supervised candidate ranker / N-way classifier に進む価値があるかを判定する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- train-side audit のみ。推論 port、submission、直接 TVT regression、hidden branch 置換はしない。
- 評価区間 true TVT は coverage、oracle headroom、miss-rate 診断にだけ使う。候補生成や target-free rank score には使わない。
- primary candidate set は `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb`。追加 ablation として `self_gr_ens`、`self_gr_best`、`self_gr_sc8/15/25` を入れる。
- 分岐判断は全体 RMSE だけでなく、bucket 別 topK coverage、oracle headroom、worst bucket を見る。

## 受け入れ基準

- `experiments/exp093_pf_candidate_coverage_then_ranker_audit/` に設定、補助コード、train notebook、記録ファイルがある。
- `config.yaml` に route、lineage、leakage policy、candidate sets、expected train artifacts が明記されている。
- train notebook から `pf_candidate_coverage_then_ranker_audit.py` を実行し、candidate metrics、rank metrics、bucket metrics、candidate-set bucket metrics、by-well metrics、summary JSON を保存できる。
- summary JSON に input cache SHA、decompressed SHA、exp056/exp083 context SHA、ranker readiness recommendation が含まれる。
- deterministic anchor として扱わないことが記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
