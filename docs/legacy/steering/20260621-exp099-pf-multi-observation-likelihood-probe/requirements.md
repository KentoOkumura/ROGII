# 要件

## 依頼

`pf_multi_observation_likelihood_probe` を実装する。exp093 で弱かった target-free candidate rank score を改善できるか、既存 PF/Beam/likelihood-PF 候補を複数 GR 観測点の likelihood で再採点して監査する。

## 制約

- Route: `pf_beam`
- 親実験: `exp093_pf_candidate_coverage_then_ranker_audit`
- 入力 cache は `exp072_exp063_full_replay_feature_cache` の deterministic train feature cache を固定する。
- PF/Beam / likelihood-PF は再実行しない。既存候補値を読むだけにする。
- true TVT は scoring / oracle / rank metrics のみに使う。multi-observation likelihood と candidate rank score には使わない。
- supervised ranker、N-way classifier、inference port、submission はこの実験の範囲外にする。
- 再現性: `docs/06_reproducibility.md` に従い、upstream cache SHA と gzip decompressed content SHA を記録する。

## 受け入れ基準

- `experiments/exp099_pf_multi_observation_likelihood_probe/` に config、補助コード、train / inference notebook、記録ファイルがある。
- `config.yaml` に route、lineage、leakage policy、multi-observation likelihood 設定、candidate sets、expected train artifacts が明記されている。
- train notebook から `pf_multi_observation_likelihood_probe.py` を実行し、candidate metrics、rank metrics、bucket metrics、by-well metrics、multiobs well summary、row context、candidate long、summary JSON を保存できる。
- multi-observation likelihood は raw horizontal GR、row index、finite prefix TVT_input、既存候補 TVT だけから計算する。
- deterministic anchor として扱わない。model / prediction / submission SHA は対象外であることを明記する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
