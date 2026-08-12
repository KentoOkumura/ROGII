# 要件

## 依頼

`candidate_ranker_feature_enrichment` を実装する。これは exp101 の supervised candidate selector / ranker に `tvt_dense` family と high-drift / disagreement 信頼度特徴を追加する実験。

## 制約

- Route: `pf_beam`
- 親実験: `exp101_pf_candidate_ranker_or_nway_classifier`
- 入力 cache:
  - exp099 v2 multi-observation likelihood train feature cache
  - exp072 full replay train feature cache
- Kaggle runtime は CPU。`enable_gpu=false`。
- 候補集合は exp101 の 5候補に `tvt_dense`、`tvt_densew`、`tvt_dense50` を追加する。
- `tvt_dense*` は exp099 cache に無いため、exp072 cache の `tvt_dense*_d` から `last_known_tvt + delta` で復元する。
- 追加特徴は target-free な drift、candidate disagreement、tail/near-row proxy、PF/Beam/likPF-vs-dense 差に限定する。
- true TVT、oracle best、true error、fold label は特徴量に使わない。
- 直接 TVT regressor、soft average、hard replacement、submission は作らない。
- 再現性: `docs/06_reproducibility.md` に従い、入力 SHA、feature schema SHA、model manifest SHA、prediction SHA を記録する。

## 受け入れ基準

- exp157 の実験フォルダ、config、notebook、補助モジュール、README、result、SESSION_NOTES、metrics 初期状態が作成されている。
- `candidate_ranker_feature_enrichment.py` が exp099 cache と exp072 cache を join し、dense candidate と追加特徴を作れる。
- Kaggle train push 前コストとして CPU runtime、active experiment 1、LightGBM family 3、fold 5、合計 booster 15、control 再学習なしが `SESSION_NOTES.md` に記録されている。
- `python3 -m py_compile`、notebook JSON check、`uv run python scripts/validate_experiment.py --experiment exp157_candidate_ranker_feature_enrichment` が通る。
- Kaggle package prepare を行う場合は、metadata に exp099 と exp072 の kernel source が入る。
- deterministic anchor とは扱わない。gzip 生成物を比較する場合は decompressed content SHA を主証拠として記録する。
