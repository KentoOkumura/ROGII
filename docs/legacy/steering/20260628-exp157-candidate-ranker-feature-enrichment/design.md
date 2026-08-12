# 設計

## アプローチ

exp101 の LightGBM multiclass / candidate-long binary / candidate-long error ranker を維持し、入力特徴だけを拡張する。exp099 v2 cache で既存候補と multiobs score を読み、exp072 full replay cache から `tvt_dense_d`、`tvt_densew_d`、`tvt_dense50_d`、`dense_std`、`dense_dist`、`pf_vs_dense` を `id` で join する。

`tvt_dense*` 候補は `last_known_tvt + tvt_dense*_d` で復元する。追加特徴は `crfe_` prefix で生成し、dense family dispersion、candidate drift per MD、likPF/Beam/PF と dense の差、tail rank、near flag、high-disagreement proxy を含める。

## 実験範囲

- 対象実験: `exp157_candidate_ranker_feature_enrichment`
- Route: `pf_beam`
- 親実験: `exp101_pf_candidate_ranker_or_nway_classifier`
- 変更する変数: candidate set、ranker feature columns
- 固定する変数: GroupKFold split、LightGBM model family、exp099 cache、exp072 cache、train-side pseudo-tail validation

## 再現性設計

- seed policy: exp101 と同じ GroupKFold seed 42、LightGBM seed fold offset、candidate-long sampling seed を使う。
- stochastic 処理の有無: exp157 の feature join / feature generation は乱数なし。LightGBM と long-frame row subsampling は exp101 と同じ固定 seed。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。保存済み exp099 / exp072 cache を読む。
- 並列処理と乱数の関係: feature generation は deterministic pandas/numpy 処理。LightGBM は CPU histogram training で deterministic anchor とは扱わない。
- CPU/GPU runtime と deterministic flags: Kaggle CPU。`enable_gpu=false`。
- train cache / test feature regeneration の SHA 記録方針: exp099 cache、exp072 auxiliary cache、schema SHA、decompressed SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: model manifest SHA、OOF prediction decompressed SHA、variant prediction SHA を記録する。submission は作らない。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks.py --strict` 後に metadata の kernel sources と config を確認する。

## リスク

- リークリスク: oracle label は supervised target だけに使い、feature には入れない。dense candidate も true TVT ではなく target-free delta から復元する。
- CV/LB 不一致リスク: train-side pseudo-tail selector audit なので、改善しても raw-test parity と continuity verifier なしに submit しない。
- ランタイム/メモリリスク: 候補数が 5 から 8 へ増え、candidate-long frame が大きくなる。`max_train_rows_per_fold=650000` で cap する。
- 再現性リスク: 上流 cache は stochastic provenance を持つため、exp157 は deterministic submission anchor にしない。
