# 要件

## 依頼

`exp098_topn_related_feature_prune` を、最新番号を接頭辞にした `exp108_topn_related_feature_prune` として実装する。

exp098 の全 260 features、つまり exp073/exp072 由来の既存 196 features + exp098 追加 rank-slot 64 features を対象に、top-n selector と関係が薄い candidate family / rank-slot signal を静的 column set で落とす。

## 制約

- Route: `ml_model`
- 親実験: `exp098_selector_rank_slot_features_on_exp073`
- base surface: exp098 と同じ exp073/exp072 196-feature cache。
- control は `exp098_full_260` として config に残すが、GPU 節約のため学習対象から外す。
- ablation 候補は `top1_related_pruned_260`、`top2_related_pruned_260`、`top3_related_pruned_260`、`non_candidate_context_plus_topn_related` として config に残す。
- 実際に学習する active variant は、既存 exp098 の rank-slot distribution と feature importance から決めた `top3_related_pruned_260` のみにする。
- feature schema を `base_196_candidate_family`、`base_196_non_candidate_context`、`rank_slot_topn_slot_features`、`rank_slot_global_candidate_stats`、`rank_slot_source_flags`、`rank_slot_pairwise_disagreement` に分類できること。
- row-wise dynamic masking は使わず、候補 family / rank-slot group 単位の静的 column set に限定する。
- direct selector、soft average、candidate TVT path replacement は行わない。
- inference は train-side review まで disabled にする。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `experiments/exp108_topn_related_feature_prune/` が存在する。
- `config.yaml` に control と prune variants 4 本が定義され、active variant は `top3_related_pruned_260` のみになっている。
- 学習 notebook が Kaggle train で同一 GroupKFold ablation を実行できる。
- feature schema に base feature group と rank-slot feature group が記録される。
- inference notebook は train-side audit only の guard で停止する。
- deterministic anchor として扱わない。Kaggle train 実行後は feature content SHA、model SHA、prediction SHA、Kaggle kernel version を記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録する。
