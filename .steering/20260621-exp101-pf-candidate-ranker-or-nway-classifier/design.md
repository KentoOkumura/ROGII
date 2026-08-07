# Design

exp099 v2 の wide feature cache は `id`, `well`, `target`, `last_known_tvt`, 5候補の絶対 TVT、multiobs likelihood feature を持つ。exp101 はこの cache を読み、`true_tvt = last_known_tvt + target` を scoring と oracle label 作成にだけ使う。

## Candidate Selector

候補集合:

- `pf_ancc`
- `beam_mean`
- `likpf_mean`
- `sc_ens`
- `hyb`

各 row で `abs(candidate_tvt - true_tvt)` が最小の候補 index を `oracle_label` とする。

特徴量:

- source context: `last_known_tvt`, `pf_ancc_std`, `beam_mean_d`, `sc_ens_d`, `hyb_d`, `eval_len`, `md_since`, `likpf_mean_d`
- multiobs: `multiobs_score_*`, `multiobs_mae_*`, `multiobs_ncc_*`, `multiobs_score_max`, `multiobs_score_mean`, `multiobs_score_gap`, `multiobs_top1_source_id`
- engineered: candidate delta from last anchor、pairwise candidate absolute differences、candidate mean/std/range

`target`, `true_tvt`, `oracle_label`, `oracle_candidate` は feature に入れない。

## Models

GroupKFold by `well` で以下を OOF 比較する。

- `lgb_multiclass`: row-wide feature から 5-class classification。
- `lgb_candidate_binary`: candidate-long feature から oracle candidate かどうかを binary scoringし、row内 argmax。
- `lgb_candidate_error_ranker`: candidate-long feature から candidate absolute error を予測し、row内 argmin。

candidate-long model は full 5x rows を作ると重いため、train 側 row を fold ごとに固定 seed で上限 sampling する。validation 側は全 candidate を score する。

## Metrics

- selected TVT RMSE / MAE
- within 1/2/5/10 ft
- oracle label accuracy
- selection distribution, especially `pf_ancc` rate
- distance / tail / eval length / PF std / likPF delta bucket metrics
- by-well RMSE and path switch count
- feature importance

## Decision

`likpf_mean` 単体と target-free `multiobs_score_top1` を baseline にする。OOF が `likpf_mean` を明確に超え、`pf_ancc` を非自明に選べる場合だけ follow-up の continuity / raw-test parity audit に進む。CV 改善だけでは提出しない。

## Reproducibility

- exp099 input gzip raw SHA と decompressed SHA を summary に保存する。
- schema SHA を保存する。
- LightGBM model files と manifest SHA を保存する。
- OOF selected predictions gzip raw SHA と decompressed SHA、variant 別 prediction SHA を保存する。
