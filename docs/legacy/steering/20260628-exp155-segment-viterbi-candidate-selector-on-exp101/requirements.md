# 要件

## 依頼

`segment_viterbi_candidate_selector_on_exp101` を実装する。

## 制約

- Route: `pf_beam`
- 親実験: `exp101_pf_candidate_ranker_or_nway_classifier`
- 入力 cache: `exp099_pf_multi_observation_likelihood_probe` v2 train feature cache
- 新規 LightGBM 学習は行わず、exp101 の saved booster を OOF fold ごとに再適用する。
- 候補集合は exp101 と同じ `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb` に固定する。
- `tvt_dense` は exp101/exp099 の score schema に無いため、この実装では追加しない。
- Viterbi selection に true TVT、oracle best、true error を使わない。
- 再現性: `docs/06_reproducibility.md` に従い、入力 SHA、exp101 model manifest SHA、prediction SHA を記録する。

## 受け入れ基準

- exp155 の実験フォルダ、config、notebook、補助モジュール、README、result、SESSION_NOTES が作成されている。
- `selector.viterbi_grid` で switch penalty、jump penalty、delta cap、PF std cap、md_since gate、minimum segment length を比較できる。
- `likpf_mean_single`、`exp101_error_ranker_rowwise`、Viterbi variants、oracle の metrics / distribution / by-well / bucket metrics が保存される。
- `make validate-exp EXP=exp155_segment_viterbi_candidate_selector_on_exp101` が通る。
- Kaggle push 前に、追加 variant 数、新規 booster 数 0、control 再学習なしを `SESSION_NOTES.md` に記録する。
