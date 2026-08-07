# exp158_segment_continuity_selector_on_exp157

## 状態

Kaggle train v1 完了。提出なし。train-side supported。

## 仮説

exp157 の row-wise `lgb_candidate_error_ranker` は `likpf_mean` を大きく超え、dense family も使えた。一方で row-wise switch が大きく、そのまま inference / submit するには path continuity のリスクが高い。

exp157 の per-candidate predicted-error surface を使い、well 内の候補 path を Viterbi で連続化すれば、exp157 の RMSE 改善を残しつつ過剰な候補切替を抑えられる可能性がある。

## 検証方針

exp099 v2 cache、exp072 dense feature cache、exp157 saved booster から OOF score surface を復元する。候補集合は exp157 と同じ 8 候補に固定し、`likpf_mean_single`、`exp157_error_ranker_rowwise`、Viterbi variants、oracle を比較する。

Viterbi は predicted-error local cost に、switch penalty、candidate TVT jump penalty、`likpf_mean` からの delta cap、`pf_ancc_std` cap、`md_since` gate、minimum segment length pruning を加える。

## 判定

best Viterbi は RMSE 10.789163、within10 0.792647。`likpf_mean_single` から RMSE -0.805734、exp157 row-wise から -0.006590 改善した。

path switch は exp157 row-wise の 277,110 から 11,767 に減った。max well path switch も 357.199/1000 rows から 24.092/1000 rows に下がった。

## 所見

exp158 は train-side continuity audit として supported。ただし worst well `86454a6f` は RMSE 57.836738 とまだ重く、提出候補にする前に raw-test parity、hidden-like stress、worst-well guard を同じ exp158 内で確認する。

## 主な生成物

- `exp158_segment_continuity_selector_on_exp157_metrics.csv`
- `exp158_segment_continuity_selector_on_exp157_oof_predictions.csv.gz`
- `exp158_segment_continuity_selector_on_exp157_selection_distribution.csv`
- `exp158_segment_continuity_selector_on_exp157_by_well.csv`
- `exp158_segment_continuity_selector_on_exp157_bucket_metrics.csv`
- `exp158_segment_continuity_selector_on_exp157_viterbi_params.csv`
- `exp158_segment_continuity_selector_on_exp157_score_summary.csv`
- `exp158_segment_continuity_selector_on_exp157_summary.json`
