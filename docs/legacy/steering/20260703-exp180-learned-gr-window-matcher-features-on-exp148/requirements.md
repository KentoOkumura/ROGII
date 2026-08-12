# 要件

## 依頼

`learned_gr_window_matcher_features_on_exp148` backlog を実装する。exp178 の learned GR match probability / expected-error signal を、exp148 の LightGBM anchor へ add-only confidence feature として評価できる状態にする。

## 制約

- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- control 再学習なし。比較は exp148 の既存 CV / Public LB を historical baseline とする。
- Active variant は `learned_gr_window_matcher_addonly` の 1 つだけにする。
- 候補 TVT の hard switch、softmax weighted TVT、midpoint、direct correction、PF/Beam 再生成はしない。
- observed finite `TVT_input` prefix rows だけを matcher label として使い、評価 tail true TVT、NaN `TVT_input` rows、oracle rank を feature source に使わない。
- 再現性: `docs/06_reproducibility.md` に従い、feature cache の decompressed SHA、model/prediction SHA、Kaggle kernel version を実行時に記録する。

## 受け入れ基準

- `experiments/exp180_learned_gr_window_matcher_features_on_exp148/` が作成され、config/README/SESSION_NOTES/result/metrics が TODO なしで記載されている。
- feature group `learned_gr_window_matcher` が実装され、probability、expected-error、top1/top2 margin、entropy、real-vs-shuffled/no-GR gap、candidate-family indicator、`md_since` interaction を出力できる。
- train feature cache は fold-safe by well scorer fitting を既定にする。
- `gr_matcher_features` notebook kind を Kaggle package 化できる。
- `py_compile`、`ruff --select F821`、Jupytext 変換/test、`make validate-exp` が通る。
