# 設計

## アプローチ

exp148 の feature surface を固定し、same-well row-neighbor context を add-only feature group として追加する。well 内を `md_since` 順に並べ、input/candidate/confidence columns から lag/lead delta と centered rolling mean/std を作る。

## 実験範囲

- 対象実験: `exp220_row_neighbor_input_context_features_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 比較対象: exp148 CPU/GPU、exp193、exp198
- 変更する変数: `row_neighbor_input_context` feature group の追加
- 固定する変数: exp072 full replay cache、exp092 U-projection、exp145 learned likelihood features、GroupKFold by well、LightGBM config family、residual target

## Feature 設計

- prefix: `rnic_`
- order: `md_since`、欠損時は id suffix
- lag/lead periods: 1 / 3 / 5
- rolling window: 5
- default source columns: `gr`, `dzdmd`, `md_since`, `beam_mean_d`, `likpf_mean_d`, `ll_candidate_tvt_std`, `ll_learned_prob_entropy`, `uproj_source_u_std`
- max feature count: 60

## 再現性設計

- seed policy: GroupKFold seed 42。row-neighbor feature generation 自体は RNG なし。
- stochastic 処理の有無: 新規 feature generation には無し。upstream PF/Beam / learned-likelihood cache は固定 artifact として扱う。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成しない。exp072 / exp145 artifact を読む。
- 並列処理と乱数の関係: row-neighbor feature generation は pandas/numpy deterministic 操作のみ。LightGBM は CPU deterministic flags、固定 `n_jobs/num_threads=8`。
- CPU/GPU runtime と deterministic flags: CPU only。Kaggle metadata は `enable_gpu=false`。
- train cache / test feature regeneration の SHA 記録方針: gzip は decompressed content SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: split notebook ごとの manifest / prediction SHA を記録し、inference 化する場合は merged manifest または split source の SHA を追加する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --notebook train_lgb{0,1,2} --strict` 後、package 内 config と metadata の CPU/source 設定を確認する。

## リスク

- リークリスク: lead / centered rolling は future rows を見るため、hidden inference で同一 well 全体が見える前提が崩れると使えない。`TVT_input` や model prediction は source から除外する。
- CV/LB 不一致リスク: exp193 は CV 改善ほど LB が伸びなかったため、global CV だけでは submit 候補にしない。
- ランタイム/メモリリスク: CPU で 15 boosters 一括は timeout しやすいため 3 split に分ける。
- 再現性リスク: split output の集約時に manifest/source version の取り違えが起きやすい。inference 化前に split artifact の SHA を記録する。
