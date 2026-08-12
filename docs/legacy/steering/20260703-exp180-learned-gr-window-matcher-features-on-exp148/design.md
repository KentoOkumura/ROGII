# 設計

## アプローチ

exp148 の既存 feature surface、U-projection、exp145 learned likelihood confidence は固定する。新規に、observed `TVT_input` prefix rows から supervised GR window pair scorer を学習し、既存候補 (`pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb`) ごとの match probability / expected-error / negative-control gap を feature 化する。

## 実験範囲

- 対象実験: `exp180_learned_gr_window_matcher_features_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 変更する変数: `learned_gr_window_matcher` feature group の追加
- 固定する変数: exp072/exp092 base surface、exp145 learned likelihood features、LightGBM config family、GroupKFold by well、exp148 historical baseline

## Feature 設計

- scorer training pairs:
  - positive: observed prefix row の true `TVT_input`
  - negative: fixed offset decoys
  - inputs: real GR window descriptors、shuffled-GR negative-control descriptors、no-GR context
- train cache:
  - GroupKFold by well の fold ごとに、validation well を除外した prefix pairs で scorer を fit する。
  - validation well の full rows に対して既存 candidate TVT を score する。
- emitted features:
  - per-candidate probability / shuffled probability / no-GR probability
  - per-candidate expected-error
  - probability margin、entropy、expected-error margin
  - real-vs-shuffled/no-GR gap
  - top1/min-error candidate family indicator
  - `md_since` interaction

## 再現性設計

- seed policy: fixed seed 42。shuffled-GR roll は experiment name + well id から SHA256 stable seed を作る。
- stochastic 処理の有無: sklearn logistic / HistGradientBoostingRegressor と LightGBM。いずれも seed を固定する。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。上流 exp072 / exp145 cache を固定入力として使う。
- 並列処理と乱数の関係: scorer feature generation は global RNG を使わない。LightGBM は deterministic / force_col_wise / fixed threads。
- CPU/GPU runtime と deterministic flags: CPU `cpu_deterministic_threads8` を既定とする。
- train cache / test feature regeneration の SHA 記録方針: gzip は decompressed content SHA を主証拠として manifest に記録する。
- model manifest / prediction / submission SHA 記録方針: train 完了時に model SHA と OOF prediction SHA を記録する。submission は submit 候補になった場合のみ記録する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後に package metadata と support config を確認する。

## リスク

- リークリスク: observed prefix label は hidden test でも利用可能だが、tail true TVT を scorer training に混ぜると leakage になる。train cache は fold-safe by well で厳しめに評価する。
- CV/LB 不一致リスク: GR matcher confidence が Public LB 側で過適合する可能性があるため、near-row、longtail、worst-well、hidden-like stress を確認してから submit 判断する。
- ランタイム/メモリリスク: full-row feature cache と 15 boosters は重い。feature cache と split train notebooks を分離できる構成にする。
- 再現性リスク: upstream PF/Beam cache は stochastic parent 由来なので、この実験単独を deterministic submission anchor とは扱わない。
