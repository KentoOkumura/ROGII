# 設計

## アプローチ

exp172 の 2段階 cache/train 構成を再利用し、prefix feature generator を last50-first rebuild に拡張する。各 well の raw horizontal known prefix を finite `TVT_input` anchor まで読み、source frame を last 50 known-prefix rows に切ってから TVT aggregate/stat、X/Y/Z trajectory/geometry、GR quality、typewell GR calibration、SC/NCC、candidate multiobs score/MAE/NCC、candidate-vs-prefix range/outside flag を作る。

active variant は exp148 surface から full-prefix-derived base columns と exp145 learned multiobs columns を落とし、last50-first rebuild group を追加する。exp148 control は再学習しない。

## 実験範囲

- 対象実験: `exp185_last50_first_prefix_feature_rebuild_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 変更する変数: prefix-derived feature source frame と、それに基づく feature group
- 固定する変数: target rows、fold split、PF/Beam 候補値、U-projection、learned probability/error model、exp148 historical baseline

## 再現性設計

- seed policy: GroupKFold seed 42。新しい stochastic feature generation はない。
- stochastic 処理の有無: LightGBM GPU 学習のみ非 bitwise リスクあり。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。exp072 / exp145 の保存済み deterministic cache を読む。
- 並列処理と乱数の関係: feature cache は乱数を使わない。LightGBM は fixed `n_jobs` / `num_threads` と deterministic flags を設定する。
- CPU/GPU runtime と deterministic flags: split train は Kaggle GPU、`gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、threads 8。
- train cache / test feature regeneration の SHA 記録方針: gzip は decompressed content SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: train output manifest、prediction SHA、model SHA を記録する。submit する場合のみ submission SHA を追加する。
- Kaggle package bootstrap 確認方針: prepare 後に metadata と bootstrap 内 config が GPU mode / kernel sources / run_on_push と整合しているか確認する。

## リスク

- リークリスク: valid/test true TVT、oracle best、true-error rank は使わない。raw known prefix と target-free candidate values だけを使う。
- CV/LB 不一致リスク: prefix 近傍で改善しても hidden-like / worst-well が崩れる可能性があるため、global OOF 小改善だけでは submit しない。
- ランタイム/メモリリスク: full-row 3.78M 行で feature cache が重い。train は `lgb0/1/2` split にする。
- 再現性リスク: GPU LightGBM は bitwise anchor と扱わない。採用候補になる場合は rerun/SHA 比較を別途検討する。
