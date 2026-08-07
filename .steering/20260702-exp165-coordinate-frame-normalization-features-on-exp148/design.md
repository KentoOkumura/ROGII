# 設計

## アプローチ

exp148 の feature surface を固定し、raw horizontal well の `MD/X/Y/Z` を known-prefix 末尾 anchor を原点とする座標系へ変換する。prefix-tail azimuth で XY を `along_track` / `cross_track` に回転し、well-local scale で正規化する。さらに raw trajectory の slope / curvature / roughness と、既存 PF/Beam/likPF disagreement との interaction を add-only feature として評価する。

## 実験範囲

- 対象実験: `exp165_coordinate_frame_normalization_features_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 変更する変数: coordinate-frame normalization feature group を exp148 に add-only する。
- 固定する変数: exp072 full replay cache、exp092 U-projection surface、exp145 learned likelihood confidence features、residual target、GroupKFold seed 42、LightGBM config family。

## 再現性設計

- seed policy: GroupKFold seed 42。coordinate feature generation は RNG を使わない。
- stochastic 処理の有無: 新規 stochastic feature generation なし。LightGBM は既存 config の fixed seed / random_state を使う。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規 PF/Beam 再生成なし。exp072 / exp145 upstream cache を読む。
- 並列処理と乱数の関係: CPU deterministic LightGBM、`force_col_wise=true`、`n_jobs=8`、`num_threads=8`。
- CPU/GPU runtime と deterministic flags: GPU 無効、CPU notebook を `lgb0/lgb1/lgb2` に分割する。
- train cache / test feature regeneration の SHA 記録方針: input gzip は decompressed content SHA、生成物は summary / manifest / prediction SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: split notebook ごとに manifest と prediction SHA を保存する。提出しない限り submission SHA は不要。
- Kaggle package bootstrap 確認方針: prepare 後に CPU / internet off / source dataset / run-on-push metadata を確認する。

## リスク

- リークリスク: raw `X/Y/Z/MD` は target-free だが、正規化・interaction・selection に validation/test true TVT、oracle best、true-error rank を混ぜると leakage になる。
- CV/LB 不一致リスク: coordinate regime feature が Public LB に転移する保証はない。exp148 historical CV/LB を主比較にし、global OOF だけでは submit しない。
- ランタイム/メモリリスク: exp148 full-row CPU train に 30-40 features を追加するため、LightGBM config ごとに notebook を分割する。
- 再現性リスク: raw row id suffix と horizontal well row index の alignment が崩れると feature parity が壊れる。Kaggle train/inference 両方で id suffix bounds と anchor consistency を検査する。
