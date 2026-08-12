# 設計

## アプローチ

exp073 の deterministic full replay ML surface を固定し、同一 horizontal well 内で既知 prefix GR tail と評価区間 GR query を multi-scale NCC / z-normalized L2 で照合する。生成するのは match score、matched prefix TVT offset、scale disagreement、GR missingness/length context だけで、評価区間 true TVT や外部 typewell match は使わない。

## 実験範囲

- 対象実験: `exp090_lateral_self_gr_match_pseudotail_probe`
- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 変更する変数: self-GR match summary feature の有無と feature group
- 固定する変数: exp072 196-feature cache、target `TVT - last_known_tvt`、exp073 LightGBM config family、GroupKFold by well、非加重 RMSE

## 再現性設計

- seed policy: LightGBM seed は exp073/exp089 系の config family を維持する。
- stochastic 処理の有無: self-GR feature generation は deterministic。乱数は使わない。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。exp072 cache 内の既存 feature surface は読むが、今回の追加特徴は PF/Beam に依存しない。
- 並列処理と乱数の関係: self-GR generation は逐次処理で global RNG なし。
- CPU/GPU runtime と deterministic flags: train は `gpu_repro_guard_dp_threads8` を主にし、`gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、`num_threads=8` を使う。
- train cache / test feature regeneration の SHA 記録方針: exp072 cache source SHA、schema SHA、self-GR feature schema、OOF prediction SHA を summary / manifest に記録する。
- model manifest / prediction / submission SHA 記録方針: fold model SHA、OOF prediction SHA、model manifest を保存する。submission は未選択なので作らない。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --notebook train --strict` 後に generated package と bootstrap manifest を確認する。

## リスク

- リークリスク: 評価区間 true TVT を match に使うと leakage になるため禁止。使用するのは same-well horizontal GR、finite prefix `TVT_input`、row index のみ。
- CV/LB 不一致リスク: GR alignment 系は過去に悪化しているため、pooled RMSE だけで inference port しない。
- ランタイム/メモリリスク: well ごとの NCC candidate matrix が大きくなりうるため、prefix candidate は tail 1024 rows、stride 3 に制限する。
- 再現性リスク: Kaggle source cache と local cache の所在差があるため、package bootstrap と kernel source version を記録する。
