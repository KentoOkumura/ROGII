# 設計

## アプローチ

exp092 の U-projection correction / disagreement surface を維持し、exp114 の fold-safe spatial prior OOF diagnostics を add-only confidence features として結合する。

追加するのは `xy_only_k8` と `xy_plus_trajectory_shape_k8` の prior value、prior quality、candidate disagreement、exp118 best gate proxy、near/longtail interaction に限定する。spatial prior を直接補正値や selector output としては使わない。

## 実験範囲

- 対象実験: `exp159_spatial_prior_confidence_features_on_exp092`
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- 変更する変数: spatial prior confidence add-only feature set
- 固定する変数: exp072 base cache、exp092 U-projection feature generation、residual target、GroupKFold by well、LightGBM config family

## 再現性設計

- seed policy: fixed GroupKFold seed 42、new PF/Beam RNG なし
- stochastic 処理の有無: feature merge は deterministic、LightGBM GPU training は stochastic component として記録
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。exp072 cache と exp114 OOF cache を固定入力として使用
- 並列処理と乱数の関係: LightGBM deterministic flags、fixed `n_jobs=8` / `num_threads=8`
- CPU/GPU runtime と deterministic flags: Colab high-memory GPU 前提、`gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`
- train cache / test feature regeneration の SHA 記録方針: gzip は decompressed content SHA を主証拠として記録
- model manifest / prediction / submission SHA 記録方針: model manifest と OOF prediction SHA を記録。submission は生成しない
- Kaggle package bootstrap 確認方針: 今回は Colab runner を優先し、canonical notebook は Kaggle-compatible なまま維持する

## リスク

- リークリスク: exp114 OOF prior が fold-safe であることに依存する。validation true TVT や oracle rank は使わない
- CV/LB 不一致リスク: spatial prior は train-side neighbor 構造に依存するため raw-test/full-train parity なしに submit しない
- ランタイム/メモリリスク: exp114 OOF gzip は大きいため Colab DriveFS 直読みを避け、`/content` にコピーする
- 再現性リスク: GPU LightGBM は deterministic anchor として扱わず、SHA と runtime metadata を記録する
