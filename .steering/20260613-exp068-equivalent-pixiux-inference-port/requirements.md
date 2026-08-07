# 要件

## 依頼

バックログ化されていた Pixiux equivalent inference port 候補を、正しい次番号 `exp068_equivalent_pixiux_inference_port` として実装する。目的は `exp039 型 branch` の価値を `exp063` 上で再評価すること。

## 制約

- Route: `ml_model`
- `exp067` は削除済みのため再利用せず、次は `exp068` とする。
- 新しい `exp039` experiment / notebook / directory を作らない。
- 親は `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit` とし、モデル実装は exp063 Pixiux LightGBM config family を使う。
- CV は exp039/exp038 系の `leave_one_original_fold_out` と `well_hash_holdout` を使う。
- PF/Beam/tracker features は exp063 output `ravaghi_vs_pixiux_public_replay_tracker_features_train.csv.gz` を使い、exp068 では再生成しない。
- 既存 exp063 v2 の単純再提出ではなく、exp039 CV metrics / public sample branch summary / submission diff / SHA / submit-check を保存する。
- static visible override や train/test same-well 前提は採用根拠にしない。

## 受け入れ基準

- `experiments/exp068_equivalent_pixiux_inference_port/` が存在し、主要ファイル名と config の experiment name が exp068 になっている。
- 学習 notebook が exp039 CV surface と exp063 tracker/PF/Beam output features を join し、exp063 LightGBM config family を exp039 CV で再学習評価する。
- 推論 notebook が exp063 inference prediction artifact を読み、submission summary と diff を保存する。
- `task validate-exp EXP=exp068_equivalent_pixiux_inference_port` が通る。
- 旧バックログ名が残らない。
