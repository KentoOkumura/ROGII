# 設計

## アプローチ

exp063 の public replay 実装をそのまま同梱し、Kaggle CPU notebook 上で raw competition train files から train feature frame を生成する。

生成対象:
- `pixiux_likpf_public_replay` train full features
- feature schema と summary JSON

Notebook は学習を一切行わない。exp063 の `build_replay_train_frames()` と `feature_columns_for_variant()` を再利用する。test features は後続実験の inference notebook 内で current raw test files から再生成する。

## 実験範囲

- 対象実験: `exp072_exp063_full_replay_feature_cache`
- Route: `pf_beam`
- 親実験: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- 変更する変数: 保存する train feature cache artifact の粒度と命名
- 固定する変数: exp063 public replay feature generation code, raw train data, PF seeds/particles, train feature family

## リスク

- リークリスク: raw train files だけを使う。test files はこの notebook では読まない。学習はしない。
- CV/LB 不一致リスク: 後続の train/inference が同じ feature cache を読む前提になるため、schema と feature count を明示して保存する。
- ランタイム/メモリリスク: full train feature generation は exp063 と同等に重い。GPU は使わないが CPU notebook の実行時間は長い。
