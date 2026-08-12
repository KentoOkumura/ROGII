# 設計

## アプローチ

exp063 の `public_notebook_replay_audit.py` を同梱し、raw train files から exp063 と同じ `build_replay_train_frames()` を実行する。
そのうち `pixiux_likpf_public_replay` frame の full feature list を `feature_columns_for_variant()` で取得し、196 features のまま LightGBM CV を回す。

LightGBM configs は exp063 public LGBM 3 configs を維持し、mode override として GPU double precision / deterministic / fixed threads を適用する。
CV は exp063 と同じ GroupKFold by `well`、target は `TVT - last_known_tvt`。

Inference は exp063 の `build_replay_test_frame()` で current raw test の full test features を再生成し、exp071 train の saved booster manifest を読み込んで `last_known_tvt + pred_delta` を提出値にする。

## 実験範囲

- 対象実験: `exp071_exp063_full_replay_reproducibility_guard`
- Route: `ml_model`
- 親実験: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- 変更する変数: LightGBM device / gpu_use_dp / deterministic / force_col_wise / num_threads / n_jobs
- 固定する変数: raw feature replay implementation, feature set, target definition, GroupKFold split, public LGBM parameter family, train/test raw data, inference feature generation family

## リスク

- リークリスク: exp063 と同じ train-only raw feature replay を使い、test files は train-side CV に使わない。GroupKFold は `well` で分ける。
- CV/LB 不一致リスク: exp063 の full public replay CV と比較する。LB は raw test regeneration inference のみで確認する。
- ランタイム/メモリリスク: full train feature generation は exp063 と同等に重い。GPU quota を消費するため、package 生成後の Kaggle push は明示的な実行判断を記録する。
