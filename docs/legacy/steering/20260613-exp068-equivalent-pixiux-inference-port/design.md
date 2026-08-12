# 設計

## アプローチ

`exp063` の整理済み public replay 実装を親にし、exp039/exp038 系の CV surface 上で exp063 の Pixiux LightGBM model family を再学習評価する。PF/Beam/tracker features は exp063 output を `id` join で使い、exp068 では PF/Beam を再生成しない。exp063 の実装ファイルは変更しない。

LightGBM は GPU 実行を使用しつつ、`deterministic=true` / `force_col_wise=true` を設定する。

追加する `exp063_branch_audit.py` は、train で次を保存する。

- `exp063_model_exp039_cv_metrics.csv`
- `exp063_model_exp039_cv_by_well.csv`
- `exp063_model_exp039_cv_predictions.csv.gz`
- `exp063_model_exp039_cv_summary.json`

inference では exp063 inference predictions を使い、次を保存する。

- `submission.csv`
- `exp063_branch_inference_summary.json`
- optional `exp063_branch_submission_diff.csv`

## 実験範囲

- 対象実験: `exp068_equivalent_pixiux_inference_port`
- Route: `ml_model`
- 親実験: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- 参照 branch: `exp039_ravaghi_single_lgbm_inference_submit`
- 変更する変数: exp063 Pixiux LightGBM model family を exp039 CV surface で評価する監査面
- 固定する変数: exp039/exp038 系 CV surface、exp063 output tracker/PF/Beam features、exp063 実装ファイル

## リスク

- リークリスク: exp039 CV surface の `target_tvt` は train-side scoring label としてのみ使う。PF/Beam/tracker features は exp063 output artifact 由来で、exp068 では再生成しない。
- CV/LB 不一致リスク: exp063 の direct LB 8.811 と exp027 8.781 の差は小さいため、提出前に差分が exp063 と同一かを確認する。
- ランタイム/メモリリスク: exp039 CV surface と exp063 tracker features の join、および 2 audit x 3 LightGBM config x 5 splits の学習が重い。Kaggle GPU notebook で実行する。
