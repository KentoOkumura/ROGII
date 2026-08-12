# 設計

## アプローチ

`exp063` の `metrics.json` を source of truth として読み、strict replay CV と inference v2 の metadata を検証する。実 submission file がローカル `/tmp/kaggle-output/.../submission.csv` に存在する場合は、CSV を直接読み直して行数、列、欠損、重複、SHA256、予測範囲を再計算する。存在しない環境では `exp063` metrics 内の submit-check と summary を使う。

補助 leakage gate として `exp064_train_test_well_id_assert_probe` の `hidden_code_submission.status == complete` かつ interpretation に no overlap 検出が記録されていることを確認する。

すべての required rule が通った場合、`approved_for_code_submit = true` とし、提出対象は `kentookumura/exp063-ravaghi-pixiux-strict-replay-infer` version 2 の `submission.csv` と記録する。`exp066` 自体は提出用 notebook ではなく、提出判断の記録用実験とする。

## 実験範囲

- 対象実験: `exp066_cv_submit_gate`
- Route: `ml_model`
- 親実験: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- 変更する変数: submit gate のしきい値と required rule
- 固定する変数: `exp063` の saved booster inference v2、`pixiux_likpf_public_replay` `lgb_mean`、feature generation、training config、submission output

## リスク

- リークリスク: `exp063` は public notebook replay 由来で PF/likelihood features を使う。train/test same well static override は使わないが、public implementation の転用リスクは残るため、gate は `exp064` no-overlap probe と strict replay exclusions を記録する。
- CV/LB 不一致リスク: `exp063` の CV 9.63 は強いが、Public LB 基準 `exp027` 8.781 や ML route LB 基準 `exp039` 11.740 と評価条件が違う。gate 通過は「提出して LB を確認する価値がある」という判断であり、LB anchor 更新ではない。
- ランタイム/メモリリスク: `exp066` は軽量 JSON/CSV audit のみ。提出対象の inference runtime は `exp063` v2 で 127.648 sec と記録済み。
