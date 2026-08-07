# exp036_test_time_prefix_calibration_audit セッションノート

## 目的

exp026 の fixed `exp014_bucket_shrink_params` を control に、見えない test well の既知 `TVT_input` prefix だけから軽量な per-well 補正を推定できるか監査する。

## 現在の状態

- 状態: 完了
- 親実験: `exp026_pseudo_tail_bucket_shrink_inference_submit`
- control: `exp026_bucket_shrink_control`
- control CV reference: 12.870780
- control Public LB reference: 12.102
- 比較候補:
  - `prefix_bias_add`
  - `prefix_error_slope`
  - `prefix_global_residual_shrink`
  - `prefix_distance_bucket_shrink`
  - `prefix_near_continuity_decay`

## コマンドログ

- 2026-06-08: `uv run python scripts/new_steering.py --experiment exp036_test_time_prefix_calibration_audit` で steering docs を作成。
- 2026-06-08: `uv run python scripts/new_experiment.py --name exp036_test_time_prefix_calibration_audit --source experiments/exp026_pseudo_tail_bucket_shrink_inference_submit` で exp026 から実験を作成。
- 2026-06-08: `test_time_prefix_calibration_audit.py` を追加。valid well の finite `TVT_input` prefix 内に疑似 cutoff を作り、calibration zone だけで補正を fit して元の hidden tail で評価する実装にした。
- 2026-06-08: `uv run python -m py_compile experiments/exp036_test_time_prefix_calibration_audit/test_time_prefix_calibration_audit.py experiments/exp036_test_time_prefix_calibration_audit/settings.py experiments/exp036_test_time_prefix_calibration_audit/baseline.py experiments/exp036_test_time_prefix_calibration_audit/pseudo_tail_augmentation.py` が通過。
- 2026-06-08: `uv run python scripts/validate_experiment.py --experiment exp036_test_time_prefix_calibration_audit` が通過。
- 2026-06-08: `uv run pytest` が通過。10 tests passed。
- 2026-06-08: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp036_test_time_prefix_calibration_audit --notebook train --run-on-push --strict` が通過。
- 2026-06-08: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp036_test_time_prefix_calibration_audit --notebook inference --strict` が通過。inference は guard notebook。
- 2026-06-08: `uv run python scripts/record_experiment.py --experiment exp036_test_time_prefix_calibration_audit --status planned --metric rmse ...` で `metrics.json` と `experiment_summary.md` に planned 実験として記録。
- 2026-06-08: 初回 `kaggle kernels push -p experiments/exp036_test_time_prefix_calibration_audit/kaggle/train` は Kaggle API 400。原因は title が kernel id slug に resolve しないこと。
- 2026-06-08: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp036_test_time_prefix_calibration_audit --notebook train --run-on-push --strict --title "exp036 test time prefix calibration audit train"` で title を kernel id slug と整合する形に修正。
- 2026-06-08: `kaggle kernels push -p experiments/exp036_test_time_prefix_calibration_audit/kaggle/train` で Kaggle train version 1 を push。Kernel URL: `https://www.kaggle.com/code/kentookumura/exp036-test-time-prefix-calibration-audit-train`。
- 2026-06-08: `kaggle kernels pull kentookumura/exp036-test-time-prefix-calibration-audit-train -p /tmp/kaggle-pull/exp036-test-time-prefix-calibration-audit-train -m` で同じ kernel id の存在確認が成功。
- 2026-06-08: CLI の `kaggle kernels status` は `GetKernelSessionStatus` 500。通常 `logs` / `output` はこの時点では空で返ったが、Kaggle UI 側では実行中ログが出ていることを確認済み。
- 2026-06-08: `kaggle kernels logs kentookumura/exp036-test-time-prefix-calibration-audit-train` で完了ログを確認。5 folds が完了し、control CV 12.870780、raw pseudo-tail CV 12.942938、selected method は `exp026_bucket_shrink_control`。
- 2026-06-08: `kaggle kernels output kentookumura/exp036-test-time-prefix-calibration-audit-train -p /tmp/kaggle-output/exp036_test_time_prefix_calibration_audit/train` で output を取得。
- 2026-06-08: Kaggle artifacts を `experiments/exp036_test_time_prefix_calibration_audit/artifacts/` に同期。
- 2026-06-08: Kaggle output の `metrics.json` をローカル `metrics.json` に反映し、`uv run python scripts/record_experiment.py --experiment exp036_test_time_prefix_calibration_audit --status completed --cv 12.87078 ...` で `experiment_summary.md` を更新。

## 変更点

- `config.yaml` を exp036 audit 用に更新。
- `settings.py` の `EXPERIMENT_NAME` を exp036 に更新。
- train notebook を prefix calibration audit 実行用に更新。
- inference notebook は未採用時のガード notebook として残す。

## Artifacts

Kaggle train audit で以下を保存した。

- `artifacts/prefix_calibration_summary.json`
- `artifacts/prefix_calibration_candidate_metrics.csv`
- `artifacts/prefix_calibration_bucket_summary.csv`
- `artifacts/prefix_calibration_fold_metrics.csv`
- `artifacts/prefix_calibration_well_holdout_metrics.csv`
- `artifacts/prefix_calibration_cutoff_summary.csv`
- `artifacts/prefix_calibration_selection.csv`
- `artifacts/pseudo_tail_source_summary.csv`
- `artifacts/pseudo_tail_feature_importance.csv`
- `artifacts/exp036-test-time-prefix-calibration-audit-train.log`

## 結果

- `exp026_bucket_shrink_control`: 12.870780
- `prefix_near_continuity_decay`: 12.916015、control 比 +0.045236
- `raw_pseudo_tail`: 12.942938、control 比 +0.072158
- `prefix_distance_bucket_shrink`: 13.119153、control 比 +0.248373
- `prefix_global_residual_shrink`: 13.119682、control 比 +0.248902
- `prefix_bias_add`: 15.551284、control 比 +2.680505
- `prefix_error_slope`: 19.540902、control 比 +6.670123

leave-one-original-fold-out selection と well-hash holdout selection は全 holdout で `exp026_bucket_shrink_control` を選択し、どちらも 12.870780。`clean_prefix_calibration_supported=false`。

bucket 別には `prefix_near_continuity_decay` が rows 250-999 で 6.217497 -> 6.214033 とわずかに改善したが、rows 0-49 は 0.820889 -> 7.643347、rows 50-249 は 2.986194 -> 4.815873 と大きく悪化したため採用しない。

## 次のアクション

1. prefix calibration は inference 化しない。
2. exp026 self-route anchor と exp027 public replay anchor を維持する。
