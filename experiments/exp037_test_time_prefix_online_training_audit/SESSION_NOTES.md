# exp037_test_time_prefix_online_training_audit セッションノート

## 目的

exp026 の fixed `exp014_bucket_shrink_params` を control に、見えない test well の既知 `TVT_input` prefix を小さい重みで online training rows として追加できるか監査する。

## 現在の状態

- 状態: 完了
- 親実験: `exp026_pseudo_tail_bucket_shrink_inference_submit`
- control: `exp026_bucket_shrink_control`
- control CV reference: 12.870780
- control Public LB reference: 12.102
- 比較候補:
  - `online_weight_0_05`
  - `online_weight_0_10`
  - `online_weight_0_20`

## コマンドログ

- 2026-06-08: `uv run python scripts/new_steering.py --experiment exp037_test_time_prefix_online_training_audit` で steering docs を作成。
- 2026-06-08: `uv run python scripts/new_experiment.py --name exp037_test_time_prefix_online_training_audit --source experiments/exp036_test_time_prefix_calibration_audit` で exp036 から実験を作成。
- 2026-06-08: notebook と audit script を exp037 名へ rename。
- 2026-06-08: `config.yaml` を test-time prefix online training audit 用に更新。
- 2026-06-08: `test_time_prefix_online_training_audit.py` を追加。fold 内 base training rows に validation-well finite prefix 由来の online rows を小さい重みで追加し、hidden tail だけで control と比較する実装。
- 2026-06-08: `uv run python -m py_compile experiments/exp037_test_time_prefix_online_training_audit/test_time_prefix_online_training_audit.py experiments/exp037_test_time_prefix_online_training_audit/settings.py experiments/exp037_test_time_prefix_online_training_audit/baseline.py experiments/exp037_test_time_prefix_online_training_audit/pseudo_tail_augmentation.py` が通過。
- 2026-06-08: `uv run python scripts/validate_experiment.py --experiment exp037_test_time_prefix_online_training_audit` が通過。
- 2026-06-08: `uv run ruff format experiments/exp037_test_time_prefix_online_training_audit/test_time_prefix_online_training_audit.py experiments/exp037_test_time_prefix_online_training_audit/settings.py` と `uv run ruff check experiments/exp037_test_time_prefix_online_training_audit/test_time_prefix_online_training_audit.py experiments/exp037_test_time_prefix_online_training_audit/settings.py` が通過。
- 2026-06-08: `uv run pytest` が通過。10 tests passed。
- 2026-06-08: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp037_test_time_prefix_online_training_audit --notebook train --run-on-push --strict --title "exp037 test time prefix online training audit train"` が通過。
- 2026-06-08: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp037_test_time_prefix_online_training_audit --notebook inference --strict --title "exp037 test time prefix online training audit inference"` が通過。inference は guard notebook。
- 2026-06-08: `uv run python scripts/record_experiment.py --experiment exp037_test_time_prefix_online_training_audit --status planned --metric rmse ...` で `metrics.json` と `experiment_summary.md` に planned 実験として記録。
- 2026-06-08: 初回 `kaggle kernels push -p experiments/exp037_test_time_prefix_online_training_audit/kaggle/train` は Kaggle API 400。原因は kernel id slug が 51 文字で長すぎた可能性が高い。
- 2026-06-08: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp037_test_time_prefix_online_training_audit --notebook train --run-on-push --strict --title "exp037 prefix online audit train" --kernel-id kentookumura/exp037-prefix-online-audit-train` で短い canonical kernel id に変更。
- 2026-06-08: `kaggle kernels push -p experiments/exp037_test_time_prefix_online_training_audit/kaggle/train` で Kaggle train version 1 を push。Kernel URL: `https://www.kaggle.com/code/kentookumura/exp037-prefix-online-audit-train`。
- 2026-06-08: `kaggle kernels pull kentookumura/exp037-prefix-online-audit-train -p /tmp/kaggle-pull/exp037-prefix-online-audit-train -m` で同じ kernel id の存在確認が成功。
- 2026-06-08: `kaggle kernels logs kentookumura/exp037-prefix-online-audit-train` で完了ログを確認。5 folds が完了し、control CV 12.870780、best same-OOF は `online_weight_0_20` 12.844383。
- 2026-06-08: `kaggle kernels output kentookumura/exp037-prefix-online-audit-train -p /tmp/kaggle-output/exp037_test_time_prefix_online_training_audit/train` で output を取得。
- 2026-06-08: Kaggle artifacts を `experiments/exp037_test_time_prefix_online_training_audit/artifacts/` に同期。
- 2026-06-08: Kaggle output の `metrics.json` をローカル `metrics.json` に反映。

## 変更点

- `config.yaml` を exp037 audit 用に更新。
- `settings.py` の `EXPERIMENT_NAME` を exp037 に更新。
- train notebook を prefix online-training audit 実行用に更新。
- inference notebook は未採用時のガード notebook として残す。

## Artifacts

Kaggle train audit で以下を保存した。

- `artifacts/prefix_online_training_summary.json`
- `artifacts/prefix_online_training_candidate_metrics.csv`
- `artifacts/prefix_online_training_bucket_summary.csv`
- `artifacts/prefix_online_training_fold_metrics.csv`
- `artifacts/prefix_online_training_well_holdout_metrics.csv`
- `artifacts/prefix_online_training_online_rows.csv`
- `artifacts/prefix_online_training_selection.csv`
- `artifacts/pseudo_tail_source_summary.csv`
- `artifacts/pseudo_tail_feature_importance.csv`
- `artifacts/exp037-prefix-online-audit-train.log`

## 結果

Kaggle train version 1 は完了。same-OOF では online training が少し改善したが、held-out candidate selection は control より悪化したため採用しない。

- `online_weight_0_20`: 12.844383、control 比 -0.026396
- `online_weight_0_05`: 12.855228、control 比 -0.015552
- `exp026_bucket_shrink_control`: 12.870780
- `online_weight_0_10`: 12.909600、control 比 +0.038820
- `raw_pseudo_tail`: 12.942938、control 比 +0.072158

selection audit:

- leave-one-original-fold-out selection: 12.999364、control 比 +0.128584
- well-hash holdout selection: 12.970333、control 比 +0.099553

`clean_prefix_online_training_supported=false`。selected method は `exp026_bucket_shrink_control`。

bucket 別には `online_weight_0_20` が rows 50-249、250-999、1000-2499、2500+ で control より改善した一方、holdout selection で fold / well への転移が不安定だった。organizer approval 未確認の rules risk もあるため inference 化しない。

## 次のアクション

1. prefix online training は inference 化しない。
2. exp026 self-route anchor と exp027 public replay anchor を維持する。
