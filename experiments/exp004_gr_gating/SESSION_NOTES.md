# exp004_gr_gating セッションノート

## 目的

`exp002_drift_minimal` を control baseline とし、`exp003_residual_ablation` の no-GR 改善を条件付きで使う。GR signal features を全削除せず、inference-safe な well 条件で no-GR prediction へ hard/soft に寄せる gating を検証する。

## 現在の状態

- 状態: Kaggle train CV 完了
- 親実験: `exp003_residual_ablation`
- control baseline: `exp002_drift_minimal` CV 14.124569 / public LB 12.533
- alternate comparison: `exp003_residual_ablation` CV 13.882944 / public LB 12.852
- selected candidate: `gate_low_gr_coverage_hard`
- CV: `gate_low_gr_coverage_hard` 13.932968
- LB: public 12.730 (`ref=53247991`)

## コマンドログ

- 2026-05-31: `uv run python scripts/new_steering.py --experiment exp004_gr_gating --title gr-gating` で steering docs を作成。
- 2026-05-31: `uv run python scripts/new_experiment.py --name exp004_gr_gating --source experiments/exp003_residual_ablation` で exp003 から実験を作成。
- 2026-05-31: notebook 名、`settings.py`、`config.yaml`、README、SESSION_NOTES、steering docs を exp004 用に更新。
- 2026-05-31: `baseline.py` に GR gated model bundle を追加。gated variant は fold ごとに `all` と `no_gr_signal` の 2 モデルを train fold だけで学習し、well 条件に応じて blend する。
- 2026-05-31: train notebook を gated variant runner に更新。`well_metrics.csv` に `gr_gate_weight` と `condition_*` columns を出す。
- 2026-05-31: inference notebook を selected gated variant の final fit / prediction に対応。
- 2026-05-31: `uv run python scripts/validate_experiment.py --experiment exp004_gr_gating` が通過。
- 2026-05-31: `uv run ruff check experiments/exp004_gr_gating/baseline.py` が通過。
- 2026-05-31: `uv run pytest` が通過。9 tests passed。
- 2026-05-31: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp004_gr_gating --notebook train --run-on-push --title "exp004 gr gating train" --strict` が通過。
- 2026-05-31: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp004_gr_gating --notebook inference --run-on-push --title "exp004 gr gating inference" --strict` が通過。
- 2026-05-31: local/generate Kaggle notebooks の code cell AST parse と同梱 `baseline.py` / `settings.py` の `py_compile` が通過。
- 2026-05-31: notebook 実行ではなく baseline 関数 smoke として、4 wells subset で `gate_high_gr_shift_soft` の fit/predict 経路を確認。1,800 sampled rows、4,296 predictions、gate weight 0.0。
- 2026-05-31: `kaggle kernels push -p experiments/exp004_gr_gating/kaggle/train` は sandbox 内 DNS 制限で失敗後、承認済み escalated 実行で成功。Kaggle kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp004-gr-gating-train
- 2026-05-31: `kaggle kernels status kentookumura/exp004-gr-gating-train` を監視し、`KernelWorkerStatus.COMPLETE` を確認。
- 2026-05-31: `kaggle kernels output kentookumura/exp004-gr-gating-train -p /tmp/kaggle-output/exp004_gr_gating/train` は一部 DNS エラー後、承認済み escalated 再試行で成功。
- 2026-05-31: Kaggle output の `metrics.json`、`artifacts/*.csv`、kernel log を `experiments/exp004_gr_gating/` に反映。
- 2026-05-31: full CV では `control_exp003_no_gr` が 13.882944 で全体最良、gating variants では `gate_low_gr_coverage_hard` が 13.932968 で最良。`ablation.selected_variant` を `gate_low_gr_coverage_hard` に更新。
- 2026-05-31: selected variant 更新後、`uv run python scripts/validate_experiment.py --experiment exp004_gr_gating` が通過。
- 2026-05-31: selected variant 更新後、`uv run python scripts/prepare_kaggle_notebooks.py --experiment exp004_gr_gating --notebook inference --run-on-push --title "exp004 gr gating inference" --strict` が通過。
- 2026-05-31: `kaggle kernels push -p experiments/exp004_gr_gating/kaggle/inference` は sandbox 内 DNS 制限で失敗後、承認済み escalated 実行で成功。Kaggle kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp004-gr-gating-inference
- 2026-05-31: `kaggle kernels status kentookumura/exp004-gr-gating-inference` を監視し、`KernelWorkerStatus.COMPLETE` を確認。
- 2026-05-31: `kaggle kernels output kentookumura/exp004-gr-gating-inference -p /tmp/kaggle-output/exp004_gr_gating/inference` で output と kernel log を取得。
- 2026-05-31: `.agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp004_gr_gating/inference/submission.csv --sample data/raw/sample_submission.csv` は PASS。14,151 rows、`id,tvt`、欠損/重複なし。
- 2026-05-31: visible duplicate well sanity を `artifacts/visible_submission_well_comparison.csv` に保存。aggregate visible RMSE は exp002 7.916353、exp003 8.472623、exp004 7.948310。
- 2026-06-01: `kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp004-gr-gating-inference -v 1 -f submission.csv -m "exp004_gr_gating gate_low_gr_coverage_hard CV 13.932968"` は sandbox 内 DNS 制限で失敗後、承認済み escalated 実行で成功。
- 2026-06-01: submission ref `53247991` は `SubmissionStatus.COMPLETE`、public LB 12.730。
- 2026-06-01: `uv run python scripts/record_submission.py --experiment exp004_gr_gating --file /tmp/kaggle-output/exp004_gr_gating/inference/submission.csv --cv 13.932968 --public-lb 12.730 ...` で `SUBMISSIONS.md` に v004 を記録。

## 変更点

- `control_exp002_all`: exp002 と同じ all-GR residual model。
- `control_exp003_no_gr`: exp003 selected variant と同じ no-GR residual model。
- `gate_high_gr_shift_soft`: `prefix_gr_std >= 35` または `gr_delta_abs_mean >= 25` の well だけ no-GR 予測へ 50% blend。
- `gate_high_gr_shift_hard`: 同じ GR shift tags の well で no-GR 予測へ 100% gate。
- `gate_low_gr_coverage_hard`: `prefix_gr_missing_rate >= 0.35` または `eval_gr_missing_rate >= 0.40` の well で no-GR 予測へ 100% gate。

## 次のアクション

1. Public LB は exp002 12.533 がまだ最良なので、次の実験は exp002 を LB anchor として扱う。
2. GR coverage gate を続ける場合は、`000d7d20` のような visible/public 寄り well を no-GR へ倒しすぎない条件へ見直す。
