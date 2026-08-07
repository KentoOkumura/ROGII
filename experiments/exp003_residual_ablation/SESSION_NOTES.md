# exp003_residual_ablation セッションノート

## 目的

`exp002_drift_minimal` の residual model をベースに、sampling cap、residual shrink、feature set の one-at-a-time ablation を同じ well holdout CV で比較する。

## 現在の状態

- 状態: train CV 完了
- 親実験: `exp002_drift_minimal`
- 優先アイデア: exp002 residual model の sampling / shrink / feature ablation
- CV: `feature_no_gr_signal` 13.882944
- LB: public 12.852 (`ref=53213975`)

## コマンドログ

- 2026-05-31: `task new-steering EXP=exp003_residual_ablation` は `task` 未インストールで失敗。
- 2026-05-31: `uv run python scripts/new_steering.py --experiment exp003_residual_ablation` で steering docs を作成。
- 2026-05-31: `uv run python scripts/new_experiment.py --name exp003_residual_ablation --source experiments/exp002_drift_minimal` で exp002 から実験を作成。
- 2026-05-31: exp002 由来の notebook 名と実験名を exp003 に置換。
- 2026-05-31: `config.yaml` に ablation variants を追加。
- 2026-05-31: `baseline.py` に feature set 切替を追加。
- 2026-05-31: train notebook を variant 別 CV runner に更新。
- 2026-05-31: inference notebook を `ablation.selected_variant` 適用に対応。
- 2026-05-31: `uv run python scripts/validate_experiment.py --experiment exp003_residual_ablation` が通過。
- 2026-05-31: `uv run ruff check experiments/exp003_residual_ablation/baseline.py` が通過。
- 2026-05-31: `uv run pytest` が通過。9 tests passed。
- 2026-05-31: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp003_residual_ablation --notebook train --run-on-push --title "exp003 residual ablation train" --strict` が通過。
- 2026-05-31: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp003_residual_ablation --notebook inference --run-on-push --title "exp003 residual ablation inference" --strict` が通過。
- 2026-05-31: train/inference notebook の code cell AST parse と `py_compile` が通過。
- 2026-05-31: `uv run python scripts/update_experiment_summary.py` で exp003 を実験サマリーに追加。
- 2026-05-31: `KAGGLE_DIRECTION.md` の先頭バックログを exp003 full CV 取得待ちに更新。
- 2026-05-31: `kaggle kernels push -p experiments/exp003_residual_ablation/kaggle/train` は sandbox 内 DNS 制限で失敗後、承認済み escalated 実行で成功。Kaggle kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp003-residual-ablation-train
- 2026-05-31: `kaggle kernels status kentookumura/exp003-residual-ablation-train` は `KernelWorkerStatus.RUNNING`。
- 2026-05-31: 複数回 status を確認し、train kernel は継続して `KernelWorkerStatus.RUNNING`。Kaggle output は未取得。
- 2026-05-31: `kaggle kernels status kentookumura/exp003-residual-ablation-train` は `KernelWorkerStatus.COMPLETE`。
- 2026-05-31: `kaggle kernels output kentookumura/exp003-residual-ablation-train -p /tmp/kaggle-output/exp003_residual_ablation/train` で output を取得。
- 2026-05-31: Kaggle output の `metrics.json` と `artifacts/*.csv` を `experiments/exp003_residual_ablation/` に反映。
- 2026-05-31: `feature_no_gr_signal` が best CV 13.882944 だったため、`ablation.selected_variant` を `feature_no_gr_signal` に更新。
- 2026-05-31: `uv run python scripts/validate_experiment.py --experiment exp003_residual_ablation` が通過。
- 2026-05-31: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp003_residual_ablation --notebook inference --run-on-push --title "exp003 residual ablation inference" --strict` が通過。
- 2026-05-31: `kaggle kernels push -p experiments/exp003_residual_ablation/kaggle/inference` は sandbox 内 DNS 制限で失敗後、承認済み escalated 実行で成功。Kaggle kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp003-residual-ablation-inference
- 2026-05-31: `kaggle kernels status kentookumura/exp003-residual-ablation-inference` は `KernelWorkerStatus.RUNNING`。
- 2026-05-31: `kaggle kernels status kentookumura/exp003-residual-ablation-inference` は `KernelWorkerStatus.COMPLETE`。
- 2026-05-31: `kaggle kernels output kentookumura/exp003-residual-ablation-inference -p /tmp/kaggle-output/exp003_residual_ablation/inference` で `submission.csv` を取得。
- 2026-05-31: `.agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp003_residual_ablation/inference/submission.csv --sample data/raw/sample_submission.csv` は PASS。14,151 rows、`id,tvt`、欠損/重複なし。
- 2026-05-31: `kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp003-residual-ablation-inference -v 1 -f submission.csv -m "exp003_residual_ablation feature_no_gr_signal CV 13.882944"` を実行。
- 2026-05-31: submission ref `53213975` は `SubmissionStatus.PENDING`。
- 2026-05-31: `kaggle competitions submissions rogii-wellbore-geology-prediction` で ref `53213975` が `SubmissionStatus.COMPLETE`、public LB 12.852 と確認。
- 2026-05-31: `uv run python scripts/record_experiment.py --experiment exp003_residual_ablation --status completed --cv 13.882944 --public-lb 12.852 --metric rmse ... --no-summary` で metrics を更新。
- 2026-05-31: `uv run python scripts/record_submission.py --experiment exp003_residual_ablation --file /tmp/kaggle-output/exp003_residual_ablation/inference/submission.csv --cv 13.882944 --public-lb 12.852 ...` で `submissions/SUBMISSIONS.md` に v003 を記録。

## 変更点

- `control_exp002`: exp002 設定の再実行。
- `sample_per_well_400`: `max_train_rows_per_well` を 800 から 400 に下げる。
- `sample_total_200k`: CV total cap を 300k から 200k に下げ、inference final cap を 450k から 300k に下げる。
- `shrink_100`: `residual_shrink` を 0.85 から 1.00 に変更。
- `feature_no_gr_signal`: GR raw / derived signal features を外す。

## 結果

Kaggle train full CV:

| Variant | CV | Mean Fold RMSE | exp002 差分 |
| --- | ---: | ---: | ---: |
| `feature_no_gr_signal` | 13.882944 | 13.859376 | -0.241625 |
| `sample_per_well_400` | 14.122145 | 14.099246 | -0.002424 |
| `control_exp002` | 14.124569 | 14.101909 | 0.000000 |
| `shrink_100` | 14.127689 | 14.106161 | +0.003120 |
| `sample_total_200k` | 14.183193 | 14.159896 | +0.058624 |

Selected variant: `feature_no_gr_signal`

Kaggle inference / submission:

| Item | Value |
| --- | --- |
| Kernel | `kentookumura/exp003-residual-ablation-inference` |
| Version | 1 |
| Submission ref | 53213975 |
| Public LB | 12.852 |
| Private LB | - |

CV は exp002 から改善したが、public LB は exp002 の 12.533 から 12.852 へ悪化した。

## CV/LB 逆転調査

- 2026-05-31: `experiments/exp003_residual_ablation/artifacts/exp002_exp003_well_delta.csv` を作成。exp002 vs exp003 selected variant の well-level OOF 差分を保存。
- 2026-05-31: `experiments/exp003_residual_ablation/artifacts/visible_submission_well_comparison.csv` を作成。visible duplicate 3 wells の exp002/exp003 submission 差分を local train truth で確認。
- OOF では 773 wells のうち 408 wells が改善、365 wells が悪化。weighted SSE delta は -25.6M で net 改善。
- 改善は一部 hard wells に集中している。上位 20 改善 well の SSE 改善だけで net 改善量を上回り、悪化 well による相殺も大きい。
- `exp002` が `last_anchor` より悪い 244 wells では exp003 が大きく改善。`exp002` が `last_anchor` より良い 529 wells では median RMSE delta が +0.076692 で典型的には少し悪化。
- visible duplicate 3 wells の local truth では、exp003 は `00e12e8b` で改善したが、`00bbac68` で +1.239759、`000d7d20` で +0.119149 悪化。OOF ではこの 3 wells すべてで exp003 が改善していたため、fold holdout と final full-train inference の挙動差もある。
- 結論: 実装バグより、GR feature removal が hard wells の過補正を抑える一方で、GR feature が効く public/visible 寄り wells を悪化させた可能性が高い。

## 次のアクション

1. exp003 で悪化した OOF wells の条件をタグ付けする。
2. GR feature を全削除ではなく、hard well 判定に応じた gating / shrink にできるか検証する。
3. GR を戻す場合は raw rolling ではなく local matcher / typewell alignment として再設計する。
