# exp011_tracker_divergence_features セッションノート

## 目的

PF / beam / deterministic DTW 系の candidate path を standalone 予測ではなく、HGB residual model の補助特徴として検証する。

## 現在の状態

- 状態: Kaggle full CV 完了、提出なし
- 親実験: `exp010_trajectory_drift_ablation`
- selected candidate: `tracker_divergence_no_gr`
- CV: `tracker_divergence_no_gr` 14.903823
- LB: 未提出

## コマンドログ

- 2026-06-03: `uv run python scripts/new_steering.py --experiment exp011_tracker_divergence_features` で steering docs を作成。
- 2026-06-03: `uv run python scripts/new_experiment.py --name exp011_tracker_divergence_features --source experiments/exp010_trajectory_drift_ablation` で exp010 から実験を作成。
- 2026-06-03: train / inference notebook を exp011 名にリネーム。
- 2026-06-03: `baseline.py` に deterministic tracker feature group を追加。
- 2026-06-03: `config.yaml` を exp011 用に更新し、tracker divergence variants を定義。
- 2026-06-03: `uv run python scripts/validate_project.py` が通過。
- 2026-06-03: `python3 -m py_compile experiments/exp011_tracker_divergence_features/baseline.py experiments/exp011_tracker_divergence_features/settings.py` が通過。
- 2026-06-03: `uv run ruff check experiments/exp011_tracker_divergence_features/baseline.py experiments/exp011_tracker_divergence_features/settings.py` が通過。
- 2026-06-03: `uv run python scripts/validate_experiment.py --experiment exp011_tracker_divergence_features` が通過。
- 2026-06-03: `uv run pytest` が通過。9 tests passed。
- 2026-06-03: 1 train well `c50b42f6` で `tracker_divergence_no_gr` の feature frame sanity check を実行。`rows=6014`、`active_features=87`、`tracker_available=1.0`、`tracker_best_score=0.408529` を確認。
- 2026-06-03: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp011_tracker_divergence_features --notebook train --run-on-push --title "exp011 tracker divergence features train" --strict` が通過。
- 2026-06-03: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp011_tracker_divergence_features --notebook inference --run-on-push --title "exp011 tracker divergence features inference" --strict` が通過。
- 2026-06-03: `uv run python scripts/update_experiment_summary.py` で `experiment_summary.md` に exp011 scaffold row を追加。
- 2026-06-03: `kaggle kernels push -p experiments/exp011_tracker_divergence_features/kaggle/train` で version 1 を push。URL: https://www.kaggle.com/code/kentookumura/exp011-tracker-divergence-features-train
- 2026-06-03: `kaggle kernels status kentookumura/exp011-tracker-divergence-features-train` を監視し、`KernelWorkerStatus.COMPLETE` を確認。
- 2026-06-03: `kaggle kernels output kentookumura/exp011-tracker-divergence-features-train -p /tmp/kaggle-output/exp011_tracker_divergence_features/train` で output と kernel log を取得。
- 2026-06-03: Kaggle output の `metrics.json`、`artifacts/ablation_metrics.csv`、`fold_metrics.csv`、`fold_model_training.csv`、`tracker_group_summary.csv`、`well_metrics.csv`、train log を `experiments/exp011_tracker_divergence_features/` に反映。

## 変更点

- `tracker_divergence_no_gr`: exp003 の `no_gr_signal` に deterministic tracker features を追加。
- `tracker_divergence_all`: exp002 の all-GR feature set に deterministic tracker features を追加。
- `tracker_direction_no_gr`: tracker features に trajectory direction の限定 group を追加。
- train notebook は `tracker_group_summary.csv` を出力し、exp010 audit で悪化した hard/no-GR、steep trajectory、high GR missing、long eval group を確認する。

## リーク対策

- 同一 well は fold 間で分割しない。
- `TVT_input` の既知 prefix 以外から target-derived feature を作らない。
- train-only formation columns は使わない。
- tracker path は hidden test でも利用できる `MD`、`Z`、`GR`、`TVT_input` prefix、paired typewell `GR` だけから作る。
- stochastic PF は使わず、scale-specific deterministic path と bounded shift search に固定する。

## 結果

| Variant | Feature Set | CV | exp002 差分 |
| --- | --- | ---: | ---: |
| `control_exp003_no_gr` | `no_gr_signal` | 13.882944 | -0.241625 |
| `control_exp002_all` | `all` | 14.124569 | 0.000000 |
| `tracker_divergence_no_gr` | `no_gr_signal_plus_tracker` | 14.903823 | +0.779254 |
| `tracker_direction_no_gr` | `no_gr_signal_plus_tracker_direction` | 14.918102 | +0.793533 |
| `tracker_divergence_all` | `all_plus_tracker` | 14.955276 | +0.830707 |

selected `tracker_divergence_no_gr` は CV 14.903823。best は `control_exp003_no_gr` 13.882944 で、tracker variants はすべて exp002 / exp003 controls より悪化した。提出しない。

`tracker_group_summary.csv` では `tracker_divergence_no_gr` が exp010 audit の注意 group で悪化した。hard-no-GR candidate は 10.675618 -> 11.334843、steep trajectory は 12.078737 -> 13.375446、high GR missing は 10.746502 -> 11.387577、long eval は 12.253799 -> 13.171136。

Kaggle train log timestamp は最終出力が約 1,914 秒。tracker variants は各 8-9 分程度で、見えない test well 推論 に入れる前に feature pruning / routing が必要。

## 次のアクション

1. exp011 は提出しない。
2. 現行 deterministic tracker add-only features は凍結する。
3. 次候補は `exp012_model_diversity_or_postprocess`。tracker を再検討する場合は、まず failure audit / routing 診断に限定する。
