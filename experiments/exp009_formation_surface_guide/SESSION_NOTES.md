# exp009_formation_surface_guide セッションノート

## 目的

train-only formation columns を直接使わず、fold-safe な KNN surface guide に変換して residual model の補助特徴として検証する。

## 現在の状態

- 状態: Kaggle full CV 完了、提出なし
- 親実験: `exp008_gr_ncc_matcher`
- selected candidate: `formation_knn_no_gr`
- CV: 14.558630
- LB: 未提出

## コマンドログ

- 2026-06-02: `uv run python scripts/new_steering.py --experiment exp009_formation_surface_guide` で steering docs を作成。
- 2026-06-02: `uv run python scripts/new_experiment.py --name exp009_formation_surface_guide --source experiments/exp008_gr_ncc_matcher` で exp008 から実験を作成。
- 2026-06-02: train / inference notebook を exp009 名にリネーム。
- 2026-06-02: `config.yaml` を exp009 用に更新し、`control_exp002_all`、`control_exp003_no_gr`、`formation_knn_no_gr`、`formation_knn_all` を定義。
- 2026-06-02: `baseline.py` に fold-safe KNN formation guide、formation feature sets、model への guide 保持を実装。
- 2026-06-02: train / inference notebook の valid/test feature frame 作成時に fit 済み `formation_guide_` を渡すよう更新。
- 2026-06-02: 6 train wells で guide fit、1 valid well で feature frame 作成の sanity check を実行。`active_features=49`、`eval_rows=6234`、`formation_available=[1.0]` を確認。
- 2026-06-02: `python3 -m py_compile experiments/exp009_formation_surface_guide/baseline.py experiments/exp009_formation_surface_guide/settings.py` が通過。
- 2026-06-02: `uv run ruff check experiments/exp009_formation_surface_guide/baseline.py experiments/exp009_formation_surface_guide/settings.py` が通過。
- 2026-06-02: `uv run python scripts/validate_experiment.py --experiment exp009_formation_surface_guide` が通過。
- 2026-06-02: `uv run pytest` が通過。9 tests passed。
- 2026-06-02: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp009_formation_surface_guide --notebook train --run-on-push --title "exp009 formation surface guide train" --strict` が通過。
- 2026-06-02: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp009_formation_surface_guide --notebook inference --run-on-push --title "exp009 formation surface guide inference" --strict` が通過。
- 2026-06-02: `uv run python scripts/update_experiment_summary.py` で `experiment_summary.md` に exp009 scaffold row を追加。
- 2026-06-02: `kaggle kernels push -p experiments/exp009_formation_surface_guide/kaggle/train` で version 1 を push。URL: https://www.kaggle.com/code/kentookumura/exp009-formation-surface-guide-train
- 2026-06-02: `kaggle kernels status kentookumura/exp009-formation-surface-guide-train` を監視し、`KernelWorkerStatus.COMPLETE` を確認。
- 2026-06-02: `kaggle kernels output kentookumura/exp009-formation-surface-guide-train -p /tmp/kaggle-output/exp009_formation_surface_guide/train` で output と kernel log を取得。
- 2026-06-02: Kaggle output の `metrics.json`、`artifacts/ablation_metrics.csv`、`fold_metrics.csv`、`fold_model_training.csv`、`well_metrics.csv`、train log を `experiments/exp009_formation_surface_guide/` に反映。

## 変更点

- `formation_knn_no_gr`: exp003 の `no_gr_signal` feature set に fold-safe KNN formation guide を追加。
- `formation_knn_all`: exp002 の raw-GR feature set に同じ guide を追加。
- KNN guide は train-fold wells の formation columns から `X,Y -> formation surface Z` を学習し、valid/test では推定 surface と `Z` の距離特徴だけを使う。

## リーク対策

- 同一 well は fold 間で分割しない。
- valid fold の formation columns は guide fitting に使わない。
- inference 時は hidden test に存在しない formation columns を入力に要求しない。
- GR NCC は exp008 で悪化したため再投入しない。

## 結果

| Variant | Feature Set | CV | exp002 差分 |
| --- | --- | ---: | ---: |
| `control_exp003_no_gr` | `no_gr_signal` | 13.882944 | -0.241625 |
| `control_exp002_all` | `all` | 14.124569 | 0.000000 |
| `formation_knn_no_gr` | `no_gr_signal_plus_formation_guide` | 14.558630 | +0.434061 |
| `formation_knn_all` | `all_plus_formation_guide` | 14.739226 | +0.614657 |

selected `formation_knn_no_gr` は CV 14.558630 で、best control の `control_exp003_no_gr` より 0.675686 悪く、exp002 control より 0.434061 悪い。formation guide は現行設計では採用しない。

## 次のアクション

1. exp009 は提出しない。
2. `experiment_summary.md` と `backlog/KAGGLE_DIRECTION.md` を更新する。
3. 次は `exp010_trajectory_drift_ablation` に進む。
