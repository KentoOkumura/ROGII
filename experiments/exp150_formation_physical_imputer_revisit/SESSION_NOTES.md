# exp150_formation_physical_imputer_revisit セッションノート

## 目的

`formation_physical_imputer_revisit` backlog を実装する。Sunny 系の formation contact physical branch を、hidden test で直接使えない train-only formation columns に依存しない形へ再構成し、直接 TVT 置換ではなく confidence / weak-prior feature 候補として監査する。

## 現在の状態

- 状態: 実装済み、Kaggle full train 未実行
- Route: `ml_model`
- 親実験: `exp138_ancc_surface_predictability_audit`
- GPU/booster: CPU-only、LightGBM 0 config、0 fold boosters、合計 booster 0
- 提出: なし

## コマンドログ

### 2026-06-27

```bash
uv run python scripts/new_steering.py --experiment exp150_formation_physical_imputer_revisit
uv run python scripts/new_experiment.py --name exp150_formation_physical_imputer_revisit --source experiments/exp138_ancc_surface_predictability_audit
```

- `docs/legacy/steering/20260627-exp150-formation-physical-imputer-revisit/` を作成。
- `experiments/exp150_formation_physical_imputer_revisit/` を exp138 から作成。
- `formation_physical_imputer_revisit.py` を追加。
- `config.yaml`、`README.md`、`result.md`、`metrics.json` を exp150 用に更新。
- train / inference notebook を exp150 名にリネームし、監査内容へ差し替え。
- `uv run ruff check experiments/exp150_formation_physical_imputer_revisit/formation_physical_imputer_revisit.py experiments/exp150_formation_physical_imputer_revisit/settings.py` が通過。
- `uv run python -m py_compile experiments/exp150_formation_physical_imputer_revisit/formation_physical_imputer_revisit.py experiments/exp150_formation_physical_imputer_revisit/settings.py` が通過。
- `uv run python scripts/validate_experiment.py --experiment exp150_formation_physical_imputer_revisit` が通過。
- 8 wells smoke を `/tmp/exp150_smoke2` に出力して通過。
  - best smoke candidate: `well_plane_knn` / `contact_best_prefix`
  - smoke RMSE: 26.967070
  - smoke MAE: 18.333600
  - smoke max well RMSE: 47.601074
- `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp150_formation_physical_imputer_revisit --notebook train --kernel-id kentookumura/exp150-formation-physical-imputer-revisit-train --title "exp150 formation physical imputer revisit train" --run-on-push --strict` が通過。
- `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp150_formation_physical_imputer_revisit --notebook inference --kernel-id kentookumura/exp150-formation-physical-imputer-revisit-inference --title "exp150 formation physical imputer revisit inference" --run-on-push --strict` が通過。
- `uv run pytest` は最初、既存 test が `build_metadata(machine_shape)` を渡していないため 2 件失敗。`tests/test_kaggle_notebooks.py` を現行 API に合わせて `machine_shape=None` を追加し、再実行で 13 tests passed。
- `uv run python scripts/update_experiment_summary.py` で `experiment_summary.md` に exp150 を追加。
- `make push-kaggle-train EXP=exp150_formation_physical_imputer_revisit` で Kaggle train version 1 を push。
  - Kernel: `kentookumura/exp150-formation-physical-imputer-revisit-train`
  - URL: https://www.kaggle.com/code/kentookumura/exp150-formation-physical-imputer-revisit-train
  - push 後の pull 確認は成功。
  - 最後の status 確認では `KernelWorkerStatus.RUNNING`。
  - logs / output はまだ空。ユーザー指示により監視停止。
- ユーザーから完了連絡後、status が `KernelWorkerStatus.COMPLETE` であることを確認。
- `kaggle kernels logs kentookumura/exp150-formation-physical-imputer-revisit-train` で logs を取得。
  - runtime は log 上で約 801 sec。
  - best: `well_plane_knn` / `contact_best_prefix`
  - RMSE: 28.23389701015109
  - MAE: 18.2813938515376
  - max well RMSE: 136.52671800693165
  - score rows: 3,746,966
- `kaggle kernels output ... -p experiments/exp150_formation_physical_imputer_revisit/kaggle/output/train` を実行。
  - lightweight artifacts は取得済み。
  - row-level `features/formation_physical_oof_features.csv` は大きく、取得途中で 0 byte placeholder になったためローカル記録には使わない。
- `artifacts/*.csv` を `experiments/exp150_formation_physical_imputer_revisit/artifacts/` に反映。
- `metrics.json` と `result.md` を full result で更新。

## 実装メモ

- surface method:
  - `global_median`
  - `row_knn_xy`
  - `well_plane_knn`
- formation contacts:
  - `ANCC`
  - `ASTNU`
  - `ASTNL`
  - `EGFDU`
  - `EGFDL`
  - `BUDA`
- physical candidate:
  - `contact_median`
  - `contact_prefix_weighted`
  - `contact_best_prefix`
- prediction input:
  - `X`, `Y`, `Z`
  - known-prefix `TVT_input` for offset calibration only
- scoring-only:
  - validation-fold true `TVT`
  - validation-fold true formation columns

## リーク対策

- Same well は fold 間で分割しない。
- valid fold の formation columns は surface fitting に使わない。
- valid fold の eval true `TVT` は scoring にしか使わない。
- known-prefix `TVT_input` 以外の future/eval target は candidate generation に使わない。
- hidden test にない horizontal formation columns を inference input に要求しない。

## 出力予定

- `features/formation_physical_oof_features.csv`
- `artifacts/formation_prefix_calibration.csv`
- `artifacts/candidate_metrics.csv`
- `artifacts/distance_bucket_metrics.csv`
- `artifacts/confidence_bucket_metrics.csv`
- `artifacts/surface_proxy_metrics.csv`

## 次のアクション

1. direct TVT candidate / inference port / submit は行わない。
2. 後続に使う場合は、`formation_pred_spread`、`neighbor_dist`、`prefix_mae_best` を exp092 系 add-only confidence feature として低優先で小さく評価する。
