# exp001_baseline セッションノート

## 目的

現状調査で強い null model とされている `last_known_TVT` anchor を、leak-safe な初期ベースラインとして実装する。

## 現在の状態

- 状態: 完了
- 学習ロジック: prefix-only baseline 実装済み
- 推論ロジック: `sample_submission.csv` order に合わせて `id,tvt` を生成
- CV: full 773 wells、`well_id` GroupKFold、`TVT_input` NaN 行で RMSE。
- LB: public 15.883 (`ref=53206637`)

## コマンドログ

- 2026-05-27: Kaggle API から ROGII 公式ページ、ファイル一覧、leaderboard metadata を取得。
- 2026-05-27: `data/raw/sample_submission.csv` と `000d7d20` の schema 確認用 CSV を取得。
- 2026-05-27: `make validate-config` と `make validate-exp EXP=exp001_baseline` が通過。
- 2026-05-28: train notebook debug 実行。30 wells debug CV RMSE 12.145321。
- 2026-05-28: train notebook full 実行。full CV RMSE 15.909853。
- 2026-05-28: inference notebook 実行。当時はローカル提出 CSV を生成し、提出形式検証が通過。現在はローカル提出ファイルを削除済み。
- 2026-05-31: Kaggle inference notebook `kentookumura/exp001-baseline-inference-cpu-smoke` version 1 の `submission.csv` を提出。`ref=53206637`、public LB 15.883。
- 2026-05-31: notebook 実行前提を Kaggle に統一。`settings.py` は Kaggle runtime で `/kaggle/input` を優先し、notebook は明示 override なしのローカル実行を停止する。
- 2026-05-31: `uv run python scripts/validate_experiment.py --experiment exp001_baseline`、`uv run python scripts/prepare_kaggle_notebooks.py --experiment exp001_baseline --strict`、`uv run python scripts/validate_project.py`、`uv run pytest` が通過。

再現用コマンド。

```bash
task validate-config
task validate-exp EXP=exp001_baseline
task prepare-kaggle-notebooks EXP=exp001_baseline EXTRA_ARGS="--strict"
task push-kaggle-infer EXP=exp001_baseline
task kaggle-status KERNEL=kentookumura/exp001-baseline-inference-cpu-smoke
```

## 変更点

- `baseline.py` を追加し、well の known prefix だけから tail を予測する処理を共通化。
- train notebook で `last_anchor` と参考 `recent_linear` の CV を記録。
- inference notebook で visible / hidden test の sample IDs に合わせた提出生成を実装。
- `artifacts/well_metrics.csv` に well 単位の診断を出力。
- 2026-05-31 以降、notebook の通常実行先は Kaggle。ローカル notebook 実行は `--allow-local` 付き smoke debug のみ。

## 結果

| Strategy | OOF RMSE | Mean Fold RMSE | Rows |
| --- | ---: | ---: | ---: |
| `last_anchor` | 15.909853 | 15.894391 | 3,783,989 |
| `recent_linear` | 41.022355 | 40.903046 | 3,783,989 |

`recent_linear` は大きく悪化したため、提出生成は `last_anchor` のまま。

Kaggle public LB は 15.883。CV 15.909853 と近く、exp001 は比較用 baseline として利用可能。

## 次のアクション

1. `exp002_drift_minimal` を作り、`TVT - last_anchor_tvt` target の tabular baseline を実装する。
2. prefix slope、GR rolling、trajectory (`dZ/dMD`, azimuth) を入れる。
3. fold-safe な formation / NCC features は exp003 以降で段階追加する。
