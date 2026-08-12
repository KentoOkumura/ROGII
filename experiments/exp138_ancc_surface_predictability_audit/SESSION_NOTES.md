# exp138_ancc_surface_predictability_audit セッションノート

## 目的

train-only の `ANCC` formation surface を LightGBM なしで fold-safe に推定できるかを監査する。
後続の `ANCC_hat` residual target ablation に進む前に、surface predictability と
target 分布が十分に安定しているかを確認する。

## 現在の状態

- Route: ml_model
- 状態: 完了
- CV: ANCC surface audit only
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 2026-06-26

```bash
uv run python scripts/new_steering.py --experiment exp138_ancc_surface_predictability_audit
uv run python scripts/new_experiment.py --name exp138_ancc_surface_predictability_audit
```

- `docs/legacy/steering/20260626-exp138-ancc-surface-predictability-audit/` を作成。
- `experiments/exp138_ancc_surface_predictability_audit/` を作成。
- `ancc_surface_predictability_audit.py` を追加。
- train notebook を fold-safe ANCC surface audit 用に更新。
- inference notebook は提出なしの no-op に更新。

```bash
uv run ruff check experiments/exp138_ancc_surface_predictability_audit/ancc_surface_predictability_audit.py experiments/exp138_ancc_surface_predictability_audit/settings.py
uv run python -m py_compile experiments/exp138_ancc_surface_predictability_audit/ancc_surface_predictability_audit.py experiments/exp138_ancc_surface_predictability_audit/settings.py
uv run python scripts/validate_experiment.py --experiment exp138_ancc_surface_predictability_audit
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp138_ancc_surface_predictability_audit --notebook train --run-on-push --title "exp138 ancc surface predictability audit train" --strict
make push-kaggle-train EXP=exp138_ancc_surface_predictability_audit
```

- Kaggle train version 1 を push。URL: https://www.kaggle.com/code/kentookumura/exp138-ancc-surface-predictability-audit-train
- v1 はしばらく `RUNNING` だが logs が空で、`well_plane_knn` の row-wise Python loop が長時間化している可能性が高いと判断。
- `well_plane_knn` を chunked batch solve に変更し、`surface.well_plane_knn.chunk_size: 50000` を追加。
- ruff / py_compile / validate-exp / prepare-kaggle-notebooks を再実行して通過。
- Kaggle train version 2 を push。ユーザー指示により監視は停止し、Kaggle 側の実行は継続。
- version 2 は `np.linalg.solve(lhs, rhs)` の batched RHS shape mismatch で `ERROR`。`rhs[..., None]` を渡して squeeze するよう修正。
- ruff / py_compile / prepare-kaggle-notebooks を再実行して通過。
- Kaggle train version 3 を push。以後の継続監視はしない。

### 2026-06-27

```bash
kaggle kernels status kentookumura/exp138-ancc-surface-predictability-audit-train
kaggle kernels logs kentookumura/exp138-ancc-surface-predictability-audit-train
kaggle kernels output kentookumura/exp138-ancc-surface-predictability-audit-train -p experiments/exp138_ancc_surface_predictability_audit/kaggle/output/train
```

- v3 status は `COMPLETE`。
- runtime は Kaggle log 上で約 354 sec。
- `method_metrics.csv`、`bucket_metrics.csv`、`target_distribution_summary.csv` を取得して `artifacts/` に反映。
- `ancc_surface_oof_predictions.csv` は `kaggle/output/train/features/` に取得済み。サイズが約 1.44GB のため `features/` には重複コピーしない。
- `metrics.json` は取得済み CSV と log の SHA からローカル再構成。

### 取得後の生成物

```bash
experiments/exp138_ancc_surface_predictability_audit/artifacts/method_metrics.csv
experiments/exp138_ancc_surface_predictability_audit/artifacts/bucket_metrics.csv
experiments/exp138_ancc_surface_predictability_audit/artifacts/target_distribution_summary.csv
experiments/exp138_ancc_surface_predictability_audit/kaggle/output/train/features/ancc_surface_oof_predictions.csv
```

## 変更点

- LightGBM は使用しない。
- `global_median`、`row_knn_xy`、`well_plane_knn` を比較する。
- fold ごとに train wells だけで `ANCC_hat` surface を fit し、valid wells の score rows と anchor row に予測する。
- `ANCC_hat` absolute error と anchor-relative delta error を評価する。
- target ablation は行わず、`TVT - last_known_TVT`、`TVT - ANCC_hat`、anchor-relative residual の分布 summary だけを保存する。

## 再現性メモ

- seed policy: fixed global seed 42 + fold offset
- stochastic components: `row_knn_xy` の deterministic row subsampling のみ
- CPU/GPU runtime: CPU only
- Kaggle kernel id / version: `kentookumura/exp138-ancc-surface-predictability-audit-train`, v3 complete
- input / feature schema SHA: 未実行
- feature content SHA: `metrics.json` に記録済み
- model manifest / model SHA: persistent model なし
- prediction SHA: OOF prediction CSV SHA `fbbccb2f8d924ef01529dd48deb9200b1e443f93338417052ef1df735e48e60f`
- submission SHA: submission なし
- rerun check: 未実行

## 次のアクション

1. `ancc_surface_predictability_audit` は完了として backlog から外す。
2. `ancc_hat_residual_target_ablation_on_exp073` は現設計では実施しない。
