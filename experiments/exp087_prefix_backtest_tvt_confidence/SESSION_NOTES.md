# exp087_prefix_backtest_tvt_confidence セッションノート

## 目的

`KAGGLE_DIRECTION.md` の最優先 backlog `prefix_backtest_tvt_confidence` を実装する。PF/Beam / likelihood-PF の confidence / disagreement signal が、TVT error の大きい row / distance bucket を fold-safe に識別できるかを診断する。

## 現在の状態

- Route: `pf_beam`
- 状態: `implemented`
- CV: Kaggle train 未実行
- LB: なし
- Submit: なし

## コマンドログ

```bash
uv run python scripts/new_steering.py --experiment exp086_prefix_backtest_tvt_confidence
mv .steering/20260620-exp086-prefix-backtest-tvt-confidence .steering/20260620-exp087-prefix-backtest-tvt-confidence
uv run python scripts/new_experiment.py --name exp087_prefix_backtest_tvt_confidence --source experiments/exp083_pf_beam_true_tvt_2d_well_eda
```

注: `exp086_oof_feature_importance_error_readout` が既に存在したため、今回の実験番号は `exp087` に変更した。

実装内容:

- `config.yaml` を `exp087_prefix_backtest_tvt_confidence`、route `pf_beam`、diagnostic-only に更新。
- `prefix_backtest_tvt_confidence.py` を追加。
- exp072 cache の候補列を TVT 空間に戻す処理を追加。
- well 内 row fraction で calibration / holdout phase を付与。
- calibration phase のみで ridge confidence model を fit し、well-hash fold 外で `expected_tvt_error` を出す処理を追加。
- candidate metrics、confidence bin metrics、bucket metrics、phase metrics、fold metrics、signal correlations、row-level predictions を保存する。
- train notebook を source check、audit 実行、生成物確認の構成に更新。
- inference notebook は no-op policy check に更新。

## 再現性メモ

- seed policy: 乱数なし。well-hash fold は stable `blake2b(well_id) % n_folds`。
- stochastic components: なし。PF/Beam は新規生成しない。
- CPU/GPU runtime: CPU only。
- source artifact: exp072 `exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz`
- 実行時に raw SHA と decompressed content SHA を summary JSON / metrics.json に記録する。

## 次のアクション

## 実装後の検証

```bash
uv run ruff check experiments/exp087_prefix_backtest_tvt_confidence/prefix_backtest_tvt_confidence.py experiments/exp087_prefix_backtest_tvt_confidence/settings.py
uv run python -m py_compile experiments/exp087_prefix_backtest_tvt_confidence/prefix_backtest_tvt_confidence.py experiments/exp087_prefix_backtest_tvt_confidence/settings.py
uv run python scripts/validate_experiment.py --experiment exp087_prefix_backtest_tvt_confidence
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp087_prefix_backtest_tvt_confidence --notebook train --kernel-id kentookumura/exp087-prefix-backtest-tvt-confidence-train --title "exp087 prefix backtest tvt confidence train" --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp087_prefix_backtest_tvt_confidence --notebook inference --kernel-id kentookumura/exp087-prefix-backtest-tvt-confidence-infer --title "exp087 prefix backtest tvt confidence infer" --run-on-push --strict
```

- `ruff check`: pass
- `py_compile`: pass
- `validate_experiment`: pass

## v2 push

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp087_prefix_backtest_tvt_confidence --notebook train --kernel-id kentookumura/exp087-prefix-backtest-tvt-confidence-train --title "exp087 prefix backtest tvt confidence train" --run-on-push --strict
kaggle kernels push -p experiments/exp087_prefix_backtest_tvt_confidence/kaggle/train
kaggle kernels pull kentookumura/exp087-prefix-backtest-tvt-confidence-train -p /tmp/kaggle-pull/exp087-prefix-backtest-tvt-confidence-train-v2 -m
```

- v2 package: prepared successfully.
- v2 push: success.
- kernel id: `kentookumura/exp087-prefix-backtest-tvt-confidence-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp087-prefix-backtest-tvt-confidence-train`
- existence check: pull succeeded.
- monitoring: not started by request. Check logs/output later with the same kernel id.

## v2 completed

ユーザー確認で v2 完了後、logs / output を取得した。

```bash
kaggle kernels logs kentookumura/exp087-prefix-backtest-tvt-confidence-train
kaggle kernels output kentookumura/exp087-prefix-backtest-tvt-confidence-train -p /tmp/kaggle-output/exp087_prefix_backtest_tvt_confidence/train_v2
```

- v2 status: completed.
- output: `/tmp/kaggle-output/exp087_prefix_backtest_tvt_confidence/train_v2`
- rows / wells: 3,783,989 / 773
- source gzip SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- source decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- primary candidate: `pf_ancc`
- primary PF RMSE: 14.493050690
- primary PF MAE: 8.921559334
- expected error vs absolute error Pearson: 0.519681049
- high-error threshold: 15.234375
- unstable expected-error threshold: 13.889335522
- unstable flag rate: 0.200000053
- unstable flag high-error rate: 0.538377480
- stable flag high-error rate: 0.115413927
- top-vs-bottom confidence bin observed MAE lift: 7.745937780

confidence bin metrics:

- `bin_0_low`: rows 756,798 / observed MAE 2.460268 / high-error rate 0.015496
- `bin_1_mid`: rows 756,798 / observed MAE 5.386803 / high-error rate 0.076981
- `bin_2_mid`: rows 756,797 / observed MAE 7.663323 / high-error rate 0.136118
- `bin_3_mid`: rows 756,798 / observed MAE 10.040319 / high-error rate 0.233061
- `bin_4_high`: rows 756,798 / observed MAE 19.057079 / high-error rate 0.538377

top signal correlations:

- `pf_likpf_abs`: Pearson 0.589850 / Spearman 0.523773
- `md_since`: Pearson 0.288224 / Spearman 0.322989
- `pf_beam_abs`: Pearson 0.359971 / Spearman 0.264001
- `beam_likpf_abs`: Pearson 0.272138 / Spearman 0.245015
- `likpf_delta_abs`: Pearson 0.247383 / Spearman 0.211492

output synced:

- `artifacts/prefix_backtest_tvt_confidence_summary.json`
- `artifacts/prefix_backtest_tvt_confidence_candidate_metrics.csv`
- `artifacts/prefix_backtest_tvt_confidence_confidence_bin_metrics.csv`
- `artifacts/prefix_backtest_tvt_confidence_bucket_metrics.csv`
- `artifacts/prefix_backtest_tvt_confidence_phase_metrics.csv`
- `artifacts/prefix_backtest_tvt_confidence_fold_metrics.csv`
- `artifacts/prefix_backtest_tvt_confidence_signal_correlations.csv`
- `artifacts/exp087-prefix-backtest-tvt-confidence-train.log`
- `metrics.json`

`prefix_backtest_tvt_confidence_predictions.csv.gz` は 124MB のため repo 配下には同期しない。必要なら `/tmp/kaggle-output/exp087_prefix_backtest_tvt_confidence/train_v2/artifacts/` を参照する。

判断:

- high-error row / bucket の fold-safe confidence 識別は支持された。
- PF/Beam 予測値を直接置換せず、`pf_beam_disagreement_sample_weight` の feature / sample-weight 候補として `pf_likpf_abs`、`pf_beam_abs`、`beam_likpf_abs`、`likpf_delta_abs`、`md_since` を吸収する。
- `prepare_kaggle_notebooks` train: pass
- `prepare_kaggle_notebooks` inference: pass
- train metadata: `enable_gpu=false`, `enable_internet=false`, `kernel_sources=["kentookumura/exp072-exp063-full-replay-feature-cache-train"]`
- inference metadata: no-op notebook。`submission.csv` は作らない。

合成 CSV smoke:

- `/tmp/exp087_synthetic_source.csv.gz` を一時作成し、source materialize、well-hash fold confidence、artifact 出力を確認した。
- smoke 結果: 960 rows / 12 wells scored、`expected_error_abs_error_pearson=0.730688`、top-vs-bottom confidence bin observed MAE lift `5.264550`。
- smoke artifact は本番結果と混ざらないよう削除し、`metrics.json` は `implemented_not_run` に戻した。

## 次のアクション

Kaggle train 実行:

```bash
kaggle kernels push -p experiments/exp087_prefix_backtest_tvt_confidence/kaggle/train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp087-prefix-backtest-tvt-confidence-train
kaggle kernels output kentookumura/exp087-prefix-backtest-tvt-confidence-train -p /tmp/kaggle-output/exp087_prefix_backtest_tvt_confidence/train_v1
```

- v1 push: success。
- kernel id: `kentookumura/exp087-prefix-backtest-tvt-confidence-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp087-prefix-backtest-tvt-confidence-train`
- existence check: `kaggle kernels pull kentookumura/exp087-prefix-backtest-tvt-confidence-train -p /tmp/kaggle-pull/exp087-prefix-backtest-tvt-confidence-train -m` succeeded.
- monitoring: `logs -f` は約数分間ログ本文なし。ユーザー指示でローカル監視を停止した。Kaggle notebook 自体は Kaggle 側で実行中または完了待ちの可能性がある。

## v1 failure

```bash
kaggle kernels logs kentookumura/exp087-prefix-backtest-tvt-confidence-train
kaggle kernels output kentookumura/exp087-prefix-backtest-tvt-confidence-train -p /tmp/kaggle-output/exp087_prefix_backtest_tvt_confidence/train_v1_failed
```

- v1 status: failed.
- failure: `DeadKernelError: Kernel died` during the audit execution cell.
- source check had succeeded and found the exp072 cache at `/kaggle/input/notebooks/kentookumura/exp072-exp063-full-replay-feature-cache-train/artifacts/exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz`.
- likely cause: `read_source_frame` read the full exp072 feature cache with all 196 columns, then created multiple copies during materialization / feature generation. This likely exceeded the Kaggle CPU notebook memory budget.
- output synced to `/tmp/kaggle-output/exp087_prefix_backtest_tvt_confidence/train_v1_failed`; no audit artifacts were produced.

Fix for v2:

- `read_source_frame` now reads only required source columns via `usecols`.
- numeric columns are downcast after read.
- synthetic CSV smoke confirmed `usecols` path still produces fold-safe confidence predictions: 960 rows / 12 wells, `expected_error_abs_error_pearson=0.730692`.

Validation after fix:

```bash
uv run ruff check experiments/exp087_prefix_backtest_tvt_confidence/prefix_backtest_tvt_confidence.py experiments/exp087_prefix_backtest_tvt_confidence/settings.py
uv run python -m py_compile experiments/exp087_prefix_backtest_tvt_confidence/prefix_backtest_tvt_confidence.py experiments/exp087_prefix_backtest_tvt_confidence/settings.py
uv run python scripts/validate_experiment.py --experiment exp087_prefix_backtest_tvt_confidence
```

- `ruff check`: pass
- `py_compile`: pass
- `validate_experiment`: pass
