# exp034_public_sel15_pf_meta_stack セッションノート

## 目的

`public_sel15_pf_meta_stack` を実装する。`exp029` の public sel15 PF/Beam OOF-like artifact 上で、`exp026` 相当の pseudo-tail bucket-shrink anchor を fold-safe に再生成し、PF/Beam feature と合わせた 2nd stage が clean CV で安定して効くか監査する。

## 現在の状態

- Route: pf_beam
- 状態: completed
- CV: 14.313668 (`ridge_meta_residual_shrink0p75_clip60p0`, original-fold OOF)
- LB: なし
- 親: `exp029_public_sel15_pf_oof_feature_generation`
- anchor: `exp026_pseudo_tail_bucket_shrink_inference_submit` clean CV 12.870780
- feature source: `experiments/exp029_public_sel15_pf_oof_feature_generation/features/public_sel15_pf_oof_features.csv.gz`
- Kaggle kernel: `kentookumura/exp034-sel15-pf-meta-stack-train` version 2
- output path: `/tmp/kaggle-output/exp034_public_sel15_pf_meta_stack/train_v2`
- artifact path: `experiments/exp034_public_sel15_pf_meta_stack/artifacts/`

## コマンドログ

### 実行済み

```bash
uv run python scripts/new_steering.py --experiment exp034_public_sel15_pf_meta_stack
uv run python scripts/new_experiment.py --name exp034_public_sel15_pf_meta_stack --source experiments/exp032_public_sel15_pf_residual_correction
```

### 実行済み

```bash
uv run ruff check experiments/exp034_public_sel15_pf_meta_stack/meta_stack_audit.py experiments/exp034_public_sel15_pf_meta_stack/baseline.py experiments/exp034_public_sel15_pf_meta_stack/pseudo_tail_augmentation.py
uv run python -m py_compile experiments/exp034_public_sel15_pf_meta_stack/meta_stack_audit.py experiments/exp034_public_sel15_pf_meta_stack/baseline.py experiments/exp034_public_sel15_pf_meta_stack/pseudo_tail_augmentation.py experiments/exp034_public_sel15_pf_meta_stack/settings.py
uv run python experiments/exp034_public_sel15_pf_meta_stack/meta_stack_audit.py --max-wells 10 --max-train-rows 2000 --skip-hgb --base-estimator HistGradientBoostingRegressor --output-dir /tmp/exp034_smoke
uv run python scripts/validate_experiment.py --experiment exp034_public_sel15_pf_meta_stack
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp034_public_sel15_pf_meta_stack --notebook train --kernel-id kentookumura/exp034-sel15-pf-meta-stack-train --title "exp034 sel15 pf meta stack train" --run-on-push --strict
uv run pytest tests/test_kaggle_notebooks.py tests/test_scaffold.py
```

### 実行済み

```bash
kaggle kernels push -p experiments/exp034_public_sel15_pf_meta_stack/kaggle/train
kaggle kernels pull kentookumura/exp034-sel15-pf-meta-stack-train -p /tmp/kaggle-pull/exp034-sel15-pf-meta-stack-train -m
kaggle kernels logs kentookumura/exp034-sel15-pf-meta-stack-train
kaggle kernels output kentookumura/exp034-sel15-pf-meta-stack-train -p /tmp/kaggle-output/exp034_public_sel15_pf_meta_stack/train
kaggle kernels push -p experiments/exp034_public_sel15_pf_meta_stack/kaggle/train
timeout 20 kaggle kernels logs -f --interval 5 kentookumura/exp034-sel15-pf-meta-stack-train
kaggle kernels output kentookumura/exp034-sel15-pf-meta-stack-train -p /tmp/kaggle-output/exp034_public_sel15_pf_meta_stack/train_v2
cp /tmp/kaggle-output/exp034_public_sel15_pf_meta_stack/train_v2/metrics.json experiments/exp034_public_sel15_pf_meta_stack/metrics.json
cp /tmp/kaggle-output/exp034_public_sel15_pf_meta_stack/train_v2/artifacts/meta_stack_*.csv /tmp/kaggle-output/exp034_public_sel15_pf_meta_stack/train_v2/artifacts/meta_stack_summary.json experiments/exp034_public_sel15_pf_meta_stack/artifacts/
```

## 変更点

- `config.yaml` を exp034 / pf_beam / meta-stack audit 用に更新。
- `settings.py` の `EXPERIMENT_NAME` を exp034 に更新。
- `baseline.py` と `pseudo_tail_augmentation.py` を exp026 から同梱し、exp026 相当の pseudo-tail anchor を再生成できるようにした。
- `meta_stack_audit.py` を追加。
  - exp029 PF/Beam feature artifact を読む。
  - original-fold / well-hash の各 audit split で validation wells を除外して pseudo-tail LightGBM を fit する。
  - exp029 pseudo cutoff rows に対して exp026 bucket-shrink prediction を作る。
  - fixed blend、ridge residual meta、shallow HGB residual meta を比較する。
  - overall / distance bucket / split / well metrics と source summary を保存する。
- train notebook を exp034 用に更新。
- inference notebook は audit-only guard に変更し、提出生成しない。
- `--base-estimator` CLI override を追加し、ローカル smoke だけ HGB に差し替えられるようにした。Kaggle full audit は config 通り LGBM を使う。
- 明示 `--output-dir` の smoke では実験本体の `metrics.json` を上書きしないようにした。
- v1 failure を受けて、exp029 の `cutoff_row` 定義に合わせ `predict_well_cutoff` では `cutoff_row` から `TVT_input` を NaN 化するよう修正した。

## 結果

- ruff: PASS
- py_compile: PASS
- notebook code cell compile: PASS
- local smoke: PASS (`--max-wells 10 --max-train-rows 2000 --skip-hgb --base-estimator HistGradientBoostingRegressor --output-dir /tmp/exp034_smoke`)
  - smoke rows / wells: 21,974 / 10
  - smoke artifact: `/tmp/exp034_smoke/meta_stack_summary.json`
  - local smoke は配線確認だけで、exp026 LGBM anchor の正式 CV ではない。
- validation: `scripts/validate_experiment.py --experiment exp034_public_sel15_pf_meta_stack` PASS
- prepare: Kaggle train package `experiments/exp034_public_sel15_pf_meta_stack/kaggle/train` 生成 PASS
- tests: `tests/test_kaggle_notebooks.py tests/test_scaffold.py` は 10 passed
- Kaggle metadata:
  - kernel id: `kentookumura/exp034-sel15-pf-meta-stack-train`
  - competition source: `rogii-wellbore-geology-prediction`
  - kernel source: `kentookumura/exp029-sel15-pf-oof-train`
- Kaggle train v1: FAILED
  - error: `ValueError: missing exp026 predictions for 404c4384__horizontal_well.csv rows: 1698`
  - 原因: exp029 artifact は `row_idx == cutoff_row` を eval_step 0 として出力するが、exp034 側は `cutoff_row + 1` から NaN にしていたため、最初の評価行が exp026-style prediction に含まれなかった。
- 修正: `predict_well_cutoff` で `TVT_input[int(cutoff_row):] = np.nan` に変更し、local smoke PASS。
- Kaggle train v2: COMPLETED
  - rows / wells: 1,782,279 / 773
  - required control by audit: original-fold 15.089532、well-hash 15.089532
  - selected: `ridge_meta_residual_shrink0p75_clip60p0`
  - original-fold OOF: 14.313668
  - well-hash holdout: 14.172010
  - original-fold delta vs exp026-row control / public PF / pf090_hold010: -1.330065 / -0.858969 / -0.775864
  - well-hash delta vs exp026-row control / public PF / pf090_hold010: -1.654092 / -1.000626 / -0.917521
  - selected は両 audit の全 distance bucket と全 split で exp026-row reference より改善。
  - artifact と metrics は `experiments/exp034_public_sel15_pf_meta_stack/artifacts/` と `metrics.json` に同期済み。

## logs 調査メモ

- `kaggle kernels logs` は Kaggle CLI 2.1.2 の実装上、`ListKernelSessionOutput` API の `response.log` をそのまま表示するだけ。
- v2 実行開始直後は通常 `logs` と `output` が空を返した。これは API response に log がまだ入っていなかったためで、slug 不一致や pull 失敗ではない。`kaggle kernels pull kentookumura/exp034-sel15-pf-meta-stack-train -m` は成功していた。
- `kaggle kernels status` は既知どおり `GetKernelSessionStatus` が 500 を返したため、完了判定には使っていない。
- `kaggle kernels logs -f --interval 5` で polling すると、実行中ログと最終ログを取得できた。今回の完了確認はこの follow logs と `kernels output` の artifact 取得を根拠にした。

## 次のアクション

1. `ridge_meta_residual_shrink0p75_clip60p0` を exp027 public sel15 inference flow に移植する別実験を作る。
2. output diff、range、start continuity、submit-check を確認する。
3. Public LB anchor 8.781 に対して提出価値があるか判断する。
