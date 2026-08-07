# exp033_public_sel15_pf_residual_inference_port セッションノート

## 目的

`exp032` で supported になった `ridge_residual_shrink0p5_clip20p0` を、`exp027/exp031` の公開 sel15 inference flow に移植し、提出前に output diff / range / changed wells / submit-check を監査できる状態にする。

## 現在の状態

- Route: pf_beam
- 状態: completed; code submission complete
- CV: なし
- Public LB: 14.961
- 親: `exp031_public_sel15_pf_hold_blend_inference_audit`
- 変更: 見えない test well の `tvt_selector` に `0.5 * clip(ridge_residual, -20, 20)` を加える
- residual training source: `kentookumura/exp029-sel15-pf-oof-train` の `public_sel15_pf_oof_features.csv.gz`
- Kaggle kernel: `kentookumura/exp033-sel15-pf-residual-infer` version 1
- output path: `/tmp/kaggle-output/exp033_public_sel15_pf_residual_inference_port/inference_retry`
- artifact path: `experiments/exp033_public_sel15_pf_residual_inference_port/artifacts/`
- submission sha256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
- submit-check: PASS
- competition submit: ref `53444678`
- Public LB: 14.961

## コマンドログ

### 実行済み

```bash
uv run python scripts/new_steering.py --experiment exp033_public_sel15_pf_residual_inference_port
uv run python scripts/new_experiment.py --name exp033_public_sel15_pf_residual_inference_port --source experiments/exp031_public_sel15_pf_hold_blend_inference_audit
uv run python scripts/validate_experiment.py --experiment exp033_public_sel15_pf_residual_inference_port
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp033_public_sel15_pf_residual_inference_port --notebook inference --kernel-id kentookumura/exp033-sel15-pf-residual-infer --title "exp033 sel15 pf residual infer" --run-on-push --strict
uv run python scripts/update_experiment_summary.py
uv run pytest tests/test_kaggle_notebooks.py tests/test_scaffold.py
kaggle kernels push -p experiments/exp033_public_sel15_pf_residual_inference_port/kaggle/inference
kaggle kernels pull kentookumura/exp033-sel15-pf-residual-infer -p /tmp/kaggle-pull/exp033-sel15-pf-residual-infer -m
kaggle kernels logs kentookumura/exp033-sel15-pf-residual-infer
kaggle kernels output kentookumura/exp033-sel15-pf-residual-infer -p /tmp/kaggle-output/exp033_public_sel15_pf_residual_inference_port/inference_retry
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp033_public_sel15_pf_residual_inference_port/inference_retry/submission.csv
kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp033-sel15-pf-residual-infer -v 1 -f submission.csv -m "exp033 ridge residual hidden branch"
kaggle competitions submissions rogii-wellbore-geology-prediction
uv run python scripts/record_submission.py --experiment exp033_public_sel15_pf_residual_inference_port --file experiments/exp033_public_sel15_pf_residual_inference_port/artifacts/submission.csv --cv - --public-lb - --private-lb - --notes "ref=53444678; kernel=kentookumura/exp033-sel15-pf-residual-infer v1; code-submit hidden ridge_residual_shrink0p5_clip20p0 branch; status=PENDING; sample identical to exp027/031 SHA; submit-check PASS"
kaggle competitions submissions rogii-wellbore-geology-prediction
```

### 予定

```bash
kaggle competitions submissions rogii-wellbore-geology-prediction
```

## 変更点

- `config.yaml` を exp033 / pf_beam / selected residual inference port 用に更新。
- `settings.py` の `EXPERIMENT_NAME` を exp033 に更新。
- train/inference notebook を exp033 名にリネーム。
- inference notebook に exp032 selected Ridge residual model の fit と 見えない test well 用補正 を追加。
- 監査 artifact として元 selector submission、row-level diff、summary JSON を出力。
- Kaggle 用 inference notebook を `experiments/exp033_public_sel15_pf_residual_inference_port/kaggle/inference/` に生成。

## 結果

- 構文チェック: train / inference notebook と prepared Kaggle notebook の code cell は `ast.parse` 通過。
- feature columns: inference notebook の `RESIDUAL_FEATURE_COLUMNS` は `exp032` config の `model.features` と一致。
- validation: `uv run python scripts/validate_experiment.py --experiment exp033_public_sel15_pf_residual_inference_port` 通過。
- prepare: `uv run python scripts/prepare_kaggle_notebooks.py ... --strict` 通過。
- Kaggle metadata: competition source `rogii-wellbore-geology-prediction` と kernel source `kentookumura/exp029-sel15-pf-oof-train` を含む。
- tests: `uv run pytest tests/test_kaggle_notebooks.py tests/test_scaffold.py` は 10 passed。
- push: `kaggle kernels push -p experiments/exp033_public_sel15_pf_residual_inference_port/kaggle/inference` が成功し、version 1 を push。
- pull: `kaggle kernels pull kentookumura/exp033-sel15-pf-residual-infer -p /tmp/kaggle-pull/exp033-sel15-pf-residual-infer -m` で存在確認。
- Kaggle log: 約 184 秒で `submission.csv` と監査 artifact を生成。
- residual model fit: exp029 feature artifact 1,782,279 rows から 473,950 rows を sampling、51 features、`ridge_residual_shrink0p5_clip20p0`。
- output rows: 14,151
- prediction range: 11587.038593 - 12240.016066
- prediction mean: 11903.630073
- missing values: 0
- duplicate IDs: 0
- submit-check: PASS
- original selector submission との差分: min 0.000000、max 0.000000、mean 0.000000、abs mean 0.000000、RMSE 0.000000
- output SHA256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
- 監査 summary: changed_rows 0、changed_wells 0、diff_rmse 0.000000
- public sample は全 3 wells が visible train wells で physical-model branch を使うため、見えない test well 用 residual 処理 は sample output では発火しない。
- code submit: ref `53444678`、Public LB 14.961。
- exp027 Public LB 8.781 より +6.180 悪化。
- exp031 Public LB 8.956 より +6.005 悪化。
- exp032 train well の途中以降を隠した疑似 test residual correction は 見えない test well 評価の LB に転移しなかったため、residual branch は採用しない。
- `submissions/SUBMISSIONS.md` に v014 として記録。

## 次のアクション

1. exp027 anchor 8.781 を維持する。
2. public sel15 PF residual / hold branch の追加チューニングには進まない。
