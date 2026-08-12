# exp035_public_sel15_pf_meta_inference_port セッションノート

## 目的

`exp034` で supported になった `ridge_meta_residual_shrink0p75_clip60p0` を、`exp027` 系の公開 sel15 inference flow に移植し、提出前に output diff / range / changed wells / submit-check を監査できる状態にする。

## 現在の状態

- Route: pf_beam
- 状態: completed; code submission complete
- CV: なし
- Public LB: 13.738
- 親: `exp034_public_sel15_pf_meta_stack`
- 変更: 見えない test well の base を exp026-style pseudo-tail anchor にし、`0.75 * clip(ridge_meta_residual, -60, 60)` を加える
- meta training source: `kentookumura/exp029-sel15-pf-oof-train` の `public_sel15_pf_oof_features.csv.gz`
- supporting evidence: exp034 original-fold 14.313668、well-hash 14.172010
- Kaggle kernel: `kentookumura/exp035-sel15-pf-meta-infer` version 1
- output path: `/tmp/kaggle-output/exp035_public_sel15_pf_meta_inference_port/inference_v1`
- artifact path: `experiments/exp035_public_sel15_pf_meta_inference_port/artifacts/`
- submission sha256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
- submit-check: PASS
- competition submit: ref `53452712`
- submission status: COMPLETE

## コマンドログ

### 実行済み

```bash
uv run python scripts/new_steering.py --experiment exp035_public_sel15_pf_meta_inference_port
uv run python scripts/new_experiment.py --name exp035_public_sel15_pf_meta_inference_port --source experiments/exp033_public_sel15_pf_residual_inference_port
python3 -c "... notebook code cell ast.parse ..."
uv run python -m py_compile experiments/exp035_public_sel15_pf_meta_inference_port/baseline.py experiments/exp035_public_sel15_pf_meta_inference_port/pseudo_tail_augmentation.py experiments/exp035_public_sel15_pf_meta_inference_port/settings.py
python3 -c "... yaml.safe_load(config.yaml) ..."
uv run python scripts/validate_experiment.py --experiment exp035_public_sel15_pf_meta_inference_port
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp035_public_sel15_pf_meta_inference_port --notebook inference --kernel-id kentookumura/exp035-sel15-pf-meta-infer --title "exp035 sel15 pf meta infer" --run-on-push --strict
python3 -c "... prepared notebook code cell ast.parse ..."
uv run pytest tests/test_kaggle_notebooks.py tests/test_scaffold.py
kaggle kernels push -p experiments/exp035_public_sel15_pf_meta_inference_port/kaggle/inference
kaggle kernels pull kentookumura/exp035-sel15-pf-meta-infer -p /tmp/kaggle-pull/exp035-sel15-pf-meta-infer -m
kaggle kernels output kentookumura/exp035-sel15-pf-meta-infer -p /tmp/kaggle-output/exp035_public_sel15_pf_meta_inference_port/inference_v1
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp035_public_sel15_pf_meta_inference_port/inference_v1/submission.csv
kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp035-sel15-pf-meta-infer -v 1 -f submission.csv -m "exp035 ridge meta hidden branch"
kaggle competitions submissions rogii-wellbore-geology-prediction
uv run python scripts/record_submission.py --experiment exp035_public_sel15_pf_meta_inference_port --file experiments/exp035_public_sel15_pf_meta_inference_port/artifacts/submission.csv --cv - --public-lb - --private-lb - --notes "ref=53452712; kernel=kentookumura/exp035-sel15-pf-meta-infer v1; code-submit ridge_meta_residual_shrink0p75_clip60p0 hidden branch; status=PENDING; sample unchanged physical branch SHA; submit-check PASS"
kaggle competitions submissions rogii-wellbore-geology-prediction
```

## 変更点

- `config.yaml` を exp035 / pf_beam / selected meta inference port 用に更新。
- `settings.py` の `EXPERIMENT_NAME` を exp035 に更新。
- train/inference notebook を exp035 名にリネーム。
- `baseline.py` と `pseudo_tail_augmentation.py` を exp034 から同梱し、inference notebook 内で exp026-style anchor を生成できるようにした。
- inference notebook に次を追加。
  - exp026-style pseudo-tail anchor model を train wells から fit。
  - exp029 train well の途中以降を隠した疑似 test sampled rows に exp026 anchor prediction を再生成。
  - exp034 selected Ridge meta residual model を fit。
  - hidden test wells に `exp026_anchor + 0.75 * clip(meta_residual, -60, 60)` を適用。
  - `public_sel15_exp026_anchor_submission.csv`、`public_sel15_meta_corrected_diff.csv`、`public_sel15_meta_corrected_summary.json` を出力。
- visible public sample wells の physical-model branch は変更しない。

## 結果

- notebook code cell AST: PASS。
- `baseline.py` / `pseudo_tail_augmentation.py` / `settings.py` py_compile: PASS。
- YAML parse: PASS。
- validation: `scripts/validate_experiment.py --experiment exp035_public_sel15_pf_meta_inference_port` PASS。
- prepare: Kaggle inference package `experiments/exp035_public_sel15_pf_meta_inference_port/kaggle/inference` 生成 PASS。
- prepared notebook code cell AST: PASS。
- Kaggle metadata:
  - kernel id: `kentookumura/exp035-sel15-pf-meta-infer`
  - competition source: `rogii-wellbore-geology-prediction`
  - kernel source: `kentookumura/exp029-sel15-pf-oof-train`
  - GPU/internet: false / false
- tests: `tests/test_kaggle_notebooks.py tests/test_scaffold.py` は 10 passed。
- push: `kaggle kernels push -p experiments/exp035_public_sel15_pf_meta_inference_port/kaggle/inference` が成功し、version 1 を push。
- pull: `kaggle kernels pull kentookumura/exp035-sel15-pf-meta-infer -p /tmp/kaggle-pull/exp035-sel15-pf-meta-infer -m` で存在確認。
- Kaggle log: 約 442 秒で `submission.csv` と監査 artifact を生成。
- exp026-style anchor fit: train wells 773、train rows 242,843、source rows 788。
- meta model fit: exp029 feature artifact 1,782,279 rows から 503,950 rows を sampling、54 features、`ridge_meta_residual_shrink0p75_clip60p0`。
- output rows: 14,151
- prediction range: 11587.038593 - 12240.016066
- prediction mean: 11903.630073
- missing values: 0
- duplicate IDs: 0
- submit-check: PASS
- exp026 anchor submission との差分: min 0.000000、max 0.000000、mean 0.000000、abs mean 0.000000、RMSE 0.000000
- output SHA256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
- 監査 summary: changed_rows 0、changed_wells 0、diff_rmse 0.000000
- public sample は全 3 wells が visible train wells で physical branch を使うため、見えない test well 用 meta 処理 は sample output では発火しない。
- code submit: ref `53452712`、Public LB 13.738。
- exp027 Public LB 8.781 より +4.957 悪化。
- exp031 Public LB 8.956 より +4.782 悪化。
- exp033 Public LB 14.961 よりは -1.223 改善したが、public sel15 anchor には大きく届かない。
- exp034 train well の途中以降を隠した疑似 test meta stack は 見えない test well 評価の LB に転移しなかったため、見えない test well 用 meta 処理 は採用しない。
- `SUBMISSIONS.md` に v016 として記録。

## 次のアクション

```bash
uv run python scripts/update_experiment_summary.py
```

exp027 anchor 8.781 を維持する。public sel15 の見えない test well 用処理に対する hold / residual / meta 補正は追加チューニングしない。
