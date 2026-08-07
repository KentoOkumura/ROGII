# exp045_public_pf_meta_strict_parity_audit セッションノート

## 目的

`exp035` の見えない test well 用 meta 処理を最小コピーし、見えない test well 推論側の PF diagnostics を exp029 parity の `16 seeds / 250 particles` に揃える。exp034 の疑似 test 条件と exp035 の本番採点条件の feature 分布不一致が Public LB 悪化の主因かを監査する。

## 現在の状態

- Route: pf_beam
- 状態: completed; code submission complete
- CV: なし
- Public LB: 19.177
- 親: `exp035_public_sel15_pf_meta_inference_port`
- 変更: 見えない test well 用 PF 診断値 を `128 seeds / 500 particles` から `16 seeds / 250 particles` に変更
- fixed: exp026-style anchor、exp034 selected Ridge meta residual、meta feature columns、selector rules、beam configs、visible physical branch
- meta training source: `kentookumura/exp029-sel15-pf-oof-train` の `public_sel15_pf_oof_features.csv.gz`
- comparison anchors: exp035 Public LB 13.738、exp027 Public LB 8.781
- Kaggle kernel: `kentookumura/exp045-pf-meta-parity-infer` version 1
- output path: `/tmp/kaggle-output/exp045_public_pf_meta_strict_parity_audit/inference_v1`
- artifact path: `experiments/exp045_public_pf_meta_strict_parity_audit/artifacts/`
- submission sha256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
- submit-check: PASS
- competition submit: ref `53502516`
- submission status: COMPLETE

## コマンドログ

### 実行済み

```bash
uv run python scripts/new_steering.py --experiment exp045_public_pf_meta_strict_parity_audit
uv run python scripts/new_experiment.py --name exp045_public_pf_meta_strict_parity_audit --source experiments/exp035_public_sel15_pf_meta_inference_port
python3 -c "... inference notebook code cell ast.parse ..."
python3 -c "... train notebook code cell ast.parse ..."
uv run python -m py_compile experiments/exp045_public_pf_meta_strict_parity_audit/baseline.py experiments/exp045_public_pf_meta_strict_parity_audit/pseudo_tail_augmentation.py experiments/exp045_public_pf_meta_strict_parity_audit/settings.py
python3 -c "... yaml.safe_load(config.yaml) ..."
uv run python scripts/validate_experiment.py --experiment exp045_public_pf_meta_strict_parity_audit
uv run pytest tests/test_kaggle_notebooks.py tests/test_scaffold.py
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp045_public_pf_meta_strict_parity_audit --notebook inference --kernel-id kentookumura/exp045-pf-meta-parity-infer --title "exp045 pf meta parity infer" --run-on-push --strict
python3 -c "... prepared inference notebook code cell ast.parse ..."
kaggle kernels push -p experiments/exp045_public_pf_meta_strict_parity_audit/kaggle/inference
kaggle kernels pull kentookumura/exp045-pf-meta-parity-infer -p /tmp/kaggle-pull/exp045-pf-meta-parity-infer -m
kaggle kernels logs -f --interval 5 kentookumura/exp045-pf-meta-parity-infer
kaggle kernels logs kentookumura/exp045-pf-meta-parity-infer
kaggle kernels output kentookumura/exp045-pf-meta-parity-infer -p /tmp/kaggle-output/exp045_public_pf_meta_strict_parity_audit/inference_v1
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp045_public_pf_meta_strict_parity_audit/inference_v1/submission.csv
sha256sum /tmp/kaggle-output/exp045_public_pf_meta_strict_parity_audit/inference_v1/submission.csv /tmp/kaggle-output/exp045_public_pf_meta_strict_parity_audit/inference_v1/public_sel15_meta_corrected_summary.json
cp /tmp/kaggle-output/exp045_public_pf_meta_strict_parity_audit/inference_v1/submission.csv /tmp/kaggle-output/exp045_public_pf_meta_strict_parity_audit/inference_v1/public_sel15_exp026_anchor_submission.csv /tmp/kaggle-output/exp045_public_pf_meta_strict_parity_audit/inference_v1/public_sel15_meta_corrected_diff.csv /tmp/kaggle-output/exp045_public_pf_meta_strict_parity_audit/inference_v1/public_sel15_meta_corrected_summary.json /tmp/kaggle-output/exp045_public_pf_meta_strict_parity_audit/inference_v1/exp045-pf-meta-parity-infer.log experiments/exp045_public_pf_meta_strict_parity_audit/artifacts/
kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp045-pf-meta-parity-infer -v 1 -f submission.csv -m "exp045 strict parity meta hidden branch"
kaggle competitions submissions rogii-wellbore-geology-prediction
uv run python scripts/record_submission.py --experiment exp045_public_pf_meta_strict_parity_audit --file experiments/exp045_public_pf_meta_strict_parity_audit/artifacts/submission.csv --cv - --public-lb 19.177 --private-lb - --notes "ref=53502516; kernel=kentookumura/exp045-pf-meta-parity-infer v1; strict PF parity hidden meta branch 16 seeds/250 particles; status=COMPLETE; Public LB worse than exp035 13.738 by +5.439 and exp027 8.781 by +10.396; submit-check PASS; sample SHA identical"
```

## 変更点

- exp035 を `exp045_public_pf_meta_strict_parity_audit` にコピー。
- `config.yaml` を exp045 / strict parity audit 用に更新。
- `settings.py` の `EXPERIMENT_NAME` を exp045 に更新。
- train/inference notebook を exp045 名にリネーム。
- inference notebook の 見えない test well 用 PF 診断値 を config 由来にし、exp045 config で `n_particles=250`、`n_seeds=16` を使うようにする。
- summary に実際の 見えない test well 用 PF 診断値 settings を保存する。

## 結果

- inference notebook AST: PASS。
- train notebook AST: PASS。
- `baseline.py` / `pseudo_tail_augmentation.py` / `settings.py` py_compile: PASS。
- YAML parse: PASS。
- validation: `scripts/validate_experiment.py --experiment exp045_public_pf_meta_strict_parity_audit` PASS。
- tests: `tests/test_kaggle_notebooks.py tests/test_scaffold.py` は 10 passed。
- prepare: Kaggle inference package `experiments/exp045_public_pf_meta_strict_parity_audit/kaggle/inference` 生成 PASS。
- prepared inference notebook AST: PASS。
- Kaggle metadata:
  - kernel id: `kentookumura/exp045-pf-meta-parity-infer`
  - competition source: `rogii-wellbore-geology-prediction`
  - kernel source: `kentookumura/exp029-sel15-pf-oof-train`
  - GPU/internet: false / false
- push: initial sandboxed push failed with `Expecting value: line 1 column 1 (char 0)`; escalated rerun succeeded and pushed version 1.
- pull: `kaggle kernels pull kentookumura/exp045-pf-meta-parity-infer -m` で存在確認。
- Kaggle log: 約 180 秒で `submission.csv` と監査 artifact を生成。
- exp026-style anchor fit: train wells 773、train rows 242,843、source rows 788。
- meta model fit: exp029 feature artifact 1,782,279 rows から 503,950 rows を sampling、54 features、`ridge_meta_residual_shrink0p75_clip60p0`。
- 見えない test well 用 PF 診断値の条件一致 log: `seeds=16 particles=250 source=exp029_public_sel15_pf_oof_feature_generation_default`。
- output rows: 14,151
- prediction range: 11587.038593 - 12240.016066
- prediction mean: 11903.630073
- submit-check: PASS
- output SHA256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
- summary SHA256: `c4851b6b45cad48224e44fdb5b882df67a4432308ab861742246648764161fc1`
- 監査 summary: changed_rows 0、changed_wells 0、diff_rmse 0.000000、meta_residual_abs_mean 0.000000。
- public sample は全 3 wells が visible train wells で physical branch を使うため、見えない test well 用 meta 処理 は sample output では発火しない。
- code submit: ref `53502516`、Public LB 19.177。
- exp035 Public LB 13.738 より +5.439 悪化。
- exp027 Public LB 8.781 より +10.396 悪化。
- strict PF parity でも見えない test well 用 meta 処理は転移せず、exp034/035-style の meta stack 追加チューニングは止める。
- `submissions/SUBMISSIONS.md` に v018 として記録。
- `git status` は不可。この環境では `.git` が空の管理ディレクトリとして見えている。

## 次のアクション

exp027 anchor 8.781 を維持する。public sel15 PF/Beam train well の途中以降を隠した疑似 test 生成物 に基づく exp026 anchor + Ridge meta 見えない test well 用 residual 処理 の追加チューニングには進まない。
