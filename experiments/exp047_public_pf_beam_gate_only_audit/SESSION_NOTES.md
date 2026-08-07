# exp047_public_pf_beam_gate_only_audit セッションノート

## 目的

`public_pf_beam_gate_only_audit` を実装する。PF/Beam を直接予測値や自由な残差補正として使うのではなく、`base + w * (candidate - base)` の重み調整だけに制限し、train well の途中以降を隠した疑似 test 条件で監査する。

## 現在の状態

- Route: pf_beam
- 状態: completed; Kaggle train version 1 complete
- CV: 14.527279 (`exp026_to_pf_gate_w0p30`, original-fold surrogate)
- Public LB: なし
- 親: `exp046_hidden_branch_surrogate_audit`
- 入力: `experiments/exp029_public_sel15_pf_oof_feature_generation/features/public_sel15_pf_oof_features.csv.gz`
- 監査対象: exp026 anchor、public PF、`pf090_hold010`、fixed gate、learned gate
- artifact path: `experiments/exp047_public_pf_beam_gate_only_audit/artifacts/`
- smoke output: `/tmp/exp047_smoke`
- Kaggle train package: `experiments/exp047_public_pf_beam_gate_only_audit/kaggle/train`
- Kaggle train kernel id: `kentookumura/exp047-pf-beam-gate-audit-train`
- Kaggle version: 1
- output path: `/tmp/kaggle-output/exp047_public_pf_beam_gate_only_audit/train_v1`

## コマンドログ

### 実行済み

```bash
uv run python scripts/new_steering.py --experiment exp047_public_pf_beam_gate_only_audit
uv run python scripts/new_experiment.py --name exp047_public_pf_beam_gate_only_audit --source experiments/exp046_hidden_branch_surrogate_audit
uv run python -m py_compile experiments/exp047_public_pf_beam_gate_only_audit/gate_only_audit.py experiments/exp047_public_pf_beam_gate_only_audit/hidden_branch_surrogate_audit.py experiments/exp047_public_pf_beam_gate_only_audit/meta_stack_audit.py experiments/exp047_public_pf_beam_gate_only_audit/baseline.py experiments/exp047_public_pf_beam_gate_only_audit/pseudo_tail_augmentation.py experiments/exp047_public_pf_beam_gate_only_audit/settings.py
uv run ruff check experiments/exp047_public_pf_beam_gate_only_audit/gate_only_audit.py experiments/exp047_public_pf_beam_gate_only_audit/hidden_branch_surrogate_audit.py experiments/exp047_public_pf_beam_gate_only_audit/meta_stack_audit.py experiments/exp047_public_pf_beam_gate_only_audit/baseline.py experiments/exp047_public_pf_beam_gate_only_audit/pseudo_tail_augmentation.py experiments/exp047_public_pf_beam_gate_only_audit/settings.py
uv run python scripts/validate_experiment.py --experiment exp047_public_pf_beam_gate_only_audit
uv run python experiments/exp047_public_pf_beam_gate_only_audit/gate_only_audit.py --max-wells 5 --max-train-rows 300 --base-estimator HistGradientBoostingRegressor --output-dir /tmp/exp047_smoke
uv run pytest tests/test_kaggle_notebooks.py tests/test_scaffold.py
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp047_public_pf_beam_gate_only_audit --notebook train --kernel-id kentookumura/exp047-pf-beam-gate-audit-train --title "exp047 pf beam gate audit train" --run-on-push --strict
uv run python -c "... prepared train notebook code cells ast.parse ..."
uv run python scripts/record_experiment.py --experiment exp047_public_pf_beam_gate_only_audit --status running --cv - --public-lb - --private-lb - --metric rmse --key-idea "Gate-only surrogate audit for public PF/Beam: constrain candidates to base + clipped w*(candidate-base), avoiding direct residual/meta branches." --notes "implemented; py_compile/ruff/validate/pytest PASS; local smoke PASS on 5 wells; Kaggle train package prepared; full audit pending"
kaggle kernels push -p experiments/exp047_public_pf_beam_gate_only_audit/kaggle/train
kaggle kernels pull kentookumura/exp047-pf-beam-gate-audit-train -p /tmp/kaggle-pull/exp047-pf-beam-gate-audit-train -m
kaggle kernels logs kentookumura/exp047-pf-beam-gate-audit-train
kaggle kernels output kentookumura/exp047-pf-beam-gate-audit-train -p /tmp/kaggle-output/exp047_public_pf_beam_gate_only_audit/train_v1
cp /tmp/kaggle-output/exp047_public_pf_beam_gate_only_audit/train_v1/artifacts/public_pf_beam_gate_only_*.csv /tmp/kaggle-output/exp047_public_pf_beam_gate_only_audit/train_v1/artifacts/public_pf_beam_gate_only_summary.json experiments/exp047_public_pf_beam_gate_only_audit/artifacts/
cp /tmp/kaggle-output/exp047_public_pf_beam_gate_only_audit/train_v1/metrics.json experiments/exp047_public_pf_beam_gate_only_audit/metrics.json
cp /tmp/kaggle-output/exp047_public_pf_beam_gate_only_audit/train_v1/exp047-pf-beam-gate-audit-train.log experiments/exp047_public_pf_beam_gate_only_audit/artifacts/
uv run python scripts/record_experiment.py --experiment exp047_public_pf_beam_gate_only_audit --status completed --cv 14.527279 --public-lb - --private-lb - --metric rmse --key-idea "Gate-only surrogate audit for public PF/Beam; fixed exp026_to_pf_gate_w0p30 is best across original-fold, well-hash, and stratified-group surrogate splits." --notes "Kaggle train v1 COMPLETE; rows=1782279 wells=773; best original=14.527279, well-hash=14.620835, stratified=14.353489; all distance buckets improved vs exp026 anchor; synced artifacts; no submission"
```

## 変更点

- `config.yaml` を exp047 / gate-only audit 用に更新。
- `settings.py` の `EXPERIMENT_NAME` を exp047 に更新。
- train/inference notebook を exp047 名にリネームし、train は `gate_only_audit.py` を呼ぶ構成に更新。
- `gate_only_audit.py` を追加。
  - `exp046` の split / exp026 anchor 再生成 / metrics 出力を再利用。
  - fixed gate と learned gate を `base + w * (candidate - base)` の形で生成。
  - learned gate は `optimal_weight` または `candidate_wins` の bounded target を学習し、`w` を 0.2-0.4 以下に clip。
  - overall / segment / well / diff / gate stats / exp026 source summary を保存。
- inference notebook は `NO_SUBMISSION.txt` だけを生成する audit-only guard に更新。

## 結果

- py_compile: PASS
- ruff: PASS
- validation: `scripts/validate_experiment.py --experiment exp047_public_pf_beam_gate_only_audit` PASS
- notebook code cell AST: source train/inference PASS、prepared train PASS
- tests: `tests/test_kaggle_notebooks.py tests/test_scaffold.py` は 10 passed
- local smoke: PASS
  - command: `--max-wells 5 --max-train-rows 300 --base-estimator HistGradientBoostingRegressor --output-dir /tmp/exp047_smoke`
  - rows / wells: 11,207 / 5
  - output files: `public_pf_beam_gate_only_metrics.csv`、`public_pf_beam_gate_only_segment_metrics.csv`、`public_pf_beam_gate_only_diff_metrics.csv`、`public_pf_beam_gate_only_well_metrics.csv`、`public_pf_beam_gate_only_gate_stats.csv`、`public_pf_beam_gate_only_summary.json`
  - smoke best:
    - leave-one-original-fold: `public_pf_selector` 15.845874
    - well-hash: `public_pf_selector` 15.845874
    - stratified-group: `learned_pf_gate_ridge_wmax0p40` 13.930404
- prepare: Kaggle train package 生成 PASS
  - kernel id: `kentookumura/exp047-pf-beam-gate-audit-train`
  - competition source: `rogii-wellbore-geology-prediction`
  - kernel source: `kentookumura/exp029-sel15-pf-oof-train`
  - GPU/internet: false / false
- Kaggle train v1: COMPLETE
  - runtime: notebook log time 約 1,668 秒
  - rows / wells: 1,782,279 / 773
  - output path: `/tmp/kaggle-output/exp047_public_pf_beam_gate_only_audit/train_v1`
  - synced artifact files:
    - `public_pf_beam_gate_only_metrics.csv`
    - `public_pf_beam_gate_only_segment_metrics.csv`
    - `public_pf_beam_gate_only_diff_metrics.csv`
    - `public_pf_beam_gate_only_well_metrics.csv`
    - `public_pf_beam_gate_only_gate_stats.csv`
    - `public_pf_beam_gate_only_exp026_source_summary.csv`
    - `public_pf_beam_gate_only_well_metadata.csv`
    - `public_pf_beam_gate_only_summary.json`
    - `exp047-pf-beam-gate-audit-train.log`
- Full surrogate best:
  - original-fold: `exp026_to_pf_gate_w0p30` 14.527279
  - well-hash: `exp026_to_pf_gate_w0p30` 14.620835
  - stratified-group: `exp026_to_pf_gate_w0p30` 14.353489
- Full surrogate deltas for `exp026_to_pf_gate_w0p30`:
  - original-fold: vs exp026 -1.116453、vs public PF -0.645357、vs pf090_hold010 -0.562252
  - well-hash: vs exp026 -1.205268、vs public PF -0.551801、vs pf090_hold010 -0.468697
  - stratified-group: vs exp026 -1.118336、vs public PF -0.819147、vs pf090_hold010 -0.736043
- Distance bucket: `rows_0_49` から `rows_2500_plus` まで全 bucket / 全 split system で exp026 anchor より改善。
- Learned gate: `learned_pf_gate_ridge_wmax0p40` は全 split で 2 位。固定 `w=0.30` が一貫して上回った。
- `experiment_summary.md` に `completed` として記録済み。

## 次のアクション

1. `exp027` Public LB 8.781 は維持する。
2. 推論側へ進める場合は、learned gate ではなく固定 `exp026_to_pf_gate_w0p30` だけを候補にする。
3. code submit 前に、exp031/033/035/045 と同じ Public LB 転移失敗リスクを前提に、別実験として inference port / output diff / public sample blind spot を監査する。
