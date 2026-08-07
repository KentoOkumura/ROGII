# exp046_hidden_branch_surrogate_audit セッションノート

## 目的

`hidden_branch_surrogate_audit` を実装する。ここでいう「見えない test well 用処理」は、Kaggle の本番採点で出てくる train に存在しない test well だけに使う推論処理のこと。public sample では visible train well 用処理が使われるため `exp031/033/035/045` の見えない test well 用処理が発火せず、`changed_rows=0` や SHA 一致だけでは安全性を確認できない。train well の途中以降を隠した疑似 test 条件でその処理を強制適用し、提出前の代理監査に使える生成物を作る。

## 現在の状態

- Route: pf_beam
- 状態: completed; Kaggle train version 1 complete
- CV: 14.313668 (`exp035_ridge_meta_residual_shrink0p75_clip60p0`, original-fold surrogate)
- Public LB: なし
- 親: `exp045_public_pf_meta_strict_parity_audit`
- 入力: `experiments/exp029_public_sel15_pf_oof_feature_generation/features/public_sel15_pf_oof_features.csv.gz`
- 監査対象: exp026 anchor、public PF、`pf090_hold010`、`exp033` PF residual、`exp035/045` meta residual
- smoke output: `/tmp/exp046_smoke`
- Kaggle train package: `experiments/exp046_hidden_branch_surrogate_audit/kaggle/train`
- Kaggle kernel id: `kentookumura/exp046-hidden-branch-surrogate-audit-train`
- Kaggle version: 1
- output path: `/tmp/kaggle-output/exp046_hidden_branch_surrogate_audit/train_v1`
- artifact path: `experiments/exp046_hidden_branch_surrogate_audit/artifacts/`

## コマンドログ

### 実行済み

```bash
uv run python scripts/new_steering.py --experiment exp046_hidden_branch_surrogate_audit
uv run python scripts/new_experiment.py --name exp046_hidden_branch_surrogate_audit --source experiments/exp034_public_sel15_pf_meta_stack
uv run python -m py_compile experiments/exp046_hidden_branch_surrogate_audit/hidden_branch_surrogate_audit.py experiments/exp046_hidden_branch_surrogate_audit/meta_stack_audit.py experiments/exp046_hidden_branch_surrogate_audit/baseline.py experiments/exp046_hidden_branch_surrogate_audit/pseudo_tail_augmentation.py experiments/exp046_hidden_branch_surrogate_audit/settings.py
uv run ruff check experiments/exp046_hidden_branch_surrogate_audit/hidden_branch_surrogate_audit.py experiments/exp046_hidden_branch_surrogate_audit/meta_stack_audit.py experiments/exp046_hidden_branch_surrogate_audit/baseline.py experiments/exp046_hidden_branch_surrogate_audit/pseudo_tail_augmentation.py experiments/exp046_hidden_branch_surrogate_audit/settings.py
uv run python scripts/validate_experiment.py --experiment exp046_hidden_branch_surrogate_audit
uv run python experiments/exp046_hidden_branch_surrogate_audit/hidden_branch_surrogate_audit.py --max-wells 5 --max-train-rows 300 --base-estimator HistGradientBoostingRegressor --output-dir /tmp/exp046_smoke
uv run pytest tests/test_kaggle_notebooks.py tests/test_scaffold.py
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp046_hidden_branch_surrogate_audit --notebook train --kernel-id kentookumura/exp046-hidden-branch-surrogate-audit-train --title "exp046 hidden branch surrogate audit train" --run-on-push --strict
uv run python - <<'PY'
import ast, json
from pathlib import Path
p = Path('experiments/exp046_hidden_branch_surrogate_audit/kaggle/train/exp046_hidden_branch_surrogate_audit_train.ipynb')
nb = json.loads(p.read_text())
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        ast.parse(''.join(cell.get('source', [])), filename=f'{p}:{i}')
print('prepared train notebook code cells parse')
PY
uv run python scripts/record_experiment.py --experiment exp046_hidden_branch_surrogate_audit --status running --cv - --public-lb - --private-lb - --metric rmse --key-idea "Train-side surrogate audit for hidden-branch candidates that do not fire on the visible public sample." --notes "implemented; local smoke PASS on 5 wells; validation/tests/prepare PASS; full Kaggle train audit pending"
kaggle kernels push -p experiments/exp046_hidden_branch_surrogate_audit/kaggle/train
kaggle kernels pull kentookumura/exp046-hidden-branch-surrogate-audit-train -p /tmp/kaggle-pull/exp046-hidden-branch-surrogate-audit-train -m
kaggle kernels logs kentookumura/exp046-hidden-branch-surrogate-audit-train
kaggle kernels output kentookumura/exp046-hidden-branch-surrogate-audit-train -p /tmp/kaggle-output/exp046_hidden_branch_surrogate_audit/train_v1
cp /tmp/kaggle-output/exp046_hidden_branch_surrogate_audit/train_v1/artifacts/hidden_branch_surrogate_*.csv /tmp/kaggle-output/exp046_hidden_branch_surrogate_audit/train_v1/artifacts/hidden_branch_surrogate_summary.json /tmp/kaggle-output/exp046_hidden_branch_surrogate_audit/train_v1/exp046-hidden-branch-surrogate-audit-train.log experiments/exp046_hidden_branch_surrogate_audit/artifacts/
cp /tmp/kaggle-output/exp046_hidden_branch_surrogate_audit/train_v1/metrics.json experiments/exp046_hidden_branch_surrogate_audit/metrics.json
```

## 変更点

- `config.yaml` を exp046 / 見えない test well 用処理の代理監査用に更新。
- `settings.py` の `EXPERIMENT_NAME` を exp046 に更新。
- train/inference notebook を exp046 名にリネーム。
- `hidden_branch_surrogate_audit.py` を追加。
  - `exp029` の train well の途中以降を隠した疑似 test PF/Beam 生成物を読む。
  - original-fold / well-hash / stratified-group fold で `exp026` anchor を fold-safe に再生成する。
  - 提出済みの見えない test well 用処理候補を同じ代理検証 rows に強制適用する。
  - overall / segment / well / diff/range metrics と summary JSON を保存する。
- train notebook は設定確認、入力 preview、監査実行、生成物確認の読める構成に更新。
- inference notebook は `NO_SUBMISSION.txt` だけを生成する audit-only guard に更新。

## 結果

- py_compile: PASS
- ruff: PASS
- validation: `scripts/validate_experiment.py --experiment exp046_hidden_branch_surrogate_audit` PASS
- tests: `tests/test_kaggle_notebooks.py tests/test_scaffold.py` は 10 passed
- local smoke: PASS
  - command: `--max-wells 5 --max-train-rows 300 --base-estimator HistGradientBoostingRegressor --output-dir /tmp/exp046_smoke`
  - rows / wells: 11,207 / 5
  - split systems: original-fold、well-hash、stratified-group fold
  - best by smoke audit:
    - original-fold: `public_pf_selector` 15.845874
    - well-hash: `public_pf_selector` 15.845874
    - stratified-group: `exp026_pseudo_tail_bucket_shrink` 15.119003
  - output files: `hidden_branch_surrogate_metrics.csv`、`hidden_branch_surrogate_segment_metrics.csv`、`hidden_branch_surrogate_diff_metrics.csv`、`hidden_branch_surrogate_well_metrics.csv`、`hidden_branch_surrogate_summary.json`
- prepare: Kaggle train package 生成 PASS
  - kernel id: `kentookumura/exp046-hidden-branch-surrogate-audit-train`
  - competition source: `rogii-wellbore-geology-prediction`
  - kernel source: `kentookumura/exp029-sel15-pf-oof-train`
  - GPU/internet: false / false
- prepared train notebook code cell AST: PASS
- `experiment_summary.md` に `running` として記録済み。
- Kaggle train v1: COMPLETE
  - kernel id: `kentookumura/exp046-hidden-branch-surrogate-audit-train`
  - runtime: notebook log time 約 1,315 秒
  - rows / wells: 1,782,279 / 773
  - output path: `/tmp/kaggle-output/exp046_hidden_branch_surrogate_audit/train_v1`
  - synced artifact files:
    - `hidden_branch_surrogate_metrics.csv`
    - `hidden_branch_surrogate_segment_metrics.csv`
    - `hidden_branch_surrogate_diff_metrics.csv`
    - `hidden_branch_surrogate_well_metrics.csv`
    - `hidden_branch_surrogate_exp026_source_summary.csv`
    - `hidden_branch_surrogate_well_metadata.csv`
    - `hidden_branch_surrogate_summary.json`
    - `exp046-hidden-branch-surrogate-audit-train.log`
- Full surrogate best:
  - original-fold: `exp035_ridge_meta_residual_shrink0p75_clip60p0` 14.313668
  - well-hash: `exp035_ridge_meta_residual_shrink0p75_clip60p0` 14.172010
  - stratified-group: `exp035_ridge_meta_residual_shrink0p75_clip60p0` 14.022803
- Full surrogate deltas for `exp035_ridge_meta_residual_shrink0p75_clip60p0`:
  - original-fold: vs exp026 -1.330065、vs public PF -0.858969、vs pf090_hold010 -0.775864
  - well-hash: vs exp026 -1.654092、vs public PF -1.000626、vs pf090_hold010 -0.917521
  - stratified-group: vs exp026 -1.449021、vs public PF -1.149833、vs pf090_hold010 -1.066728
- 既知の見えない test well 用処理 outcome:
  - `exp031_pf090_hold010_hidden_branch`: Public LB 8.956, exp027 から +0.175
  - `exp033_ridge_residual_shrink0p5_clip20p0`: Public LB 14.961, exp027 から +6.180
  - `exp035_ridge_meta_residual_shrink0p75_clip60p0`: Public LB 13.738, exp027 から +4.957
  - `exp045_strict_parity_meta_same_feature_surrogate`: Public LB 19.177, exp027 から +10.396
- 解釈: exp046 の代理検証では meta residual が一貫して最良だが、同じ候補は実 Public LB で悪化済み。代理監査は public sample の `changed_rows=0` blind spot を可視化する目的では有効だが、exp034/035-style の見えない test well 用 meta 処理の採用根拠にはならない。

## 次のアクション

1. exp027 anchor 8.781 を維持する。
2. exp034/035-style の見えない test well 用 meta 処理と PF residual branch の追加チューニングには進まない。
3. 次に public PF/Beam を使う場合は、直接残差補正ではなく `public_pf_beam_gate_only_audit` のような保守的な重み調整に限定する。
