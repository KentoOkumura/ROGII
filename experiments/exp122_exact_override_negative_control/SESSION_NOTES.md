# exp122_exact_override_negative_control セッションノート

## 目的

Pilkwang replay に含まれる exact-match recovery / guarded overlap override を、hidden-safe な改善根拠から除外するための negative control を実装する。

## 現在の状態

- Route: pf_beam
- 状態: Kaggle train v1 完了
- CV: まだなし
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 実行済み

```bash
uv run python scripts/new_steering.py --experiment exp122_exact_override_negative_control
uv run python scripts/new_experiment.py --name exp122_exact_override_negative_control --source templates/experiment
python3 -m py_compile experiments/exp122_exact_override_negative_control/exact_override_negative_control.py
uv run ruff check experiments/exp122_exact_override_negative_control/exact_override_negative_control.py
uv run python scripts/validate_experiment.py --experiment exp122_exact_override_negative_control
python3 experiments/exp122_exact_override_negative_control/exact_override_negative_control.py --summary
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp122_exact_override_negative_control --notebook train --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp122_exact_override_negative_control --notebook inference --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp122_exact_override_negative_control --notebook train --run-on-push --strict --title 'exp122 exact override negative control train'
kaggle kernels push -p experiments/exp122_exact_override_negative_control/kaggle/train
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp122-exact-override-negative-control-train
kaggle kernels output kentookumura/exp122-exact-override-negative-control-train -p experiments/exp122_exact_override_negative_control/kaggle/output/train_v1
```

### 予定なし

Kaggle train v1 の output は取得済み。

## 変更点

- `config.yaml` を pf_beam route の diagnostic audit に更新した。
- `exact_override_negative_control.py` を追加した。
- train / inference notebook を、入力証拠確認、audit 実行、metrics / artifacts 確認の構成に差し替えた。
- Kaggle 実行時にも証拠が欠落しないよう、`evidence/` に Pilkwang notebook、exp079 summary / submission summary / pairwise、exp064 metrics を snapshot として同梱した。

## ローカル smoke 結果

- status: `negative_control_passed_current_evidence`
- adoption: `exclude_same_well_exact_or_guarded_override`
- confidence: `medium`
- `final_equals_base`: true
- `hidden_assertion_not_triggered`: true
- `guard_changed`: false
- `guard_rows`: 0
- caveat: archived notebook source には same-well shortcut flags が enabled として見える一方、exp079 source spec は exact/override disabled check を期待している。この矛盾は改善根拠ではなく risk として記録する。
- summary: `artifacts/exact_override_negative_control_summary.json`
- notebook risk summary: `artifacts/notebook_risk_summary.csv`
- guard output inventory: `artifacts/guard_output_inventory.csv`

## Kaggle train v1 結果

- Kernel: `kentookumura/exp122-exact-override-negative-control-train`
- Version: 1
- URL: https://www.kaggle.com/code/kentookumura/exp122-exact-override-negative-control-train
- Output: `experiments/exp122_exact_override_negative_control/kaggle/output/train_v1`
- status: `negative_control_passed_current_evidence`
- adoption: `exclude_same_well_exact_or_guarded_override`
- confidence: `medium`
- `final_equals_base`: true
- `hidden_assertion_not_triggered`: true
- `guard_changed`: false
- `guard_rows`: 0
- summary: `experiments/exp122_exact_override_negative_control/kaggle/output/train_v1/artifacts/exact_override_negative_control_summary.json`
- metrics: `experiments/exp122_exact_override_negative_control/kaggle/output/train_v1/metrics.json`
- note: 初回 push は title slug mismatch で SaveKernel 400。title を `exp122 exact override negative control train` に短縮して同じ canonical kernel id へ v1 push した。

## 再現性メモ

- seed policy: `none_deterministic_file_audit`
- stochastic components: なし
- CPU/GPU runtime: CPU file audit。GPU 不要
- Kaggle kernel id / version: `kentookumura/exp122-exact-override-negative-control-train` v1
- input / feature schema SHA: notebook、exp079 summary、exp064 metrics、guard output CSV の SHA を audit summary に記録する
- feature content SHA: feature は生成しない
- model manifest / model SHA: model は生成しない
- prediction SHA: before/after submission が見つかった場合だけ記録する
- submission SHA: submission は生成しない
- rerun check: Kaggle train v1 でローカル smoke と同じ decision を確認

## 次のアクション

1. 完了。追加提出や inference run は不要。
