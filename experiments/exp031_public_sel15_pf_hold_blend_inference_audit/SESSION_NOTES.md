# exp031_public_sel15_pf_hold_blend_inference_audit セッションノート

## 目的

`exp030` で fold 外でも支持があった fixed `pf090_hold010` を、`exp027` の公開 sel15 inference flow に移植し、提出前に差分監査できる状態にする。

## 現在の状態

- Route: pf_beam
- 状態: completed; code submission scored; not adopted
- CV: なし
- LB: 8.956
- 親: `exp027_public_replay_needless090_sel15_spread3`
- 変更: 見えない test well の `tvt_selector` を `0.90 * tvt_selector + 0.10 * last_known_TVT_input` に変更
- Kaggle kernel: `kentookumura/exp031-public-sel15-pf-hold-blend-inference-audit` version 1
- output path: `/tmp/kaggle-output/exp031_public_sel15_pf_hold_blend_inference_audit/inference`
- artifact path: `experiments/exp031_public_sel15_pf_hold_blend_inference_audit/artifacts/`
- submission sha256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
- submit-check: PASS
- competition submit: ref `53443300`
- Public LB: 8.956

## コマンドログ

### 実行済み

```bash
uv run python scripts/new_steering.py --experiment exp031_public_sel15_pf_hold_blend_inference_audit
uv run python scripts/new_experiment.py --name exp031_public_sel15_pf_hold_blend_inference_audit --source experiments/exp027_public_replay_needless090_sel15_spread3
uv run python scripts/validate_experiment.py --experiment exp031_public_sel15_pf_hold_blend_inference_audit
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp031_public_sel15_pf_hold_blend_inference_audit --notebook inference --kernel-id kentookumura/exp031-public-sel15-pf-hold-blend-inference-audit --run-on-push --title 'exp031 public sel15 pf hold blend inference audit' --strict
kaggle kernels push -p experiments/exp031_public_sel15_pf_hold_blend_inference_audit/kaggle/inference
kaggle kernels pull kentookumura/exp031-public-sel15-pf-hold-blend-inference-audit -p /tmp/kaggle-pull/exp031-public-sel15-pf-hold-blend-inference-audit -m
kaggle kernels logs kentookumura/exp031-public-sel15-pf-hold-blend-inference-audit
kaggle kernels output kentookumura/exp031-public-sel15-pf-hold-blend-inference-audit -p /tmp/kaggle-output/exp031_public_sel15_pf_hold_blend_inference_audit/inference
.venv/bin/python scripts/validate_submission.py --submission /tmp/kaggle-output/exp031_public_sel15_pf_hold_blend_inference_audit/inference/submission.csv
kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp031-public-sel15-pf-hold-blend-inference-audit -v 1 -f submission.csv -m "exp031 pf090 hold010 hidden branch audit"
kaggle competitions submissions rogii-wellbore-geology-prediction
uv run python scripts/record_submission.py --experiment exp031_public_sel15_pf_hold_blend_inference_audit --file experiments/exp031_public_sel15_pf_hold_blend_inference_audit/artifacts/submission.csv --cv - --public-lb - --private-lb - --notes "ref=53443300; kernel=kentookumura/exp031-public-sel15-pf-hold-blend-inference-audit v1; code-submit hidden pf090_hold010 branch audit; status=PENDING; public sample identical to exp027 SHA; submit-check PASS"
```

### 予定

```bash
kaggle competitions submissions rogii-wellbore-geology-prediction
```

## 変更点

- `config.yaml` を exp031 / pf_beam / fixed `pf090_hold010` inference audit 用に更新。
- `settings.py` の `EXPERIMENT_NAME` を exp031 に更新。
- train/inference notebook を exp031 名にリネーム。
- inference notebook 末尾で 見えない test well に fixed `pf090_hold010` を適用。
- 監査 artifact として元 selector submission、row-level diff、summary JSON を出力。
- Kaggle 用 inference notebook を `experiments/exp031_public_sel15_pf_hold_blend_inference_audit/kaggle/inference/` に生成。

## 結果

- 構文チェック: train / inference notebook の code cell は `ast.parse` 通過。
- validation: `uv run python scripts/validate_experiment.py --experiment exp031_public_sel15_pf_hold_blend_inference_audit` 通過。
- prepare: `uv run python scripts/prepare_kaggle_notebooks.py ... --strict` 通過。
- push: `kaggle kernels push -p experiments/exp031_public_sel15_pf_hold_blend_inference_audit/kaggle/inference` が成功し、version 1 を push。進捗 URL: https://www.kaggle.com/code/kentookumura/exp031-public-sel15-pf-hold-blend-inference-audit
- Kaggle log: 約 191 秒で `submission.csv` と監査 artifact を生成。
- output rows: 14,151
- prediction range: 11587.038593 - 12240.016066
- prediction mean: 11903.630073
- missing values: 0
- duplicate IDs: 0
- submit-check: PASS
- exp027 submission との差分: min 0.000000、max 0.000000、mean 0.000000、abs mean 0.000000、RMSE 0.000000、corr 1.000000
- output SHA256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
- 監査 summary: changed_rows 0、changed_wells 0、diff_rmse 0.000000
- code submit: ref `53443300`、Public LB 8.956。
- exp027 Public LB 8.781 より +0.175 悪化したため、fixed `pf090_hold010` の見えない test well 用処理は採用しない。
- `exp030` の supporting diagnostic:
  - raw public PF selector: 15.172636
  - fixed `pf090_hold010` same-OOF: 15.089532
  - original-fold selection: 15.141132
  - well-hash selection: 15.131490

## 次のアクション

1. exp027 Public LB 8.781 を anchor として維持する。
2. fixed hidden `pf090_hold010` は採用せず、train-side residual/meta-stack validation に戻る。
