# exp047_gate_w030_inference_port_audit セッションノート

## 目的

`exp047_public_pf_beam_gate_only_audit` で最良だった固定 `exp026_to_pf_gate_w0p30` を、public sel15 inference flow の見えない test well branch に移植し、提出前に output diff / range / hidden branch summary / submit-check を監査する。

## 現在の状態

- Route: pf_beam
- 状態: completed; Kaggle inference version 2 complete; UI code submission complete
- CV: 14.527279 (`exp026_to_pf_gate_w0p30`, parent exp047 original-fold surrogate)
- Public LB: 11.056
- 親: `exp047_public_pf_beam_gate_only_audit`
- 実装親: `exp045_public_pf_meta_strict_parity_audit`
- 変更: exp045 の見えない test well 用 Ridge meta residual branch を削除し、固定 gate `exp026_anchor + 0.30 * (pf_pred - exp026_anchor)` に置換
- fixed: exp026-style anchor、public visible physical branch、PF selector settings 16 seeds / 250 particles、distance bucket shrink
- Kaggle inference kernel id 予定: `kentookumura/exp047-gate-w030-infer`
- Kaggle inference kernel id: `kentookumura/exp047-gate-w030-infer`
- Kaggle version: 2
- output path: `/tmp/kaggle-output/exp047_gate_w030_inference_port_audit/inference_v2`
- artifact path: `experiments/exp047_gate_w030_inference_port_audit/artifacts/`
- submission sha256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
- competition submit: ref `53509425`
- submission status: COMPLETE

## コマンドログ

### 実行済み

```bash
uv run python scripts/new_steering.py --experiment exp047_gate_w030_inference_port_audit
uv run python scripts/new_experiment.py --name exp047_gate_w030_inference_port_audit --source experiments/exp045_public_pf_meta_strict_parity_audit
uv run python -m py_compile experiments/exp047_gate_w030_inference_port_audit/baseline.py experiments/exp047_gate_w030_inference_port_audit/pseudo_tail_augmentation.py experiments/exp047_gate_w030_inference_port_audit/settings.py
uv run ruff check experiments/exp047_gate_w030_inference_port_audit/baseline.py experiments/exp047_gate_w030_inference_port_audit/pseudo_tail_augmentation.py experiments/exp047_gate_w030_inference_port_audit/settings.py
uv run python scripts/validate_experiment.py --experiment exp047_gate_w030_inference_port_audit
uv run pytest tests/test_kaggle_notebooks.py tests/test_scaffold.py
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp047_gate_w030_inference_port_audit --notebook inference --kernel-id kentookumura/exp047-gate-w030-infer --title "exp047 gate w030 infer" --run-on-push --strict
uv run python -c "... prepared inference notebook code cells ast.parse ..."
uv run python scripts/record_experiment.py --experiment exp047_gate_w030_inference_port_audit --status running --cv 14.527279 --public-lb - --private-lb - --metric rmse --key-idea "Inference port audit for fixed exp047 exp026_to_pf_gate_w0p30; hidden wells only use exp026_anchor + 0.30*(public_pf_pred-exp026_anchor)." --notes "Implemented; AST/py_compile/ruff/validate/pytest PASS; Kaggle inference package prepared as kentookumura/exp047-gate-w030-infer; Kaggle run/output/submit-check pending"
kaggle kernels push -p experiments/exp047_gate_w030_inference_port_audit/kaggle/inference
kaggle kernels pull kentookumura/exp047-gate-w030-infer -p /tmp/kaggle-pull/exp047-gate-w030-infer -m
kaggle kernels logs kentookumura/exp047-gate-w030-infer
kaggle kernels logs -f --interval 5 kentookumura/exp047-gate-w030-infer
uv run python scripts/validate_experiment.py --experiment exp047_gate_w030_inference_port_audit
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp047_gate_w030_inference_port_audit --notebook inference --kernel-id kentookumura/exp047-gate-w030-infer --title "exp047 gate w030 infer" --run-on-push --strict
kaggle kernels push -p experiments/exp047_gate_w030_inference_port_audit/kaggle/inference
kaggle kernels pull kentookumura/exp047-gate-w030-infer -p /tmp/kaggle-pull/exp047-gate-w030-infer -m
kaggle kernels logs -f --interval 5 kentookumura/exp047-gate-w030-infer
kaggle kernels output kentookumura/exp047-gate-w030-infer -p /tmp/kaggle-output/exp047_gate_w030_inference_port_audit/inference_v2
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp047_gate_w030_inference_port_audit/inference_v2/submission.csv
make submit-check EXP=exp047_gate_w030_inference_port_audit SUBMISSION=/tmp/kaggle-output/exp047_gate_w030_inference_port_audit/inference_v2/submission.csv
sha256sum /tmp/kaggle-output/exp047_gate_w030_inference_port_audit/inference_v2/submission.csv /tmp/kaggle-output/exp047_gate_w030_inference_port_audit/inference_v2/public_sel15_gate_w030_summary.json
cp /tmp/kaggle-output/exp047_gate_w030_inference_port_audit/inference_v2/submission.csv /tmp/kaggle-output/exp047_gate_w030_inference_port_audit/inference_v2/public_sel15_exp026_anchor_submission.csv /tmp/kaggle-output/exp047_gate_w030_inference_port_audit/inference_v2/public_sel15_gate_w030_diff.csv /tmp/kaggle-output/exp047_gate_w030_inference_port_audit/inference_v2/public_sel15_gate_w030_summary.json /tmp/kaggle-output/exp047_gate_w030_inference_port_audit/inference_v2/exp047-gate-w030-infer.log experiments/exp047_gate_w030_inference_port_audit/artifacts/
kaggle competitions submissions rogii-wellbore-geology-prediction
uv run python scripts/record_submission.py --experiment exp047_gate_w030_inference_port_audit --file experiments/exp047_gate_w030_inference_port_audit/artifacts/submission.csv --cv 14.527279 --public-lb 11.056 --private-lb - --notes "ref=53509425; kernel=kentookumura/exp047-gate-w030-infer v2; UI code submit fixed exp026_to_pf_gate_w0p30 hidden branch; status=COMPLETE; Public LB worse than exp027 8.781 by +2.275 and worse than exp031 8.956 by +2.100; submit-check PASS; sample SHA identical"
uv run python scripts/record_experiment.py --experiment exp047_gate_w030_inference_port_audit --status completed --cv 14.527279 --public-lb 11.056 --private-lb - --metric rmse --key-idea "Fixed exp047 exp026_to_pf_gate_w0p30 hidden branch completed Kaggle inference and UI code submit; Public LB did not transfer to exp027 anchor." --notes "UI submit ref=53509425 COMPLETE; Public LB 11.056; worse than exp027 8.781 by +2.275 and worse than exp031 8.956 by +2.100; sample output SHA matched exp027; stop public PF gate inference ports"
```

## 変更点

- `config.yaml` を固定 gate inference port audit 用に更新。
- `settings.py` の `EXPERIMENT_NAME` を exp047_gate_w030 に更新。
- train/inference notebook を exp047_gate_w030 名にリネーム。
- inference notebook は exp045 の meta residual 学習/適用を使わず、見えない test well のみ固定 `w=0.30` gate を適用する構成に更新。
- inference notebook は `public_sel15_exp026_anchor_submission.csv`、`public_sel15_gate_w030_diff.csv`、`public_sel15_gate_w030_summary.json`、`submission.csv` を保存する。
- Kaggle inference package を `experiments/exp047_gate_w030_inference_port_audit/kaggle/inference` に生成。

## 結果

- notebook code cell AST: source train/inference PASS、prepared inference PASS。
- py_compile: PASS。
- ruff: PASS。
- validation: `scripts/validate_experiment.py --experiment exp047_gate_w030_inference_port_audit` PASS。
- tests: `tests/test_kaggle_notebooks.py tests/test_scaffold.py` は 10 passed。
- prepare: Kaggle inference package 生成 PASS。
  - kernel id: `kentookumura/exp047-gate-w030-infer`
  - competition source: `rogii-wellbore-geology-prediction`
  - kernel source: `kentookumura/exp029-sel15-pf-oof-train`
  - GPU/internet: false / false
- Kaggle inference v2:
  - push: success、kernel version 2
  - pull: success、Kaggle 側の存在確認済み
  - logs: `logs -f` で完了ログを確認
  - output: `/tmp/kaggle-output/exp047_gate_w030_inference_port_audit/inference_v2`
  - runtime: notebook log time 約 122 秒
  - exp026-style anchor fit: train wells 773、train rows 242,843、source rows 788
  - output rows: 14,151
  - visible sample wells: 3 wells all `physical_visible`
  - hidden_rows / hidden_wells: 0 / 0
  - changed_rows / changed_wells: 0 / 0
  - diff_rmse: 0.000000
  - prediction range: 11587.038593 - 12240.016066
  - PF settings: 16 seeds / 250 particles、selector scales 3/5/8/12
  - submit-check: PASS (`validate_submission.py` and `make submit-check`)
  - submission SHA256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
  - summary SHA256: `37a726ed5bad59acc715116439fb88a2ee16721f3a23bc4e1af7370e35328a54`
  - synced artifact files:
    - `submission.csv`
    - `public_sel15_exp026_anchor_submission.csv`
    - `public_sel15_gate_w030_diff.csv`
    - `public_sel15_gate_w030_summary.json`
    - `exp047-gate-w030-infer.log`
- Code submit:
  - method: Kaggle UI
  - ref: `53509425`
  - status: COMPLETE
  - Public LB: 11.056
  - delta vs exp027 8.781: +2.275 worse
  - delta vs exp031 8.956: +2.100 worse
  - conclusion: fixed `exp026_to_pf_gate_w0p30` hidden branch did not transfer; stop public PF gate inference ports.
- Kaggle inference v1:
  - push: success、kernel version 1
  - pull: success、Kaggle 側の存在確認済み
  - logs: push 直後の通常 logs は空、`logs -f` で失敗を確認
  - failure: `IndexError: list index out of range` at `pseudo_tail_augmentation.bucket_labels`
  - cause: `config.yaml` に `audit.distance_buckets` がなく、exp026 anchor 学習の distance balancing が空 bucket list を参照した。
  - fix: `audit.distance_buckets` を 5 bucket で復元し、validation と Kaggle inference package 再生成 PASS。

## 次のアクション

1. exp027 anchor 8.781 を維持する。
2. public PF gate inference port は停止する。固定 `w=0.30` は exp031 より保守的でも Public LB 11.056 と大きく悪化した。
3. 次は PF/Beam hidden branch ではなく、ML route の XGBoost / LGBM micro tune / seed bagging へ戻る。
