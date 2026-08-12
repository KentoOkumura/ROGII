# 要件

## 依頼

`exp047_gate_w030_inference_port_audit` を実装する。`exp047_public_pf_beam_gate_only_audit` で最良だった固定 `exp026_to_pf_gate_w0p30` を public sel15 inference flow の見えない test well branch に移植し、提出前に output diff と submit-check を監査できる状態にする。

## 制約

- Route: `pf_beam`
- learned gate は使わない。
- Ridge meta residual / exp034-style meta stack は使わない。
- visible public sample wells は既存の physical branch を維持する。
- hidden / unseen test well のみ `exp026_anchor + 0.30 * (public_pf_pred - exp026_anchor)` を適用する。
- Kaggle Notebook 実行を正とし、ローカル notebook 実行はしない。

## 受け入れ基準

- config / settings / notebook が `exp047_gate_w030_inference_port_audit` 名で整合する。
- inference notebook が `submission.csv`、`public_sel15_exp026_anchor_submission.csv`、`public_sel15_gate_w030_diff.csv`、`public_sel15_gate_w030_summary.json` を出力する。
- summary に gate weight、hidden PF settings、changed rows / wells、prediction range、diff stats、exp026 anchor info が記録される。
- `scripts/validate_experiment.py` と notebook code cell AST check が通る。
- Kaggle inference package を生成できる。
