# exp047_gate_w030_inference_port_audit

## 状態

- ルート: pf_beam
- 状態: completed
- CV: 14.527279 (parent exp047 surrogate original-fold)
- Public LB: 11.056
- Private LB: -
- Submit ID: 53509425
- 作成日: 2026-06-09
- 親実験: `exp047_public_pf_beam_gate_only_audit`
- 実装親: `exp045_public_pf_meta_strict_parity_audit`

## 仮説

`exp047_public_pf_beam_gate_only_audit` で最良だった固定 `exp026_to_pf_gate_w0p30` を、見えない test well 用の inference branch にだけ移植する。learned gate や meta residual は使わず、`exp026_anchor + 0.30 * (public_pf_pred - exp026_anchor)` に限定する。

## 検証方針

Kaggle inference notebook で `submission.csv`、exp026 anchor submission、gate diff、summary を生成し、submit-check と output sanity を確認する。public sample は visible train well branch のため changed rows が 0 になり得る。

## 所見

Kaggle inference version 2 が完了し、submit-check は PASS。public sample は visible branch のみで hidden gate は発火せず、submission SHA は exp027 系と同一。UI code submit ref `53509425` の Public LB は 11.056 で、exp027 8.781 から悪化した。

## 参照ファイル

- 設定: `config.yaml`
- セッションノート: `SESSION_NOTES.md`
- 結果: `result.md`
- メトリクス: `metrics.json`
- 学習 notebook: `exp047_gate_w030_inference_port_audit_train.ipynb`
- 推論 notebook: `exp047_gate_w030_inference_port_audit_inference.ipynb`

## 読み方

仮説、変更点、実行コマンド、出力、次のアクションは `SESSION_NOTES.md` と `result.md` を正とする。
