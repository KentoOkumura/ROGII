# exp047_public_pf_beam_gate_only_audit

## 状態

- ルート: pf_beam
- 状態: completed
- CV: 14.527279
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-09
- 親実験: exp046_hidden_branch_surrogate_audit

## 仮説

exp031/033/035/045 の見えない test well 用処理は、PF/Beam を直接置換や残差補正として使うと Public LB に転移しなかった。PF/Beam の値を直接当てにせず、`exp026` anchor から PF/Beam 方向へ小さく動かす重み `w` だけを学習すれば、破壊的な外れを抑えつつ PF/Beam が明確に勝つ行だけを拾える可能性がある。

## 検証方針

`exp046` と同じ train-side surrogate surface を使う。`exp029` の public sel15 PF/Beam 生成物を読み、original-fold、well-hash、stratified-group fold の各 split で `exp026` anchor を fold-safe に再生成する。候補は必ず `base + w * (candidate - base)` の形に限定し、固定 gate と learned gate の `w` を 0.2-0.4 以下に clip する。

## 所見

Kaggle train version 1 で full audit 完了。固定 `exp026_to_pf_gate_w0p30` が original-fold 14.527279、well-hash 14.620835、stratified-group 14.353489 で全 split system の最良だった。提出なし。

## 参照ファイル

- 設定: `config.yaml`
- 監査スクリプト: `gate_only_audit.py`
- セッションノート: `SESSION_NOTES.md`
- 結果: `result.md`
- メトリクス: `metrics.json`
- 学習 notebook: `exp047_public_pf_beam_gate_only_audit_train.ipynb`
- 推論 notebook: `exp047_public_pf_beam_gate_only_audit_inference.ipynb`

## 読み方

この README は実験フォルダの入口です。仮説、変更点、実行コマンド、出力、失敗理由、次のアクションは `SESSION_NOTES.md` と `result.md` を正とします。
