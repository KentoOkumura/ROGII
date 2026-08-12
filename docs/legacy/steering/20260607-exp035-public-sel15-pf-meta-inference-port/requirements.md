# 要件

## 依頼

`public_sel15_pf_meta_inference_port` を実装する。`exp034_public_sel15_pf_meta_stack` で selected になった `ridge_meta_residual_shrink0p75_clip60p0` を、公開 sel15 inference flow に移植する。

## 制約

- Route: `pf_beam`
- 親実験は `exp034_public_sel15_pf_meta_stack` とする。
- 公開 sel15 replay の visible physical-model branch は変更しない。
- 見えない test well 用処理だけに exp026-style pseudo-tail anchor と Ridge meta residual 補正を適用する。
- meta model training は exp029 train well の途中以降を隠した疑似 test feature artifact を使う。
- 見えない test の評価区間 TVT は参照しない。
- 初回の正式実行は Kaggle Notebook で行う。

## 受け入れ基準

- `experiments/exp035_public_sel15_pf_meta_inference_port/` が作成されている。
- `config.yaml` に route、parent、selected candidate、feature columns、Kaggle kernel source が記録されている。
- inference notebook が `submission.csv` と audit artifacts を出力する。
- notebook code cell が構文検証に通る。
- `task validate-exp EXP=exp035_public_sel15_pf_meta_inference_port` が通る。
