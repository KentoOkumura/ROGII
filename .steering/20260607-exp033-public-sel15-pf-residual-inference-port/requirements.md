# 要件

## 依頼

`KAGGLE_DIRECTION.md` の高優先 backlog `public_sel15_pf_residual_inference_port` を `exp033_public_sel15_pf_residual_inference_port` として実装する。`exp032` で supported になった `ridge_residual_shrink0p5_clip20p0` だけを、公開 sel15 PF/Beam inference flow に移植する。

## 制約

- Route: `pf_beam`
- 親実験は `exp031_public_sel15_pf_hold_blend_inference_audit` とし、公開 sel15 replay の PF/Beam/selector logic は維持する。
- residual model training は `exp029` の train well の途中以降を隠した疑似 test train feature artifact を使う。
- 見えない test well の evaluation zone 真値、未来 `TVT_input`、train-only formation columns を使わない。
- visible public sample wells の physical-model branch は変更しない。
- 追加チューニングはしない。移植対象は `ridge_residual_shrink0p5_clip20p0` に限定する。
- 提出前に Kaggle output diff、prediction range、changed wells、submit-check を記録する。

## 受け入れ基準

- `experiments/exp033_public_sel15_pf_residual_inference_port/` に config、settings、train/inference notebook、記録ファイルがある。
- inference notebook が exp029 feature artifact から Ridge residual model を fit し、見えない test well branch だけ `tvt_selector + 0.5 * clip(residual, -20, 20)` を適用する。
- notebook が `submission.csv`、original selector submission、row-level diff CSV、summary JSON を保存する。
- `task validate-exp EXP=exp033_public_sel15_pf_residual_inference_port` が通る。
