# 要件

## 依頼

`public_pf_meta_strict_parity_audit` を `exp045_public_pf_meta_strict_parity_audit` として実装する。`exp035` の見えない test well 用 meta 処理を最小コピーし、見えない test well 推論側の PF diagnostics を exp029 の軽量 public sel15 feature 定義に揃えて、exp034 の疑似 test 条件と exp035 の本番採点条件の feature 分布不一致が meta 失敗原因かを切り分ける。

## 制約

- Route: `pf_beam`
- visible public sample wells の physical branch は変更しない。
- meta training artifact は exp035 と同じ exp029 `public_sel15_pf_oof_features.csv.gz` を使う。
- meta model は exp034 selected `ridge_meta_residual_shrink0p75_clip60p0` のまま固定する。
- hidden diagnostic PF は `16 seeds / 250 particles`、selector scales `[3, 5, 8, 12]`、14 beam configs に揃える。
- 目的は提出候補化ではなく監査。Public LB は exp035 13.738 と exp027 8.781 の比較材料として扱う。

## 受け入れ基準

- `exp045_public_pf_meta_strict_parity_audit` の実験フォルダ、config、notebook、記録ファイルが存在する。
- inference notebook が exp045 名で prepare 可能で、見えない test well 用 PF 診断値 の seeds/particles が config 由来で summary に記録される。
- `scripts/validate_experiment.py` が PASS する。
- notebook code cell が Python AST parse できる。
