# exp148 feature correlation audit run manifest

このディレクトリには、exp148の294特徴量を対象にしたKaggle実行のmetadataを置きます。人間向けの結果、削除候補、注意点は、[exp148・exp092特徴量置換監査](../../docs/surveys/exp148_exp092_feature_replacement_audit_20260704.md)を正とします。

## 実行条件

- Kaggle kernel: `kentookumura/exp148-feature-correlation-audit` v2
- 入力: exp072 full replay train feature cache、exp145 full-train learned likelihood feature cache、exp148 train feature schema / feature importance
- 対象: 全3,783,989行からseed 148で600,000行をuniform sample
- projection featuresは全行で構築してからsample
- kernel metadata: `kernel-metadata.json`
- 取得時の出力先: `/tmp/kaggle-output/exp148_feature_correlation_audit_v2/`

一時出力の表を再取得した場合も、結論をこのREADMEへ複製せずsurveyを更新します。
