# Feature replacement audit inputs and outputs

このディレクトリには、exp148・exp092系の特徴量重複監査を再実行するコードと、その生の出力を置きます。人間向けの結論と判断は、[exp148・exp092特徴量置換監査](../../docs/surveys/exp148_exp092_feature_replacement_audit_20260704.md)を正とします。

## 再実行

保存済みのfeature correlation、train/inference schema、feature importance、raw-test schemaを読み直すno-training readoutです。Kaggle GPU学習、推論、提出は行いません。

```bash
uv run python studies/feature_replacement_audit/corr_prune_sanity_readout.py
```

## 生の出力

`outputs/corr_prune_sanity_readout_on_exp148/`に、候補表、設定断片、コード参照、schema差分、summary JSONを保存します。各ファイルの解釈と後続実験への制約はsurveyを参照してください。
