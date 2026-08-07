# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ先頭にある
`public_postprocess_ablation` を実装する。

公開 notebook 系の後処理候補のうち、現時点で exp013 の OOF artifact
だけから再現できる候補を同じ評価面で切り分ける。

## 制約

- 親実験は `exp013_model_diversity_or_postprocess`。
- 新しいモデル学習は行わず、`exp013` の `lightgbm_no_gr` OOF を読み込む。
- evaluation-zone の `TVT` は候補評価と fold 外 selection audit にだけ使う。
- 同一 OOF 上の単純 best score と、fold 外 selection score を分けて記録する。
- PF / beam の `w_pf` は exp013 OOF に候補予測がないため、この実験では対象外にする。

## 受け入れ基準

- raw LightGBM no-GR、last anchor、SG smoothing、prediction-start fade-in、
  hold-last-known blend、alpha/tau shrink、exp013 bucket shrink を比較できる。
- original fold holdout と stable well-hash holdout の selection audit を出力する。
- distance bucket 別 RMSE を artifact として残す。
- `config.yaml`、`SESSION_NOTES.md`、`result.md`、`metrics.json` に結果を記録する。
