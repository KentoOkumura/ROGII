# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ先頭にある `distance_weighted_training_audit` を `exp020_distance_weighted_training_audit` として実装する。

## 制約

- 親実験は raw anchor の `exp013_model_diversity_or_postprocess` とする。
- `exp013` の `lightgbm_no_gr` OOF を clean CV anchor として扱い、same-OOF postprocess score と混同しない。
- 学習・検証は well 単位 GroupKFold を維持する。
- `TVT_input` の evaluation zone 真値、train-only formation columns、validation well の疑似情報を特徴に混ぜない。
- ローカル notebook 実行はしない。authoritative run は Kaggle train notebook で行う。
- 実験ディレクトリに 1GB 級の row OOF artifact は常設しない。

## 受け入れ基準

- `exp020_distance_weighted_training_audit` に config、settings、train/inference notebook、監査スクリプト、docs が揃っている。
- `exp013` OOF から raw / `last_anchor` / `recent_linear` / exp014 bucket shrink の距離 bucket 別 RMSE を出せる。
- raw residual の bias、error std、target residual std を距離 bucket 別に artifact 化できる。
- LightGBM no-GR control、near downweight、far upweight、near+far、near/mid/far segmented model を同一 fold で比較できる。
- `task validate-exp` と Kaggle train package generation が通る。
