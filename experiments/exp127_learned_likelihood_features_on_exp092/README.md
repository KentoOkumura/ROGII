# exp127_learned_likelihood_features_on_exp092

## 状態

Kaggle train v1 完了。提出なし。

## 仮説

exp112 の learned likelihood は hard selector / PF weight としては危険だが、target-free confidence signal として exp092 系 LightGBM に add-only で渡すと、exp092 の U-projection correction / disagreement surface では捉えきれない候補信頼度を補える可能性がある。

## 検証方針

exp092 と同じ exp072 full replay 196 features、U-projection correction、U-disagreement features、LightGBM config family、`TVT - last_known_tvt` target を固定する。

exp112 feature cache は 155 wells subset のため、exp072/exp092 surface を exp112 feature が存在する shared rows に inner join し、同じ shared rows で次を比較する。

- `exp092_shared_row_control`: exp092 相当の feature surface を shared rows だけで再学習。
- `learned_likelihood_confidence_addonly`: control に exp112 の probability、predicted-error、margin、entropy、candidate disagreement、weighted TVT proxy features を追加。

## 所見

shared rows 757,738 rows / 155 wells で、`learned_likelihood_confidence_addonly` は `exp092_shared_row_control` を全 pooled model で改善した。`lgb_mean` は 9.847052694 -> 9.727317518、delta -0.119735177。distance bucket も全 bucket で改善した。

ただし exp112 feature cache が存在する subset 評価であり、exp092 full CV や hidden-like split の直接証拠ではない。worst-well regression も最大 +1.071012 残るため、direct inference port / submit はしない。次に使う場合は exp115 hidden-like stress、raw-test/full-train feature parity、worst-well guard を確認する。

## 生成物

- `exp127_learned_likelihood_features_on_exp092_metrics.csv`
- `exp127_learned_likelihood_features_on_exp092_by_well.csv`
- `exp127_learned_likelihood_features_on_exp092_bucket_metrics.csv`
- `exp127_learned_likelihood_features_on_exp092_projection_feature_summary.csv`
- `exp127_learned_likelihood_features_on_exp092_learned_feature_summary.csv`
- `exp127_learned_likelihood_features_on_exp092_feature_importance.csv`
- `exp127_learned_likelihood_features_on_exp092_feature_importance_mean.csv`
- `exp127_learned_likelihood_features_on_exp092_feature_importance_mean_top.png`
- `exp127_learned_likelihood_features_on_exp092_predictions.csv.gz`
- `exp127_learned_likelihood_features_on_exp092_feature_schema.csv`
- `exp127_learned_likelihood_features_on_exp092_lgb_models/manifest.json`
- `exp127_learned_likelihood_features_on_exp092_summary.json`
