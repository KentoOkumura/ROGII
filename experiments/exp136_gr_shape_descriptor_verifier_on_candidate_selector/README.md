# exp136_gr_shape_descriptor_verifier_on_candidate_selector

## 状態

Kaggle train v2 完了。提出なし。

## 仮説

exp131 では GR shape descriptor に candidate-long AUC signal はあったが、score argmax は `pf_ancc` 偏重で崩壊した。そこで descriptor を selector 本体ではなく、exp101/exp102 系の低頻度 candidate switch を承認または veto する verifier として使う。

## 検証方針

exp099 v2 cache と exp101 saved booster から OOF score surface を復元する。raw train GR と visible `TVT_input` prefix から exp131 相当の descriptor score を再計算し、`likpf_mean` default のまま、`pf_ancc` / `beam_mean` への切替だけを低頻度に評価する。

## 判定

`likpf_mean_single`、`exp101_error_ranker_rowwise`、`oracle`、descriptor verifier variants を比較する。主指標は RMSE / within10 / switch rate。採用判断には path switch、near-row、1000+ longtail、worst-well regression、bucket metrics も使う。

この実験では inference port や `submission.csv` は作らない。

## 所見

best RMSE gate は `likpf_mean` から RMSE -0.009782 と小改善したが、within10 は -0.000063 悪化し、最大 well regression +3.542 RMSE が残った。RMSE と within10 を両方改善する gate もあるが改善幅は小さい。direct inference port / submit はせず、descriptor は diagnostic / ML add-only confidence feature 材料に留める。

## 主な生成物

- `exp136_gr_shape_descriptor_verifier_on_candidate_selector_metrics.csv`
- `exp136_gr_shape_descriptor_verifier_on_candidate_selector_oof_predictions.csv.gz`
- `exp136_gr_shape_descriptor_verifier_on_candidate_selector_selection_distribution.csv`
- `exp136_gr_shape_descriptor_verifier_on_candidate_selector_by_well.csv`
- `exp136_gr_shape_descriptor_verifier_on_candidate_selector_bucket_metrics.csv`
- `exp136_gr_shape_descriptor_verifier_on_candidate_selector_score_summary.csv`
- `exp136_gr_shape_descriptor_verifier_on_candidate_selector_descriptor_well_summary.csv`
- `exp136_gr_shape_descriptor_verifier_on_candidate_selector_summary.json`
