# exp202_heatmap_mdn_candidate_generator_probe

## 目的

exp182/184 の heatmap MTP 系を、PF/Beam 候補の direct replacement ではなく、候補生成器として評価する。5ch heatmap CNN/MTP が出す topK TVT/path 候補を既存 PF/Beam/likPF/sc/hyb 候補 union に追加し、oracle coverage と selector headroom が増えるかを train-side で確認する。

## 仮説

exp182 の heatmap encoder と MTP head は real GR signal を拾えているため、既存 PF/Beam/likPF/sc/hyb 候補が外す row に対して別 mode の TVT/path 候補を出せる可能性がある。topK 候補を union に追加して oracle RMSE / within10 が改善するなら、後続の selector や confidence feature に進める価値がある。

## 方針

- Route: `pf_beam`
- 入力: exp182 と同じ 5ch heatmap
- Head: `K=10` path/logit head、closest-mode loss
- 学習: `candidate_real_w128_b64_fullfold` 1 spec x 5 folds = 5 CNN models
- 比較: exp099 の `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb` を既存 union とする
- dense family は cache に存在すれば評価し、なければ missing candidate として記録する

## 範囲外

direct TVT replacement、softmax weighted average、PF weight replacement、postprocess blend、inference port、submit はしない。positive の場合だけ、後続で `heatmap_mdn_candidates_into_exp158_selector` または `heatmap_mdn_confidence_features_on_exp148` に分ける。

## 生成物

- `*_heatmap_candidates.csv.gz`
- `*_candidate_union_metrics.csv`
- `*_candidate_union_by_well.csv`
- `*_candidate_union_distance_bucket_metrics.csv`
- `*_validation_predictions.csv.gz`
- `*_model_manifest.json`
- `*_summary.json`

## 検証方針

Kaggle T4 train-side で 5-fold heatmap candidate generator を学習し、valid pseudo-tail rows で heatmap topK candidate を保存する。同じ `id` で exp099 train-side candidate cache に join し、既存 union と `existing_plus_heatmap_topK` の oracle RMSE、within10、new-best candidate rate を overall / by-well / distance bucket で比較する。

## 所見

未実行。実装と静的検証のみ完了。

## 状態

実装済み、Kaggle train 未実行。push 前に GPU コスト確認が必要。
