# exp157_candidate_ranker_feature_enrichment

## 目的

`candidate_ranker_feature_enrichment` backlog の実験。`exp101_pf_candidate_ranker_or_nway_classifier` の supervised candidate selector を親にし、候補集合と特徴量を `tvt_dense` family で拡張する。

## 状態

実装済み、未実行。Kaggle train push 前の静的検証中。

## 仮説

exp101 は `pf_ancc` を選択できるようになったが、`likpf_mean` 単体を超えなかった。`tvt_dense` family と dense disagreement / drift / continuity 系特徴を加えることで、候補選択の oracle gap を少しでも縮められる可能性がある。

## 変更点

- 既存候補: `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb`
- 追加候補: `tvt_dense`、`tvt_densew`、`tvt_dense50`
- 追加特徴: dense drift、dense family dispersion、PF/Beam/likPF-vs-dense 差、tail / near-row flag、high-disagreement proxy
- 入力: exp099 v2 multiobs cache と exp072 full replay feature cache

## 検証方針

GroupKFold by well の train-side pseudo-tail OOF で、`likpf_mean_single`、`multiobs_score_top1`、oracle、3種類の LightGBM selector を比較する。RMSE、within10、oracle label accuracy、selection distribution、path switch、bucket metrics、feature importance を見る。

## 実行方針

Kaggle CPU train notebook で実行する。GPU は使わない。LightGBM は exp101 と同じ 3 family x 5 folds = 15 boosters。exp101 control を別途再学習せず、保存済み exp101 / likPF metrics を比較基準にする。

## 所見

未実行のためスコア所見はない。exp099 cache に `tvt_dense*` が無いことを確認済みなので、exp072 full replay cache を補助 source として使う。

## 非対象

- direct TVT regressor
- candidate の soft average / blend
- hard replacement submission
- true validation TVT、oracle label、true error rank を feature に使う処理
