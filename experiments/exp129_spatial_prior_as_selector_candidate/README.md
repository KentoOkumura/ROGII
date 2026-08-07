# exp129_spatial_prior_as_selector_candidate

## 状態

Kaggle train v1 完了。不採用。

## 仮説

exp114/118 で spatial prior は hard correction としての改善幅は小さいが、信頼できる regime では候補 path として選べる可能性がある。exp099/101 の PF/Beam candidate surface に spatial prior TVT を追加し、oracle/topK headroom と Viterbi-smoothed selector で価値を確認する。

## 検証方針

exp099 v2 cache と exp114 fold-safe OOF spatial prior を固定入力にする。base 5候補に `xy_plus_trajectory_shape_k8_prior_tvt` と `xy_only_k8_prior_tvt` を追加し、expanded oracle、spatial oracle selection rate、true-error topK、predicted-error ranker、Viterbi switch penalty grid、by-well continuity、bucket metrics を見る。

## 所見

expanded oracle は RMSE 6.709127 で、base-only oracle 7.434030 から -0.724903 改善した。spatial 候補は oracle top1 で合計 21.53% 選ばれ、headroom はある。

一方で best OOF selector は `lgb_error_ranker_rowwise` の RMSE 13.793157 で、`likpf_mean_single` 11.594898 より +2.198259 悪い。Viterbi smoothing は path switch を下げるが RMSE は改善せず、direct selector としては不採用。

## 注意

この実験は train-side audit 専用で、推論化や提出は行わない。spatial prior は candidate path selector ではなく、exp092 系 ML の confidence / add-only feature 側に戻す。
