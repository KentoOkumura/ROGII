# exp101_pf_candidate_ranker_or_nway_classifier

## 状態

Kaggle train v1 完了。不採用。

## 仮説

PF/Beam/likelihood-PF の 5 候補から、train pseudo-tail の oracle best candidate index を教師にして supervised selector を学習する実験。

入力は exp099 v2 の multi-observation likelihood feature cache に固定する。比較対象は `likpf_mean` 単体、target-free `multiobs_score_top1`、oracle、LightGBM multiclass、candidate-long binary scorer、candidate-long error ranker。

## 検証方針

GroupKFold by `well` で OOF selected TVT を作り、RMSE、within 1/2/5/10 ft、oracle label accuracy、候補選択率、bucket metrics、by-well path switch count を見る。

## 所見

best OOF は `lgb_candidate_error_ranker` で RMSE 11.600097。`likpf_mean` 単体 RMSE 11.594898 を超えられず、summary recommendation は `ranker_not_supported`。`pf_ancc` は 35.98% 選べたが、path switch が多く、提出候補化しない。

## 注意

この実験は train-side audit 専用で、推論化や提出は行わない。改善した場合も path continuity、worst-well、raw-test feature parity を別途確認してから次へ進む。
