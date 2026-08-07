# exp176_typewell_late_range_pfbeam_candidate_prior

## 目的

`typewell_late_range_pfbeam_candidate_prior` backlog の実験。PF/Beam / dense 候補ごとの `candidate_pct` を ranker / verifier の prior として使い、late prefix なのに typewell TVT 前半へ戻る候補を低信頼扱いにできるか確認する。

## 状態

実装済み、Kaggle train 未実行。

## 仮説

`known_last_pct` が高い well では、visible prefix が typewell range 後半まで進んでいる。その後の候補が `candidate_pct < 0.5/0.6/0.7` に戻る場合、GR 周期性や別 mode への落下を疑える。ただし exp174 では ML hard clip が悪化したため、候補を捨てずに弱い selector feature としてだけ使う。

## 変更点

- 親実験: `exp157_candidate_ranker_feature_enrichment`
- 候補集合: exp157 と同じ 8 候補
- 追加 row-level feature: `tlp_known_last_pct`、candidate pct summary、fixed lower-bound flag、late-prefix interaction
- 追加 candidate-long feature: `candidate_tlp_candidate_pct`、`candidate_tlp_candidate_pct_minus_known_last_pct`、fixed / dynamic lower-bound flag、risk score
- 入力: exp099 v2 multiobs cache、exp072 full replay feature cache、raw train typewell / horizontal prefix

## 検証方針

GroupKFold by well の train-side pseudo-tail OOF で、`likpf_mean_single`、`multiobs_score_top1`、oracle、3種類の LightGBM selector を比較する。exp157 の best OOF 10.795800 と exp158 segment continuity 10.789163 を比較基準にする。

## 実行方針

Kaggle CPU train notebook で実行する。GPU は使わない。LightGBM は 3 family x 5 folds = 15 boosters。exp157 control は再学習せず、保存済み exp157 / exp158 metrics を比較基準にする。

## 所見

未実行のため CV 所見はない。synthetic smoke では typewell context join と late-range prior feature 生成が通り、row-level 77 列、candidate-long 20 列が有限値として生成された。

## 非対象

- PF/Beam 再生成
- candidate hard invalid / direct clip
- direct TVT regressor
- soft-average candidate blending
- inference port / submission
- true validation TVT、oracle best、true error rank を feature に使う処理
