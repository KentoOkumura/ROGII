# 設計

## アプローチ

exp176 の saved LightGBM boosters と feature schema を読み、GroupKFold by well の fold-held-out score surface を再構成する。local cost は exp176 `lgb_candidate_error_ranker` の predicted error とし、exp158 と同じ well-local Viterbi dynamic programming で候補 path を選ぶ。

exp176 v3 は row-level `tlp_` feature を multiclass には使い、candidate-long model では `tlp_` を複製せず `candidate_tlp_` feature だけを使った。そのため exp191 helper でも raw train typewell context から `tlp_` を復元し、long-frame では `row_feature_exclude_prefixes: [tlp_]` と exp176 の `candidate_tlp_` feature 生成順を再現する。

## 実験範囲

- 対象実験: `exp191_typewell_late_range_continuity_selector_on_exp176`
- Route: `ensemble`
- 親実験: `exp176_typewell_late_range_pfbeam_candidate_prior`
- 変更する変数: Viterbi switch penalty、jump penalty、delta cap、minimum segment length。
- 固定する変数: 8-candidate set、exp099 cache、exp072 dense cache、exp176 typewell late-range prior thresholds、exp176 saved boosters、GroupKFold by well、candidate-long sampled row seed。
- 比較対象: `likpf_mean_single`、`exp176_error_ranker_rowwise`、Viterbi variants、oracle。

## 再現性設計

- seed policy: exp176 と同じ GroupKFold seed 42 と candidate-long sampled train row seed を使う。Viterbi 自体は乱数なし。
- stochastic 処理の有無: 新規 stochastic feature generation / model training はない。exp176 saved booster の score 復元で、学習時の deterministic row subsampling を再現する。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規実行なし。exp099 / exp072 の既存 cache と exp176 boosters を入力として使う。
- 並列処理と乱数の関係: 新規並列 RNG なし。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU disabled。Viterbi は NumPy / pandas の deterministic 処理。
- train cache / test feature regeneration の SHA 記録方針: exp099 / exp072 gzip は decompressed SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: exp176 model manifest SHA、resolved model SHA、OOF prediction decompressed SHA、variant prediction SHA を記録する。submission は作らない。
- Kaggle package bootstrap 確認方針: prepare 後に loose config と notebook bootstrap 内 config、kernel sources、GPU false、internet false を確認する。

## リスク

- リークリスク: true TVT と oracle label は scoring / oracle baseline のみ。Viterbi local cost と guards には使わない。
- CV/LB 不一致リスク: exp176 は train-side supported だが row-wise switch が高い。continuity が global RMSE を改善しても raw-test parity、worst-well、near-row、`1000_plus` が弱ければ閉じる。
- ランタイム/メモリリスク: 新規学習はないが、3,783,989 rows x 8 candidates の long score 復元と 180 Viterbi variants で CPU runtime は長い。
- 再現性リスク: exp176 output archive がローカル未取得でも Kaggle source から解決できる前提。Kaggle source version と manifest SHA を記録する。
