# 設計

## アプローチ

exp157 の row-wise supervised selector は RMSE を改善したが、candidate path が行単位で頻繁に切り替わる。exp158 では exp157 の保存済み booster から OOF の per-candidate predicted-error surface を復元し、well ごとに Viterbi dynamic programming で最小 cost path を選ぶ。

local cost は exp157 `lgb_candidate_error_ranker` の predicted error。transition cost は switch penalty と候補 TVT path の jump penalty。追加 guard として `likpf_mean` からの delta cap、`pf_ancc_std` cap、`md_since` gate、minimum segment length pruning を適用する。

## 実験範囲

- 対象実験: `exp158_segment_continuity_selector_on_exp157`
- Route: `pf_beam`
- 親実験: `exp157_candidate_ranker_feature_enrichment`
- 変更する変数: Viterbi continuity constraints、switch penalty、jump penalty、delta cap、minimum segment length。
- 固定する変数: exp099 train cache、exp072 dense train cache、exp157 feature schema、exp157 saved boosters、GroupKFold by `well`、candidate-long row sampling seed、候補集合 8件。
- 比較対象: `likpf_mean_single`、`exp157_error_ranker_rowwise`、oracle。

## 再現性設計

- seed policy: exp157 と同じ GroupKFold seed 42 と candidate-long sampled train row seed を使う。Viterbi 自体は乱数なし。
- stochastic 処理の有無: 新規 stochastic feature generation はない。score 復元時のみ exp157 学習時と同じ deterministic row subsampling を再現する。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規には実行しない。exp099 / exp072 の既存 cache を入力として使う。
- 並列処理と乱数の関係: 新規 training はないため LightGBM histogram training の非決定性は追加しない。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU disabled。Viterbi は NumPy / pandas の deterministic 処理。
- train cache / test feature regeneration の SHA 記録方針: exp099 gzip は decompressed SHA、exp072 gzip は decompressed SHA、schema SHA を summary / result に記録する。
- model manifest / prediction / submission SHA 記録方針: exp157 model manifest SHA、resolved model SHA、OOF prediction decompressed SHA、variant 別 prediction SHA を記録する。submission は生成しない。
- Kaggle package bootstrap 確認方針: prepare 後に loose `config.yaml`、`kernel-metadata.json`、notebook bootstrap manifest に exp158 module / config / kernel sources が含まれることを確認する。

## リスク

- リークリスク: true TVT と oracle label は scoring / oracle baseline のみ。Viterbi local cost と guards には使わない。dense 候補は `last_known_tvt + tvt_dense*_d` で target-free に復元する。
- CV/LB 不一致リスク: train pseudo-tail OOF は hidden test の path continuity と完全一致しない。良い結果でも raw-test parity と worst-well guard を確認するまでは submit しない。
- ランタイム/メモリリスク: 8 candidates x 3 model families x 5 fold の score 復元と 180 Viterbi variants を CPU で行う。新規 booster training はないが、long frame 復元にメモリを使う。
- 再現性リスク: exp157 saved booster の feature schema と enrichment feature 生成が一致しないと score が再現できないため、missing feature check を strict にする。
