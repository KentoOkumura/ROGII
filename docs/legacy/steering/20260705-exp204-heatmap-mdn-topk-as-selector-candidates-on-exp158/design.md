# 設計

## アプローチ

exp203 をコピーし、exp202 reader / hmdn feature block を再利用する。exp203 では heatmap MDN topK を feature-only として扱ったが、exp204 では row-interpolation 後の `hmdn_top1_tvt` ... `hmdn_top10_tvt` を `ranker.candidates` に追加する。候補追加後も hmdn distance feature が trivial にならないよう、row-level `hmdn_*_vs_candidate` は既存 8 候補との距離として計算する。

## 実験範囲

- 対象実験: `exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158`
- Route: `pf_beam`
- 親実験: `exp203_heatmap_mdn_candidates_into_selector_features`
- 比較対象: `exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`、`exp158_segment_continuity_selector_on_exp157`、exp202 existing+heatmap top10 oracle
- 変更する変数: selector candidate set、candidate-long hmdn family/rank/score features、Viterbi allowed switch candidates
- 固定する変数: exp099/exp072/exp182/exp202 inputs、fold、LightGBM config、Viterbi grid、long-model memory cap、GPU disabled、parent/control retraining なし

## 再現性設計

- seed policy: fixed GroupKFold seed + LightGBM random_state + candidate-long row subsample local RNG
- stochastic 処理の有無: exp204 自体は CPU LightGBM と deterministic interpolation。ただし upstream exp182/exp202 GPU heatmap models は stochastic source として扱う。
- PF/Beam / likelihood-PF / seed bagging の有無: exp099/exp072 saved candidate cache を読む。exp204 内では PF/Beam を再生成しない。
- 並列処理と乱数の関係: feature generation は global RNG なし。long-model subsample は fold seed の local RNG。
- CPU/GPU runtime と deterministic flags: Kaggle CPU runtime、`runtime.kaggle.enable_gpu=false`。deterministic submission anchor とは扱わない。
- train cache / test feature regeneration の SHA 記録方針: 入力 cache と gzip 出力は decompressed content SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: train 実行後に model manifest SHA、OOF prediction SHA、feature schema SHA を記録。submission SHA は対象外。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後に metadata と bootstrap config の整合を確認する。

## リスク

- リークリスク: exp202 target/abs-error/oracle columns を読まない。candidate labels は train target から作るが feature には入れない。
- CV/LB 不一致リスク: train-side selector audit のため、global OOF が良くても raw-test heatmap generation / sparse coverage / hidden-like stress 確認までは inference しない。
- ランタイム/メモリリスク: candidate count が 8 から 18 に増え、candidate-long rows と pairwise features が増える。exp183/184/203 と同じ 120k rows/fold cap と chunked prediction を維持する。
- 再現性リスク: upstream GPU heatmap prediction に依存するため deterministic anchor ではない。採用検討時は upstream artifact SHA と Kaggle kernel version を必須記録する。
