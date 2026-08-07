# 設計

## アプローチ

exp101 の OOF score surface を復元し、`lgb_candidate_error_ranker` の per-candidate predicted error を Viterbi local cost にする。well ごとに row order で DP を解き、switch penalty と candidate TVT jump penalty で不自然な row-wise switch を抑える。短すぎる non-default segment は `likpf_mean` に戻す。

## 実験範囲

- 対象実験: `exp155_segment_viterbi_candidate_selector_on_exp101`
- Route: `pf_beam`
- 親実験: `exp101_pf_candidate_ranker_or_nway_classifier`
- 変更する変数: Viterbi switch penalty、non-default bias、jump penalty、jump free threshold、delta cap、PF std cap、md_since gate、minimum segment length
- 固定する変数: exp099 v2 train cache、exp101 feature schema、exp101 saved boosters、GroupKFold split、candidate set

## 再現性設計

- seed policy: exp101 と同じ GroupKFold seed と candidate-long sampling seed を使い、Viterbi 自体は乱数を使わない。
- stochastic 処理の有無: exp155 ではなし。上流 exp072/exp099/exp101 の生成物は stochastic provenance を持つため deterministic submission anchor とは呼ばない。
- PF/Beam / likelihood-PF / seed bagging の有無: exp155 は生成済み PF/Beam cache を読むだけで新規生成しない。
- 並列処理と乱数の関係: Viterbi は単一プロセス deterministic DP。global RNG や thread scheduling 依存なし。
- CPU/GPU runtime と deterministic flags: Kaggle CPU posthoc audit。GPU は使わない。
- train cache / test feature regeneration の SHA 記録方針: exp099 cache の raw gzip SHA と decompressed SHA、schema SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: exp101 model manifest SHA、resolved booster SHA、保存対象 OOF prediction decompressed SHA、variant prediction SHA を記録する。submission は作らない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後に metadata と bootstrap config が exp155 の config を含むことを確認する。

## リスク

- リークリスク: Viterbi penalty grid を OOF に合わせ込みすぎると hidden に弱い。selection 条件に true TVT、oracle label、true error を入れない。
- CV/LB 不一致リスク: train-side pseudo-tail audit なので、改善しても raw-test parity と hidden-like stress が必要。
- ランタイム/メモリリスク: exp136 は大きな grid と by-well/bucket 評価で DeadKernel になった。exp155 は descriptor 計算を削り、初回 audit は 16 Viterbi variants に限定する。
- 再現性リスク: 上流 PF/Beam cache と exp101 booster は保存済み artifact 依存。SHA を記録し、exp155 単体を deterministic submission anchor と扱わない。
