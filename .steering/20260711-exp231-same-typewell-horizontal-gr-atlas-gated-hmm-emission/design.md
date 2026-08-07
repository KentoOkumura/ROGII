# 設計

## アプローチ

fold `f` の validation well `w` に対し、exp065 `native_overlap / threshold=1` の同じ typewell group から、fold `f` の training wells だけを選ぶ。各 source well の raw horizontal GR を局所的に robust normalize した `64/128/256` rows patch にし、source true TVT の2 ft binごとに平均・分散・source patch数・source well数を蓄積する。

評価 row の GR patch と各 candidate TVT state の atlas patch distribution の距離を peer score にする。state 方向に中心化して clip した score を、peer support・match quality・TVT state uniqueness・base emission ambiguity/innovation・GR change point による target-free confidence で gate する。query/state を粗い格子で計算して row/state 方向に補間し、3スケールの patch 比較をCPU runtime内に収める。

## 実験範囲

- 対象実験: `exp231_same_typewell_horizontal_gr_atlas_gated_hmm_emission`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 参照: exp065 group assignment、exp201 readout、exp115 hidden-like、exp223/225/230 negative evidence。
- 変更する変数: fold-safe peer atlas と gated auxiliary emission `alpha=0.01/0.025/0.05`。
- 固定する変数: exp209 HMM transition/grid/base emission、saved exp072 comparison cache、comparison metric、CPU runtime policy。
- 非対象: exp072/exp209 control再生成、raw-test peer atlas、inference、submission、PF weight・direct replacement・LB tuning。

## 再現性設計

- seed policy: sorted well list を `np.random.default_rng(42)` で一度だけ shuffle して 5 fold に固定。source center/query centerは固定strideであり乱数なし。
- stochastic 処理の有無: なし。GR patch normalization、TVT binning、atlas aggregation は決定的。
- PF/Beam / likelihood-PF / seed bagging の有無: exp209 exact HMM は no RNG。exp072 likelihood-PF は再実行せず保存済み cache を比較だけに使う。
- 並列処理と乱数の関係: atlas は並列実行前にfoldごとに構築する。well HMM をthread並列しても atlas membership/value は変化しない。
- CPU/GPU runtime と deterministic flags: CPU only、internet off、`outer_workers=2` / `numba_num_threads=2`。GPU/LightGBMなし。
- train cache / test feature regeneration の SHA: cluster assignment SHA、atlas fold summary SHA、feature schema SHA、raw gzip SHA、decompressed feature content SHA を記録する。test regenerationは未実装。
- model manifest / prediction / submission SHA: model/prediction/submissionなし。inferenceが承認された時点で別途追加する。
- Kaggle package bootstrap: push前に generated bootstrap 内 config で route、atlas variants、seed、fold数、GPU false、kernel sources を検査する。

## リスク

- リーク: validation/same-fold valid peerのtrue TVTがatlasに入るとリークになる。foldごとにsource集合をassertし、atlas summaryに `validation_in_source_count=0` を残す。
- CV/LB不一致: train-fold peer atlas のcoverageがraw testで再現されない可能性がある。train-side採用だけではinference/submitへ進めない。
- ランタイム/メモリ: 3 alpha variants × 5 fold atlas × full HMM は重い。patchを16点へ縮約し、source/query strideとbin prototypeを固定し、per-fold source patch数とwall timeを出力する。
- 再現性: gzip timestamp差は主証拠にしない。source assignment、atlas summary、decompressed content SHAを保存する。
