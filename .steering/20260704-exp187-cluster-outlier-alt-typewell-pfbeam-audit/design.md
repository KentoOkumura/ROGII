# 設計

## アプローチ

exp065 の native-overlap typewell cluster assignment と exp114 の well geometry summary から、cluster 外れ well を target-free に抽出する。対象 well ごとに次の参照 typewell strategy を作る。

- `own_typewell`: query well 自身の typewell。
- `nearest_other_cluster_composite`: query well から最も近い別 cluster center の member typewell を TVT bin で結合した composite typewell。
- `nearby_majority_cluster_composite_k8`: 近傍 8 well の inverse-distance weighted majority cluster の member typewell を TVT bin で結合した composite typewell。own cluster と異なる場合だけ有効。

各 strategy で、exp072/099 と同じ train `TVT_input` 欠損相当 row に対して PF と Beam を再生成する。true TVT は exp072 cache の `last_known_tvt + target` として scoring にだけ使い、strategy 選択、PF/Beam likelihood、Beam cost には使わない。

## 実験範囲

- 対象実験: `exp187_cluster_outlier_alt_typewell_pfbeam_audit`
- Route: `pf_beam`
- 親実験: `cluster_outlier_alt_typewell_pfbeam_audit` backlog
- 参照: `exp065_typewell_supertype_cluster_cv_audit`、`exp114_spatial_neighbor_prior_signal_audit`、`exp186_typewell_late_range_pfbeam_generation_soft_prior`
- 変更する変数: PF/Beam の参照 typewell source。
- 固定する変数: query horizontal well、exp072 eval rows、particles、seed count、seed derivation、likelihood temperature、Beam size/cost/move radius。
- 対象数: config の `model.validation_surface.max_target_wells` で制限し、初期値は 64 wells。

## 再現性設計

- seed policy: `stable_sha256_per_query_well_seed_index_shared_across_typewell_strategies`
- stochastic 処理の有無: PF particle initialization、transition noise、resampling にあり。
- PF/Beam / likelihood-PF / seed bagging の有無: PF seed bagging と Beam top1 を行う。likelihood-PF は PF seed log-likelihood weighted mean として扱う。
- 並列処理と乱数の関係: 初期 audit は sequential。global RNG を使わず、well id と seed index から `np.random.default_rng` を作る。
- CPU/GPU runtime と deterministic flags: CPU only、GPU disabled、internet disabled。
- train cache / test feature regeneration の SHA 記録方針: train-side row candidates と metrics の SHA を記録し、gzip は decompressed content SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: model と submission は作らない。row-candidate prediction content SHA を記録する。
- Kaggle package bootstrap 確認方針: push 前に `prepare-kaggle-notebooks --notebook train --strict` を使い、bootstrap 内 support files を再生成する。

## リスク

- リークリスク: cluster/geometry/composite source selection は target-free だが、train audit の scoring には exp072 target true TVT を使う。selection には scoring result を戻さない。
- CV/LB 不一致リスク: train-side eval-zone 診断なので raw hidden test にそのまま移植しない。
- ランタイム/メモリリスク: PF/Beam 再生成は高コスト。初期値は 64 target wells、exp072 eval-zone 全 row、260 particles、8 seeds、3 strategy 程度に制限する。
- 再現性リスク: strategy によって likelihood/resampling timing が変わるため、同じ seed derivationでも乱数消費は分岐する。実験は deterministic submission anchor ではなく scoped diagnostic として扱う。
