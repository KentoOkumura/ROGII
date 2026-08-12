# 設計

## アプローチ

exp065 の `common_typewell_cluster_assignments.csv` から `native_overlap` / `exact_hash` group を読み、exp099 v2 train feature cache の pseudo-tail rows に対して OOF の neighbor prior を作る。

各 validation well は自分の cluster ID を query として使う。同じ cluster に属する train-fold wells の `true_tvt - last_known_tvt` curve を `md_since` 軸で補間し、validation row の `md_since` における median drift を neighbor prior とする。query well の absolute prior は `last_known_tvt + median_neighbor_delta`。

この prior を直接提出候補にせず、`likpf_mean` / `pf_ancc` / `beam_mean` に対して小さい `alpha` と clip 幅を持つ後段補正として評価する。

```text
corrected = base + alpha * clip(neighbor_prior - base, -clip_ft, clip_ft)
```

補助的に、neighbor count と prior std による gated correction も出す。評価は exp099 cache と同じ rows で `rmse`、`mae`、`within10`、distance bucket、by-well worst を見る。

## 実験範囲

- 対象実験: `exp109_typewell_neighbor_prior_features`
- Route: `ensemble`
- 親実験: `exp099_pf_multi_observation_likelihood_probe`
- 参照実験: `exp065_typewell_supertype_cluster_cv_audit`、`exp072_exp063_full_replay_feature_cache`
- 変更する変数: native typewell overlap neighbor から作る prior と、その correction alpha / clip / gate 条件
- 固定する変数: exp099 v2 candidate cache、既存 `likpf_mean` / `pf_ancc` / `beam_mean` 候補、score rows、train-side pseudo-tail evaluation

## 再現性設計

- seed policy: deterministic well fold assignment with fixed seed 42。新規 PF / booster / sampling はない。
- stochastic 処理の有無: 新規 stochastic 処理なし。上流 exp072 / exp099 / exp065 の生成物は既存 artifact として SHA を記録する。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規実行なし。exp099 cache 内の候補を固定入力にする。
- 並列処理と乱数の関係: `num_workers=1`。global RNG / thread scheduling 依存なし。
- CPU/GPU runtime と deterministic flags: CPU notebook。GPU 不使用。
- train cache / test feature regeneration の SHA 記録方針: exp099 cache は raw gzip SHA と decompressed SHA、exp065 cluster assignment は file SHA を summary JSON に記録する。exp109 OOF gzip も raw / decompressed SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: model なし、submission なし。OOF prediction gzip SHA と metrics SHA を記録する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --notebook train --strict` 後に生成 package の `config.yaml` と補助 `.py` が含まれることを `validate-exp` で確認する。

## リスク

- リークリスク: valid wells を neighbor source に混ぜると TVT drift prior が答えを見てしまう。実装では fold ごとに train wells のみを source にする。
- CV/LB 不一致リスク: 本番 test では full train を neighbor source にできる一方、CV では fold-out に制約するため coverage / prior quality が変わる。改善しても raw-test parity audit なしでは submit しない。
- ランタイム/メモリリスク: 3.78M rows に対して cluster neighbor interpolation を実行するため、group max が大きい native overlap ではメモリ増に注意する。実装は well 単位に stack して巨大 all-row matrix を避ける。
- 再現性リスク: 上流 cache の stochastic 性は exp109 では制御しない。exp109 は deterministic posthoc audit として扱い、submission anchor にはしない。
