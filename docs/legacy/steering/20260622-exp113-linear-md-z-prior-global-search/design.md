# 設計

## アプローチ

exp080/exp095 では U-space / prefix-line 系の supervised target 変更が大きく悪化した。一方で、最後の既知点以降の TVT drift が `MD` と `Z` の低次幾何だけで部分的に説明できる well があるなら、強い target 変更ではなく、弱い prior / confidence feature / clipped correction として使える可能性がある。

今回の初期実装では、exp099 train feature cache と raw train horizontal/typewell CSV、exp065 common typewell cluster assignment を入力にする。raw horizontal から各 score row の `MD`、`Z`、`GR` と最後の既知点の `MD`、`Z` を復元し、`dMD`、`dZ` を作る。固定 grid の `a_values x b_values` に対して `last_tvt + a*dMD + b*dZ` を作り、GroupKFold by well で fold 内 train wells のみから RMSE 最良の `(a,b)` を選ぶ。選ばれた candidate は valid wells にだけ適用し、fold-out OOF prediction として評価する。

GR loss は horizontal GR と predicted TVT 位置の typewell GR の sampled MAE として記録する。exp065 native overlap / exact hash group ごとに、posthoc well-oracle best `(a,b)` の一貫性も集計する。

## 実験範囲

- 対象実験: `exp113_linear_md_z_prior_global_search`
- Route: `ml_model`
- 親実験: `exp099_pf_multi_observation_likelihood_probe`
- 診断親: `exp065_typewell_supertype_cluster_cv_audit`
- 変更する変数: 線形 prior の `a,b`、fold-safe selection、weak clipped correction 候補
- 固定する変数: exp099 cache、raw train files、exp065 cluster assignment、GroupKFold by well、submission なし

## 検証

- 主指標: fold-selected linear prior の TVT RMSE。
- 比較: `last_anchor_tvt`、`pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb`。
- bucket: `md_since` の 0-50、50-100、100-250、250-500、500-1000、1000+ ft。
- worst-well: candidate ごとの by-well RMSE と baseline 差分。
- consistency: native typewell group 内の best candidate mode share、`a,b` spread。

## 再現性設計

- seed policy: deterministic GroupKFold fixed grid。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: upstream exp099 cache に含まれるが、この実験では再生成しない。
- 並列処理と乱数の関係: 並列 RNG なし。
- CPU/GPU runtime と deterministic flags: CPU-only audit。GPU 不要。
- train cache / test feature regeneration の SHA 記録方針: exp099 gzip は raw SHA と decompressed SHA、exp065 assignment は SHA を summary JSON に記録する。
- model manifest / prediction / submission SHA 記録方針: model なし、submission なし。OOF prediction gzip は raw SHA と decompressed SHA を記録する。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks` 後に train metadata と source kernel を確認する。

## リスク

- リークリスク: fold-selected `(a,b)` に valid true TVT を混ぜると過大評価になる。selection は fold train のみ、posthoc group consistency は診断専用にする。
- CV/LB 不一致リスク: exp099 surface 上の改善が exp073/exp092 ML surface に転移する保証はない。改善しても add-only feature / raw-test parity を次段確認する。
- ランタイム/メモリリスク: candidate grid x 3.8M rows の演算になる。candidate 数は 56 に抑え、GR loss は sampled diagnostic にする。
- 再現性リスク: upstream cache 由来の stochastic 要素はこの実験では固定入力として扱う。
