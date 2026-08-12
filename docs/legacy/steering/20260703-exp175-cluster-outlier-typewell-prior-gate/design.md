# 設計

## アプローチ

exp065 の `native_overlap=1` cluster assignment と exp114 の well geometry summary から、well ごとの cluster 外れ度を作る。主な gate feature は `own_cluster_dist_z`、`nearest_other_cluster_dist < own_cluster_dist`、`nearby K wells majority cluster != own_cluster`。これを exp109 typewell prior、exp114 spatial prior、exp092 / exp148 OOF prediction に join し、cluster 外れ well だけに prior と base prediction の差分を小さく足す posthoc grid を評価する。

## 実験範囲

- 対象実験: `exp175_cluster_outlier_typewell_prior_gate`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- fallback 比較: `exp092_u_projection_correction_disagreement_fullrun`
- 変更する変数: cluster outlier gate、prior family、prior quality gate、correction alpha / clip。
- 固定する変数: upstream OOF prediction、exp109 / exp114 prior、cluster assignment、raw geometry、評価 metric。

## 再現性設計

- seed policy: no new RNG。posthoc grid は deterministic CSV join と numpy 演算だけ。
- stochastic 処理の有無: exp175 内ではなし。upstream LightGBM / prior generation は固定 output として扱う。
- PF/Beam / likelihood-PF / seed bagging の有無: exp175 内ではなし。upstream prior の由来としてのみ記録する。
- 並列処理と乱数の関係: exp175 内では並列 RNG なし。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU disabled。LightGBM booster は学習しない。
- train cache / test feature regeneration の SHA 記録方針: input CSV の file SHA、gzip decompressed SHA、cluster feature CSV SHA、prediction CSV SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: model manifest と submission はなし。top gated prediction は raw gzip SHA と decompressed SHA を記録する。
- Kaggle package bootstrap 確認方針: push 前に `prepare-kaggle-notebooks --strict` で metadata と kernel sources を生成し、bootstrap config が正の `config.yaml` を含むことを確認する。

## リスク

- リークリスク: cluster gate は target-free だが、exp109 / exp114 prior の fold-safe 前提に依存する。true TVT は scoring のみに使う。
- CV/LB 不一致リスク: posthoc OOF の微小改善は hidden test に移らない可能性が高い。global 改善だけで submit しない。
- ランタイム/メモリリスク: row 数は 3,783,989。policy grid は numpy の sparse correction score で計算し、詳細診断は top policy だけに絞る。
- 再現性リスク: upstream output が Kaggle input として mount されない場合は実行不可。source path と SHA を summary に残す。
