# 要件

## 依頼

`pf_z_unified_velocity_observation_prior` を実装する。

## 制約

- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`
- 対象は train pseudo-tail の Stage 1 ablation とし、提出用 inference port は作らない。
- 既存 `pf_z` の Z-aware velocity + GR observation を control として残す。
- 追加 prior は弱く掛ける。hard constraint、target 変更、評価区間 true TVT を使った coefficient fit は禁止する。
- 再現性: `docs/06_reproducibility.md` に従い、PF 乱数は well id / variant から stable seed を作る。

## 受け入れ基準

- `pf_z_control`、単独 3 本、組み合わせ 4 本の Stage 1 variant を同じ pseudo-tail 条件で評価できる。
- 出力に variant metrics、bucket metrics、by-well metrics、candidate wide/long、summary JSON が含まれる。
- RMSE / within10 / path switch / smoothness / worst well / distance bucket を比較できる。
- `config.yaml` に route、lineage、variant、再現性方針、expected artifacts が記録されている。
- gzip 生成物を比較する場合は raw `.csv.gz` SHA と decompressed content SHA を分けて記録する。
