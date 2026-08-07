# 設計

## アプローチ

既存 `pf_z` を train pseudo-tail 上で再実行し、粒子速度と GR observation likelihood に弱い補助 prior を足す。Stage 1 は切り分けを優先し、`XY slope`、`prefix slope`、`GR affine calibration` の単独と組み合わせだけを比較する。

## 実験範囲

- 対象実験: `exp100_pf_z_unified_velocity_observation_prior`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`
- 変更する変数: PF 粒子重みに掛ける velocity / observation prior
- 固定する変数: raw train split、finite `TVT_input` prefix、PF particle count、GR grid、noise scale、score rows

## Variant

- `pf_z_control`: 既存 Z-aware velocity + GR observation
- `pf_z_xy_slope`: `dTVT/dMD ~= a*dZ/dMD + b*dXY/dMD + c`
- `pf_z_prefix_slope`: known prefix 末尾 50/all の `dTVT/dMD` を expected velocity に弱く blend
- `pf_z_gr_calibrated`: prefix で `horizontal_GR ~= cal_a * typewell_GR(TVT) + cal_b` を fit し、評価区間の GR likelihood に適用
- `pf_z_xy_plus_prefix`
- `pf_z_xy_plus_gr_calibrated`
- `pf_z_prefix_plus_gr_calibrated`
- `pf_z_xy_prefix_gr_calibrated`

## 再現性設計

- seed policy: `experiment_name`, variant, well id から SHA256 stable seed を生成する。
- stochastic 処理の有無: あり。PF particle initialization、process noise、resampling。
- PF/Beam / likelihood-PF / seed bagging の有無: PF のみ。Beam と supervised model は使わない。
- 並列処理と乱数の関係: `num_workers=1` を既定にし、global RNG 消費順序への依存を避ける。
- CPU/GPU runtime と deterministic flags: CPU only。GPU は使わない。
- train cache / test feature regeneration の SHA 記録方針: raw input file SHA、candidate wide/long の raw gzip SHA と decompressed SHA を summary JSON に記録する。
- model manifest / prediction / submission SHA 記録方針: model / submission は作らない。candidate 生成物 SHA と metrics SHA だけを記録する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --notebook train --strict` 後に `validate-exp` を通し、support file SHA は package manifest に残す。

## リスク

- リークリスク: 評価区間 true TVT を coefficient fit に使うと漏れるため、fit 対象は finite `TVT_input` prefix のみ。
- CV/LB 不一致リスク: train-side pseudo-tail のみなので、改善しても即 submit しない。
- ランタイム/メモリリスク: 8 variant x PF rerun のため exp099 より重い。`max_wells` と `n_particles` を config で絞れるようにする。
- 再現性リスク: numba 内 RNG を使うため bitwise anchor ではなく、stable seed による diagnostic reproducibility として扱う。
