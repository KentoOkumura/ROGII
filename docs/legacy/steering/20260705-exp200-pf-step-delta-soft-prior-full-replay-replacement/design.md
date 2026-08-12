# 設計

## アプローチ

exp072 の full replay builder をベースにし、PF 系の粒子重みだけに transition prior を足す。
exp186 は full replay wrapper と記録形式の参照に使うが、typewell late-range prior の実装は持ち込まない。

## 実験範囲

- 対象実験: `exp200_pf_step_delta_soft_prior_full_replay_replacement`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`
- 参照実験: `exp186_typewell_late_range_pfbeam_generation_soft_prior`
- 変更する変数: PF_ANCC / PF_Z / likelihood-PF の particle weight に入る per-step TVT delta prior。
- 固定する変数: Beam search、raw input、feature schema、PF seed derivation、PF seeds 128、particles 500、LightGBM なし。

## prior

Selected variant:

```text
name = delta_free010_cost0025_scale003
dtvt = current_tvt - previous_particle_tvt
excess = max(0, abs(dtvt) - 0.10)
prior = 0.025 * (excess / 0.03)^2
likelihood *= exp(-prior)
```

2 個目の候補 `delta_free008_cost005_scale003` は config に残すが、初回 default run では inactive。

## 再現性設計

- seed policy: `stable_seed("pf_ancc" / "pf_z" / "likpf", split, well)`。
- stochastic 処理の有無: PF particle propagation / resampling と likelihood-PF seed ensemble が stochastic。
- PF/Beam / likelihood-PF / seed bagging の有無: PF_ANCC、PF_Z、Beam、128-seed likelihood-PF を生成する。Beam は deterministic。
- 並列処理と乱数の関係: joblib threads で well ごとに独立 stable seed を使うため、thread scheduling で乱数系列を共有しない。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU disabled。
- train cache / test feature regeneration の SHA 記録方針: gzip raw SHA と decompressed content SHA を分け、decompressed content SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: model / prediction / submission は生成しない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後、metadata と bootstrap config を validate-exp で確認する。

## リスク

- リークリスク: train TVT は generated target と direct comparison の真値にだけ使い、PF/Beam generation には使わない。
- CV/LB 不一致リスク: この実験は direct PF/Beam train-side comparison のみで、CV/LB は主張しない。
- ランタイム/メモリリスク: exp186 full replay generation は約 14,053 sec。default run は 1 variant に限定する。
- 再現性リスク: PF stochastic output は seed fixed だが、submission anchor ではない。test 側は downstream inference で同じ code から再生成する必要がある。
