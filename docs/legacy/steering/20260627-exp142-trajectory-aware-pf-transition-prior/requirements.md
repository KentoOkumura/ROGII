# 要件

## 依頼

`trajectory_aware_pf_transition_prior` を実装する。PF 生成時の transition prior に `dZ/dMD`、`d2Z/dMD2`、MD step、prefix slope を反映し、Z-driven 区間で追加価値がある PF-Z 候補を train pseudo-tail 上で監査する。

## 制約

- Route: `pf_beam`
- 親実験: `exp106_strict_exp072_pf_z_multiseed_scale_cache`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 再現性: `docs/06_reproducibility.md` に従い、PF/Beam の seed、parallel RNG、feature SHA、gzip decompressed SHA を記録する。
- true TVT は scoring と oracle diagnostics にだけ使う。transition prior、gate、candidate generation には使わない。
- 初回は train-side diagnostic only。raw-test-compatible inference port と submit は、候補の headroom と collapse 監査が通った場合だけ同じ exp142 内で追加する。
- 直接 `likpf_mean` を超えることを初回の必須条件にしない。Z-driven bucket、PF worst50、representative wells、oracle/gate headroom、ESS、mode diversity、path roughness を見る。

## 受け入れ基準

- `experiments/exp142_trajectory_aware_pf_transition_prior/` に実装、config、train/inference notebook、記録ファイルがある。
- strict exp072 PF-Z parity candidate と trajectory-aware variants が同じ pseudo-tail rows で比較される。
- trajectory variants が `dZ/dMD` と `d2Z/dMD2` を transition mean / process noise / likelihood sigma に使う。
- candidate metrics、bucket metrics、by-well metrics、strict PF quality、trajectory PF quality、parity diff、candidate wide、summary JSON が保存される。
- ESS、resample count、collapse rate、particle std が variant/well 単位で記録される。
- deterministic anchor として扱わず、train-side diagnostic として記録される。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
