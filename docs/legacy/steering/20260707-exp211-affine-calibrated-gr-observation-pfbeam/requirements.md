# 要件

## 依頼

KAGGLE_DIRECTION の backlog `affine_calibrated_gr_observation_pfbeam` を `exp211_affine_calibrated_gr_observation_pfbeam` として実装する。known prefix だけで horizontal GR と typewell GR の affine calibration を fit し、PF/Beam の observation likelihood に使った場合の train-side pseudo-tail 品質を raw baseline と比較する。

## 制約

- Route: `pf_beam`
- 親/参照: `exp072_exp063_full_replay_feature_cache`、`exp189_denoised_gr_pfbeam_generation_audit`、`exp170_heel_calibrated_shift_scan_pfbeam_audit`、公開 notebook catch-up memo。
- 実行面: LightGBM なし、fold なし、booster 0。Kaggle CPU / internet off の train-side diagnostic とする。
- 比較条件: raw/classic、affine/classic、raw/prefix structural、affine/prefix structural の 2x2 を同一 target wells、同一 PF seeds、同一 particles、同一 Beam 幅で比較する。
- calibration fit は known prefix の `TVT_input` と `GR`、typewell の `TVT` と `GR` だけを使う。eval tail の true TVT、target、oracle best、true-error rank を fit やvariant選択に使わない。
- prefix が短い、typewell GR 分散が低い、slope が非正または極端、prefix RMSE が高い場合は raw observation に fallback し、fallback rate と理由を記録する。
- direct replacement、inference port、submit は対象外。良かった場合でも raw-test-safe regeneration と hidden-like stress を別途確認する。

## 受け入れ基準

- `config.yaml` に `experiment.route: pf_beam`、lineage、2x2 variants、fallback guard、seed policy、Kaggle runtime が明記されている。
- train notebook が setup、入力確認、variant contract、PF/Beam audit 実行、metrics/diagnostics/生成物表示をセル単位で追える。
- helper が raw/affine observation と classic/prefix structural transition を同一 scoring surface で生成できる。
- 出力に RMSE/MAE/within10、distance bucket、by-well、worst-well regression、PF diagnostics、affine slope/intercept/prefix RMSE/fallback、row candidates、summary JSON が含まれる。
- deterministic anchor としては扱わず、gzip 生成物は decompressed content SHA を主証拠として記録する設計になっている。
