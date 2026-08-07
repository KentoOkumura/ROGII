# 要件

## 依頼

`well_scaled_z_dz_features_on_exp218` を `exp224_well_scaled_z_dz_features_on_exp218` として実装する。

## 制約

- Route: `ml_model`
- 親: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- parent/control 再学習はしない。exp218 の保存済み CV / Public LB を baseline とする。
- `z` / `dz` / `dzdmd` / `slp_z` の raw 4 列は残す。
- target-derived scaler、direct correction、candidate replacement、blend、postprocess、hard selector、sample-weight 変更は禁止。
- `likpf_tvt = last_known_tvt + likpf_mean_d` の p05-p95 range は scale feature のみに使い、candidate correction には使わない。
- interaction は増やしすぎず、scaled z/dz x raw GR / 指定 GRWR confidence に限定する。
- GPU 枯渇対策として CPU 実行にする。
- タイムアウト対策として LightGBM config を `train_lgb0` / `train_lgb1` / `train_lgb2` に分割する。
- 再現性: `docs/06_reproducibility.md` に従い、CPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `well_scaled_z_dz` feature group が train / inference の両方で同じ builder から生成される。
- active variant が `well_scaled_z_dz_addonly` だけで、CPU split の planned boosters が各 split 5、合計 15 と記録されている。
- Kaggle train 前に notebook 変換、構文チェック、ruff F821/F401、experiment validation が通る。
- `train_lgb0` / `train_lgb1` / `train_lgb2` の Kaggle package metadata が `enable_gpu=false` である。
- Kaggle train 後に CV、feature importance、bucket metrics、by-well metrics、model manifest が記録される。
- 100-1000 bucket と worst-well regression を exp218 と比較できる。
