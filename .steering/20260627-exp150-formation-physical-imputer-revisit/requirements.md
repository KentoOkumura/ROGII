# 要件

## 依頼

`formation_physical_imputer_revisit` backlog を実装する。Sunny 系の formation contact physical branch を、hidden test で成立する fold-safe imputer / confidence diagnostic として再構成する。

## 制約

- Route: `ml_model`
- 親実験: `exp138_ancc_surface_predictability_audit`
- train-only formation columns (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`) を直接 inference feature にしない。
- validation fold の formation columns と true `TVT` は scoring にしか使わない。
- 直接 TVT 置換、target conditioning、提出候補化は今回の範囲外にする。
- GPU 学習、LightGBM 学習、PF/Beam 生成は行わない。

## 受け入れ基準

- `experiments/exp150_formation_physical_imputer_revisit/` に config、実装、train/inference notebook、記録ファイルがある。
- fold-safe surface fitting と known-prefix calibration が実装されている。
- `candidate_metrics.csv`、`distance_bucket_metrics.csv`、`confidence_bucket_metrics.csv`、`surface_proxy_metrics.csv`、`formation_prefix_calibration.csv` を出力できる。
- static check と experiment validation が通る。
- full run 前の状態では、`SESSION_NOTES.md` と `result.md` に未実行であることが明記されている。
