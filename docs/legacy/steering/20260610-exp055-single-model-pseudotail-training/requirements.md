# 要件

## 依頼

`single_model_pseudotail_training` を `exp055_single_model_pseudotail_training` として実装する。

## 制約

- Route: `ml_model`
- 親実験は ML route Public LB 基準の `exp039_ravaghi_single_lgbm_inference_submit` とする。
- 実装土台は同一 single-model feature surface の監査コードを持つ `exp048_ravaghi_single_model_feature_parity_revisit` とする。
- exp039 系の LightGBM estimator、residual target、feature surface、fixed bucket-shrink は固定する。
- 変更する変数は training row policy に限定する。
- pseudo-tail policy は exp051 系の cutoff `[0.45, 0.65, 0.82]`、per pseudo-tail cap、distance-balanced sampling を表現できること。
- 現在の exp029 local artifact は cutoff 0.65 のみなので、multi-cutoff artifact がない場合は available cutoff に fallback し、missing cutoff を記録する。
- direct PF/Beam replacement、Ridge/meta stack、hidden branch router、追加 Ravaghi feature family は含めない。

## 受け入れ基準

- `experiments/exp055_single_model_pseudotail_training/` に self-contained な実験がある。
- `config.yaml` に control と pseudo-tail training policy の比較が明示されている。
- train notebook は設定確認、feature loading、single-model pseudo-tail training audit、metrics/生成物保存をセル単位で追える。
- static check と `validate_experiment` が通る。
