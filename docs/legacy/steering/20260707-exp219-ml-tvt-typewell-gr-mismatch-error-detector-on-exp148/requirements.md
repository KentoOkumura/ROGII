# 要件

## 依頼

`KAGGLE_DIRECTION.md` の `ml_tvt_typewell_gr_mismatch_error_detector_on_exp148` を exp219 として実装する。exp148 の ML OOF 予測 TVT を typewell TVT 軸上の仮位置として使い、horizontal GR と typewell GR の局所 mismatch が exp148 high-error row を検出できるか no-training readout で確認する。

## 制約

- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 初期実装では新規 LightGBM を学習しない。
- 初期実装では inference / submit を行わない。
- `best_offset` を hard correction、direct replacement、row-wise switch、blend、PF weight replacement に使わない。
- validation/test true TVT、oracle best、true-error rank、OOF absolute error を feature source に漏らさない。
- 再現性: `docs/06_reproducibility.md` に従い、input / feature cache / schema の SHA を記録する。

## 受け入れ基準

- exp148 train v1 `lgb_mean` OOF prediction を読み、`pred_tvt + offset` の typewell GR window と horizontal GR window を比較する feature cache を生成できる。
- `score_at_ml`、`best_offset`、`best_score`、`score_gap`、`entropy`、`decoy_gap`、`derivative_ncc`、`raw_vs_denoised_score_gap`、`local_z_mse`、candidate disagreement interaction を含む。
- `abs_error_gt10` AUC、high-mismatch bucket error lift、distance bucket、worst-well、exp115 hidden-like subgroup、diagnostic correction を出力する。
- Kaggle train 実行前の planned boosters は 0 で、control / parent retraining がない。
- `metrics.json`、`SESSION_NOTES.md`、`result.md` に現在状態と次 gate が記録されている。
