# 要件

## 依頼

`pseudo_tail_seed_bagging` を `exp053_pseudo_tail_seed_bagging` として実装する。ベースは古い `exp026` ではなく、直近で pseudo-tail 自前系の通常 CV と Public LB を更新した `exp051/exp052` 系を再考して採用する。

## 制約

- Route: `ml_model`
- 親実験は `exp051_pseudo_tail_lgbm_param_micro_tune` とする。
- 実装親も `exp051` とし、train-side OOF 監査を先に行う。
- base model は exp051 best の `LGBMRegressor(num_leaves=47, min_child_samples=60)` とする。
- pseudo-tail cutoff quantiles `[0.45, 0.65, 0.82]`、distance-balanced sampling、no-GR feature set、residual shrink、fixed `exp014_bucket_shrink_params` は維持する。
- 変更する変数は pseudo-tail row sampling seed と LightGBM random_state の平均化だけに限定する。
- 推論 port / submission は、この実験の full CV 結果を確認するまで行わない。
- exp027 全体 / PF route Public LB 基準とは混ぜず、ML route / pseudo-tail 自前系の比較として扱う。

## 受け入れ基準

- `kind: seed_bagging` の training variant が実装され、seed member ごとに sampling RNG と model seed を変えられる。
- valid fold の raw prediction は seed member 平均で作り、その後 fixed bucket-shrink candidate も評価できる。
- same-run single seed control と 3-seed bagging の raw / fixed bucket-shrink RMSE が同じ output CSV に保存される。
- static check と `validate_experiment` が通る。
- Kaggle train package を `run_on_push=true` で作成できる。
