# exp100_pf_z_unified_velocity_observation_prior

## 状態

- ルート: pf_beam
- 状態: completed_train_side_audit
- CV: best `pf_z_xy_slope` RMSE 29.404162 / within10 0.655593
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-21
- 親実験: `exp072_exp063_full_replay_feature_cache`

## 仮説

既存 `pf_z` は known prefix から `dTVT/dMD ~= beta*dZ/dMD + intercept` を推定し、粒子速度をその期待値に寄せている。Z だけでなく XY trajectory、prefix 末尾の TVT slope、prefix で fit した GR affine calibration を弱い prior として粒子重みに入れれば、PF 単体の pseudo-tail 精度と path smoothness が改善する可能性がある。

## 変更点

- raw train の horizontal / typewell を読み、finite `TVT_input` prefix だけで velocity / GR calibration を fit する。
- `pf_z_control`、単独 3 variant、組み合わせ 4 variant を同じ PF 条件で比較する。
- 評価区間 true TVT は metrics の scoring にだけ使う。
- Stage 2 の dense prior、multi-scale GR mixture、acceleration penalty は未実装とし、切り分け不能な全部入りを避ける。
- inference port / submission は作らない。

## 検証方針

- 検証: train well pseudo-tail PF z-prior ablation
- Group: well
- Score rows: `TVT_input` missing rows
- Metrics: RMSE、MAE、within1/2/5/10、bucket RMSE、worst-well、path switch、smoothness
- Leakage Check: coefficient fit / particle weighting は finite `TVT_input` prefix と raw GR / trajectory のみを使う

## 実行入口

- 学習 notebook: `exp100_pf_z_unified_velocity_observation_prior_train.ipynb`
- 推論 notebook: `exp100_pf_z_unified_velocity_observation_prior_inference.ipynb`
- Kaggle 準備:

```bash
make validate-exp EXP=exp100_pf_z_unified_velocity_observation_prior
make prepare-kaggle-notebooks EXP=exp100_pf_z_unified_velocity_observation_prior EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp100-pf-z-unified-prior-train --title 'exp100 pf z unified prior train' --run-on-push --strict"
```

Kaggle Notebook 実行を正とする。ローカル実行は smoke debug が必要な場合だけ `--max-wells` / `--n-particles` を絞って行う。

## 期待生成物

- `exp100_pf_z_unified_velocity_observation_prior_variant_metrics.csv`
- `exp100_pf_z_unified_velocity_observation_prior_bucket_metrics.csv`
- `exp100_pf_z_unified_velocity_observation_prior_by_well.csv`
- `exp100_pf_z_unified_velocity_observation_prior_well_fit_summary.csv`
- `exp100_pf_z_unified_velocity_observation_prior_candidate_wide.csv.gz`
- `exp100_pf_z_unified_velocity_observation_prior_candidate_long.csv.gz`
- `exp100_pf_z_unified_velocity_observation_prior_summary.json`

## 所見

Kaggle train v2 完了。best は `pf_z_xy_slope` で RMSE 29.404162 / within10 0.655593。standalone control の `pf_z_control` RMSE 163.301551 からは大幅に改善した。

ただし、既存 `likpf_mean` RMSE 11.594897 より大きく弱いため、直接 inference port / submit 候補にはしない。Stage 2 の dense prior、multi-scale GR mixture、acceleration penalty に進む根拠も不足している。

### リスク / 注意

- 8 variant で PF を再実行するため、Kaggle runtime は exp099 より長くなる。
- train pseudo-tail 改善が出ても hidden test で同じ prior が効く保証はないため、直接 submit せず inference port を別途確認する。
- `pf_z_control` は exp100 standalone rerun 内の control で、exp072 保存済み `pf_z` との strict parity ではない。

## 次

1. `pf_z_unified_velocity_observation_prior` は backlog から外す。
2. PF 側を続けるなら、先に exp072 public replay と strict parity の `pf_z_control` を小さく再現確認する。
