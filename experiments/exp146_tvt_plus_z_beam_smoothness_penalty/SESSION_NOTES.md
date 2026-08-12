# exp146_tvt_plus_z_beam_smoothness_penalty セッションノート

## 目的

`tvt_plus_z_beam_smoothness_penalty` backlog を正しく独立実装する。exp142 v2 のように trajectory PF に beam reference penalty を混ぜるのではなく、Beam search の cost 自体に `U = TVT + Z - (T0 + Z0)` と `dU/dMD = dTVT/dMD + dZ/dMD` の penalty を入れる。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_train_side_beam_improved_not_adopted_no_submit`
- CV: train-side pseudo-tail audit RMSE 11.594897672 (`likpf_mean` baseline)
- LB: 未提出
- blocked: none

## 変更点

- `docs/legacy/steering/20260627-exp146-tvt-plus-z-beam-smoothness-penalty/` を作成した。
- `experiments/exp146_tvt_plus_z_beam_smoothness_penalty/` を exp140 から作成した。
- 実装ファイルを `tvt_plus_z_beam_smoothness_penalty.py` に変更した。
- exp140 の posthoc correction 実装を削除し、Beam search 再生成 audit に置き換えた。
- Beam transition cost に次を追加できるようにした。
  - `U = TVT + Z - (T0 + Z0)` absolute penalty
  - `dU/dMD = dTVT/dMD + dZ/dMD` slope penalty
  - `dU/dMD` curvature penalty
- 比較対象は exp072 cache の `likpf_mean`、`beam_mean`、`pf_ancc`、`pf_z`。
- exp142 v2 は混在実装として invalid。exp146 を `tvt_plus_z_beam_smoothness_penalty` の正しい実装先にする。

## 再現性メモ

- seed policy: no new RNG
- stochastic components: upstream exp072 cache only
- CPU/GPU runtime: CPU-only、GPU 不使用
- deterministic anchor: false。train-side diagnostic only。
- gzip output: decompressed content SHA を summary JSON に記録する。

## 検証

- `uv run python -m py_compile experiments/exp146_tvt_plus_z_beam_smoothness_penalty/tvt_plus_z_beam_smoothness_penalty.py experiments/exp146_tvt_plus_z_beam_smoothness_penalty/settings.py`: PASS
- `uv run ruff check experiments/exp146_tvt_plus_z_beam_smoothness_penalty/tvt_plus_z_beam_smoothness_penalty.py experiments/exp146_tvt_plus_z_beam_smoothness_penalty/settings.py`: PASS
- `uv run ruff format --check experiments/exp146_tvt_plus_z_beam_smoothness_penalty/tvt_plus_z_beam_smoothness_penalty.py experiments/exp146_tvt_plus_z_beam_smoothness_penalty/settings.py`: PASS
- `uv run python scripts/validate_experiment.py --experiment exp146_tvt_plus_z_beam_smoothness_penalty`: PASS
- `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp146_tvt_plus_z_beam_smoothness_penalty --notebook train --kernel-id kentookumura/exp146-tvt-z-beam-smooth-train --title 'exp146 tvt z beam smooth train' --run-on-push --strict`: PASS
- packaged train notebook JSON / packaged module py_compile: PASS
- Kaggle train v1:
  - command: `make push-kaggle-train EXP=exp146_tvt_plus_z_beam_smoothness_penalty`
  - result: Kernel version 1 successfully pushed
  - URL: https://www.kaggle.com/code/kentookumura/exp146-tvt-z-beam-smooth-train
  - monitoring: not started by user request
- Kaggle train v1 completion:
  - status: COMPLETE
  - `kaggle kernels logs kentookumura/exp146-tvt-z-beam-smooth-train`: PASS
  - `kaggle kernels output kentookumura/exp146-tvt-z-beam-smooth-train -p experiments/exp146_tvt_plus_z_beam_smoothness_penalty/kaggle/output/train_v1`: PASS
  - output dir: `experiments/exp146_tvt_plus_z_beam_smoothness_penalty/kaggle/output/train_v1`

## 結果

- 評価 rows: 3,783,989
- 評価 wells: 773
- 主比較対象 `beam_mean`: RMSE 15.774327032 / MAE 10.898586486 / within10 0.591649183
- best generated Beam variant `tvt_plus_z_uslope_c100_uabs005`: RMSE 15.566811180 / MAE 10.758215973 / within10 0.597455754
- best generated delta vs `beam_mean`: RMSE -0.207515852 / MAE -0.140370513 / within10 +0.005806571
- 採用ガード `likpf_mean`: RMSE 11.594897672 / MAE 7.067632584 / within10 0.772807479
- best generated delta vs `likpf_mean`: RMSE +3.971913508
- `longtail_1000_plus`: best generated 16.920213787 vs `beam_mean` 17.132810666、delta -0.212597。ただし `likpf_mean` 12.702990216 には +4.217224 悪い。
- `beam_likpf_gap_top_quartile`: best generated 24.058046366 vs `beam_mean` 25.153127971、delta -1.095082。ただし `likpf_mean` 15.582631565 には +8.475415 悪い。
- `near_000_050`: best generated `tvt_plus_z_uslope_c100_ucurve025` 1.021850418 vs `beam_mean` 1.109429382、delta -0.087579。`likpf_mean` 1.188877518 からも -0.167027。ただし `pf_ancc` 0.452880405 の方が強い。

## 判断

- `tvt_plus_z_beam_smoothness_penalty` は従来 `beam_mean` の改良としては小幅 positive。
- ただし `likpf_mean` より大きく弱いため、direct Beam candidate、inference port、submit 候補としては不採用。
- 次に使う場合は直接置換ではなく、confidence feature / segment verifier の材料に限定する。

## 次のアクション

1. exp146 から inference port / submit は行わない。
2. exp142 の復旧版 trajectory-only v3 は Kaggle CPU session 枠が空き次第 push する。
