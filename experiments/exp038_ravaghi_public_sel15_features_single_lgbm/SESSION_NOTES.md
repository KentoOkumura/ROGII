# exp038_ravaghi_public_sel15_features_single_lgbm セッションノート

## 状態

- status: implemented
- route: `ml_model`
- parent: `exp026_pseudo_tail_bucket_shrink_inference_submit`
- supporting artifact: `exp029_public_sel15_pf_oof_feature_generation`

## 仮説

Ravaghi/public sel15 PF/Beam の候補値、likelihood、selector、beam spread、PF/Beam divergence は、直接 replay や Ridge meta-stack では 見えない test well 評価の LB に転移しにくい。一方で単体 LightGBM の feature としてなら、exp026 系 pseudo-tail model が不得意な遠方 tail の uncertainty / candidate path signal を拾える可能性がある。

## 実装メモ

- `ravaghi_single_lgbm_audit.py` を追加。
- `exp029` artifact を chunked load し、well-level split で cross-fit する。
- target は `target_tvt - last_anchor_tvt`。
- variants:
  - `base_geometry`
  - `base_plus_pf_prediction`
  - `base_plus_beam_prediction`
  - `base_plus_pf_beam_diagnostics`
- 全 variant で raw と `exp014_bucket_shrink_params` 適用後を評価する。
- `pf_error`、`last_anchor_error`、`beam_error`、`exp026_oof`、exp026 bridge columns は feature から除外。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp038_ravaghi_public_sel15_features_single_lgbm
uv run python scripts/new_experiment.py --name exp038_ravaghi_public_sel15_features_single_lgbm --source experiments/exp034_public_sel15_pf_meta_stack --skip-steering-check
uv run ruff check experiments/exp038_ravaghi_public_sel15_features_single_lgbm/ravaghi_single_lgbm_audit.py experiments/exp038_ravaghi_public_sel15_features_single_lgbm/settings.py
uv run python -m py_compile experiments/exp038_ravaghi_public_sel15_features_single_lgbm/ravaghi_single_lgbm_audit.py experiments/exp038_ravaghi_public_sel15_features_single_lgbm/settings.py
uv run python scripts/validate_experiment.py --experiment exp038_ravaghi_public_sel15_features_single_lgbm
uv run python experiments/exp038_ravaghi_public_sel15_features_single_lgbm/ravaghi_single_lgbm_audit.py --max-wells 10 --max-train-rows 2000 --skip-exp026-control --base-estimator HistGradientBoostingRegressor --output-dir /tmp/exp038_smoke
kaggle kernels push -p experiments/exp038_ravaghi_public_sel15_features_single_lgbm/kaggle/train
kaggle kernels pull kentookumura/exp038-ravaghi-sel15-lgbm-train -p /tmp/kaggle-pull/exp038-ravaghi-sel15-lgbm-train-check -m
kaggle kernels status kentookumura/exp038-ravaghi-sel15-lgbm-train
kaggle kernels logs kentookumura/exp038-ravaghi-sel15-lgbm-train
kaggle kernels output kentookumura/exp038-ravaghi-sel15-lgbm-train -p /tmp/kaggle-output/exp038_ravaghi_public_sel15_features_single_lgbm/train_v1_check
kaggle kernels logs kentookumura/exp038-ravaghi-sel15-lgbm-train
kaggle kernels output kentookumura/exp038-ravaghi-sel15-lgbm-train -p /tmp/kaggle-output/exp038_ravaghi_public_sel15_features_single_lgbm/train_v1
cp /tmp/kaggle-output/exp038_ravaghi_public_sel15_features_single_lgbm/train_v1/metrics.json experiments/exp038_ravaghi_public_sel15_features_single_lgbm/metrics.json
cp /tmp/kaggle-output/exp038_ravaghi_public_sel15_features_single_lgbm/train_v1/artifacts/single_lgbm_*.csv /tmp/kaggle-output/exp038_ravaghi_public_sel15_features_single_lgbm/train_v1/artifacts/single_lgbm_summary.json experiments/exp038_ravaghi_public_sel15_features_single_lgbm/artifacts/
```

## 結果

- Local smoke: PASS
  - command: `--max-wells 10 --max-train-rows 2000 --skip-exp026-control --base-estimator HistGradientBoostingRegressor`
  - rows: 21,974
  - wells: 10
  - output: `/tmp/exp038_smoke`
- Full Kaggle audit: PASS
- Kaggle train:
  - initial long slug `kentookumura/exp038-ravaghi-public-sel15-features-single-lgbm-train` was rejected by SaveKernel 400.
  - pushed version 1 with short canonical id `kentookumura/exp038-ravaghi-sel15-lgbm-train`.
  - URL: `https://www.kaggle.com/code/kentookumura/exp038-ravaghi-sel15-lgbm-train`
  - direct `kaggle kernels pull ... -m` succeeds and returns `id_no: 122027027`, so the kernel exists.
  - `kaggle kernels status` returned Kaggle API `GetKernelSessionStatus` 500; do not use it as completion evidence.
  - `kaggle kernels list/search` did not show this running private kernel yet, likely because it has no completed `lastRunTime` and the list index has not caught up.
  - regular logs/output were empty during the initial running period; retry the same kernel id instead of creating another slug.
  - completed output downloaded to `/tmp/kaggle-output/exp038_ravaghi_public_sel15_features_single_lgbm/train_v1`.
  - artifacts and `metrics.json` synced to `experiments/exp038_ravaghi_public_sel15_features_single_lgbm/`.
- CV:
  - selected single-LGBM feature candidate: `base_plus_pf_prediction_bucket_shrink`
  - original-fold RMSE: 15.850147
  - well-hash RMSE: 15.820850
  - best overall control: `pf090_hold010` at 15.089532
  - interpretation: PF prediction features improve strongly over `base_geometry_bucket_shrink`, but do not beat public PF controls.
- LB: not submitted.

## 次のアクション

1. `result.md`、`metrics.json`、`experiment_summary.md` に反映する。
2. PF confidence / divergence を direct candidate selection or postprocess gate として使う次実験を検討する。
