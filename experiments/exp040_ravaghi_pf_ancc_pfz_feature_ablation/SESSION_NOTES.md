# exp040_ravaghi_pf_ancc_pfz_feature_ablation セッションノート

## 状態

- status: completed
- route: `ml_model`
- parent: `exp026_pseudo_tail_bucket_shrink_inference_submit`
- supporting artifact: `exp029_public_sel15_pf_oof_feature_generation`

## 仮説

Ravaghi の `pf_ancc` / `pf_z` 系 signal は、候補値の直接置換では 見えない test well 評価の LB に転移しにくい。一方で、PF delta、PF seed/std、Z/MD cutoff-relative PFZ proxy、PF-vs-Z disagreement、likelihood margin として単体 LightGBM に渡せば、exp026 系 pseudo-tail model が不得意な遠方 tail の confidence / divergence を拾える可能性がある。

## 実装メモ

- `exp038` の `ravaghi_single_lgbm_audit.py` を土台に `exp040` 用へ変更。
- `exp029` artifact を chunked load し、well-level split で cross-fit する。
- target は `target_tvt - last_anchor_tvt`。
- variants:
  - `base_geometry`
  - `base_plus_pf_ancc_delta_proxy`
  - `base_plus_pf_z_proxy`
  - `base_plus_pf_uncertainty`
  - `base_plus_pf_ancc_pfz_core`
  - `base_plus_pf_ancc_pfz_uncertainty`
- 全 variant で raw と `exp014_bucket_shrink_params` 適用後を評価する。
- `pf_error`、`last_anchor_error`、`beam_error`、`exp026_oof`、exp026 bridge columns は feature から除外。
- train-only の `ANCC` 列は読まず、`pf_ancc` / `pf_z` は `exp029` の PF output、Z/MD、pseudo cutoff から作れる proxy に限定する。
- beam 予測は feature family から外し、report control のみにする。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp040_ravaghi_pf_ancc_pfz_feature_ablation
uv run python scripts/new_experiment.py --name exp040_ravaghi_pf_ancc_pfz_feature_ablation --source experiments/exp038_ravaghi_public_sel15_features_single_lgbm --skip-steering-check
uv run ruff check experiments/exp040_ravaghi_pf_ancc_pfz_feature_ablation/ravaghi_single_lgbm_audit.py experiments/exp040_ravaghi_pf_ancc_pfz_feature_ablation/settings.py
uv run python -m py_compile experiments/exp040_ravaghi_pf_ancc_pfz_feature_ablation/ravaghi_single_lgbm_audit.py experiments/exp040_ravaghi_pf_ancc_pfz_feature_ablation/settings.py
uv run python scripts/validate_experiment.py --experiment exp040_ravaghi_pf_ancc_pfz_feature_ablation
uv run python experiments/exp040_ravaghi_pf_ancc_pfz_feature_ablation/ravaghi_single_lgbm_audit.py --max-wells 10 --max-train-rows 2000 --skip-exp026-control --base-estimator HistGradientBoostingRegressor --output-dir /tmp/exp040_smoke
task validate-exp EXP=exp040_ravaghi_pf_ancc_pfz_feature_ablation
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp040_ravaghi_pf_ancc_pfz_feature_ablation --notebook train --kernel-id kentookumura/exp040-ravaghi-pf-ancc-pfz-train --title "exp040 ravaghi pf ancc pfz train" --run-on-push --strict
kaggle kernels push -p experiments/exp040_ravaghi_pf_ancc_pfz_feature_ablation/kaggle/train
kaggle kernels pull kentookumura/exp040-ravaghi-pf-ancc-pfz-train -p /tmp/kaggle-pull/exp040-ravaghi-pf-ancc-pfz-train-check -m
kaggle kernels logs kentookumura/exp040-ravaghi-pf-ancc-pfz-train
kaggle kernels output kentookumura/exp040-ravaghi-pf-ancc-pfz-train -p /tmp/kaggle-output/exp040_ravaghi_pf_ancc_pfz_feature_ablation/train_v1_check
kaggle kernels output kentookumura/exp040-ravaghi-pf-ancc-pfz-train -p /tmp/kaggle-output/exp040_ravaghi_pf_ancc_pfz_feature_ablation/train_v1
kaggle kernels pull kentookumura/exp040-ravaghi-pf-ancc-pfz-train -p /tmp/kaggle-pull/exp040-ravaghi-pf-ancc-pfz-train-complete -m
uv run python scripts/validate_experiment.py --experiment exp040_ravaghi_pf_ancc_pfz_feature_ablation
uv run python scripts/update_experiment_summary.py
```

## 結果

- Static checks: PASS
  - `ruff check`: PASS
  - `py_compile`: PASS
  - `scripts/validate_experiment.py --experiment exp040_ravaghi_pf_ancc_pfz_feature_ablation`: PASS
- `task validate-exp`: skipped because `task` is not installed in this local environment.
- Local smoke: PASS
  - command: `--max-wells 10 --max-train-rows 2000 --skip-exp026-control --base-estimator HistGradientBoostingRegressor`
  - rows: 21,974
  - wells: 10
  - output: `/tmp/exp040_smoke`
- Kaggle train:
  - prepared train notebook with short canonical id `kentookumura/exp040-ravaghi-pf-ancc-pfz-train`.
  - pushed version 1 successfully.
  - URL: `https://www.kaggle.com/code/kentookumura/exp040-ravaghi-pf-ancc-pfz-train`
  - direct `kaggle kernels pull ... -m` succeeds and returns `id_no: 122072787`, so the kernel exists.
  - During the run, the UI showed execution logs while local `kaggle kernels logs` was empty because Kaggle API `ListKernelSessionOutput` returned `{'log': ''}` for the running session.
  - During the run, `kaggle kernels logs -f` exited silently because the CLI polled `ListKernelSessionOutput` first, then called `GetKernelSessionStatus`; the status call returned Kaggle API 500 and the CLI broke out of follow mode.
  - During the run, output download was still empty.
- Full Kaggle audit: completed on Kaggle train notebook version 1.
  - rows: 1,782,279
  - wells: 773
  - output: `/tmp/kaggle-output/exp040_ravaghi_pf_ancc_pfz_feature_ablation/train_v1`
  - synced local artifacts:
    - `metrics.json`
    - `artifacts/single_lgbm_metrics.csv`
    - `artifacts/single_lgbm_bucket_metrics.csv`
    - `artifacts/single_lgbm_exp026_source_summary.csv`
    - `artifacts/single_lgbm_feature_importance.csv`
    - `artifacts/single_lgbm_split_metrics.csv`
    - `artifacts/single_lgbm_summary.json`
    - `artifacts/single_lgbm_train_summary.csv`
    - `artifacts/single_lgbm_well_metrics.csv`
  - best overall report control: `pf090_hold010`, RMSE 15.089532.
  - selected supported bucket-shrink feature candidate: `base_plus_pf_ancc_pfz_uncertainty_bucket_shrink`.
  - selected original-fold RMSE: 16.011790.
  - selected well-hash RMSE: 15.645900.
  - selected vs `base_geometry_bucket_shrink`: -3.077619 original-fold / -3.313418 well-hash.
  - selected vs `exp026_regenerated_bucket_shrink`: -0.471838 original-fold / -0.783713 well-hash.
  - selected vs `public_pf_selector`: +0.839154 original-fold / +0.473264 well-hash.
  - selected vs `pf090_hold010`: +0.922258 original-fold / +0.556368 well-hash.
- LB: not submitted.

## 次のアクション

1. `ravaghi_pf_ancc_pfz_feature_ablation` は完了扱いにする。
2. inference port / submit は行わない。PF ANCC/PFZ proxy は direct replacement ではなく confidence / divergence feature family として後続候補に限定する。
