# exp041_ravaghi_beam_exact_feature_ablation セッションノート

## 状態

- status: implemented
- route: `ml_model`
- parent: `exp040_ravaghi_pf_ancc_pfz_feature_ablation`
- supporting artifact: `exp029_public_sel15_pf_oof_feature_generation`

## 仮説

`exp038` では aggregate beam feature、`exp040` では PF ANCC/PFZ proxy が base single-LGBM reference を改善したが、direct public PF controls には届かなかった。Ravaghi notebook の元実装は 7 本の named beam path (`cons`, `loose`, `vcons`, `sm5`, `vloose`, `mid`, `stiff`) を持つため、aggregate ではなく exact path / cost / spread / PF disagreement として渡すと、beam tracker が有効な条件を LightGBM がより細かく拾える可能性がある。

## 実装メモ

- `exp040` を土台に `exp041` を作成。
- `exp029` train well の途中以降を隠した疑似 test rows を入力に使い、各 `(well_id, cutoff_row)` で train horizontal/typewell CSV から exact Ravaghi beam paths を再生成する。
- exact beam は、pseudo cutoff 以降の `TVT_input` を隠し、visible prefix の last TVT、horizontal `GR`、typewell `TVT/GR` だけで計算する。
- variants:
  - `base_geometry`
  - `base_plus_public_beam_aggregate`
  - `base_plus_beam_exact_paths`
  - `base_plus_beam_exact_diagnostics`
  - `base_plus_beam_exact_disagreement`
  - `base_plus_beam_exact_pf_context`
  - `base_plus_public_and_exact_beam`
- `pf_error`、`last_anchor_error`、`beam_error`、`exp026_oof`、exp026 bridge columns は feature から除外。
- train-only の `ANCC` など formation columns は読まない。
- direct `beam` / `public_pf_selector` / `pf090_hold010` は report control として残し、model feature candidate とは分ける。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp041_ravaghi_beam_exact_feature_ablation
uv run python scripts/new_experiment.py --name exp041_ravaghi_beam_exact_feature_ablation --source experiments/exp040_ravaghi_pf_ancc_pfz_feature_ablation
uv run ruff check experiments/exp041_ravaghi_beam_exact_feature_ablation/ravaghi_single_lgbm_audit.py experiments/exp041_ravaghi_beam_exact_feature_ablation/settings.py
uv run python -m py_compile experiments/exp041_ravaghi_beam_exact_feature_ablation/ravaghi_single_lgbm_audit.py experiments/exp041_ravaghi_beam_exact_feature_ablation/settings.py
uv run python scripts/validate_experiment.py --experiment exp041_ravaghi_beam_exact_feature_ablation
uv run python experiments/exp041_ravaghi_beam_exact_feature_ablation/ravaghi_single_lgbm_audit.py --max-wells 5 --max-train-rows 1000 --skip-exp026-control --base-estimator HistGradientBoostingRegressor --output-dir /tmp/exp041_smoke
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp041_ravaghi_beam_exact_feature_ablation --notebook train --kernel-id kentookumura/exp041-ravaghi-beam-exact-train --title "exp041 ravaghi beam exact train" --run-on-push --strict
kaggle kernels push -p experiments/exp041_ravaghi_beam_exact_feature_ablation/kaggle/train
kaggle kernels pull kentookumura/exp041-ravaghi-beam-exact-train -p /tmp/kaggle-pull/exp041-ravaghi-beam-exact-train-check -m
kaggle kernels logs kentookumura/exp041-ravaghi-beam-exact-train
kaggle kernels output kentookumura/exp041-ravaghi-beam-exact-train -p /tmp/kaggle-output/exp041_ravaghi_beam_exact_feature_ablation/train_v1_probe
kaggle kernels logs kentookumura/exp041-ravaghi-beam-exact-train
kaggle kernels output kentookumura/exp041-ravaghi-beam-exact-train -p /tmp/kaggle-output/exp041_ravaghi_beam_exact_feature_ablation/train_v1
kaggle kernels pull kentookumura/exp041-ravaghi-beam-exact-train -p /tmp/kaggle-pull/exp041-ravaghi-beam-exact-train-complete -m
uv run python scripts/update_experiment_summary.py
```

## 結果

- Static checks: PASS
  - `ruff check`: PASS
  - `py_compile`: PASS
  - `scripts/validate_experiment.py --experiment exp041_ravaghi_beam_exact_feature_ablation`: PASS
- Local smoke: PASS
  - command: `--max-wells 5 --max-train-rows 1000 --skip-exp026-control --base-estimator HistGradientBoostingRegressor`
  - rows: 11,207
  - wells: 5
  - exact beam generated for 5 `(well_id, cutoff_row)` groups
  - output: `/tmp/exp041_smoke`
  - selected smoke candidate: `base_plus_beam_exact_pf_context_bucket_shrink`
  - smoke score is not recorded as CV because it uses 5 wells and a lightweight estimator.
- Kaggle train:
  - prepared train notebook with canonical id `kentookumura/exp041-ravaghi-beam-exact-train`.
  - pushed version 1 successfully.
  - URL: `https://www.kaggle.com/code/kentookumura/exp041-ravaghi-beam-exact-train`
  - direct `kaggle kernels pull ... -m` succeeds and returns `id_no: 122077235`, so the kernel exists.
  - `kaggle kernels status` returned Kaggle API 500, consistent with the known status API issue.
  - shortly after push, normal `logs`, `logs -f`, and `output` returned empty, treated as API/session output lag or queued/running state rather than a failure.
  - monitoring stopped at user request; user will report completion.
- Full Kaggle audit: completed on Kaggle train notebook version 1.
  - rows: 1,782,279
  - wells: 773
  - runtime from log: about 4,495 seconds
  - output: `/tmp/kaggle-output/exp041_ravaghi_beam_exact_feature_ablation/train_v1`
  - synced local artifacts:
    - `metrics.json`
    - `artifacts/exp041-ravaghi-beam-exact-train.log`
    - `artifacts/single_lgbm_metrics.csv`
    - `artifacts/single_lgbm_bucket_metrics.csv`
    - `artifacts/single_lgbm_exp026_source_summary.csv`
    - `artifacts/single_lgbm_feature_importance.csv`
    - `artifacts/single_lgbm_split_metrics.csv`
    - `artifacts/single_lgbm_summary.json`
    - `artifacts/single_lgbm_train_summary.csv`
    - `artifacts/single_lgbm_well_metrics.csv`
  - best overall report control: `pf090_hold010`, RMSE 15.089532.
  - selected supported bucket-shrink feature candidate: `base_plus_beam_exact_disagreement_bucket_shrink`.
  - selected original-fold RMSE: 15.527268.
  - selected well-hash RMSE: 15.727948.
  - selected vs `base_geometry_bucket_shrink`: -3.562141 original-fold / -3.231370 well-hash.
  - selected vs `exp026_regenerated_bucket_shrink`: -0.956360 original-fold / -0.701665 well-hash.
  - selected vs `public_pf_selector`: +0.354631 original-fold / +0.555312 well-hash.
  - selected vs `pf090_hold010`: +0.437736 original-fold / +0.638416 well-hash.
- LB: not submitted.

## 次のアクション

1. `ravaghi_beam_exact_feature_ablation` は完了扱いにする。
2. inference port / submit は行わない。Exact beam disagreement は direct PF controls より弱いため、後続では confidence / divergence feature family または raw candidate input として扱う。
