# exp043_ravaghi_feature_family_ablation_matrix セッションノート

## 状態

- status: implemented
- route: `ml_model`
- parent: `exp042_ravaghi_ncc_gr_match_features`
- supporting artifact: `exp029_public_sel15_pf_oof_feature_generation`

## 仮説

`exp038` から `exp042` で個別に見た Ravaghi 由来 feature family を、
同一 train well の途中以降を隠した疑似 test rows と同一 split surface で横並び比較する。PF
prediction、PF uncertainty、exact beam disagreement、NCC/GR diagnostics、
見えない test で使える spatial/formation proxy を分け、強い family と弱い family を
distance bucket / fold / well group / feature importance で確認する。

## 実装メモ

- `exp042` を土台に `exp043` を作成。
- `exp029` train well の途中以降を隠した疑似 test rows を入力に使う。
- exact beam と NCC/GR match は pseudo cutoff 以降の `TVT_input` を隠して再生成する。
- spatial/formation proxy は train-only formation columns を読まず、X/Y/Z/MD と public PF/beam disagreement から作る。
- candidate variants:
  - `base_geometry`
  - `base_plus_pf_prediction`
  - `base_plus_pf_uncertainty`
  - `base_plus_pf_prediction_uncertainty`
  - `base_plus_public_beam_aggregate`
  - `base_plus_exact_beam_disagreement`
  - `base_plus_pf_exact_beam_disagreement`
  - `base_plus_ncc_disagreement`
  - `base_plus_ncc_gr_match`
  - `base_plus_ncc_gr_match_pf_context`
  - `base_plus_spatial_formation_proxy`
  - `base_plus_supported_ravaghi_core`
  - `base_plus_all_ravaghi_families`
- `pf_error`、`last_anchor_error`、`beam_error`、`exp026_oof`、exp026 bridge columns は feature から除外。
- direct `beam` / `public_pf_selector` / `pf090_hold010` は report control として残し、model feature candidate とは分ける。
- `single_lgbm_family_matrix.csv` を追加し、candidate ごとの family flag と overall RMSE を保存する。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp043_ravaghi_feature_family_ablation_matrix
uv run python scripts/new_experiment.py --name exp043_ravaghi_feature_family_ablation_matrix --source experiments/exp042_ravaghi_ncc_gr_match_features
uv run ruff check experiments/exp043_ravaghi_feature_family_ablation_matrix/ravaghi_single_lgbm_audit.py experiments/exp043_ravaghi_feature_family_ablation_matrix/settings.py
uv run python -m py_compile experiments/exp043_ravaghi_feature_family_ablation_matrix/ravaghi_single_lgbm_audit.py experiments/exp043_ravaghi_feature_family_ablation_matrix/settings.py
uv run python scripts/validate_experiment.py --experiment exp043_ravaghi_feature_family_ablation_matrix
uv run python experiments/exp043_ravaghi_feature_family_ablation_matrix/ravaghi_single_lgbm_audit.py --max-wells 3 --max-train-rows 500 --skip-exp026-control --base-estimator HistGradientBoostingRegressor --output-dir /tmp/exp043_smoke
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp043_ravaghi_feature_family_ablation_matrix --notebook train --kernel-id kentookumura/exp043-ravaghi-family-matrix-train --title "exp043 ravaghi family matrix train" --run-on-push --strict
kaggle kernels push -p experiments/exp043_ravaghi_feature_family_ablation_matrix/kaggle/train
kaggle kernels pull kentookumura/exp043-ravaghi-family-matrix-train -p /tmp/kaggle-pull/exp043-ravaghi-family-matrix-train-check -m
kaggle kernels logs kentookumura/exp043-ravaghi-family-matrix-train
kaggle kernels output kentookumura/exp043-ravaghi-family-matrix-train -p /tmp/kaggle-output/exp043_ravaghi_feature_family_ablation_matrix/train_v1
kaggle kernels pull kentookumura/exp043-ravaghi-family-matrix-train -p /tmp/kaggle-pull/exp043-ravaghi-family-matrix-train-complete -m
```

## 結果

- Static checks: PASS
  - `ruff check`: PASS
  - `py_compile`: PASS
  - `scripts/validate_experiment.py --experiment exp043_ravaghi_feature_family_ablation_matrix`: PASS
- Local smoke: PASS
  - command: `--max-wells 3 --max-train-rows 500 --skip-exp026-control --base-estimator HistGradientBoostingRegressor`
  - rows: 6,727
  - wells: 3
  - exact beam features generated for 3 `(well_id, cutoff_row)` groups
  - NCC/GR match features generated for 3 `(well_id, cutoff_row)` groups
  - output: `/tmp/exp043_smoke`
  - smoke score is not recorded as CV because it uses 3 wells and a lightweight estimator.
- Kaggle train package: prepared
  - path: `experiments/exp043_ravaghi_feature_family_ablation_matrix/kaggle/train`
  - kernel id: `kentookumura/exp043-ravaghi-family-matrix-train`
- Kaggle train:
  - pushed version 1 successfully.
  - URL: `https://www.kaggle.com/code/kentookumura/exp043-ravaghi-family-matrix-train`
  - direct `kaggle kernels pull ... -m` succeeds, so the kernel exists on Kaggle.
  - shortly after push, normal `logs` and `output` returned empty; treated as API/session output lag rather than a failure.
  - `logs -f` first failed locally with DNS resolution error, then connected after network approval but still had no CLI output before the user interrupted polling.
- Full Kaggle audit: completed on Kaggle train notebook version 1.
  - rows: 1,782,279
  - wells: 773
  - runtime from log: about 3,900 seconds
  - output: `/tmp/kaggle-output/exp043_ravaghi_feature_family_ablation_matrix/train_v1`
  - synced local artifacts:
    - `metrics.json`
    - `artifacts/exp043-ravaghi-family-matrix-train.log`
    - `artifacts/single_lgbm_metrics.csv`
    - `artifacts/single_lgbm_bucket_metrics.csv`
    - `artifacts/single_lgbm_exp026_source_summary.csv`
    - `artifacts/single_lgbm_family_matrix.csv`
    - `artifacts/single_lgbm_feature_importance.csv`
    - `artifacts/single_lgbm_split_metrics.csv`
    - `artifacts/single_lgbm_summary.json`
    - `artifacts/single_lgbm_train_summary.csv`
    - `artifacts/single_lgbm_well_metrics.csv`
  - best overall report control: `pf090_hold010`, RMSE 15.089532.
  - `public_pf_selector`: 15.172636.
  - best overall single-LGBM feature candidate by original-fold: `base_plus_ncc_gr_match_pf_context_raw`, original-fold 15.526227 / well-hash 15.406654.
  - selected supported bucket-shrink candidate: `base_plus_ncc_gr_match_pf_context_bucket_shrink`, original-fold 15.634436 / well-hash 15.485651.
  - selected bucket candidate vs `base_geometry_bucket_shrink`: -3.454973 original-fold / -3.473667 well-hash.
  - selected bucket candidate vs `public_pf_selector`: +0.461800 original-fold / +0.313015 well-hash.
  - selected bucket candidate vs `pf090_hold010`: +0.544904 original-fold / +0.396119 well-hash.
  - `base_plus_pf_prediction_bucket_shrink`: original-fold 15.862071 / well-hash 15.804236.
  - `base_plus_pf_uncertainty_bucket_shrink`: original-fold 16.023513 / well-hash 15.596025.
  - `base_plus_exact_beam_disagreement_bucket_shrink`: original-fold 15.986100 / well-hash 16.020314.
  - `base_plus_spatial_formation_proxy_bucket_shrink`: original-fold 17.280459 / well-hash 16.959015.
  - `base_plus_ncc_disagreement_bucket_shrink`: original-fold 17.730920 / well-hash 17.703975.
  - family matrix confirms add-only Ravaghi features improve over the weak base single-LGBM control, but none beat direct public PF controls.
- LB: not submitted.

## 次のアクション

1. inference port / submit は行わない。`base_plus_ncc_gr_match_pf_context_bucket_shrink` は direct PF controls に届かない。
2. Ravaghi family は direct replacement ではなく confidence/gate feature、または XGBoost/CatBoost probe 用の候補として扱う。
