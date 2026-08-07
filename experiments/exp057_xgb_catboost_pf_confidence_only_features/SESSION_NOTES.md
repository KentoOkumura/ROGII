# exp057_xgb_catboost_pf_confidence_only_features セッションノート

## 目的

高優先 backlog `xgb_catboost_pf_confidence_only_features` を実装する。PF/Beam
予測の直接置換ではなく、信頼度・不一致だけを XGBoost / CatBoost residual
model の特徴量として追加し、対応する geometry control と比較する。

## 現在の状態

- status: implemented
- route: `ml_model`
- parent: `exp049_xgboost_pseudo_tail_residual`
- implementation parent: `exp055_single_model_pseudotail_training`
- supporting artifact: `exp029_public_sel15_pf_oof_feature_generation`
- selected variant: none
- CV: not run
- LB: not submitted

## 実装メモ

- `exp055` を土台に `exp057` を作成。
- `ravaghi_single_lgbm_audit.py` を `pf_confidence_model_audit.py` に rename。
- variant ごとに `estimator` / `params` を指定できるようにし、`XGBRegressor` と `CatBoostRegressor` を追加。
- feature family を `geometry_anchor_context` と `pf_beam_confidence_only` に分離。
- PF/Beam の raw prediction と exp026_oof raw prediction は model feature から除外。
- `abs_pf_pred_minus_exp026_oof` は `PF-exp052 diff` が未保存のため proxy として採用。
- supported 判定は confidence-only candidate が対応する geometry control を original-fold / well-hash の両方で上回ること。
- output は `artifacts/pf_confidence_*.csv` と `artifacts/pf_confidence_summary.json` に保存する。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp057_xgb_catboost_pf_confidence_only_features
uv run python scripts/new_experiment.py --name exp057_xgb_catboost_pf_confidence_only_features --source experiments/exp055_single_model_pseudotail_training
```

## 検証状況

- Static checks: PASS
  - `uv run python -m py_compile experiments/exp057_xgb_catboost_pf_confidence_only_features/pf_confidence_model_audit.py experiments/exp057_xgb_catboost_pf_confidence_only_features/baseline.py experiments/exp057_xgb_catboost_pf_confidence_only_features/pseudo_tail_augmentation.py experiments/exp057_xgb_catboost_pf_confidence_only_features/settings.py`
  - `uv run ruff check experiments/exp057_xgb_catboost_pf_confidence_only_features/pf_confidence_model_audit.py experiments/exp057_xgb_catboost_pf_confidence_only_features/baseline.py experiments/exp057_xgb_catboost_pf_confidence_only_features/pseudo_tail_augmentation.py experiments/exp057_xgb_catboost_pf_confidence_only_features/settings.py`
  - `uv run python scripts/validate_experiment.py --experiment exp057_xgb_catboost_pf_confidence_only_features`
- Required feature columns: PASS
  - feature path: `experiments/exp029_public_sel15_pf_oof_feature_generation/features/public_sel15_pf_oof_features.csv.gz`
  - required columns: 44
  - missing columns: none
- Kaggle train package: prepared
  - command: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp057_xgb_catboost_pf_confidence_only_features --notebook train --kernel-id kentookumura/exp057-xgb-catboost-pf-conf-train --title "exp057 xgb catboost pf confidence train" --run-on-push --strict`
  - path: `experiments/exp057_xgb_catboost_pf_confidence_only_features/kaggle/train`
  - kernel id: `kentookumura/exp057-xgb-catboost-pf-conf-train`
  - title: `exp057 xgb catboost pf confidence train`
  - run_on_push: true
  - enable_gpu: false
  - enable_internet: false
- Kaggle train: completed
  - push command: `kaggle kernels push -p experiments/exp057_xgb_catboost_pf_confidence_only_features/kaggle/train`
  - version: 1
  - canonical kernel id: `kentookumura/exp057-xgb-catboost-pf-confidence-train`
  - URL: `https://www.kaggle.com/code/kentookumura/exp057-xgb-catboost-pf-confidence-train`
  - push warning: local metadata id `kentookumura/exp057-xgb-catboost-pf-conf-train` did not resolve from the title slug; Kaggle created / updated canonical slug `exp057-xgb-catboost-pf-confidence-train`.
  - pull existence check: `kaggle kernels pull kentookumura/exp057-xgb-catboost-pf-confidence-train -p /tmp/kaggle-pull/exp057-xgb-catboost-pf-confidence-train -m` succeeded.
  - initial normal `logs` and `output` were empty; short `logs -f` polling returned full logs and completion output.
  - output: `/tmp/kaggle-output/exp057_xgb_catboost_pf_confidence_only_features/train_v1`
  - synced local artifacts:
    - `metrics.json`
    - `artifacts/exp057-xgb-catboost-pf-confidence-train.log`
    - `artifacts/pf_confidence_bucket_metrics.csv`
    - `artifacts/pf_confidence_family_matrix.csv`
    - `artifacts/pf_confidence_feature_importance.csv`
    - `artifacts/pf_confidence_feature_parity_report.csv`
    - `artifacts/pf_confidence_metrics.csv`
    - `artifacts/pf_confidence_split_metrics.csv`
    - `artifacts/pf_confidence_summary.json`
    - `artifacts/pf_confidence_train_summary.csv`
    - `artifacts/pf_confidence_well_metrics.csv`

## 結果

- rows / wells: 1,782,279 / 773
- best original-fold candidate: `pf090_hold010` 15.089532
- best well-hash candidate: `pf090_hold010` 15.089532
- selected paired-control candidate: `catboost_pf_confidence_only_raw`
- `catboost_pf_confidence_only_raw`: original-fold 15.609958 / well-hash 15.440706
- `catboost_geometry_control_raw`: original-fold 18.508503 / well-hash 18.502741
- CatBoost raw confidence delta vs paired control: -2.898545 original-fold / -3.062035 well-hash
- `catboost_pf_confidence_only_bucket_shrink`: original-fold 15.645771 / well-hash 15.451603
- `xgb_pf_confidence_only_raw`: original-fold 15.836064 / well-hash 15.662857
- `xgb_pf_confidence_only_bucket_shrink`: original-fold 15.958626 / well-hash 15.777023
- direct PF controls remained stronger:
  - `pf090_hold010`: 15.089532 / 15.089532
  - `public_pf_selector`: 15.172636 / 15.172636
- requested cutoffs `[0.45, 0.65, 0.82]` のうち available は 0.65 のみ。0.45 / 0.82 は missing として train summary に記録された。

Interpretation:

- PF/Beam confidence-only features are useful model features on this exp029 pseudo-test surface.
- The effect is large versus geometry-only XGB/CatBoost controls, but still does not beat direct public PF controls.
- This is not enough to justify direct inference port / submit. Follow-up should test confidence-only diagnostics in the stronger exp051/052 LGBM capacity pseudo-tail surface, or use disagreement for sample weighting / residual clipping.

## 次のアクション

1. `experiment_summary.md` と `KAGGLE_DIRECTION.md` を更新する。
2. 直接 submit はしない。
3. 次に進めるなら `lgbm_pf_confidence_only_features` または `pf_beam_disagreement_sample_weight` を優先する。
