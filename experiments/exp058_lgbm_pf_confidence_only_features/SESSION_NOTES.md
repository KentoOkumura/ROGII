# exp058_lgbm_pf_confidence_only_features セッションノート

## 目的

高優先 backlog `lgbm_pf_confidence_only_features` を実装する。`exp051/052` の
LightGBM capacity pseudo-tail 設定を固定し、PF/Beam 予測の直接置換ではなく、
信頼度・不一致だけを residual model の特徴量として追加して same-run geometry
control と比較する。

## 現在の状態

- status: implemented
- route: `ml_model`
- parent: `exp051_pseudo_tail_lgbm_param_micro_tune`
- inference parent: `exp052_lgbm_capacity_pseudotail_inference_submit`
- implementation parent: `exp057_xgb_catboost_pf_confidence_only_features`
- supporting artifact: `exp029_public_sel15_pf_oof_feature_generation`
- selected variant: none
- CV: not run
- LB: not submitted

## 実装メモ

- `exp057` を土台に `exp058` を作成。
- `pf_confidence_model_audit.py` を再利用し、variant を LightGBM capacity の paired control に限定。
- `lgbm_capacity_geometry_control`: `geometry_anchor_context` のみ。
- `lgbm_capacity_pf_confidence_only`: `geometry_anchor_context` と `pf_beam_confidence_only` を追加。
- LightGBM params は `exp051` selected capacity 設定の `num_leaves=47`、`min_child_samples=60`、`n_estimators=800`、`learning_rate=0.03`。
- cutoff filtering、distance-balanced sampling、row cap、residual shrink、fixed bucket-shrink postprocess は `exp051/052` と同じ設定。
- PF/Beam の raw prediction と exp026_oof raw prediction は model feature から除外。
- `abs_pf_pred_minus_exp026_oof` は `PF-exp052 diff` が未保存のため proxy として採用。
- supported 判定は confidence-only candidate が対応する LightGBM geometry control を original-fold / well-hash の両方で上回ること。
- output は `artifacts/pf_confidence_*.csv` と `artifacts/pf_confidence_summary.json` に保存する。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp058_lgbm_pf_confidence_only_features
uv run python scripts/new_experiment.py --name exp058_lgbm_pf_confidence_only_features --source experiments/exp057_xgb_catboost_pf_confidence_only_features
```

## 検証状況

- Static checks: PASS
  - `uv run python -m py_compile experiments/exp058_lgbm_pf_confidence_only_features/pf_confidence_model_audit.py experiments/exp058_lgbm_pf_confidence_only_features/baseline.py experiments/exp058_lgbm_pf_confidence_only_features/pseudo_tail_augmentation.py experiments/exp058_lgbm_pf_confidence_only_features/settings.py`
  - `uv run ruff check experiments/exp058_lgbm_pf_confidence_only_features/pf_confidence_model_audit.py experiments/exp058_lgbm_pf_confidence_only_features/baseline.py experiments/exp058_lgbm_pf_confidence_only_features/pseudo_tail_augmentation.py experiments/exp058_lgbm_pf_confidence_only_features/settings.py`
  - `uv run python scripts/validate_experiment.py --experiment exp058_lgbm_pf_confidence_only_features`
- Required feature columns: PASS
  - feature path: `experiments/exp029_public_sel15_pf_oof_feature_generation/features/public_sel15_pf_oof_features.csv.gz`
  - required columns: 44
  - missing columns: none
- Kaggle train package: prepared
  - command: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp058_lgbm_pf_confidence_only_features --notebook train --kernel-id kentookumura/exp058-lgbm-pf-confidence-train --title "exp058 lgbm pf confidence train" --run-on-push --strict`
  - path: `experiments/exp058_lgbm_pf_confidence_only_features/kaggle/train`
  - kernel id: `kentookumura/exp058-lgbm-pf-confidence-train`
  - title: `exp058 lgbm pf confidence train`
  - run_on_push: true
  - enable_gpu: false
  - enable_internet: false
- Kaggle train: completed
  - push command: `kaggle kernels push -p experiments/exp058_lgbm_pf_confidence_only_features/kaggle/train`
  - version: 1
  - kernel id: `kentookumura/exp058-lgbm-pf-confidence-train`
  - URL: `https://www.kaggle.com/code/kentookumura/exp058-lgbm-pf-confidence-train`
  - pull existence check: `kaggle kernels pull kentookumura/exp058-lgbm-pf-confidence-train -p /tmp/kaggle-pull/exp058-lgbm-pf-confidence-train -m` succeeded.
  - initial normal `logs` was empty; short `logs -f` polling returned full logs and completion output.
  - output: `/tmp/kaggle-output/exp058_lgbm_pf_confidence_only_features/train_v1`
  - synced local artifacts:
    - `metrics.json`
    - `artifacts/exp058-lgbm-pf-confidence-train.log`
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
- selected paired-control candidate: `lgbm_capacity_pf_confidence_only_raw`
- `lgbm_capacity_pf_confidence_only_raw`: original-fold 15.945678 / well-hash 15.569216
- `lgbm_capacity_geometry_control_raw`: original-fold 18.647176 / well-hash 18.841697
- raw confidence delta vs paired control: -2.701498 original-fold / -3.272481 well-hash
- `lgbm_capacity_pf_confidence_only_bucket_shrink`: original-fold 16.071589 / well-hash 15.671243
- `lgbm_capacity_geometry_control_bucket_shrink`: original-fold 18.877985 / well-hash 19.119130
- bucket-shrink confidence delta vs paired control: -2.806396 original-fold / -3.447887 well-hash
- direct PF controls as diagnostic ceiling values:
  - `pf090_hold010`: 15.089532 / 15.089532
  - `public_pf_selector`: 15.172636 / 15.172636
- requested cutoffs `[0.45, 0.65, 0.82]` のうち available は 0.65 のみ。0.45 / 0.82 は missing として train summary に記録された。

Interpretation:

- PF/Beam confidence-only features are useful model features on this exp029 pseudo-test surface.
- The effect is large versus the LightGBM capacity geometry-only control, so this is a positive ML-route feature audit.
- Direct public PF controls are not the primary route comparison; they are reported as ceiling diagnostics for this exp029 pseudo-test surface.
- This is not enough to justify direct inference port / submit because the ML-anchor disagreement feature is missing and the fixed exp014 bucket-shrink is mismatched to this surface.

## Inference / submit

- User requested submit despite the train-side caution above, so the selected raw candidate was ported inside the same exp058 experiment.
- Inference notebook: `experiments/exp058_lgbm_pf_confidence_only_features/exp058_lgbm_pf_confidence_only_features_inference.ipynb`
- Candidate: `lgbm_capacity_pf_confidence_only_raw_hidden`
- Hidden-branch train source: `kentookumura/exp029-sel15-pf-oof-train/features/public_sel15_pf_oof_features.csv.gz`
- PF settings for hidden branch: 250 particles / 16 seeds, matching the exp029 artifact scale.
- Postprocess: raw residual prediction only; exp014 bucket-shrink is intentionally not applied.
- Kaggle package command:
  - `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp058_lgbm_pf_confidence_only_features --notebook inference --kernel-id kentookumura/exp058-lgbm-pf-confidence-inference --title "exp058 lgbm pf confidence inference" --run-on-push --strict`
- Kaggle inference:
  - initial push used id `kentookumura/exp058-lgbm-pf-confidence-infer` with title `exp058 lgbm pf confidence inference`, which produced a slug warning and created the URL slug `kentookumura/exp058-lgbm-pf-confidence-inference`.
  - local prepared metadata was regenerated with the canonical id `kentookumura/exp058-lgbm-pf-confidence-inference` to avoid future slug mismatch.
  - version: 1
  - URL: `https://www.kaggle.com/code/kentookumura/exp058-lgbm-pf-confidence-inference`
  - output: `/tmp/kaggle-output/exp058_lgbm_pf_confidence_only_features/inference_v1`
  - log: `/tmp/kaggle-output/exp058_lgbm_pf_confidence_only_features/inference_v1/exp058-lgbm-pf-confidence-inference.log`
- Inference output:
  - rows: 14,151
  - columns: `id,tvt`
  - SHA256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
  - range: 11587.038593 - 12240.016066
  - public sample changed rows: 0 / 14,151
  - public sample changed wells: 0
  - reason: the public sample wells are visible train wells, so the physical replay branch is used locally; hidden code execution is still present for hidden wells.
- Submit-check: PASS
  - command: `python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp058_lgbm_pf_confidence_only_features/inference_v1/submission.csv --sample data/raw/sample_submission.csv`
- Code submission:
  - command: `kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp058-lgbm-pf-confidence-inference -v 1 -f submission.csv -m "exp058 lgbm pf confidence raw hidden branch"`
  - ref: `53535327`
  - submitted_at: `2026-06-10 12:26:23.747000`
  - status: `COMPLETE`
  - Public LB: 12.778
  - outcome: worse than exp052 12.076 by +0.702 and exp054 11.856 by +0.922; do not adopt this hidden branch.

## 次のアクション

1. 別実験として、`PF-vs-exp052/054` 差分を fold-safe に使う feature surface と、exp058 surface 専用の fold-out bucket shrink を実装する。
2. exp058 hidden branch は exp052/054 ML-route anchor を大きく下回ったため採用しない。
