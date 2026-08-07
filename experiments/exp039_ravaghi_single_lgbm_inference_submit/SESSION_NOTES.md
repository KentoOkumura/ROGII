# exp039_ravaghi_single_lgbm_inference_submit セッションノート

## 状態

- status: completed
- route: `ml_model`
- parent: `exp038_ravaghi_public_sel15_features_single_lgbm`
- inference parent: `exp035_public_sel15_pf_meta_inference_port`

## 仮説

exp038 selected `base_plus_pf_prediction_bucket_shrink` を public sel15 inference flow に移植し、見えない test well 用処理 で single-LGBM residual prediction を使う。

## 実装メモ

- exp035 の physical visible branch / PF feature generation / Beam / selector logic を流用。
- exp035 の Ridge meta residual branch は使わない。
- exp029 artifact から `base_plus_pf_prediction` feature set で final single LightGBM を fit。
- 見えない test well 用処理は `last_anchor + 0.85 * clip(lgbm_residual, -80, 80)` に fixed bucket shrink を適用。
- public visible sample wells は physical branch のまま変更しない。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp039_ravaghi_single_lgbm_inference_submit
uv run python scripts/new_experiment.py --name exp039_ravaghi_single_lgbm_inference_submit --source experiments/exp035_public_sel15_pf_meta_inference_port --skip-steering-check
uv run python scripts/validate_experiment.py --experiment exp039_ravaghi_single_lgbm_inference_submit
```

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp039_ravaghi_single_lgbm_inference_submit --notebook inference --kernel-id kentookumura/exp039-ravaghi-lgbm-infer --title "exp039 ravaghi lgbm infer" --run-on-push --strict
kaggle kernels push -p experiments/exp039_ravaghi_single_lgbm_inference_submit/kaggle/inference
kaggle kernels logs kentookumura/exp039-ravaghi-lgbm-infer
kaggle kernels output kentookumura/exp039-ravaghi-lgbm-infer -p /tmp/kaggle-output/exp039_ravaghi_single_lgbm_inference_submit/inference_v2
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp039_ravaghi_single_lgbm_inference_submit/inference_v2/submission.csv
kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp039-ravaghi-lgbm-infer -v 2 -f submission.csv -m "exp039 ravaghi single lgbm hidden branch"
kaggle competitions submissions rogii-wellbore-geology-prediction
```

## 結果

- Kaggle inference v1: completed, but not submitted. Log showed `PF failed: name 'run_particle_filter_diag' is not defined`; fixed by restoring the missing exp035 PF helper before rerun.
- Kaggle inference v2: completed on `kentookumura/exp039-ravaghi-lgbm-infer`.
- v2 log: physical model, PF 128-seed likelihood ensemble, Beam 14-config ensemble, and selector all completed for public sample wells `000d7d20`, `00bbac68`, `00e12e8b`.
- v2 output rows: 14,151.
- single-LGBM model: feature_count=23, train_rows=277,877, candidate=`base_plus_pf_prediction_bucket_shrink`.
- public sample diff vs exp027 anchor: changed_rows=0, changed_wells=0, diff_rmse=0.0. This is expected because the visible public sample branch stays physical/PF and hidden-only LGBM corrections are not visible in the local public sample output.
- submit-check: PASS.
- submission SHA256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`.
- Kaggle submit: submitted from kernel version 2.
- submission ref: `53464736`.
- status at final check: `SubmissionStatus.COMPLETE`.
- Public LB: 11.740.
- comparison: updates the ML route Public LB anchor from exp026 12.102 to 11.740; still worse than the overall/PF route exp027 Public LB anchor 8.781 by +2.959.

## 次のアクション

1. ML route の Public LB anchor は exp039 11.740 に更新する。
2. 全体 / PF route anchor は exp027 8.781 のまま維持する。
3. Ravaghi/PF 系は次に confidence / divergence feature や gate の材料として扱う。
