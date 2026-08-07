# exp048_ravaghi_single_model_feature_parity_revisit セッションノート

## 状態

- status: completed
- route: `ml_model`
- parent: `exp043_ravaghi_feature_family_ablation_matrix`
- supporting artifact: `exp029_public_sel15_pf_oof_feature_generation`

## 仮説

`exp038` から `exp043` の Ravaghi 由来特徴量は弱い `base_geometry` 単体 LGBM には効いたが、
`public_pf_selector` と `pf090_hold010` には届かなかった。再訪では、推論移植を前提にせず、
train 側の見えない test 風データ上で特徴列、欠損処理、PF/Beam 条件を明示的に監査し、
raw 予測、固定 bucket shrink、anchor gate、public PF blend を分けて評価する。

成功条件は `base_geometry_bucket_shrink` だけでなく、`public_pf_selector` と `pf090_hold010` を
original-fold / well-hash の両方で上回ることにする。

## 実装メモ

- `exp043` を土台に `exp048` を作成。
- `settings.py` の `EXPERIMENT_NAME` と notebook ファイル名を `exp048` に更新。
- `ravaghi_single_lgbm_audit.py` に以下を追加。
  - postprocess 候補: raw、fixed bucket shrink、anchor gate、public PF blend。
  - `audit.success_controls` による direct PF controls 超えの support 判定。
  - `single_lgbm_feature_parity_report.csv` の保存。
  - family matrix で anchor / PF blend 候補を variant と postprocess に分解。
- `config.yaml` では `public_pf_selector` と `pf090_hold010` を必須成功比較基準にした。
- direct PF/Beam replacement、Ridge/meta-stack、learned router、train-only formation columns は使わない。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp048_ravaghi_single_model_feature_parity_revisit
uv run python scripts/new_experiment.py --name exp048_ravaghi_single_model_feature_parity_revisit --source experiments/exp043_ravaghi_feature_family_ablation_matrix
uv run ruff check experiments/exp048_ravaghi_single_model_feature_parity_revisit/ravaghi_single_lgbm_audit.py experiments/exp048_ravaghi_single_model_feature_parity_revisit/settings.py
uv run python -m py_compile experiments/exp048_ravaghi_single_model_feature_parity_revisit/ravaghi_single_lgbm_audit.py experiments/exp048_ravaghi_single_model_feature_parity_revisit/settings.py
uv run python scripts/validate_experiment.py --experiment exp048_ravaghi_single_model_feature_parity_revisit
uv run python experiments/exp048_ravaghi_single_model_feature_parity_revisit/ravaghi_single_lgbm_audit.py --max-wells 3 --max-train-rows 500 --skip-exp026-control --base-estimator HistGradientBoostingRegressor --output-dir /tmp/exp048_smoke
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp048_ravaghi_single_model_feature_parity_revisit --notebook train --kernel-id kentookumura/exp048-ravaghi-parity-revisit-train --title "exp048 ravaghi parity revisit train" --run-on-push --strict
kaggle kernels push -p experiments/exp048_ravaghi_single_model_feature_parity_revisit/kaggle/train
kaggle kernels pull kentookumura/exp048-ravaghi-parity-revisit-train -p /tmp/kaggle-pull/exp048-ravaghi-parity-revisit-train-check -m
kaggle kernels logs kentookumura/exp048-ravaghi-parity-revisit-train
kaggle kernels output kentookumura/exp048-ravaghi-parity-revisit-train -p /tmp/kaggle-output/exp048_ravaghi_single_model_feature_parity_revisit/train_v1
kaggle kernels status kentookumura/exp048-ravaghi-parity-revisit-train
```

## 結果

- Static checks: PASS
  - `ruff check`: PASS
  - `py_compile`: PASS
  - `scripts/validate_experiment.py --experiment exp048_ravaghi_single_model_feature_parity_revisit`: PASS
- Local smoke: PASS
  - command: `--max-wells 3 --max-train-rows 500 --skip-exp026-control --base-estimator HistGradientBoostingRegressor`
  - rows: 6,727
  - wells: 3
  - exact beam features generated for 3 `(well_id, cutoff_row)` groups
  - NCC/GR match features generated for 3 `(well_id, cutoff_row)` groups
  - output: `/tmp/exp048_smoke`
  - generated `single_lgbm_feature_parity_report.csv`
  - smoke score is not recorded as CV because it uses 3 wells and a lightweight estimator.
- Kaggle train package: prepared
  - path: `experiments/exp048_ravaghi_single_model_feature_parity_revisit/kaggle/train`
  - kernel id: `kentookumura/exp048-ravaghi-parity-revisit-train`
  - title: `exp048 ravaghi parity revisit train`
  - run_on_push: true
- Kaggle train:
  - pushed version 1 successfully.
  - URL: `https://www.kaggle.com/code/kentookumura/exp048-ravaghi-parity-revisit-train`
  - direct `kaggle kernels pull ... -m` succeeds, so the kernel exists on Kaggle.
  - shortly after push, normal `logs` and `output` returned empty; treated as API/session output lag rather than failure.
  - supplemental `kaggle kernels status` returned `KernelWorkerStatus.RUNNING`.
  - later status returned `KernelWorkerStatus.COMPLETE`.
  - full audit completed on Kaggle train notebook version 1.
  - runtime from log: about 5,795 seconds.
  - rows: 1,782,279
  - wells: 773
  - output: `/tmp/kaggle-output/exp048_ravaghi_single_model_feature_parity_revisit/train_v1`
  - synced local artifacts:
    - `metrics.json`
    - `artifacts/exp048-ravaghi-parity-revisit-train.log`
    - `artifacts/single_lgbm_metrics.csv`
    - `artifacts/single_lgbm_bucket_metrics.csv`
    - `artifacts/single_lgbm_exp026_source_summary.csv`
    - `artifacts/single_lgbm_family_matrix.csv`
    - `artifacts/single_lgbm_feature_importance.csv`
    - `artifacts/single_lgbm_feature_parity_report.csv`
    - `artifacts/single_lgbm_split_metrics.csv`
    - `artifacts/single_lgbm_summary.json`
    - `artifacts/single_lgbm_train_summary.csv`
    - `artifacts/single_lgbm_well_metrics.csv`
  - original-fold best: `pf090_hold010`, RMSE 15.089532.
  - well-hash best: `base_plus_ncc_gr_match_pf_context_pf_blend_w0p30`, RMSE 15.019511.
  - same-surface ML control `exp026_regenerated_bucket_shrink`: original-fold 16.483627 / well-hash 16.429613.
  - standard ML route anchors for reference: usual CV `exp025` fixed bucket shrink 12.870780; Public LB `exp039` 11.740.
  - `public_pf_selector`: 15.172636 on both audit surfaces.
  - `base_plus_ncc_gr_match_pf_context_pf_blend_w0p30`: original-fold 15.122880 / well-hash 15.019511.
  - `base_plus_ncc_gr_match_pf_context_pf_blend_w0p30` vs same-surface `exp026_regenerated_bucket_shrink`: -1.360747 original-fold / -1.410102 well-hash.
  - supported candidates under the strict success rule: none.
  - selected candidate: none.
- LB: not submitted

## 次のアクション

1. exp048 は推論 port / submit しない。same-surface ML control は上回るが、direct PF controls を両 split で超える候補がなく、通常 ML route CV / LB とは評価条件も違う。
2. Ravaghi feature は LightGBM 直接置き換えでは停止し、XGBoost/CatBoost や confidence-only feature の材料に限定する。
