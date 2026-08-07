# exp055_single_model_pseudotail_training セッションノート

## 目的

高優先 backlog `single_model_pseudotail_training` を実装する。exp039 の single-model feature surface、LightGBM、residual target、fixed bucket-shrink を固定し、学習 row policy だけを exp051 pseudo-tail 方式へ寄せて比較する。

## 現在の状態

- status: completed_no_supported_candidate
- route: `ml_model`
- parent: `exp039_ravaghi_single_lgbm_inference_submit`
- implementation parent: `exp048_ravaghi_single_model_feature_parity_revisit`
- comparison anchors:
  - exp039 ML route Public LB: 11.740
  - exp051 pseudo-tail CV: 12.634392
  - exp052 pseudo-tail Public LB: 12.076
  - exp054 seed-bagged pseudo-tail Public LB: 11.856
- selected variant: none
- CV: no supported candidate
- LB: not submitted

## 実装メモ

- `exp048` を土台に `exp055` を作成。
- `config.yaml` を exp039 feature surface の 2 variant 比較に整理した。
  - `exp039_same_surface_control`
  - `single_model_pseudotail_training`
- `ravaghi_single_lgbm_audit.py` に variant-level `training_policy` を追加した。
- pseudo-tail policy は cutoff `[0.45, 0.65, 0.82]`、`max_rows_per_pseudo_tail=260`、`balanced_rows_per_bucket=60000` を表現する。
- 現在の local exp029 artifact は cutoff 0.65 のみなので、missing cutoff は train summary に記録し、available cutoff で fallback する。
- direct PF/Beam replacement、追加 Ravaghi features、Ridge/meta stack、推論 port、提出処理はこの実験範囲に含めない。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp055_single_model_pseudotail_training
uv run python scripts/new_experiment.py --name exp055_single_model_pseudotail_training --source experiments/exp048_ravaghi_single_model_feature_parity_revisit
```

## 検証状況

- Static checks: PASS
  - `ruff check`: PASS
  - `py_compile`: PASS
  - `scripts/validate_experiment.py --experiment exp055_single_model_pseudotail_training`: PASS
- Kaggle train package: prepared
  - path: `experiments/exp055_single_model_pseudotail_training/kaggle/train`
  - kernel id: `kentookumura/exp055-single-model-pseudotail-train`
  - title: `exp055 single model pseudotail train`
  - run_on_push: true
- Kaggle train: completed
  - version: 1
  - URL: `https://www.kaggle.com/code/kentookumura/exp055-single-model-pseudotail-train`
  - `kaggle kernels push -p experiments/exp055_single_model_pseudotail_training/kaggle/train` succeeded.
  - `kaggle kernels pull kentookumura/exp055-single-model-pseudotail-train -p /tmp/kaggle-pull/exp055-single-model-pseudotail-train -m` succeeded.
  - Kaggle metadata returned `id_no: 122290852`, so the kernel exists on Kaggle.
  - Normal `kaggle kernels logs` and `kaggle kernels output` were empty immediately after push; treat as Kaggle API/session output lag and retry the same kernel id later.
  - Sandboxed `logs -f` failed with DNS resolution for `api.kaggle.com`; escalated retry ran for 60 seconds and timed out with no log output.
  - Supplemental `kaggle kernels status kentookumura/exp055-single-model-pseudotail-train` returned `KernelWorkerStatus.RUNNING`.
  - Later recheck still returned `KernelWorkerStatus.RUNNING`; normal `logs` and `output` remained empty.
  - User reported completion; final `kaggle kernels status` returned `KernelWorkerStatus.COMPLETE`.
  - logs and output were retrieved successfully.
  - output: `/tmp/kaggle-output/exp055_single_model_pseudotail_training/train_v1`
  - synced local artifacts:
    - `metrics.json`
    - `artifacts/exp055-single-model-pseudotail-train.log`
    - `artifacts/single_lgbm_bucket_metrics.csv`
    - `artifacts/single_lgbm_family_matrix.csv`
    - `artifacts/single_lgbm_feature_importance.csv`
    - `artifacts/single_lgbm_feature_parity_report.csv`
    - `artifacts/single_lgbm_metrics.csv`
    - `artifacts/single_lgbm_split_metrics.csv`
    - `artifacts/single_lgbm_summary.json`
    - `artifacts/single_lgbm_train_summary.csv`
    - `artifacts/single_lgbm_well_metrics.csv`

Static check commands:

```bash
uv run ruff check experiments/exp055_single_model_pseudotail_training/ravaghi_single_lgbm_audit.py experiments/exp055_single_model_pseudotail_training/baseline.py experiments/exp055_single_model_pseudotail_training/pseudo_tail_augmentation.py experiments/exp055_single_model_pseudotail_training/settings.py
uv run python -m py_compile experiments/exp055_single_model_pseudotail_training/ravaghi_single_lgbm_audit.py experiments/exp055_single_model_pseudotail_training/baseline.py experiments/exp055_single_model_pseudotail_training/pseudo_tail_augmentation.py experiments/exp055_single_model_pseudotail_training/settings.py
uv run python scripts/validate_experiment.py --experiment exp055_single_model_pseudotail_training
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp055_single_model_pseudotail_training --notebook train --kernel-id kentookumura/exp055-single-model-pseudotail-train --title "exp055 single model pseudotail train" --run-on-push --strict
kaggle kernels push -p experiments/exp055_single_model_pseudotail_training/kaggle/train
kaggle kernels pull kentookumura/exp055-single-model-pseudotail-train -p /tmp/kaggle-pull/exp055-single-model-pseudotail-train -m
kaggle kernels logs kentookumura/exp055-single-model-pseudotail-train
kaggle kernels output kentookumura/exp055-single-model-pseudotail-train -p /tmp/kaggle-output/exp055_single_model_pseudotail_training/train_v1_probe
timeout 60 kaggle kernels logs -f --interval 5 kentookumura/exp055-single-model-pseudotail-train
kaggle kernels status kentookumura/exp055-single-model-pseudotail-train
kaggle kernels logs kentookumura/exp055-single-model-pseudotail-train
kaggle kernels output kentookumura/exp055-single-model-pseudotail-train -p /tmp/kaggle-output/exp055_single_model_pseudotail_training/train_v1
```

## 結果

- rows / wells: 1,782,279 / 773
- best original-fold candidate: `pf090_hold010` 15.089532
- best well-hash candidate: `pf090_hold010` 15.089532
- `exp039_same_surface_control_raw`: original-fold 15.722062 / well-hash 15.667445
- `single_model_pseudotail_training_raw`: original-fold 15.764607 / well-hash 15.959310
- `exp039_same_surface_control_bucket_shrink`: original-fold 15.875275 / well-hash 15.837223
- `single_model_pseudotail_training_bucket_shrink`: original-fold 15.911149 / well-hash 16.159852
- pseudo-tail raw delta vs control raw: +0.042545 original-fold / +0.291865 well-hash
- pseudo-tail bucket delta vs control bucket: +0.035874 original-fold / +0.322629 well-hash
- requested cutoffs `[0.45, 0.65, 0.82]` のうち available は 0.65 のみ。0.45 / 0.82 は missing として train summary に記録された。

Interpretation:

- pseudo-tail training policy は same-surface control を安定して上回らなかった。
- direct public PF controls (`pf090_hold010`, `public_pf_selector`) が single-model candidates より強いまま。
- Kaggle-generated `single_lgbm_summary.json` は `exp039_same_surface_control_raw` を selected としているが、これは run config で raw control を required control に入れていなかったため。解釈上は control variant であり、新規 supported candidate ではない。
- local `config.yaml` は今後の rerun 向けに `exp039_same_surface_control_raw` も required control に追加済み。

## 次のアクション

1. この exp055 は inference port / submit しない。
2. single-model surface で再訪するなら、先に exp029 public feature artifact を真の multi-cutoff `[0.45, 0.65, 0.82]` で再生成する。
3. 直近の実験優先度としては、exp051/052 capacity pseudo-tail 本体へ public/PF confidence features を限定投入する案を優先する。
