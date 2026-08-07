# exp089_pf_beam_disagreement_sample_weight セッションノート

## 現在の状態

- status: `implemented_not_run`
- route: `ml_model`
- parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- blocked: none

## 実装内容

- `.steering/20260620-exp089-pf-beam-disagreement-sample-weight/` を作成し、requirements / design / tasklist を記入。
- `experiments/exp089_pf_beam_disagreement_sample_weight/` を exp085 から作成。
- `settings.py` の experiment name を exp089 に更新。
- `config.yaml` を PF/Beam disagreement confidence feature / sample weight ablation 用に更新。
- 補助実装を `pf_beam_disagreement_sample_weight.py` に整理。
  - exp072 deterministic 196-feature train cache を読む。
  - target は exp073 と同じ `TVT - last_known_tvt` のままにする。
  - `pf_likpf_abs`、`pf_beam_abs`、`beam_likpf_abs`、`beam_std_abs`、`dense_dist_abs`、`pf_vs_dense_abs` などの target-free confidence features を作る。
  - robust rank 平均の `pfbeam_instability_score` から conservative sample weight を作り、平均 1.0 に正規化する。
  - `control_exp073_base196`、`confidence_features_core`、`sample_weight_unstable_downweight`、`confidence_features_plus_weight` を比較する。
  - sample weight は training fold の `LGBMRegressor.fit()` にだけ渡し、validation RMSE は非加重のまま保存する。
  - fold/pooled metrics、well metrics、distance/tail buckets、OOF predictions、feature schema、feature importance、confidence summary、sample weight summary、model manifest を保存する。
- train notebook を exp089 用の4セクション構成に更新。
- inference notebook は selected variant 未設定なら停止する guard notebook のままにした。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp089_pf_beam_disagreement_sample_weight
uv run python scripts/new_experiment.py --name exp089_pf_beam_disagreement_sample_weight --source experiments/exp085_u_projection_feature_ablation
```

## 次のアクション

1. 静的検証を通す。
2. Kaggle train package を作成し、bootstrap manifest の config / 補助 `.py` SHA を確認する。
3. Kaggle train を実行して、variant 別 pooled RMSE、worst-well、distance/tail bucket、feature importance、sample weight 分布を確認する。
4. 改善候補が出た場合だけ inference port と raw-test confidence feature parity を設計する。

## 検証

- `uv run python -m py_compile experiments/exp089_pf_beam_disagreement_sample_weight/pf_beam_disagreement_sample_weight.py experiments/exp089_pf_beam_disagreement_sample_weight/public_notebook_replay_audit.py experiments/exp089_pf_beam_disagreement_sample_weight/settings.py`: PASS
- `python3 -m json.tool experiments/exp089_pf_beam_disagreement_sample_weight/exp089_pf_beam_disagreement_sample_weight_train.ipynb`: PASS
- `python3 -m json.tool experiments/exp089_pf_beam_disagreement_sample_weight/exp089_pf_beam_disagreement_sample_weight_inference.ipynb`: PASS
- `uv run ruff check experiments/exp089_pf_beam_disagreement_sample_weight/pf_beam_disagreement_sample_weight.py experiments/exp089_pf_beam_disagreement_sample_weight/public_notebook_replay_audit.py experiments/exp089_pf_beam_disagreement_sample_weight/settings.py`: PASS
- `uv run ruff format --check experiments/exp089_pf_beam_disagreement_sample_weight/pf_beam_disagreement_sample_weight.py experiments/exp089_pf_beam_disagreement_sample_weight/settings.py`: PASS
- `uv run python scripts/validate_experiment.py --experiment exp089_pf_beam_disagreement_sample_weight`: PASS
- synthetic frame による `build_confidence_features()` / `build_sample_weight()` smoke test: PASS、12 rows / 15 columns、weight mean 1.0、min 0.762774、max 1.113139。
- `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp089_pf_beam_disagreement_sample_weight --notebook train --kernel-id kentookumura/exp089-pf-beam-disagreement-sample-weight-train --title "exp089 pf beam disagreement sample weight train" --run-on-push --strict`: PASS
- generated train package: `experiments/exp089_pf_beam_disagreement_sample_weight/kaggle/train`
- generated kernel id: `kentookumura/exp089-pf-beam-disagreement-sample-weight-train`
- generated metadata: GPU enabled, internet disabled, run_on_push true, competition source `rogii-wellbore-geology-prediction`, kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- generated bootstrap manifest includes `config.yaml` SHA `7fb20ac5025501b300ff98daea9c2f63286704c9ce42ab05c799f5eec10e7358` and `pf_beam_disagreement_sample_weight.py` SHA `57f32017d0d0e4f226fd7efbe3037364de8e8ecce7d4fe157890bef6057e40f8`。
- `uv run python scripts/update_experiment_summary.py`: PASS、90 experiments。

## Kaggle train v1

- 2026-06-20 02:39 UTC 頃: `kaggle kernels push -p experiments/exp089_pf_beam_disagreement_sample_weight/kaggle/train`: `Kernel version 1 successfully pushed`
- kernel id: `kentookumura/exp089-pf-beam-disagreement-sample-weight-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp089-pf-beam-disagreement-sample-weight-train`
- pull existence check: PASS at `/tmp/kaggle-pull/exp089-pf-beam-disagreement-sample-weight-train-v1`
- initial status: `KernelWorkerStatus.RUNNING`
- normal logs: empty
- `timeout 180 kaggle kernels logs -f --interval 15 ...`: no log output before timeout
- first output probe: `/tmp/kaggle-output/exp089_pf_beam_disagreement_sample_weight/train_v1_probe`, no files yet
- `timeout 300 kaggle kernels logs -f --interval 20 ...`: no log output before timeout
- status at 2026-06-20 02:48: `KernelWorkerStatus.RUNNING`
- second output probe: `/tmp/kaggle-output/exp089_pf_beam_disagreement_sample_weight/train_v1_probe2`, no files yet
- Current interpretation: Kaggle execution has started and is still running; logs/output are empty, consistent with Kaggle API lag seen in earlier runs. Do not repush under a different slug.
- User reported completion; `kaggle kernels status kentookumura/exp089-pf-beam-disagreement-sample-weight-train`: `KernelWorkerStatus.COMPLETE`
- `kaggle kernels logs kentookumura/exp089-pf-beam-disagreement-sample-weight-train`: PASS, log downloaded later with output
- `kaggle kernels output kentookumura/exp089-pf-beam-disagreement-sample-weight-train -p /tmp/kaggle-output/exp089_pf_beam_disagreement_sample_weight/train_v1`: PASS
- output root: `/tmp/kaggle-output/exp089_pf_beam_disagreement_sample_weight/train_v1`
- copied small result artifacts to ignored local artifacts directory:
  - `artifacts/exp089_pf_beam_disagreement_sample_weight_metrics.csv`
  - `artifacts/exp089_pf_beam_disagreement_sample_weight_bucket_metrics.csv`
  - `artifacts/exp089_pf_beam_disagreement_sample_weight_by_well.csv`
  - `artifacts/exp089_pf_beam_disagreement_sample_weight_confidence_feature_summary.csv`
  - `artifacts/exp089_pf_beam_disagreement_sample_weight_sample_weight_summary.csv`
  - `artifacts/exp089_pf_beam_disagreement_sample_weight_feature_importance_mean.csv`
  - `artifacts/exp089_pf_beam_disagreement_sample_weight_summary.json`
  - `artifacts/exp089-pf-beam-disagreement-sample-weight-train.log`

### Result summary

- rows: 3,783,989
- wells: 773
- model count: 60
- source cache SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- OOF predictions decompressed SHA: `d6a94e064a24e594b822a4adeed9d5a5f3c631caff4f680385a2d7eec0d805b1`
- best variant: `sample_weight_unstable_downweight`
- best lgb_mean RMSE: `9.5212120473311`
- control `control_exp073_base196` lgb_mean RMSE: `9.52637457314413`
- delta vs control: `-0.0051625258130293`
- `confidence_features_core`: `9.564240269703114`
- `confidence_features_plus_weight`: `9.562018858093628`
- sample weight summary for best variant: mean 1.0, std 0.085432, min 0.769068, p05 0.860132, p50 0.999054, p95 1.144960, max 1.249405.
- well guard for best variant vs control: improved 374, worsened 399, mean delta +0.010116, median delta +0.010215, max worsen +1.096752, max improve -1.068382.
- distance bucket: sample-weight only improved `000_050`, `050_100`, and `1000_plus`, but worsened `100_250`, `250_500`, and `500_1000`.

### Interpretation

`sample_weight_unstable_downweight` gives a small global CV improvement, but the well-level guard is weak and mid-distance buckets worsen. Confidence feature add-only and feature+weight both worsen. This is not a submit candidate; close as diagnostic and shift follow-up toward candidate ranker / observation likelihood / U-projection full-run rather than simple PF/Beam confidence feature addition.
