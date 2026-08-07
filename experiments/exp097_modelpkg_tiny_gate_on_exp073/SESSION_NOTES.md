# exp097_modelpkg_tiny_gate_on_exp073 セッションノート

## 目的

exp073 deterministic ML inference を base にし、Pilkwang model-package-only prediction を agreement-gated tiny correction として足す候補を実装する。

## 現在の状態

- Route: ml_model
- 状態: submitted_complete_public_lb_8_766
- CV: なし
- LB: Public 8.766
- Submit: ref `53897072`

## コマンドログ

```bash
uv run python scripts/new_steering.py --experiment exp097_modelpkg_tiny_gate_on_exp073
uv run python scripts/new_experiment.py --name exp097_modelpkg_tiny_gate_on_exp073
uv run ruff check experiments/exp097_modelpkg_tiny_gate_on_exp073/modelpkg_tiny_gate_on_exp073.py
uv run python scripts/validate_experiment.py --experiment exp097_modelpkg_tiny_gate_on_exp073
EXPERIMENT_ALLOW_LOCAL=1 uv run python -c "from experiments.exp097_modelpkg_tiny_gate_on_exp073.settings import ExperimentPaths, load_config; from experiments.exp097_modelpkg_tiny_gate_on_exp073.modelpkg_tiny_gate_on_exp073 import run_audit_from_config; p=ExperimentPaths(); p.ensure_output_dirs(); s=run_audit_from_config(load_config(), sample_path=p.sample_submission_path, output_dir=p.artifacts_dir); print(s['status'], s['selected_variant'], s['selected_passes_all_guards']); print(s['selected_summary'])"
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp097_modelpkg_tiny_gate_on_exp073 --notebook train --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp097_modelpkg_tiny_gate_on_exp073 --notebook inference --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp097_modelpkg_tiny_gate_on_exp073 --notebook inference --run-on-push --strict --title "exp097 modelpkg tiny gate on exp073 inference"
make push-kaggle-infer EXP=exp097_modelpkg_tiny_gate_on_exp073
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp097-modelpkg-tiny-gate-on-exp073-inference
kaggle kernels output kentookumura/exp097-modelpkg-tiny-gate-on-exp073-inference -p /tmp/kaggle-output/exp097_modelpkg_tiny_gate_on_exp073/inference_v1
make submit-check EXP=exp097_modelpkg_tiny_gate_on_exp073 SUBMISSION=/tmp/kaggle-output/exp097_modelpkg_tiny_gate_on_exp073/inference_v1/submission.csv
python3 .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp097_modelpkg_tiny_gate_on_exp073/inference_v1/submission.csv --sample data/raw/sample_submission.csv
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp097_modelpkg_tiny_gate_on_exp073 --notebook inference --run-on-push --strict --title "exp097 modelpkg tiny gate on exp073 inference"
make push-kaggle-infer EXP=exp097_modelpkg_tiny_gate_on_exp073
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp097-modelpkg-tiny-gate-on-exp073-inference
kaggle kernels output kentookumura/exp097-modelpkg-tiny-gate-on-exp073-inference -p /tmp/kaggle-output/exp097_modelpkg_tiny_gate_on_exp073/inference_v3
make submit-check EXP=exp097_modelpkg_tiny_gate_on_exp073 SUBMISSION=/tmp/kaggle-output/exp097_modelpkg_tiny_gate_on_exp073/inference_v3/submission.csv
python3 .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp097_modelpkg_tiny_gate_on_exp073/inference_v3/submission.csv --sample data/raw/sample_submission.csv
kaggle competitions submissions rogii-wellbore-geology-prediction
```

検証結果:

- `ruff check`: pass
- `validate_experiment`: pass
- `compileall`: pass
- `prepare_kaggle_notebooks` train / inference strict: pass
- local helper smoke: pass
- Kaggle inference v1: complete
- Kaggle inference v2: failed before submission. `data.raw_dir` still pointed to `data/raw`, so exp073 replay saw train/test wells as 0.
- Kaggle inference v3: complete
- submit-check: pass

## 変更点

- `modelpkg_tiny_gate_on_exp073.py` を追加。
- `config.yaml` に exp073 inference prediction、model-package-only prediction、gate grid、diff guard、選択候補を記録。
- train / inference notebook を audit / submission 生成の入口に更新。
- local smoke で `artifacts/` に aligned predictions、variant metrics、selected submission、summary JSON を保存。

## 再現性メモ

- seed policy: no_new_rng_submission_diff_postprocess
- stochastic components: upstream exp073 GPU LightGBM inference、upstream Pilkwang model package prediction
- CPU/GPU runtime: この実験自体は CPU only / no training
- Kaggle kernel id / version: `kentookumura/exp097-modelpkg-tiny-gate-on-exp073-inference` v1
- Hidden-safe fix kernel version: `kentookumura/exp097-modelpkg-tiny-gate-on-exp073-inference` v3
- input SHA: summary JSON に保存
- feature content SHA: exp073 gzip は decompressed content SHA を保存
- model manifest / model SHA: 対象外
- prediction SHA: selected `3250801c6937e3deb77d35b5aea1a3f2bcbf8cf10eca7ecfb24080f11c6f7e0e`
- selected artifact submission SHA: `2ba63b33524cfd0593f4114f019d8ab45fee2edb26b4e31698ccea3b80f473ad`
- final Kaggle `submission.csv` SHA: `9467e55c136d09063a284d8b31cf412a66130963b07642d28194e303d8ac2175`
- rerun check: local helper smoke and Kaggle inference v3 match selected summary on public rows. Hidden rerun should no longer fail on model-package row mismatch; it will disable the model-package correction and write exp073 base-only if the precomputed public CSV does not cover current sample ids.

## local smoke 結果

- selected variant: `modelpkg_gate_g005_s4p0`
- rows / wells: 14,151 / 3
- raw modelpkg diff abs p95 / max: 33.566870 / 36.557253
- gate weight mean / max: 0.000494 / 0.004924
- correction abs mean / p95 / max: 0.005103 / 0.008829 / 0.010000
- guard: pass
- 注意: 保存済み `/tmp/kaggle-output/source-check/pilkwang-rogii-target-free-tvt-geosteering/submission_model_package_only.csv` を使った smoke。提出前に Kaggle inference と submit-check が必要。

## Kaggle inference v1 結果

- kernel: `kentookumura/exp097-modelpkg-tiny-gate-on-exp073-inference`
- version: 1
- status: `inference_submission_written`
- output: `/tmp/kaggle-output/exp097_modelpkg_tiny_gate_on_exp073/inference_v1`
- selected variant: `modelpkg_gate_g005_s4p0`
- rows / wells: 14,151 / 3
- raw modelpkg diff abs p95 / max: 33.566870 / 36.557253
- correction abs mean / p95 / max: 0.005103 / 0.008829 / 0.010000
- prediction min / max / mean / std: 11593.667022 / 12241.687490 / 11905.651155 / 279.294634
- final `submission.csv` SHA: `9467e55c136d09063a284d8b31cf412a66130963b07642d28194e303d8ac2175`
- `scripts/validate_submission.py`: PASS
- `kaggle-submit-check` script: PASS

## Code submission 失敗と v3 修正

- v1 failure: code submission rerun が `Notebook Threw Exception`。原因は public-output-copy 型の実装で、hidden rerun の `sample_submission` と公開 `submission_model_package_only.csv` / exp073 public inference output の行が一致しないこと。
- v2 failure: exp073 base を current test で再生成する修正を入れたが、`data.raw_dir` が Kaggle competition input の絶対パスへ解決されず、public replay の `train_imputer_wells=0 test=0` で `AttributeError: 'DataFrame' object has no attribute 'wid'`。
- v3 fix: `run_inference_from_config` の入口で `sample_submission` の親ディレクトリを `data.raw_dir` / `train_dir` / `test_dir` に反映。exp073 base を現 test rows で再生成し、model-package CSV が現 sample ids を cover しない場合だけ `modelpkg_disabled_base_submission_written` として exp073 base-only を書く。

## Kaggle inference v3 結果

- kernel: `kentookumura/exp097-modelpkg-tiny-gate-on-exp073-inference`
- version: 3
- status: `inference_submission_written`
- output: `/tmp/kaggle-output/exp097_modelpkg_tiny_gate_on_exp073/inference_v3`
- exp073 base generation: train_imputer_wells 773 / test_wells 3 / elapsed 133.038s
- selected variant: `modelpkg_gate_g005_s4p0`
- rows / wells: 14,151 / 3
- raw modelpkg diff abs p95 / max: 33.566870 / 36.557253
- correction abs mean / p95 / max: 0.005103 / 0.008829 / 0.010000
- prediction min / max / mean / std: 11593.667022 / 12241.687490 / 11905.651155 / 279.294634
- final `submission.csv` SHA: `9467e55c136d09063a284d8b31cf412a66130963b07642d28194e303d8ac2175`
- `scripts/validate_submission.py`: PASS
- `kaggle-submit-check` script: PASS

## Code submission 結果

- ref: `53897072`
- submitted_at: 2026-06-21 00:42:53 UTC
- status: `SubmissionStatus.COMPLETE`
- Public LB: 8.766
- Private LB: -
- 解釈: exp073 raw 8.780 より -0.014 改善。ただし exp077 8.611 / exp096 8.651 より悪く、ML route anchor にはしない。

## 次のアクション

1. exp097 は提出完了として閉じる。
2. hidden で model-package CSV が行不一致になる場合、補正は無効化され exp073 base-only になる。真の hidden-compatible tiny gate が必要なら Pilkwang model package branch の直接再生成 port を別実験で実装する。
