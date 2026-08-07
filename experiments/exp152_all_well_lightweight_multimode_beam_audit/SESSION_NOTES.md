# exp152_all_well_lightweight_multimode_beam_audit セッションノート

## 目的

`all_well_lightweight_multimode_beam_audit` backlog を実装する。`exp143` の 6 well scoped audit で従来 `exp072_beam_mean` より良かった multimode PF/Beam 候補が、全 train well の軽量 tail slice でも安定して Beam を上回るかを確認する。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_all_well_tail_slice_rejected_no_submit`
- CV: train-side all-well tail500 audit 完了
- LB: 未提出
- blocked: none

## 変更点

- `.steering/20260628-exp152-all-well-lightweight-multimode-beam-audit/` を作成した。
- `experiments/exp152_all_well_lightweight_multimode_beam_audit/` を `exp143_multimode_pfbeam_local_correlation_audit` から作成した。
- 実装ファイルを `all_well_lightweight_multimode_beam_audit.py` に変更した。
- exp143 の heavy local-correlation diagnostic を config で無効化できるようにした。
- 評価対象を `exp072_pf_z`, `exp072_beam_mean`, `exp072_likpf_mean`, `multimode_pf_zacc_s010_a020_noise050_best_lik_seed` に絞った。
- 全 well 対象だが、各 well は tail 500 rows に制限する。
- `candidate_wide.csv.gz` は最小列だけ保存する。
- ML 学習、推論 port、提出は含めない。

## 再現性メモ

- seed policy: `stable_sha256_seed_from_experiment_all_well_lightweight_multimode_beam_well_variant_seed_index`
- stochastic components: strict PF-Z probe / multimode PF-Z particle initialization, process noise, resampling, upstream exp072 cache
- parallel RNG policy: well-level thread parallel; each well/variant gets stable seed vector before Numba kernel
- CPU/GPU runtime: CPU-only、GPU 不使用
- deterministic anchor: false。train-side audit only。
- gzip output: decompressed content SHA を summary JSON に記録する。
- submission / prediction SHA: 推論・提出なしのため対象外。

## コマンドログ

```bash
uv run python scripts/new_steering.py --experiment exp152_all_well_lightweight_multimode_beam_audit
uv run python scripts/new_experiment.py --name exp152_all_well_lightweight_multimode_beam_audit --source experiments/exp143_multimode_pfbeam_local_correlation_audit
```

## 検証

```bash
uv run python -m py_compile experiments/exp152_all_well_lightweight_multimode_beam_audit/all_well_lightweight_multimode_beam_audit.py experiments/exp152_all_well_lightweight_multimode_beam_audit/settings.py
python3 -m json.tool experiments/exp152_all_well_lightweight_multimode_beam_audit/exp152_all_well_lightweight_multimode_beam_audit_train.ipynb
python3 -m json.tool experiments/exp152_all_well_lightweight_multimode_beam_audit/exp152_all_well_lightweight_multimode_beam_audit_inference.ipynb
uv run ruff check experiments/exp152_all_well_lightweight_multimode_beam_audit/all_well_lightweight_multimode_beam_audit.py experiments/exp152_all_well_lightweight_multimode_beam_audit/settings.py
uv run ruff format --check experiments/exp152_all_well_lightweight_multimode_beam_audit/all_well_lightweight_multimode_beam_audit.py experiments/exp152_all_well_lightweight_multimode_beam_audit/settings.py
uv run python scripts/validate_experiment.py --experiment exp152_all_well_lightweight_multimode_beam_audit
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp152_all_well_lightweight_multimode_beam_audit --notebook train --kernel-id kentookumura/exp152-all-well-lightweight-multimode-beam-audit-train --title 'exp152 all well lightweight multimode beam audit train' --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp152_all_well_lightweight_multimode_beam_audit --notebook inference --kernel-id kentookumura/exp152-all-well-lightweight-multimode-beam-audit-infer --title 'exp152 all well lightweight multimode beam audit infer' --run-on-push --strict
uv run python -m py_compile experiments/exp152_all_well_lightweight_multimode_beam_audit/kaggle/train/all_well_lightweight_multimode_beam_audit.py experiments/exp152_all_well_lightweight_multimode_beam_audit/kaggle/train/settings.py experiments/exp152_all_well_lightweight_multimode_beam_audit/kaggle/inference/settings.py
python3 -m json.tool experiments/exp152_all_well_lightweight_multimode_beam_audit/kaggle/train/exp152_all_well_lightweight_multimode_beam_audit_train.ipynb
python3 -m json.tool experiments/exp152_all_well_lightweight_multimode_beam_audit/kaggle/inference/exp152_all_well_lightweight_multimode_beam_audit_inference.ipynb
```

- `py_compile`: PASS
- train notebook JSON: PASS
- inference notebook JSON: PASS
- `ruff check`: PASS
- `ruff format --check`: PASS
- `validate_experiment.py`: PASS
- `prepare_kaggle_notebooks.py --notebook train --strict`: PASS
- `prepare_kaggle_notebooks.py --notebook inference --strict`: PASS
- packaged support `.py` py_compile: PASS
- packaged train notebook JSON: PASS
- packaged inference notebook JSON: PASS
- generated train package: `experiments/exp152_all_well_lightweight_multimode_beam_audit/kaggle/train`
- generated inference package: `experiments/exp152_all_well_lightweight_multimode_beam_audit/kaggle/inference`
- generated train metadata:
  - kernel id: `kentookumura/exp152-all-well-lightweight-multimode-beam-audit-train`
  - title: `exp152 all well lightweight multimode beam audit train`
  - GPU: false
  - internet: false
  - run_on_push: true
  - competition source: `rogii-wellbore-geology-prediction`
  - kernel source: `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- generated inference metadata:
  - kernel id: `kentookumura/exp152-all-well-lightweight-multimode-beam-audit-infer`
  - title: `exp152 all well lightweight multimode beam audit infer`
  - GPU: false
  - internet: false
  - run_on_push: true
  - competition source: `rogii-wellbore-geology-prediction`
  - kernel source: `kentookumura/exp072-exp063-full-replay-feature-cache-train`

## Kaggle 実行計画

- Train variants: 1 lightweight multimode PF-Z candidate
- LightGBM configs: 0
- Folds: 0
- Total boosters: 0
- Control / parent retraining: なし
- GPU: なし

Kaggle push 前に `validate_experiment.py` と `prepare_kaggle_notebooks.py --notebook train --strict` を通す。

## Kaggle v1 実行

```bash
kaggle kernels push -p experiments/exp152_all_well_lightweight_multimode_beam_audit/kaggle/train
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp152_all_well_lightweight_multimode_beam_audit --notebook train --kernel-id kentookumura/exp152-allwell-light-mmbeam-audit-train --title 'exp152 allwell light mmbeam audit train' --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp152_all_well_lightweight_multimode_beam_audit --notebook inference --kernel-id kentookumura/exp152-allwell-light-mmbeam-audit-infer --title 'exp152 allwell light mmbeam audit infer' --run-on-push --strict
uv run python -m py_compile experiments/exp152_all_well_lightweight_multimode_beam_audit/kaggle/train/all_well_lightweight_multimode_beam_audit.py experiments/exp152_all_well_lightweight_multimode_beam_audit/kaggle/train/settings.py
python3 -m json.tool experiments/exp152_all_well_lightweight_multimode_beam_audit/kaggle/train/exp152_all_well_lightweight_multimode_beam_audit_train.ipynb
kaggle kernels push -p experiments/exp152_all_well_lightweight_multimode_beam_audit/kaggle/train
kaggle kernels pull kentookumura/exp152-allwell-light-mmbeam-audit-train -p /tmp/kaggle-pull/exp152-allwell-light-mmbeam-audit-train-v1 -m
kaggle kernels logs kentookumura/exp152-allwell-light-mmbeam-audit-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp152-allwell-light-mmbeam-audit-train
kaggle kernels output kentookumura/exp152-allwell-light-mmbeam-audit-train -p experiments/exp152_all_well_lightweight_multimode_beam_audit/kaggle/output/train_v1
kaggle kernels status kentookumura/exp152-allwell-light-mmbeam-audit-train
```

- 初回 canonical long slug `kentookumura/exp152-all-well-lightweight-multimode-beam-audit-train` は `SaveKernel` 400 で失敗。id/title slug は一致していたが、詳細なし 400 のため長すぎる slug と判断し、同じ exp152 のまま短い canonical slug に再 prepare した。
- short slug package:
  - train kernel id: `kentookumura/exp152-allwell-light-mmbeam-audit-train`
  - train title: `exp152 allwell light mmbeam audit train`
  - inference kernel id: `kentookumura/exp152-allwell-light-mmbeam-audit-infer`
  - inference title: `exp152 allwell light mmbeam audit infer`
- short slug packaged py_compile: PASS
- short slug packaged train notebook JSON: PASS
- `kaggle kernels push`: PASS。Kernel version 1 successfully pushed.
- URL: https://www.kaggle.com/code/kentookumura/exp152-allwell-light-mmbeam-audit-train
- `kaggle kernels pull -m`: PASS。metadata を `/tmp/kaggle-pull/exp152-allwell-light-mmbeam-audit-train-v1` に取得。`id_no=125107320`。
- `kaggle kernels logs`: 空。
- `timeout 300 kaggle kernels logs -f --interval 20`: timeout まで空。
- `kaggle kernels output`: コマンドは 0 exit だが output directory は空。実行中のため生成物未取得。
- `kaggle kernels status`: `KernelWorkerStatus.RUNNING`

## 次のアクション

完了。`exp143` scoped positive の全 well 再確認としては negative だったため、inference port / submit / full-row minimal candidate cache 生成には進めない。

## Kaggle v1 完了結果

```bash
kaggle kernels logs kentookumura/exp152-allwell-light-mmbeam-audit-train
kaggle kernels output kentookumura/exp152-allwell-light-mmbeam-audit-train -p experiments/exp152_all_well_lightweight_multimode_beam_audit/kaggle/output/train_v1
```

- `kaggle kernels logs`: PASS。fatal error なし。
- `kaggle kernels output`: PASS。
- output dir: `experiments/exp152_all_well_lightweight_multimode_beam_audit/kaggle/output/train_v1`
- status: `completed_all_well_tail_slice_train_side_audit`
- rows / wells: 386,407 / 773
- max rows per well: 500
- runtime: 780.239 sec

Generated artifacts:

- `artifacts/exp152_all_well_lightweight_multimode_beam_audit_candidate_metrics.csv`
- `artifacts/exp152_all_well_lightweight_multimode_beam_audit_bucket_metrics.csv`
- `artifacts/exp152_all_well_lightweight_multimode_beam_audit_by_well.csv`
- `artifacts/exp152_all_well_lightweight_multimode_beam_audit_strict_pf_z_quality.csv`
- `artifacts/exp152_all_well_lightweight_multimode_beam_audit_multimode_pf_z_quality.csv`
- `artifacts/exp152_all_well_lightweight_multimode_beam_audit_parity_diff.csv.gz`
- `artifacts/exp152_all_well_lightweight_multimode_beam_audit_candidate_wide.csv.gz`
- `artifacts/exp152_all_well_lightweight_multimode_beam_audit_summary.json`

Metrics:

| candidate | RMSE | MAE | within10 |
| --- | ---: | ---: | ---: |
| `exp072_likpf_mean` | 16.115835 | 10.421046 | 0.646595 |
| `exp072_beam_mean` | 19.685742 | 14.052127 | 0.478829 |
| `multimode_pf_zacc_s010_a020_noise050_best_lik_seed` | 20.110701 | 14.501435 | 0.454707 |
| `exp072_pf_z` | 24.439192 | 16.117721 | 0.488236 |

- multimode vs `exp072_beam_mean`: RMSE +0.424958、MAE +0.449309。
- multimode vs `exp072_likpf_mean`: RMSE +3.994865。
- multimode vs `exp072_pf_z`: RMSE -4.328491。
- by-well vs Beam: improved 364 wells / worsened 409 wells。
- max regression vs Beam: `f140c2fa`、+15.334797 RMSE。
- best improvement vs Beam: `a959858c`、-29.117956 RMSE。
- quality: mean mode count 1.056388、mode_count<=1 rate 0.944352。軽量設定でも多くの well で mode collapse。
- strict PF-Z parity: fail (`rmse_diff=20.554779`)。この実験では parity anchor ではなく diagnostic として扱う。

結論: `exp143` の 6 well scoped positive は全 well tail slice では再現しなかった。multimode は `exp072_pf_z` より良いが、主比較の `exp072_beam_mean` と採用 guard の `exp072_likpf_mean` に負ける。`all_well_lightweight_multimode_beam_audit` backlog は完了/不採用として閉じる。
