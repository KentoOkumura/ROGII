# exp205_exact_hmm_smoother_exp072_compatible_cache_audit セッションノート

## 2026-07-06 実装

- ユーザー依頼により `exact_hmm_smoother_exp072_compatible_cache_audit` backlog の実装を開始。
- `.steering/20260706-exp205-exact-hmm-smoother-exp072-compatible-cache-audit/` を作成し、requirements / design / tasklist を記入。
- `experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/` を template から作成。
- 参照元 notebook: `amerhu/rogii-wellbore-geology-exact-hmm-smoother`
- Route: `pf_beam`
- GPU 学習: なし
- active feature cache variant 数: 1 (`amerhu_exact_hmm_smoother_default`)
- LightGBM config 数: 0
- fold 数: 0
- 合計 booster 数: 0
- control / parent 再学習: なし
- inference / submit: なし

## 実装内容

- `exact_hmm_smoother.py`
  - amerhu notebook の `_hmm2_fb` / `run_hmm2` を、数式を大きく変えずに移植。
  - raw train horizontal/typewell から unknown suffix rows の HMM posterior mean/std/loglik cache を生成する。
  - 出力列は `id`, `well`, `target`, `last_known_tvt`, `md_since`, `hmm_mean_tvt`, `hmm_mean_d`, `hmm_std`, `hmm_loglik` など。
  - gzip raw SHA と decompressed content SHA を summary に記録する。
- `feature_cache.py`
  - `config.yaml` の HMM 設定と data path から train feature cache generation を起動する wrapper。
- `direct_hmm_comparison.py`
  - exp072 cache と HMM cache を `id` で整列し、HMM 単体、exp072 `likpf_mean`、fixed blend (`w025/w050/w075`) を比較する。
  - overall / distance bucket / by-well / HMM std calibration / step-delta rate / enriched train cache を出力する。
- `exp205_*_train.py`
  - Jupytext 起点の train notebook source。設定確認、入力確認、HMM cache generation、exp072 comparison、metrics 保存をセル分割。
- `exp205_*_inference.py`
  - train-feature-cache-only 実験であることを明示し、submission を作らない guard notebook。

## HMM contract

- default:
  - `step=0.35`
  - `n_rates=41`
  - `rate_span=0.10`
  - `sig_r=0.002`
  - `sig_p=0.02`
  - `mom=0.998`
  - `lam=1.0`
  - `sigma_mode=std`
- HMM generation input:
  - raw train horizontal/typewell files
  - observed `TVT_input` prefix
  - `MD`, `Z`, `GR`, typewell `TVT`/`GR`
- HMM generation non-input:
  - unknown suffix true `TVT`
  - exp072 cache
  - test files
  - oracle / true-error rank / absolute error features

## 再現性メモ

- seed policy: default HMM generation は no RNG。
- stochastic components: なし。
- parallel RNG policy: numba parallel、no RNG。
- CPU/GPU runtime: Kaggle CPU 想定。amerhu notebook は T4 metadata だが、本実験では GPU 不要。
- deterministic anchor: false。numba parallel floating arithmetic と gzip metadata 差を考慮し、submission anchor にはしない。
- feature SHA: gzip raw SHA と decompressed content SHA を記録し、decompressed SHA を主証拠にする。
- model / prediction / submission SHA: not applicable。

## push 前コスト確認

- Runtime: CPU (`enable_gpu=false`)
- active feature cache variant 数: 1
- target wells: all train horizontal wells (`max_wells: null`)
- expected rows: exp072 と同じ 3,783,989 rows を想定
- expected feature_count: 13
- LightGBM config 数: 0
- fold 数: 0
- total boosters: 0
- parent/control 再学習: なし
- inference / submit: なし

## コマンドログ

```bash
make new-steering EXP=exp205_exact_hmm_smoother_exp072_compatible_cache_audit
make new-exp EXP=exp205_exact_hmm_smoother_exp072_compatible_cache_audit
kaggle kernels pull amerhu/rogii-wellbore-geology-exact-hmm-smoother -p /tmp/kaggle-pull/amerhu-exact-hmm-smoother -m
```

- result: steering / experiment scaffold / source notebook pull complete。

## 次のアクション

1. Jupytext で `.py` から `.ipynb` を生成する。
2. `py_compile`、`ruff --select F821`、`validate-exp` を通す。
3. Kaggle train を push する場合は canonical kernel id を `kentookumura/exp205-exact-hmm-smoother-cache-audit-train` にする。

## local validation

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/exp205_exact_hmm_smoother_exp072_compatible_cache_audit_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/exp205_exact_hmm_smoother_exp072_compatible_cache_audit_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/exp205_exact_hmm_smoother_exp072_compatible_cache_audit_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/exp205_exact_hmm_smoother_exp072_compatible_cache_audit_inference.py
```

- result: PASS

```bash
.venv/bin/python -m py_compile \
  experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/exact_hmm_smoother.py \
  experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/feature_cache.py \
  experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/direct_hmm_comparison.py \
  experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/settings.py \
  experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/exp205_exact_hmm_smoother_exp072_compatible_cache_audit_train.py \
  experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/exp205_exact_hmm_smoother_exp072_compatible_cache_audit_inference.py
```

- result: PASS

```bash
.venv/bin/ruff check \
  experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/exact_hmm_smoother.py \
  experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/feature_cache.py \
  experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/direct_hmm_comparison.py \
  experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/settings.py \
  experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/exp205_exact_hmm_smoother_exp072_compatible_cache_audit_train.py \
  experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/exp205_exact_hmm_smoother_exp072_compatible_cache_audit_inference.py \
  --select F821
```

- result: PASS

```bash
.venv/bin/python -c "import sys; sys.path.insert(0, 'experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit'); import exact_hmm_smoother, feature_cache, direct_hmm_comparison; print('exp205_import_ok', exact_hmm_smoother.NUMBA_AVAILABLE)"
```

- result: PASS (`exp205_import_ok False`)
- local `.venv` には `numba` がないため、validation-only import guard を追加した。HMM full generation は `numba` 必須で、未導入 runtime では明示 error にする。Kaggle runtime では numba 前提。

```bash
uv run python scripts/validate_experiment.py --experiment exp205_exact_hmm_smoother_exp072_compatible_cache_audit
```

- result: PASS

```bash
make prepare-kaggle-notebooks EXP=exp205_exact_hmm_smoother_exp072_compatible_cache_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp205-exact-hmm-smoother-cache-audit-train --title 'exp205 exact hmm smoother cache audit train' --run-on-push --strict"
make prepare-kaggle-notebooks EXP=exp205_exact_hmm_smoother_exp072_compatible_cache_audit EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp205-exact-hmm-smoother-cache-audit-inference --title 'exp205 exact hmm smoother cache audit inference' --strict"
```

- result: PASS
- train metadata: `enable_gpu=false`, `enable_internet=false`, `run_on_push=true`, `kernel_sources=["kentookumura/exp072-exp063-full-replay-feature-cache-train"]`
- inference metadata: `enable_gpu=false`, `enable_internet=false`, `run_on_push=false`

## 実装完了時点の状態

- 状態: implemented / pending Kaggle train
- Kaggle train は未 push。
- output / metrics は未生成。
- `KAGGLE_DIRECTION.md` の backlog は結果待ちとして残し、実行完了後に完了/不採用/支持を判断する。

## Kaggle train v1

```bash
make push-kaggle-train EXP=exp205_exact_hmm_smoother_exp072_compatible_cache_audit
```

- result: `Kernel version 1 successfully pushed`
- kernel: `kentookumura/exp205-exact-hmm-smoother-cache-audit-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp205-exact-hmm-smoother-cache-audit-train`
- runtime: CPU (`enable_gpu=false`, `machine_shape=None`)
- internet: false
- kernel source: `kentookumura/exp072-exp063-full-replay-feature-cache-train`

```bash
kaggle kernels pull kentookumura/exp205-exact-hmm-smoother-cache-audit-train -p /tmp/kaggle-pull/exp205-exact-hmm-smoother-cache-audit-train -m
kaggle kernels logs kentookumura/exp205-exact-hmm-smoother-cache-audit-train
kaggle kernels status kentookumura/exp205-exact-hmm-smoother-cache-audit-train
```

- pull: success。Kaggle metadata は `id_no=126133209`、CPU、internet off、competition source と exp072 kernel source が付いている。
- initial logs: warning only / stdout empty。実行中 logs が空のことは既知なので失敗扱いにしない。
- initial status: `KernelWorkerStatus.RUNNING`

## Kaggle train v1 失敗

```bash
kaggle kernels status kentookumura/exp205-exact-hmm-smoother-cache-audit-train
kaggle kernels logs kentookumura/exp205-exact-hmm-smoother-cache-audit-train
```

- status: `KernelWorkerStatus.ERROR`
- first meaningful error:
  - `ValueError: No kernel name found in notebook and no override provided.`
- 原因: Jupytext 生成 notebook の metadata が `notebook_metadata_filter=-all` になっており、`kernelspec.name` が消えていた。Papermill が notebook 起動前に停止したため、HMM code は未実行。

## Kaggle train v2 復旧準備

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --set-kernel python3 \
  experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/exp205_exact_hmm_smoother_exp072_compatible_cache_audit_train.ipynb
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --set-kernel python3 \
  experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/exp205_exact_hmm_smoother_exp072_compatible_cache_audit_inference.ipynb
make prepare-kaggle-notebooks EXP=exp205_exact_hmm_smoother_exp072_compatible_cache_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp205-exact-hmm-smoother-cache-audit-train --title 'exp205 exact hmm smoother cache audit train' --run-on-push --strict"
```

- result: PASS
- package notebook metadata now has `kernelspec.name=python3`

## Kaggle train v2

```bash
make push-kaggle-train EXP=exp205_exact_hmm_smoother_exp072_compatible_cache_audit
```

- result: `Kernel version 2 successfully pushed`
- URL: `https://www.kaggle.com/code/kentookumura/exp205-exact-hmm-smoother-cache-audit-train`

```bash
kaggle kernels pull kentookumura/exp205-exact-hmm-smoother-cache-audit-train -p /tmp/kaggle-pull/exp205-exact-hmm-smoother-cache-audit-train-v2 -m
kaggle kernels status kentookumura/exp205-exact-hmm-smoother-cache-audit-train
kaggle kernels logs kentookumura/exp205-exact-hmm-smoother-cache-audit-train
```

- pull: success
- status after push: `KernelWorkerStatus.RUNNING`
- logs: warning only / stdout empty

```bash
sleep 120
kaggle kernels status kentookumura/exp205-exact-hmm-smoother-cache-audit-train
kaggle kernels logs kentookumura/exp205-exact-hmm-smoother-cache-audit-train
```

- status after 2 min: `KernelWorkerStatus.RUNNING`
- logs: still warning only / stdout empty
- interpretation: v1 の kernelspec 起動前 error は解消。HMM full cache generation は CPU 実行中と判断し、同じ kernel id のまま継続監視する。

## Kaggle train v2 完了

```bash
kaggle kernels status kentookumura/exp205-exact-hmm-smoother-cache-audit-train
kaggle kernels logs kentookumura/exp205-exact-hmm-smoother-cache-audit-train
```

- status: `KernelWorkerStatus.COMPLETE`
- kernel: `kentookumura/exp205-exact-hmm-smoother-cache-audit-train`
- version: v2
- URL: `https://www.kaggle.com/code/kentookumura/exp205-exact-hmm-smoother-cache-audit-train`
- HMM cache generation: 773 wells selected / 773 ok / 0 skipped
- rows: 3,783,989
- runtime: 15,041.783 sec
- HMM feature count: 13
- feature decompressed SHA256: `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
- feature gzip SHA256: `ca5343ca04b3774fcc4bfb95c96ba1f43a9a9ac70202e545019b3dba308b87d6`
- exp072 comparison `id_mismatches`: 0

```bash
kaggle kernels output kentookumura/exp205-exact-hmm-smoother-cache-audit-train \
  -p /tmp/kaggle-output/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/train_v2_small \
  --file-pattern '.*(metrics\.json|summary\.json|overall_metrics\.csv|distance_bucket_metrics\.csv|by_well_delta\.csv|hmm_std_calibration\.csv|step_delta_rates\.csv|feature_schema\.csv|by_well_generation_summary\.csv)$'
```

- result: PASS
- note: full HMM feature cache and enriched gzip cache were intentionally not downloaded locally. Only small JSON/CSV summaries and logs were fetched.
- local output: `/tmp/kaggle-output/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/train_v2_small`

### Overall direct comparison

| candidate | RMSE | MAE | within10 | delta RMSE vs exp072 likpf_mean |
| --- | ---: | ---: | ---: | ---: |
| `blend_likpf_hmm_w500` | 10.269699957 | 6.399213237 | 0.793202359 | -1.325197711 |
| `blend_likpf_hmm_w250` | 10.568836542 | 6.587048453 | 0.785896841 | -1.026061127 |
| `blend_likpf_hmm_w750` | 10.758296614 | 6.428404944 | 0.795246234 | -0.836601054 |
| `exp072_likpf_mean` | 11.594897668 | 7.067632583 | 0.772802194 | 0.000000000 |
| `hmm_mean_tvt` | 11.938296877 | 6.769557035 | 0.784378602 | +0.343399209 |

### Guard readout

- `blend_likpf_hmm_w500` by-well: 539 improved / 234 worsened, mean delta -0.832465 RMSE.
- `blend_likpf_hmm_w500` worst regression: `b19b0395`, +23.036816 RMSE.
- `blend_likpf_hmm_w500` best improvement: `86454a6f`, -28.133145 RMSE.
- HMM standalone by-well: 451 improved / 322 worsened, max regression `b19b0395`, +48.316191 RMSE.
- Best blend improves all distance buckets; `1000_plus` delta -1.420485 RMSE.
- `hmm_std_abs_error_corr`: 0.399484. Coarse risk signal is useful, but the lowest std bin is not strictly monotonic.
- Step smoothness improves versus `exp072_likpf_mean`: `abs_step_delta_mean` 0.029514 -> 0.017669; `|delta| > 0.1` rate 0.047355 -> 0.012700.

### Decision

exp205 is completed as a train-only direct PF/Beam audit and is supported for follow-up, but not for immediate inference / submit. The fixed HMM/likPF blend is materially better on train-side direct comparison, but raw-test-compatible HMM regeneration, hidden-like stress, and worst-well guard are not implemented in this experiment.
