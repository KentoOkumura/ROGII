# exp209_exp072_exp205_joint_exact_parity_fast_cache_generation セッションノート

## 目的

`exp072_exp205_joint_exact_parity_fast_cache_generation` backlog を新規実験番号で実装する。現行 backlog の範囲に合わせ、exp072 full replay cache と exp205 exact HMM cache / direct comparison を同一 Kaggle train notebook で生成し、行順以外の exact parity を保ったまま wall time を短縮する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle train v6 完了。v6 は v5 より遅いため不採用、best runtime は v5 `outer_workers=2`, `numba_num_threads=2`
- CV / LB: train-side best blend RMSE 10.269696146642758、LB なし
- 既定実行: exp072 full cache generation、HMM cache generation、in-memory exp072 direct comparison、reference SHA/parity summary
- GPU cost: なし。CPU-only feature generation audit。
- Booster count: 0
- Parent/control retraining: なし

## 実装メモ

- `exp072_feature_cache.py`
  - exp072 v2 の full cache logic を同梱。
  - `return_frame=True` と `return_columns` を追加し、生成済み DataFrame の comparison 必要列だけを direct comparison へ渡せるようにした。
- `direct_hmm_comparison.py`
  - `baseline_frame` optional input を追加。
  - exp072 baseline を CSV gzip から再読込せずに comparison できる。
- `exact_hmm_smoother.py`
  - output prefix を exp209 用に変更。
  - `outer_workers` を追加。既定値は `1`。
- `joint_cache_generation.py`
  - exp072 full cache、HMM cache、direct comparison、SHA/parity report、joint summary を orchestrate。
- train notebook source
  - setup、cost guard、input contract、joint run、metrics/parity 保存をセル分割。

## 固定パラメータ

- exp072: `n_jobs=8`, `pf_seeds=128`, `pf_particles=500`, full 196 features
- HMM v3: `step=0.35`, `n_rates=41`, `band_pad=100`, `numba_num_threads=4`
- HMM v4: `step=0.35`, `n_rates=41`, `band_pad=100`, `outer_workers=1`, `numba_num_threads=null`。Numba runtime default に任せたが、Kaggle summary の実効値は `numba.get_num_threads()=4`。
- HMM v5: `step=0.35`, `n_rates=41`, `band_pad=100`, `outer_workers=2`, `numba_num_threads=2`。外側 well 並列と内側 Numba threads の合計を 4 threads 目安に抑えた guarded pilot。
- HMM v6: `step=0.35`, `n_rates=41`, `band_pad=100`, `outer_workers=4`, `numba_num_threads=1`。同じ 4 threads 目安で外側 well 並列をさらに強める pilot。
- comparison: exp205 v2 と同じ candidate / blend weights

## コマンドログ

```bash
uv run python scripts/new_steering.py --experiment exp209_exp072_exp205_joint_exact_parity_fast_cache_generation
uv run python scripts/new_experiment.py --name exp209_exp072_exp205_joint_exact_parity_fast_cache_generation --source experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit
```

- result: scaffold 作成済み。

```bash
.venv/bin/python -m py_compile \
  experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/settings.py \
  experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exact_hmm_smoother.py \
  experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/feature_cache.py \
  experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/direct_hmm_comparison.py \
  experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp072_feature_cache.py \
  experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/joint_cache_generation.py \
  experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_train.py \
  experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_inference.py
```

- result: PASS

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_inference.py
.venv/bin/ruff check experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation --select F821
uv run python scripts/validate_experiment.py --experiment exp209_exp072_exp205_joint_exact_parity_fast_cache_generation
```

- result: PASS
- `validate_experiment.py`: `experiment validation passed (strict)`

```bash
uv run python scripts/prepare_kaggle_notebooks.py \
  --experiment exp209_exp072_exp205_joint_exact_parity_fast_cache_generation \
  --notebook train \
  --kernel-id kentookumura/exp209-exp072-exp205-joint-exact-parity-fast-cache-generation-train \
  --title "exp209 exp072 exp205 joint exact parity fast cache generation train" \
  --run-on-push \
  --strict
kaggle kernels push -p experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/kaggle/train
kaggle kernels status kentookumura/exp209-joint-exact-parity-train
kaggle kernels pull kentookumura/exp209-joint-exact-parity-train -p /tmp/kaggle-pull/exp209-joint-exact-parity-train-v5 -m
```

- result: static checks / JSON / `validate_experiment.py` / Jupytext test PASS
- version: 5
- URL: `https://www.kaggle.com/code/kentookumura/exp209-joint-exact-parity-train`
- status after push: `KernelWorkerStatus.RUNNING`
- metadata pull: success, `id_no=126193687`
- runtime: CPU, internet off, competition source + exp072/exp205 kernel sources present
- config check: `runtime.numba_num_threads=2`, `feature_cache.hmm.outer_workers=2`

## Kaggle train v5 完了結果

- status: `KernelWorkerStatus.COMPLETE`
- local small output: `/tmp/kaggle-output/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/train_v5_small`
- rows / wells: 3,783,989 / 773
- total elapsed: 20,203.290 sec (約 5h36m43s)
- v4 delta: -12,580.970 sec (約 -3h29m41s)
- v3 delta: -13,981.852 sec (約 -3h53m02s)
- reference simple sum: exp072 v2 17,728.972 sec + exp205 v2 HMM 15,041.783 sec = 32,770.755 sec
- runtime delta vs simple sum: -12,567.465 sec
- exp072 elapsed: 8,723.765 sec
- HMM elapsed: 11,285.868 sec
- HMM delta vs v4: -8,463.231 sec
- HMM delta vs exp205 v2: -3,755.915 sec
- HMM outer workers: `2`
- Numba threads requested / effective: `2` / `2`

Comparison:

- baseline load mode: `in_memory`
- HMM load mode: `csv_gzip`
- id mismatches: 0
- best candidate: `blend_likpf_hmm_w500`
- best RMSE: 10.269696146642758
- exp205 v2 expected RMSE: 10.269699957242537
- absolute diff: 0.0000038105997788306922
- tolerance: 0.000000001
- candidate parity: PASS
- RMSE strict tolerance parity: FAIL
- RMSE approximate parity: ACCEPT
- exp072 `likpf_mean` RMSE: 11.594897672217703
- delta vs exp072 `likpf_mean`: -1.3252015255749452
- best MAE / within10: 6.399211000927199 / 0.793202622946314

SHA / feature parity:

- exp072 generated raw gzip SHA: `cff5e56193100a8dbc2b28471b7a75404f99deb1fa6bcb1d4116f473289606a7`
- exp072 reference raw gzip SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- exp072 generated decompressed SHA: `0503de0512302b06309d26e09fc06ba5095db0ef4d610b1508afe8c8d07ca536`
- exp072 reference decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- exp072 full cache exact artifact parity: FAIL
- HMM generated raw gzip SHA: `8957e92f3e010f307ab0918316060a14e7479d5aba8225676b560272728442ba`
- HMM reference raw gzip SHA: `ca5343ca04b3774fcc4bfb95c96ba1f43a9a9ac70202e545019b3dba308b87d6`
- HMM generated decompressed SHA: `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
- HMM reference decompressed SHA: `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
- HMM feature parity: PASS

Interpretation:

- `outer_workers=2`, `numba_num_threads=2` は有効。HMM wall time は v4 から約 2h21m 短縮し、全体 runtime は約 5h36m まで下がった。
- HMM の per-well mean elapsed は `28.9516 sec` と v4 より増えているが、2 wells を同時処理しているため wall time を採用指標にする。
- HMM decompressed SHA と RMSE 近似一致は維持できた。
- exp072 full cache SHA mismatch は残るが、ユーザー確認済みの近似 RMSE 基準では exp209 v5 を runtime target 達成として完了扱いにする。

## Kaggle train v6: outer_workers=4 / numba_num_threads=1 run

User request: `outer_workers=4`, `numba_num_threads=1` にして再実行する。HMM の数式、HMM grid、exp072 PF seeds / particles、comparison 候補、blend weight は維持する。CPU-only feature generation audit であり、GPU 学習、モデル学習、booster 生成、control 再学習、提出は行わない。

- active variant: exp072 full replay cache + exp205-compatible HMM cache + direct comparison の 1 path
- model/config/fold/booster count: 0
- GPU: なし
- parent/control retraining: なし
- intended version: Kaggle kernel version 6
- runtime setting: `feature_cache.hmm.outer_workers=4`, `runtime.numba_num_threads=1`
- reproducibility note: HMM に RNG はないが、outer threads と Numba parallel の浮動小数順序により strict `1e-9` metric parity / raw gzip SHA は期待しない。HMM decompressed content SHA と best RMSE 近似一致を v5 baseline と比較する。

```bash
.venv/bin/python -m py_compile experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exact_hmm_smoother.py experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/feature_cache.py experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/joint_cache_generation.py experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_train.py
.venv/bin/ruff check experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation --select F821
python3 -m json.tool experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/metrics.json
uv run python scripts/validate_experiment.py --experiment exp209_exp072_exp205_joint_exact_parity_fast_cache_generation
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_train.py
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp209_exp072_exp205_joint_exact_parity_fast_cache_generation --notebook train --kernel-id kentookumura/exp209-joint-exact-parity-train --title "exp209 joint exact parity train" --run-on-push --strict
kaggle kernels push -p experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/kaggle/train
kaggle kernels status kentookumura/exp209-joint-exact-parity-train
kaggle kernels pull kentookumura/exp209-joint-exact-parity-train -p /tmp/kaggle-pull/exp209-joint-exact-parity-train-v6 -m
```

- result: static checks / JSON / `validate_experiment.py` / Jupytext test PASS
- version: 6
- URL: `https://www.kaggle.com/code/kentookumura/exp209-joint-exact-parity-train`
- status after push: `KernelWorkerStatus.RUNNING`
- metadata pull: success, `id_no=126193687`
- runtime: CPU, internet off, competition source + exp072/exp205 kernel sources present
- config check: `runtime.numba_num_threads=1`, `feature_cache.hmm.outer_workers=4`

## Kaggle train v6 完了結果

- status: `KernelWorkerStatus.COMPLETE`
- local small output: `/tmp/kaggle-output/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/train_v6_small`
- rows / wells: 3,783,989 / 773
- total elapsed: 28,768.406 sec (約 7h59m28s)
- v5 delta: +8,565.116 sec (約 +2h22m45s)
- v4 delta: -4,015.854 sec
- reference simple sum: exp072 v2 17,728.972 sec + exp205 v2 HMM 15,041.783 sec = 32,770.755 sec
- runtime delta vs simple sum: -4,002.349 sec
- exp072 elapsed: 13,807.238 sec
- HMM elapsed: 14,627.100 sec
- HMM delta vs v5: +3,341.232 sec
- HMM delta vs exp205 v2: -414.683 sec
- HMM outer workers: `4`
- Numba threads requested / effective: `1` / `1`

Comparison:

- baseline load mode: `in_memory`
- HMM load mode: `csv_gzip`
- id mismatches: 0
- best candidate: `blend_likpf_hmm_w500`
- best RMSE: 10.269696146642758
- exp205 v2 expected RMSE: 10.269699957242537
- absolute diff: 0.0000038105997788306922
- tolerance: 0.000000001
- candidate parity: PASS
- RMSE strict tolerance parity: FAIL
- RMSE approximate parity: ACCEPT
- exp072 `likpf_mean` RMSE: 11.594897672217703
- delta vs exp072 `likpf_mean`: -1.3252015255749452
- best MAE / within10: 6.399211000927199 / 0.793202622946314

SHA / feature parity:

- exp072 generated raw gzip SHA: `8d6d21c322a638b0946e8cf8cc615c5aa1e988c8c0e66c31f9426df71b8b4519`
- exp072 reference raw gzip SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- exp072 generated decompressed SHA: `0503de0512302b06309d26e09fc06ba5095db0ef4d610b1508afe8c8d07ca536`
- exp072 reference decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- exp072 full cache exact artifact parity: FAIL
- HMM generated raw gzip SHA: `a483e2b544021048dfb224db8306142ae0c802a7fe8303b302efa198e0ed17a5`
- HMM reference raw gzip SHA: `ca5343ca04b3774fcc4bfb95c96ba1f43a9a9ac70202e545019b3dba308b87d6`
- HMM generated decompressed SHA: `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
- HMM reference decompressed SHA: `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
- HMM feature parity: PASS

Interpretation:

- `outer_workers=4`, `numba_num_threads=1` は correctness 面では問題ないが、runtime は v5 より悪化した。
- HMM per-well mean elapsed は `74.9061 sec` で、v5 の `28.9516 sec` から大幅に増えた。4-way 外側並列で内側 Numba を 1 thread に落とす配分は、この HMM kernel では効率が悪い。
- 採用 runtime は v5 `outer_workers=2`, `numba_num_threads=2` に戻す。

- result: FAIL
- error: `400 Client Error: Bad Request ... SaveKernel`
- investigation:
  - generated notebook size was only about 86KB, so package size was not the likely cause.
  - long slug length was 67 chars.
  - `kaggle kernels pull kentookumura/exp209-exp072-exp205-joint-exact-parity-fast-cache-generation-train -m` returned 403, so no usable kernel was created.
- action: same exp209 folder, shorter slug/title pair `kentookumura/exp209-joint-exact-parity-train` / `exp209 joint exact parity train` was used to avoid Kaggle SaveKernel slug/title constraints.

```bash
uv run python scripts/prepare_kaggle_notebooks.py \
  --experiment exp209_exp072_exp205_joint_exact_parity_fast_cache_generation \
  --notebook train \
  --kernel-id kentookumura/exp209-joint-exact-parity-train \
  --title "exp209 joint exact parity train" \
  --run-on-push \
  --strict
kaggle kernels push -p experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/kaggle/train
kaggle kernels status kentookumura/exp209-joint-exact-parity-train
kaggle kernels logs kentookumura/exp209-joint-exact-parity-train
```

- version: 1
- result: FAIL
- URL: `https://www.kaggle.com/code/kentookumura/exp209-joint-exact-parity-train`
- error: `ValueError: No kernel name found in notebook and no override provided.`
- cause: generated notebook did not include `metadata.kernelspec.name`.

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_inference.py
python3 - <<'PY'
import json
from pathlib import Path
for name in ['train', 'inference']:
    p = Path(f'experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_{name}.ipynb')
    print(name, json.loads(p.read_text()).get('metadata', {}).get('kernelspec'))
PY
```

- result: PASS
- kernelspec: `{"name": "python3", "language": "python", "display_name": "Python 3 (ipykernel)"}`

```bash
uv run python scripts/prepare_kaggle_notebooks.py \
  --experiment exp209_exp072_exp205_joint_exact_parity_fast_cache_generation \
  --notebook train \
  --kernel-id kentookumura/exp209-joint-exact-parity-train \
  --title "exp209 joint exact parity train" \
  --run-on-push \
  --strict
kaggle kernels push -p experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/kaggle/train
kaggle kernels status kentookumura/exp209-joint-exact-parity-train
kaggle kernels logs kentookumura/exp209-joint-exact-parity-train
```

- version: 2
- result: FAIL
- error: `AttributeError: 'DataFrame' object has no attribute 'wid'`
- cause: exp072 replay helper expects the raw data root and appends `/train`; exp209 had passed `paths.train_data_dir`, causing it to search `/train/train` and select 0 train wells.
- action: `joint_cache_generation.py` now passes `paths.raw_data_dir` only for exp072 generation. HMM generation still uses `paths.train_data_dir`.

```bash
.venv/bin/python -m py_compile experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/joint_cache_generation.py experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_train.py
.venv/bin/ruff check experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation --select F821
uv run python scripts/validate_experiment.py --experiment exp209_exp072_exp205_joint_exact_parity_fast_cache_generation
uv run python scripts/prepare_kaggle_notebooks.py \
  --experiment exp209_exp072_exp205_joint_exact_parity_fast_cache_generation \
  --notebook train \
  --kernel-id kentookumura/exp209-joint-exact-parity-train \
  --title "exp209 joint exact parity train" \
  --run-on-push \
  --strict
kaggle kernels push -p experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/kaggle/train
kaggle kernels status kentookumura/exp209-joint-exact-parity-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp209-joint-exact-parity-train
kaggle kernels status kentookumura/exp209-joint-exact-parity-train
```

- version: 3
- result: COMPLETE
- URL: `https://www.kaggle.com/code/kentookumura/exp209-joint-exact-parity-train`
- metadata: CPU, internet off, competition source present, kernel sources exp072 + exp205 present
- `logs -f` returned no running output before timeout, which is consistent with known Kaggle CLI behavior for running notebooks.

## Kaggle train v3 完了結果

- status: `KernelWorkerStatus.COMPLETE`
- local small output: `/tmp/kaggle-output/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/train_v3_small`
- full output download: size が大きく connection reset したため未完了。採否判断に必要な `summary.json` / metrics / small CSV は `--file-pattern` で取得済み。
- rows / wells: 3,783,989 / 773
- total elapsed: 34,185.142 sec (約 9h29m45s)
- exp072 elapsed: 13,230.642 sec
- HMM elapsed: 20,619.564 sec
- reference simple sum: exp072 v2 17,728.972 sec + exp205 v2 HMM 15,041.783 sec = 32,770.755 sec
- runtime delta: exp209 v3 は simple sum より +1,414.387 sec 遅い
- HMM outer workers: `1`
- Numba threads: `4`

Comparison:

- baseline load mode: `in_memory`
- HMM load mode: `csv_gzip`
- id mismatches: 0
- best candidate: `blend_likpf_hmm_w500`
- best RMSE: 10.269696146642758
- exp205 v2 expected RMSE: 10.269699957242537
- absolute diff: 0.0000038105997788306922
- tolerance: 0.000000001
- candidate parity: PASS
- RMSE strict tolerance parity: FAIL
- RMSE approximate parity: ACCEPT
- exp072 `likpf_mean` RMSE: 11.594897672217703
- delta vs exp072 `likpf_mean`: -1.3252015255749452
- best MAE / within10: 6.399211000927199 / 0.793202622946314

SHA / feature parity:

- exp072 generated raw gzip SHA: `c05121bced73940eedb875c6e847d55fe8c1e3e6d70baf7b2c4c05d935dc1bab`
- exp072 reference raw gzip SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- exp072 generated decompressed SHA: `0503de0512302b06309d26e09fc06ba5095db0ef4d610b1508afe8c8d07ca536`
- exp072 reference decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- exp072 full cache exact artifact parity: FAIL
- HMM generated decompressed SHA: `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
- HMM reference decompressed SHA: `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
- HMM feature parity: PASS

Interpretation:

- HMM は並列化で高速化していない。実装として `outer_workers` option は入れたが、v3 完了 run は `outer_workers=1` で、外側 well 並列は未使用。
- HMM decompressed SHA は exp205 v2 と一致しており、HMM generator の exact parity は通った。
- exp072 full cache SHA が reference と一致しないため、full artifact exact parity は満たしていない。
- best RMSE 差は 3.8106e-06 で、ユーザー確認により実用上は一致として扱う。
- runtime は `< 6h` target に届かず、exp072 v2 + exp205 v2 の単純合算よりも遅かった。
- この時点では strict serial parity が通っていないため `outer_workers=2` の follow-up run は保留判断。後続のユーザー確認で RMSE 近似一致を許容し、v5/v6 の runtime 探索へ進めた。

## Kaggle train v4: Numba default threads run

User request: `numba_num_threads=4` を全 core 相当に寄せる。HMM の数式、HMM grid、exp072 PF seeds / particles、`outer_workers=1` は維持し、`runtime.numba_num_threads=null` に変更した。`exact_hmm_smoother.py` は `numba.get_num_threads()` を summary に記録するよう更新した。

```bash
.venv/bin/python -m py_compile experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exact_hmm_smoother.py experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/feature_cache.py experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/joint_cache_generation.py experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_train.py
.venv/bin/ruff check experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation --select F821
python3 -m json.tool experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/metrics.json
uv run python scripts/validate_experiment.py --experiment exp209_exp072_exp205_joint_exact_parity_fast_cache_generation
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_train.py
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp209_exp072_exp205_joint_exact_parity_fast_cache_generation --notebook train --kernel-id kentookumura/exp209-joint-exact-parity-train --title "exp209 joint exact parity train" --run-on-push --strict
kaggle kernels push -p experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/kaggle/train
kaggle kernels status kentookumura/exp209-joint-exact-parity-train
kaggle kernels logs kentookumura/exp209-joint-exact-parity-train
kaggle kernels pull kentookumura/exp209-joint-exact-parity-train -p /tmp/kaggle-pull/exp209-joint-exact-parity-train-v4 -m
```

- version: 4
- URL: `https://www.kaggle.com/code/kentookumura/exp209-joint-exact-parity-train`
- status after push: `KernelWorkerStatus.RUNNING`
- metadata pull: success, `id_no=126193687`
- runtime: CPU, internet off, competition source + exp072/exp205 kernel sources present
- config check: `runtime.numba_num_threads=null`, `feature_cache.hmm.outer_workers=1`
- logs: empty immediately after push, consistent with Kaggle CLI running-notebook behavior

## Kaggle train v4 完了結果

- status: `KernelWorkerStatus.COMPLETE`
- local small output: `/tmp/kaggle-output/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/train_v4_small`
- rows / wells: 3,783,989 / 773
- total elapsed: 32,784.260 sec (約 9h06m24s)
- v3 delta: -1,400.882 sec (約 -23m21s)
- exp072 elapsed: 12,707.958 sec
- HMM elapsed: 19,749.099 sec
- reference simple sum: exp072 v2 17,728.972 sec + exp205 v2 HMM 15,041.783 sec = 32,770.755 sec
- runtime delta: exp209 v4 は simple sum より +13.505 sec 遅い
- HMM outer workers: `1`
- Numba threads requested / effective: `null` / `4`

Comparison:

- baseline load mode: `in_memory`
- HMM load mode: `csv_gzip`
- id mismatches: 0
- best candidate: `blend_likpf_hmm_w500`
- best RMSE: 10.269696146642758
- exp205 v2 expected RMSE: 10.269699957242537
- absolute diff: 0.0000038105997788306922
- tolerance: 0.000000001
- candidate parity: PASS
- RMSE strict tolerance parity: FAIL
- RMSE approximate parity: ACCEPT
- exp072 `likpf_mean` RMSE: 11.594897672217703
- delta vs exp072 `likpf_mean`: -1.3252015255749452
- best MAE / within10: 6.399211000927199 / 0.793202622946314

SHA / feature parity:

- exp072 generated raw gzip SHA: `343029c0f932a11288d3a15269951c24b4468260ac5c4e988acda7955dbb51eb`
- exp072 reference raw gzip SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- exp072 generated decompressed SHA: `0503de0512302b06309d26e09fc06ba5095db0ef4d610b1508afe8c8d07ca536`
- exp072 reference decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- exp072 full cache exact artifact parity: FAIL
- HMM generated raw gzip SHA: `2b530d9d1857038ea06fcdd638d8a967a0e39cd18f991c7a43c74df5fdbcd1bb`
- HMM reference raw gzip SHA: `ca5343ca04b3774fcc4bfb95c96ba1f43a9a9ac70202e545019b3dba308b87d6`
- HMM generated decompressed SHA: `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
- HMM reference decompressed SHA: `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
- HMM feature parity: PASS

Interpretation:

- `runtime.numba_num_threads=null` は設定として反映されたが、Kaggle 上の Numba runtime default は実効 `4` threads だった。
- そのため v4 は all-core 化による HMM 並列度増加 run ではなく、v3 と同じ Numba 4 threads / `outer_workers=1` の再実行として扱う。
- v4 は v3 より 23 分程度速いが、HMM は exp205 v2 より 4,707.316 sec 遅く、全体は exp072 v2 + exp205 v2 simple sum とほぼ同等に留まった。

## 事後確認コマンド

## Kaggle train v5: outer_workers=2 / numba_num_threads=2 run

User request: `outer_workers=2`, `numba_num_threads=2` にして実行する。HMM の数式、HMM grid、exp072 PF seeds / particles、comparison 候補、blend weight は維持する。CPU-only feature generation audit であり、GPU 学習、モデル学習、booster 生成、control 再学習、提出は行わない。

- active variant: exp072 full replay cache + exp205-compatible HMM cache + direct comparison の 1 path
- model/config/fold/booster count: 0
- GPU: なし
- parent/control retraining: なし
- intended version: Kaggle kernel version 5
- runtime setting: `feature_cache.hmm.outer_workers=2`, `runtime.numba_num_threads=2`
- reproducibility note: HMM に RNG はないが、outer threads と Numba parallel の浮動小数順序により strict `1e-9` metric parity / raw gzip SHA は期待しない。HMM decompressed content SHA と best RMSE 近似一致を v4 serial baseline と比較する。

```bash
.venv/bin/python -m py_compile experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exact_hmm_smoother.py experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/feature_cache.py experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/joint_cache_generation.py experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_train.py
.venv/bin/ruff check experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation --select F821
python3 -m json.tool experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/metrics.json
uv run python scripts/validate_experiment.py --experiment exp209_exp072_exp205_joint_exact_parity_fast_cache_generation
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_train.py
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp209_exp072_exp205_joint_exact_parity_fast_cache_generation --notebook train --kernel-id kentookumura/exp209-joint-exact-parity-train --title "exp209 joint exact parity train" --run-on-push --strict
kaggle kernels push -p experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/kaggle/train
```

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_train.py
.venv/bin/ruff check experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation --select F821
uv run python scripts/validate_experiment.py --experiment exp209_exp072_exp205_joint_exact_parity_fast_cache_generation
```

再 push は不要。短縮 slug の train kernel は:

```bash
uv run python scripts/prepare_kaggle_notebooks.py \
  --experiment exp209_exp072_exp205_joint_exact_parity_fast_cache_generation \
  --notebook train \
  --kernel-id kentookumura/exp209-joint-exact-parity-train \
  --title "exp209 joint exact parity train" \
  --run-on-push \
  --strict
```

- result: PASS
- generated directory: `experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/kaggle/train`
- metadata:
  - `enable_gpu=false`
  - `enable_internet=false`
  - `run_on_push=true`
  - `competition_sources=["rogii-wellbore-geology-prediction"]`
  - `kernel_sources=["kentookumura/exp072-exp063-full-replay-feature-cache-train", "kentookumura/exp205-exact-hmm-smoother-cache-audit-train"]`

## 次のアクション

1. v4 完了後に HMM elapsed、effective `numba_num_threads`、HMM decompressed SHA、best RMSE を v3 と比較する。
2. exp072 full cache SHA mismatch の根本原因を追う場合は、別途列単位 / 行単位 diff の小さな調査に切り出す。
3. likPF-only/slim cache 化をやる場合は、現行 backlog とは別の新規実験として扱う。
