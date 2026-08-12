# exp176_typewell_late_range_pfbeam_candidate_prior セッションノート

## 2026-07-03 実装

- ユーザー依頼により `typewell_late_range_pfbeam_candidate_prior` の実装を開始。
- `docs/legacy/steering/20260703-exp176-typewell-late-range-pfbeam-candidate-prior/` を作成。
- `experiments/exp176_typewell_late_range_pfbeam_candidate_prior/` を `exp157_candidate_ranker_feature_enrichment` から作成。
- helper module を `typewell_late_range_pfbeam_candidate_prior.py` にリネーム。
- train / inference notebook を `exp176_typewell_late_range_pfbeam_candidate_prior_{train,inference}.py` 起点で作成し、`.ipynb` に変換した。

## 実装内容

- exp157 の 8 candidate set と LightGBM ranker 構成は維持。
- raw train typewell / horizontal prefix から `typewell_min`、`typewell_max`、`typewell_span`、`known_last_pct` を作る `read_typewell_late_range_context()` を追加。
- `add_typewell_late_range_prior()` で row-level feature を追加。
  - `tlp_known_last_pct`
  - candidate 別 `tlp_<candidate>_candidate_pct`
  - candidate pct below `0.50/0.60/0.70`
  - `known_last_pct - 0.05/0.10` dynamic lower-bound flag
  - candidate pct summary / late-prefix interaction
- `add_candidate_late_range_columns()` で candidate-long feature を追加。
  - `candidate_tlp_candidate_pct`
  - `candidate_tlp_candidate_pct_minus_known_last_pct`
  - fixed / dynamic lower-bound flag
  - `candidate_tlp_risk_score`
- hard invalid、direct clip、PF/Beam 再生成、submission candidate 作成はしない。

## Kaggle train push 前コスト確認

- Runtime: CPU (`enable_gpu=false`)
- active selector experiment 数: 1
- LightGBM model family 数: 3 (`lgb_multiclass`, `lgb_candidate_binary`, `lgb_candidate_error_ranker`)
- fold 数: 5
- 合計 booster 数: 15
- exp157 control 再学習: なし
- PF/Beam 再生成: なし
- direct TVT regression: なし
- submission / inference port: なし

## 静的検証

```bash
python3 -m py_compile \
  experiments/exp176_typewell_late_range_pfbeam_candidate_prior/typewell_late_range_pfbeam_candidate_prior.py \
  experiments/exp176_typewell_late_range_pfbeam_candidate_prior/settings.py \
  experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_train.py \
  experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_inference.py
```

- result: PASS

```bash
.venv/bin/ruff check \
  experiments/exp176_typewell_late_range_pfbeam_candidate_prior/typewell_late_range_pfbeam_candidate_prior.py \
  experiments/exp176_typewell_late_range_pfbeam_candidate_prior/settings.py \
  experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_train.py \
  experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_inference.py
```

- result: PASS

```bash
.venv/bin/ruff format --check \
  experiments/exp176_typewell_late_range_pfbeam_candidate_prior/typewell_late_range_pfbeam_candidate_prior.py \
  experiments/exp176_typewell_late_range_pfbeam_candidate_prior/settings.py \
  experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_train.py \
  experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_inference.py
```

- result: PASS

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_inference.py
```

- result: PASS

```bash
python3 -m json.tool experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_train.ipynb
python3 -m json.tool experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_inference.ipynb
```

- result: PASS

```bash
uv run python scripts/validate_experiment.py --experiment exp176_typewell_late_range_pfbeam_candidate_prior
```

- result: PASS

## Synthetic smoke

```bash
PYTHONPATH=experiments/exp176_typewell_late_range_pfbeam_candidate_prior .venv/bin/python - <<'PY'
...
PY
```

- result: PASS
- row-level late-range prior features: 77
- candidate-long late-range prior features: 20

## Kaggle package prepare

```bash
make prepare-kaggle-notebooks EXP=exp176_typewell_late_range_pfbeam_candidate_prior EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp176-typewell-late-range-pfbeam-candidate-prior-train --title 'exp176 typewell late range pfbeam candidate prior train' --run-on-push --strict"
```

- result: PASS
- output: `experiments/exp176_typewell_late_range_pfbeam_candidate_prior/kaggle/train`
- kernel id: `kentookumura/exp176-typewell-late-range-pfbeam-candidate-prior-train`
- title: `exp176 typewell late range pfbeam candidate prior train`
- metadata `enable_gpu`: false
- metadata `enable_internet`: false
- kernel sources:
  - `kentookumura/exp099-pf-multiobs-likelihood-train`
  - `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- bootstrap manifest includes `config.yaml`, `settings.py`, `typewell_late_range_pfbeam_candidate_prior.py`, train / inference `.py`, `project.yml`, and `src/`.

## 2026-07-03 Kaggle train push

```bash
make push-kaggle-train EXP=exp176_typewell_late_range_pfbeam_candidate_prior
```

- result: FAIL
- error: `400 Client Error: Bad Request ... SaveKernel`
- initial kernel id/title slug was `exp176-typewell-late-range-pfbeam-candidate-prior-train` / `exp176 typewell late range pfbeam candidate prior train`.
- slug length was 55, matching prior exp174 / exp170 SaveKernel 400 length-limit pattern.
- recovery: keep the same exp176 directory and shorten Kaggle kernel id/title only.

```bash
make prepare-kaggle-notebooks EXP=exp176_typewell_late_range_pfbeam_candidate_prior EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp176-typewell-late-candidate-prior-train --title 'exp176 typewell late candidate prior train' --run-on-push --strict"
make push-kaggle-train EXP=exp176_typewell_late_range_pfbeam_candidate_prior
```

- result: PASS
- Kaggle kernel version: v1
- URL: https://www.kaggle.com/code/kentookumura/exp176-typewell-late-candidate-prior-train
- metadata pull: PASS to `/tmp/kaggle-pull/exp176-typewell-late-candidate-prior-train`
- status after push: `KernelWorkerStatus.RUNNING`
- logs immediately after push: empty, as expected for running Kaggle notebook in this environment.
- user requested to stop monitoring and will report completion. Local `logs -f` monitor was no longer running by the time kill was attempted.

## 2026-07-03 Kaggle train v1 failure / v2 fix

```bash
kaggle kernels logs kentookumura/exp176-typewell-late-candidate-prior-train
kaggle kernels status kentookumura/exp176-typewell-late-candidate-prior-train
```

- v1 status: `KernelWorkerStatus.ERROR`
- failure:
  - `ValueError: No kernel name found in notebook and no override provided.`
  - failure occurred in papermill before notebook execution.
- root cause: generated notebook metadata lacked `metadata.kernelspec.name`.
- fix: added Jupytext notebook metadata to train / inference `.py`:
  - `display_name: Python 3`
  - `language: python`
  - `name: python3`
- regenerated train / inference `.ipynb`.
- verified generated package notebook contains kernelspec.

```bash
python3 -m py_compile experiments/exp176_typewell_late_range_pfbeam_candidate_prior/typewell_late_range_pfbeam_candidate_prior.py experiments/exp176_typewell_late_range_pfbeam_candidate_prior/settings.py experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_train.py experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_inference.py
.venv/bin/ruff check experiments/exp176_typewell_late_range_pfbeam_candidate_prior/typewell_late_range_pfbeam_candidate_prior.py experiments/exp176_typewell_late_range_pfbeam_candidate_prior/settings.py experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_train.py experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_inference.py
uv run python scripts/validate_experiment.py --experiment exp176_typewell_late_range_pfbeam_candidate_prior
```

- result: PASS

```bash
make prepare-kaggle-notebooks EXP=exp176_typewell_late_range_pfbeam_candidate_prior EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp176-typewell-late-candidate-prior-train --title 'exp176 typewell late candidate prior train' --run-on-push --strict"
make push-kaggle-train EXP=exp176_typewell_late_range_pfbeam_candidate_prior
kaggle kernels status kentookumura/exp176-typewell-late-candidate-prior-train
kaggle kernels pull kentookumura/exp176-typewell-late-candidate-prior-train -p /tmp/kaggle-pull/exp176-typewell-late-candidate-prior-train -m
```

- v2 push: PASS
- Kaggle kernel version: v2
- status after v2 push: `KernelWorkerStatus.RUNNING`
- metadata pull after v2 push: PASS
- monitoring: stopped per user request.

## 2026-07-03 Kaggle train v2 failure / v3 memory fix

```bash
kaggle kernels logs kentookumura/exp176-typewell-late-candidate-prior-train
kaggle kernels status kentookumura/exp176-typewell-late-candidate-prior-train
```

- v2 status: `KernelWorkerStatus.ERROR`
- failure:
  - fold 0 の multiclass LightGBM は開始し、best iteration 100 まで到達した。
  - その後 candidate-long binary / error ranker の実行に入る前後で `DeadKernelError: Kernel died`。
- likely root cause:
  - exp176 は row-level `tlp_` feature を 77 列追加しており、v2 ではそのまま candidate-long frame に複製していた。
  - valid fold は約 75.8 万行 x 8 candidates なので、`tlp_` row-level 77 列だけで大きな追加メモリになる。
  - `fit_impute()` の `DataFrame.replace()` も long frame 全体を複製しやすく、CPU notebook の memory pressure が高かった。
- v3 fix:
  - `ranker.long_models.row_feature_exclude_prefixes: [tlp_]` を追加し、row-level `tlp_` は long-frame へ複製しない。
  - candidate-long 用の `candidate_tlp_` feature は維持する。
  - `ranker.long_models.max_train_rows_per_fold` を 650000 から 300000 に下げる。
  - `fit_impute()` を DataFrame replace ではなく NumPy 配列上の non-finite 処理に変更する。
  - fold 後に large arrays / long frames を明示的に解放する。
- v3 Kaggle train 前コスト:
  - Runtime: CPU (`enable_gpu=false`)
  - active selector experiment 数: 1
  - LightGBM model family 数: 3
  - fold 数: 5
  - 合計 booster 数: 15
  - exp157 control 再学習: なし
  - PF/Beam 再生成: なし

## 2026-07-03 Kaggle train v3 push

```bash
python3 -m py_compile experiments/exp176_typewell_late_range_pfbeam_candidate_prior/typewell_late_range_pfbeam_candidate_prior.py experiments/exp176_typewell_late_range_pfbeam_candidate_prior/settings.py experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_train.py experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_inference.py
.venv/bin/ruff check experiments/exp176_typewell_late_range_pfbeam_candidate_prior/typewell_late_range_pfbeam_candidate_prior.py experiments/exp176_typewell_late_range_pfbeam_candidate_prior/settings.py experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_train.py experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_inference.py
.venv/bin/ruff format --check experiments/exp176_typewell_late_range_pfbeam_candidate_prior/typewell_late_range_pfbeam_candidate_prior.py experiments/exp176_typewell_late_range_pfbeam_candidate_prior/settings.py experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_train.py experiments/exp176_typewell_late_range_pfbeam_candidate_prior/exp176_typewell_late_range_pfbeam_candidate_prior_inference.py
uv run python scripts/validate_experiment.py --experiment exp176_typewell_late_range_pfbeam_candidate_prior
```

- result: PASS
- lightweight check: `select_long_row_feature_columns(..., ['a', 'tlp_known_last_pct', 'tlp_candidate_pct_mean'])` -> `['a']`

```bash
make prepare-kaggle-notebooks EXP=exp176_typewell_late_range_pfbeam_candidate_prior EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp176-typewell-late-candidate-prior-train --title 'exp176 typewell late candidate prior train' --run-on-push --strict"
make push-kaggle-train EXP=exp176_typewell_late_range_pfbeam_candidate_prior
kaggle kernels status kentookumura/exp176-typewell-late-candidate-prior-train
kaggle kernels pull kentookumura/exp176-typewell-late-candidate-prior-train -p /tmp/kaggle-pull/exp176-typewell-late-candidate-prior-train -m
```

- prepare: PASS
- generated notebook kernelspec: `python3`
- package config: `max_train_rows_per_fold=300000`, `row_feature_exclude_prefixes=[tlp_]`
- push: PASS
- Kaggle kernel version: v3
- URL: https://www.kaggle.com/code/kentookumura/exp176-typewell-late-candidate-prior-train
- status after v3 push: `KernelWorkerStatus.RUNNING`
- metadata pull: PASS
- metadata `id_no`: 125806103
- metadata `enable_gpu`: false
- metadata `enable_internet`: false
- monitoring: not started, per user request.

## 2026-07-04 Kaggle train v3 completion

```bash
kaggle kernels status kentookumura/exp176-typewell-late-candidate-prior-train
kaggle kernels logs kentookumura/exp176-typewell-late-candidate-prior-train
```

- status: `KernelWorkerStatus.COMPLETE`
- Kaggle kernel version: v3
- output archive: not downloaded. CV、SHA、生成物パスは logs の summary を根拠に記録した。
- runtime: 9329.729704 sec
- rows: 3,783,989
- wells: 773
- feature_count: 174
- long row features after v3 memory fix: 97
- long features: 124
- typewell context rows / valid rows: 773 / 773
- best OOF:
  - variant: `lgb_candidate_error_ranker`
  - RMSE: 10.64129809970173
  - MAE: 6.434563137170688
  - within10: 0.7918154624656678
  - oracle label accuracy: 0.25743917331683575
  - PF/ANCC selection rate: 0.3808161175944222
  - max path switch / 1000 rows: 330.84185680566486
- comparison:
  - vs `likpf_mean_single` 11.594897672217703: -0.9535995725159729 RMSE
  - vs exp157 best OOF 10.79579983712686: -0.15450173742512932 RMSE
  - vs exp158 best Viterbi 10.789163253: -0.14786515329826972 RMSE
  - within10 は exp157 0.7925049464995803 から -0.0006894840339125042
  - oracle label accuracy は exp157 0.25868811986504187 から -0.0012489465482061223
- decision: `ranker_supported_for_followup_continuity_audit`
- submit candidate: no
- generated artifacts:
  - `/kaggle/working/artifacts/exp176_typewell_late_range_pfbeam_candidate_prior_metrics.csv`
  - `/kaggle/working/artifacts/exp176_typewell_late_range_pfbeam_candidate_prior_feature_importance_mean.csv`
  - `/kaggle/working/artifacts/exp176_typewell_late_range_pfbeam_candidate_prior_feature_schema.csv`
- SHA:
  - metrics: `7d22a50027e003175dc789ca20c89ad48c521e83fa361aae105c8e1f3f98855c`
  - feature_schema: `f4643a36903f0a84b4f2e1f205581fcd4d6dfe18474372848a188c921126b612`
  - predictions_decompressed: `281785834923508b20258986d2d24953b754077200bc5d827f13ffd7a90e6cd8`
  - best OOF prediction (`lgb_candidate_error_ranker`): `ad9eeff7bcac5116ba77316255f162928a9117ded50f3a27b855739f068d0e66`

## 次アクション

- `typewell_late_range_pfbeam_candidate_prior` は完了済みとして backlog から外す。
- exp176 の late-range prior signal は、row-wise direct path ではなく exp158-style continuity selector または exp148/ML anchor confidence feature として使う。
- hard invalid / direct clip / PF/Beam generation soft prior / clipped candidate augmentation は、continuity guard と raw-test parity を見るまで進めない。
