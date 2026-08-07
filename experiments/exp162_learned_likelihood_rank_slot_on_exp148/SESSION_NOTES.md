# exp162_learned_likelihood_rank_slot_on_exp148 セッションノート

## 2026-06-30 実装

- `.steering/20260630-exp162-learned-likelihood-rank-slot-on-exp148/` を作成。
- `experiments/exp162_learned_likelihood_rank_slot_on_exp148/` を exp148 から作成。
- 親実験は `exp148_learned_likelihood_fulltrain_addonly_on_exp092`。
- exp148 の U-projection / learned likelihood confidence surface は残し、exp145 の candidate 別 `learned_prob_*` と `learned_pred_abs_error_*` で rank1/rank2/rank3 を作る `llrs_` 特徴を add-only する。
- 追加 feature group:
  - `learned_likelihood_rank_slot_identity`
  - `learned_likelihood_rank_slot_delta`
  - `learned_likelihood_rank_slot_u_projection`
  - `learned_likelihood_rank_slot_u_disagreement`
  - `learned_likelihood_rank_slot_exp098_compare`
- Candidate TVT path の direct selector、soft average、blend、postprocess replacement は入れない。

## CPU 実行ガード

- active variant 数: 1 (`learned_likelihood_rank_slot_addonly`)
- LightGBM config 数: 3 (`lgb0`, `lgb1`, `lgb2`)
- fold 数: 5
- 合計 booster 数: 15
- active mode: `cpu_deterministic_threads8`
- `runtime.kaggle.enable_gpu`: false
- exp148 control 再学習: なし
- baseline は保存済み exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960 を参照する。

## 検証ログ

- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp162_learned_likelihood_rank_slot_on_exp148/exp162_learned_likelihood_rank_slot_on_exp148_train.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp162_learned_likelihood_rank_slot_on_exp148/exp162_learned_likelihood_rank_slot_on_exp148_inference.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp162_learned_likelihood_rank_slot_on_exp148/exp162_learned_likelihood_rank_slot_on_exp148_train.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp162_learned_likelihood_rank_slot_on_exp148/exp162_learned_likelihood_rank_slot_on_exp148_inference.py`: PASS
- `.venv/bin/python -m py_compile experiments/exp162_learned_likelihood_rank_slot_on_exp148/learned_likelihood_rank_slot_on_exp148.py experiments/exp162_learned_likelihood_rank_slot_on_exp148/settings.py experiments/exp162_learned_likelihood_rank_slot_on_exp148/exp162_learned_likelihood_rank_slot_on_exp148_train.py experiments/exp162_learned_likelihood_rank_slot_on_exp148/exp162_learned_likelihood_rank_slot_on_exp148_inference.py`: PASS
- `uv run ruff check experiments/exp162_learned_likelihood_rank_slot_on_exp148/learned_likelihood_rank_slot_on_exp148.py experiments/exp162_learned_likelihood_rank_slot_on_exp148/settings.py experiments/exp162_learned_likelihood_rank_slot_on_exp148/exp162_learned_likelihood_rank_slot_on_exp148_train.py experiments/exp162_learned_likelihood_rank_slot_on_exp148/exp162_learned_likelihood_rank_slot_on_exp148_inference.py --select F821`: PASS
- `uv run ruff format --check experiments/exp162_learned_likelihood_rank_slot_on_exp148/learned_likelihood_rank_slot_on_exp148.py experiments/exp162_learned_likelihood_rank_slot_on_exp148/settings.py experiments/exp162_learned_likelihood_rank_slot_on_exp148/exp162_learned_likelihood_rank_slot_on_exp148_train.py experiments/exp162_learned_likelihood_rank_slot_on_exp148/exp162_learned_likelihood_rank_slot_on_exp148_inference.py`: PASS
- `make validate-exp EXP=exp162_learned_likelihood_rank_slot_on_exp148`: PASS
- `make prepare-kaggle-notebooks EXP=exp162_learned_likelihood_rank_slot_on_exp148 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp162-learned-likelihood-rank-slot-on-exp148-train --title 'exp162 learned likelihood rank slot on exp148 train' --run-on-push --strict"`: PASS
- `make prepare-kaggle-notebooks EXP=exp162_learned_likelihood_rank_slot_on_exp148 EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp162-learned-likelihood-rank-slot-on-exp148-inference --title 'exp162 learned likelihood rank slot on exp148 inference' --run-on-push --strict"`: PASS
- train / inference `kernel-metadata.json` は `enable_gpu=false`, `enable_internet=false`, `run_on_push=true` を確認済み。
- Kaggle package 内に旧 exp148 名の copied notebook/script が残っていないことを確認済み。

## 2026-06-30 compact self-contained train 修正

- ユーザー指摘により、当初の `exp162_learned_likelihood_rank_slot_on_exp148_train.ipynb` が exp148 compact self-contained train と比べて薄い通常 orchestration 版になっていたことを確認した。
- 原因は、exp148 の `exp148_learned_likelihood_fulltrain_addonly_on_exp092_train.py` 通常版を踏襲し、`settings.py` と `learned_likelihood_rank_slot_on_exp148.py` を import する形で作ったため。
- 新規 notebook 方針では compact self-contained を基本形にするため、正規 `exp162_learned_likelihood_rank_slot_on_exp148_train.py` を compact self-contained train に置き換えた。
- 現在の train notebook は exp148 compact と同じ 8 章構成:
  - Imports
  - Runtime and configuration helpers
  - Train feature assembly helpers
  - Model training and artifact helpers
  - Setup and configuration
  - Input and feature contract
  - Train learned rank-slot variant
  - Metrics and generated artifacts
- 行数比較:
  - exp148 compact self-contained train: 1702 lines
  - exp162 compact self-contained train: 2769 lines
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp162_learned_likelihood_rank_slot_on_exp148/exp162_learned_likelihood_rank_slot_on_exp148_train.py`: PASS
- `.venv/bin/python -m py_compile experiments/exp162_learned_likelihood_rank_slot_on_exp148/exp162_learned_likelihood_rank_slot_on_exp148_train.py`: PASS
- `uv run ruff check experiments/exp162_learned_likelihood_rank_slot_on_exp148/exp162_learned_likelihood_rank_slot_on_exp148_train.py --select F821`: PASS
- `uv run ruff format --check experiments/exp162_learned_likelihood_rank_slot_on_exp148/exp162_learned_likelihood_rank_slot_on_exp148_train.py`: PASS
- `make validate-exp EXP=exp162_learned_likelihood_rank_slot_on_exp148`: PASS
- `make prepare-kaggle-notebooks EXP=exp162_learned_likelihood_rank_slot_on_exp148 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp162-learned-likelihood-rank-slot-on-exp148-train --title 'exp162 learned likelihood rank slot on exp148 train' --run-on-push --strict"`: PASS
- train `kernel-metadata.json` は引き続き `enable_gpu=false`, `enable_internet=false`, `run_on_push=true`。

## 2026-06-30 Kaggle train 実行

- 初回 push:
  - command: `make push-kaggle-train EXP=exp162_learned_likelihood_rank_slot_on_exp148`
  - long slug `kentookumura/exp162-learned-likelihood-rank-slot-on-exp148-train` では Kaggle API `400 Client Error: Bad Request` で SaveKernel に失敗した。
  - kernel は作成されていなかった。
- slug を `kentookumura/exp162-ll-rank-slot-exp148-train`、title を `exp162 ll rank slot exp148 train` に短縮し、inference source も同 slug に合わせた。
- train v1:
  - command: `make push-kaggle-train EXP=exp162_learned_likelihood_rank_slot_on_exp148`
  - result: `Kernel version 1 successfully pushed`
  - URL: `https://www.kaggle.com/code/kentookumura/exp162-ll-rank-slot-exp148-train`
  - status: `KernelWorkerStatus.ERROR`
  - failure: notebook cell 3 で `NameError: name '__file__' is not defined`
  - cause: compact self-contained 化時に `settings.py` 由来の `Path(__file__)` が残っていた。Kaggle notebook では `__file__` が未定義。
- train notebook 修正:
  - `PACKAGE_DIR = Path.cwd()` に変更。
  - `find_project_root` fallback を `start` に変更。
  - config path を `PACKAGE_DIR / "config.yaml"` に変更。
  - `jupytext --to ipynb --test`: PASS
  - `py_compile`: PASS
  - `ruff F821`: PASS
  - `ruff format --check`: PASS
  - `make validate-exp EXP=exp162_learned_likelihood_rank_slot_on_exp148`: PASS
  - `make prepare-kaggle-notebooks EXP=exp162_learned_likelihood_rank_slot_on_exp148 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp162-ll-rank-slot-exp148-train --title 'exp162 ll rank slot exp148 train' --run-on-push --strict"`: PASS
- train v2:
  - command: `make push-kaggle-train EXP=exp162_learned_likelihood_rank_slot_on_exp148`
  - result: `Kernel version 2 successfully pushed`
  - URL: `https://www.kaggle.com/code/kentookumura/exp162-ll-rank-slot-exp148-train`
  - latest checked status: `KernelWorkerStatus.RUNNING`
  - `kaggle kernels logs` and 10-minute `logs -f` returned no notebook logs yet. Worker status stayed `RUNNING`; likely queue/bootstrap/logs API delay.

## 2026-07-01 Kaggle train v2 timeout readout

- User reported the run timed out and provided partial visual fold metrics.
- `kaggle kernels status kentookumura/exp162-ll-rank-slot-exp148-train`: `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`
- `kaggle kernels logs kentookumura/exp162-ll-rank-slot-exp148-train` later returned fold-level logs.
- The notebook reached all 15 fold-level LightGBM fits, but timed out/canceled before final summary/artifact completion. Treat these as logs-derived interim metrics, not a completed train summary.
- Run setup from logs:
  - rows: `3783989`
  - features: `375`
  - active mode: `cpu_deterministic_threads8`
  - active variant: `learned_likelihood_rank_slot_addonly`
  - use_gpu: `false`
- Fold metrics from logs:
  - `lgb0`:
    - fold0: RMSE `9.248265962587892`, best_iteration `4995`
    - fold1: RMSE `8.666493615621523`, best_iteration `766`
    - fold2: RMSE `7.315382134803372`, best_iteration `1648`
    - fold3: RMSE `8.303875272516935`, best_iteration `180`
    - fold4: RMSE `8.780813813130763`, best_iteration `270`
  - `lgb1`:
    - fold0: RMSE `9.137180818935946`, best_iteration `10000`
    - fold1: RMSE `8.467454524874421`, best_iteration `7264`
    - fold2: RMSE `7.233882811913086`, best_iteration `1929`
    - fold3: RMSE `8.520441931645728`, best_iteration `762`
    - fold4: RMSE `8.799596852821711`, best_iteration `1291`
  - `lgb2`:
    - fold0: RMSE `9.105696680827483`, best_iteration `9981`
    - fold1: RMSE `8.488784217479346`, best_iteration `3926`
    - fold2: RMSE `7.348655102590403`, best_iteration `2127`
    - fold3: RMSE `8.472259655164128`, best_iteration `1716`
    - fold4: RMSE `8.698947760205431`, best_iteration `1523`
- Logs-derived weighted pooled estimates using the known 5-fold validation row counts from this split:
  - `lgb0`: RMSE `8.48804924068525`
  - `lgb1`: RMSE `8.456600573816607`
  - `lgb2`: RMSE `8.443346041268542`
- Interpretation:
  - Logs-derived `lgb1` / `lgb2` single-model pooled estimates are numerically lower than exp148 `lgb_mean` CV `8.50128118189582`, but this is not a final experiment result.
  - The comparable `lgb_mean` ensemble metric for exp162 was not computed because the run timed out before final post-processing.
  - The run also did not write model artifacts, predictions, feature importance, by-well, or bucket metrics, so submit/inference decisions cannot be made from this run alone.
  - The CPU full run exceeded Kaggle runtime before final artifact write. A completed run likely needs GPU, fewer configs, fewer rows for smoke, or code changes to write fold metrics incrementally before post-processing.

## 2026-07-01 CPU split train notebook implementation

- Timeout mitigation: split the CPU train into one Kaggle notebook per LightGBM config.
- Added split train notebooks:
  - `exp162_learned_likelihood_rank_slot_on_exp148_train_lgb0.py/ipynb`
  - `exp162_learned_likelihood_rank_slot_on_exp148_train_lgb1.py/ipynb`
  - `exp162_learned_likelihood_rank_slot_on_exp148_train_lgb2.py/ipynb`
- Each split train notebook runs:
  - active variants: `learned_likelihood_rank_slot_addonly`
  - active modes: `cpu_deterministic_threads8`
  - LightGBM configs: 1 selected config only (`lgb0`, `lgb1`, or `lgb2`)
  - folds: 5
  - boosters per notebook: 5
  - total boosters across the split suite: 15
  - Kaggle metadata: `enable_gpu=false`, `enable_internet=false`, `run_on_push=true`
- Artifact separation:
  - `lgb0` writes `exp162_learned_likelihood_rank_slot_on_exp148_lgb0_lgb_models/manifest.json`
  - `lgb1` writes `exp162_learned_likelihood_rank_slot_on_exp148_lgb1_lgb_models/manifest.json`
  - `lgb2` writes `exp162_learned_likelihood_rank_slot_on_exp148_lgb2_lgb_models/manifest.json`
- Inference update:
  - `inference.model_manifest_prefixes` now lists the three split prefixes.
  - `run_saved_model_inference` accepts multiple manifest paths/prefixes.
  - `selected_model: lgb_mean` loads all matching split manifest boosters and averages them.
- Kaggle package slugs prepared:
  - `kentookumura/exp162-ll-rank-slot-exp148-lgb0-train`
  - `kentookumura/exp162-ll-rank-slot-exp148-lgb1-train`
  - `kentookumura/exp162-ll-rank-slot-exp148-lgb2-train`
  - `kentookumura/exp162-ll-rank-slot-exp148-infer`
- Validation:
  - `python3 -m py_compile ...`: PASS
  - `uv run ruff check ... --select F821`: PASS
  - `uv run ruff format --check ...`: PASS
  - `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...`: PASS
  - `make validate-exp EXP=exp162_learned_likelihood_rank_slot_on_exp148`: PASS
  - `scripts/prepare_kaggle_notebooks.py --notebook train_lgb0/train_lgb1/train_lgb2/inference --strict`: PASS

## 2026-07-01 CPU split train Kaggle push

- Pushed split train notebooks with `run_on_push=true`:
  - `kentookumura/exp162-ll-rank-slot-exp148-lgb0-train`: version 1 pushed, URL `https://www.kaggle.com/code/kentookumura/exp162-ll-rank-slot-exp148-lgb0-train`
  - `kentookumura/exp162-ll-rank-slot-exp148-lgb1-train`: version 1 pushed, URL `https://www.kaggle.com/code/kentookumura/exp162-ll-rank-slot-exp148-lgb1-train`
  - `kentookumura/exp162-ll-rank-slot-exp148-lgb2-train`: version 1 pushed, URL `https://www.kaggle.com/code/kentookumura/exp162-ll-rank-slot-exp148-lgb2-train`
- Initial `kaggle kernels status` after push:
  - `lgb0`: `KernelWorkerStatus.RUNNING`
  - `lgb1`: `KernelWorkerStatus.RUNNING`
  - `lgb2`: `KernelWorkerStatus.RUNNING`
- `kaggle kernels logs` immediately after push returned no notebook output yet.
- Pushed inference kernel with `run_on_push=false`:
  - `kentookumura/exp162-ll-rank-slot-exp148-infer`: version 1 pushed, URL `https://www.kaggle.com/code/kentookumura/exp162-ll-rank-slot-exp148-infer`
  - Kaggle warning: the three split train kernel sources were not yet valid and could not be added. Treat this as expected while the split train kernels are still running / not yet output-ready.
  - Action required after all split train kernels complete: re-prepare/re-push inference so the completed train outputs are attached as valid kernel sources before running inference.

## 2026-07-01 CPU split train completion and inference launch

- User reported the split train suite completed; CLI status confirmed:
  - `kentookumura/exp162-ll-rank-slot-exp148-lgb0-train`: `KernelWorkerStatus.COMPLETE`
  - `kentookumura/exp162-ll-rank-slot-exp148-lgb1-train`: `KernelWorkerStatus.COMPLETE`
  - `kentookumura/exp162-ll-rank-slot-exp148-lgb2-train`: `KernelWorkerStatus.COMPLETE`
- Logs confirmed `train_completed` for all split runs and artifact manifests were written:
  - `lgb0`: `exp162_learned_likelihood_rank_slot_on_exp148_lgb0_lgb_models/manifest.json`
  - `lgb1`: `exp162_learned_likelihood_rank_slot_on_exp148_lgb1_lgb_models/manifest.json`
  - `lgb2`: `exp162_learned_likelihood_rank_slot_on_exp148_lgb2_lgb_models/manifest.json`
- Split pooled metrics from logs:
  - `lgb0`: RMSE `8.48804924068525`, elapsed `12888.173` seconds
  - `lgb1`: RMSE `8.456600573816607`, elapsed `13250.118` seconds
  - `lgb2`: RMSE `8.443346041268544`, elapsed `14696.778` seconds
- Re-pushed inference after train completion:
  - command: `kaggle kernels push -p experiments/exp162_learned_likelihood_rank_slot_on_exp148/kaggle/inference`
  - result: `Kernel version 2 successfully pushed`
  - no invalid-source warning; split train kernel sources are now accepted.
- Re-prepared inference with `run_on_push=true`:
  - command: `python3 scripts/prepare_kaggle_notebooks.py --experiment exp162_learned_likelihood_rank_slot_on_exp148 --notebook inference --kernel-id kentookumura/exp162-ll-rank-slot-exp148-infer --title "exp162 ll rank slot exp148 infer" --run-on-push --strict`
  - result: PASS
- Attempted inference run push:
  - command: `kaggle kernels push -p experiments/exp162_learned_likelihood_rank_slot_on_exp148/kaggle/inference`
  - CLI returned `Kernel push error: Maximum batch CPU session count of 5 reached.`
  - Follow-up `kaggle kernels status kentookumura/exp162-ll-rank-slot-exp148-infer` returned `KernelWorkerStatus.RUNNING`; treat inference as possibly started despite the push error.
  - `kaggle kernels logs kentookumura/exp162-ll-rank-slot-exp148-infer` is currently empty, consistent with prior Kaggle CLI behavior while notebooks are still running.
- Inference v2 later failed:
  - status: `KernelWorkerStatus.ERROR`
  - failure: raw-test learned likelihood features were not available as a Kaggle input source, so the notebook fell back to generating them and failed with `ValueError: generator.candidates must not be empty`.
  - fix: add `kentookumura/exp145-inference` to `runtime.kaggle.inference_kernel_sources` and pass `data.learned_likelihood_rawtest_features_local` as `learned_feature_path`.
  - validation after fix:
    - `jupytext --to ipynb`: PASS
    - `py_compile`: PASS
    - `ruff F821`: PASS
    - `ruff format --check`: PASS
    - `jupytext --to ipynb --test`: PASS
    - `make validate-exp EXP=exp162_learned_likelihood_rank_slot_on_exp148`: PASS
- Inference v3:
  - command: `kaggle kernels push -p experiments/exp162_learned_likelihood_rank_slot_on_exp148/kaggle/inference`
  - result: `Kernel version 3 successfully pushed`
  - status immediately after push: `KernelWorkerStatus.RUNNING`
  - final status: `KernelWorkerStatus.COMPLETE`
  - raw-test learned likelihood source: `/kaggle/input/exp145-inference/artifacts/exp145_learned_likelihood_rawtest_feature_generator_parity_rawtest_ml_features.csv.gz`
  - loaded train manifests:
    - `/kaggle/input/exp162-ll-rank-slot-exp148-lgb0-train/artifacts/exp162_learned_likelihood_rank_slot_on_exp148_lgb0_lgb_models/manifest.json`
    - `/kaggle/input/exp162-ll-rank-slot-exp148-lgb1-train/artifacts/exp162_learned_likelihood_rank_slot_on_exp148_lgb1_lgb_models/manifest.json`
    - `/kaggle/input/exp162-ll-rank-slot-exp148-lgb2-train/artifacts/exp162_learned_likelihood_rank_slot_on_exp148_lgb2_lgb_models/manifest.json`
  - loaded boosters: 15
  - test rows / submission rows: `14151` / `14151`
  - predicted rows / fallback rows: `14151` / `0`
  - prediction min / max / mean / std: `11590.8623046875` / `12240.15234375` / `11905.147258047686` / `278.86038280580544`
  - prediction SHA256: `2c93fe0030206d0d9824edb368c72002730868a3ba5142f090171c2d8ccd143e`
  - submission SHA256: `7f5d9156a732531148f15680cd0583a4df4418c440b4d3e93ad2fef9336da8ea`
  - output submission path in notebook: `/kaggle/working/submission.csv`

## 2026-07-02 hidden submission failure triage

- User reported Kaggle submit failed in the hidden rerun with `Notebook Threw Exception`.
- Latest submission table entry:
  - ref: `54234540`
  - file: `submission.csv`
  - date: `2026-07-01 14:08:51.663000`
  - CLI status: `SubmissionStatus.COMPLETE`
  - public score: blank
- Root cause:
  - inference v3 attached `kentookumura/exp145-inference` and used the public raw-test learned-likelihood feature file directly.
  - Hidden rerun has different test IDs, so `learned_feature_keys_match` cannot use that public feature file and falls back to current-test feature generation.
  - exp162 config did not include the exp145 generator contract (`generator.candidates`, row-context columns, multi-observation columns, feature-cache flags) or exp111 model/schema paths needed for fallback generation.
  - The hidden rerun therefore hit `ValueError: generator.candidates must not be empty`.
- Local fix:
  - Added exp111 artifact/schema/manifest paths to `config.yaml`.
  - Added exp145-compatible `generator.candidates`, `row_context_columns`, `multiobs_global_columns`, and `feature_cache` settings to `config.yaml`.
  - Removed `kentookumura/exp145-inference` from inference kernel sources.
  - Changed inference to pass `learned_feature_path=None`, so public and hidden runs both exercise current-test feature generation instead of relying on public raw-test features.
- Validation after local fix:
  - `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp162_learned_likelihood_rank_slot_on_exp148/exp162_learned_likelihood_rank_slot_on_exp148_inference.py`: PASS
  - `python3 -m py_compile experiments/exp162_learned_likelihood_rank_slot_on_exp148/exp162_learned_likelihood_rank_slot_on_exp148_inference.py experiments/exp162_learned_likelihood_rank_slot_on_exp148/learned_likelihood_rank_slot_on_exp148.py`: PASS
  - `uv run ruff check experiments/exp162_learned_likelihood_rank_slot_on_exp148/exp162_learned_likelihood_rank_slot_on_exp148_inference.py experiments/exp162_learned_likelihood_rank_slot_on_exp148/learned_likelihood_rank_slot_on_exp148.py --select F821`: PASS
  - `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp162_learned_likelihood_rank_slot_on_exp148/exp162_learned_likelihood_rank_slot_on_exp148_inference.py`: PASS
  - `python3 scripts/prepare_kaggle_notebooks.py --experiment exp162_learned_likelihood_rank_slot_on_exp148 --notebook inference --kernel-id kentookumura/exp162-ll-rank-slot-exp148-infer --title "exp162 ll rank slot exp148 infer" --run-on-push --strict`: PASS
  - `make validate-exp EXP=exp162_learned_likelihood_rank_slot_on_exp148`: PASS
- Remote update status:
  - Pulling `kentookumura/exp162-ll-rank-slot-exp148-infer` confirmed the remote kernel is still v3: it still has `kentookumura/exp145-inference` in `kernel_sources` and passes `data.learned_likelihood_rawtest_features_local`.
  - Re-pushing the fixed inference kernel failed with `Kernel push error: Maximum batch CPU session count of 5 reached.`
  - Re-preparing with `run_on_push=false` and pushing also failed with the same error, so the block is Kaggle account CPU session capacity, not only run-on-push execution.
  - Active CPU sessions at the time of triage:
    - `kentookumura/exp161-prefix-crop-exp148-train-lgb0`: `RUNNING`
    - `kentookumura/exp161-prefix-crop-exp148-train-lgb1`: `RUNNING`
    - `kentookumura/exp163-typewell-prior-exp148-lgb0-train`: `RUNNING`
    - `kentookumura/exp163-typewell-prior-exp148-lgb1-train`: `RUNNING`
    - `kentookumura/exp163-typewell-prior-exp148-lgb2-train`: `RUNNING`
  - Do not resubmit exp162 until the fixed inference kernel is successfully pushed and completed at least one public rerun without `exp145-inference`.

## 2026-07-02 hidden-safe inference v4 execution

- CPU session capacity cleared after these notebooks completed:
  - `kentookumura/exp161-prefix-crop-exp148-train-lgb0`: `COMPLETE`
  - `kentookumura/exp161-prefix-crop-exp148-train-lgb1`: `COMPLETE`
  - `kentookumura/exp163-typewell-prior-exp148-lgb0-train`: `COMPLETE`
  - `kentookumura/exp163-typewell-prior-exp148-lgb1-train`: `COMPLETE`
  - `kentookumura/exp163-typewell-prior-exp148-lgb2-train`: `COMPLETE`
- Pushed fixed inference:
  - command: `kaggle kernels push -p experiments/exp162_learned_likelihood_rank_slot_on_exp148/kaggle/inference`
  - result: `Kernel version 4 successfully pushed`
  - kernel: `kentookumura/exp162-ll-rank-slot-exp148-infer`
- Remote metadata / notebook check:
  - `kernel_sources` no longer includes `kentookumura/exp145-inference`.
  - notebook body passes `learned_feature_path=None`.
  - `enable_gpu=false`, `enable_internet=false`.
- v4 status:
  - `kaggle kernels status kentookumura/exp162-ll-rank-slot-exp148-infer`: `KernelWorkerStatus.COMPLETE`
- v4 log evidence:
  - mode: `saved_lgb_booster_inference_with_raw_test_feature_replay`
  - generated current-test learned-likelihood feature source: `/kaggle/working/artifacts/exp162_learned_likelihood_rank_slot_on_exp148_current_test_learned_likelihood_ml_features.csv.gz`
  - source kind: `target_free_current_test_generated_learned_likelihood_ml_features`
  - learned feature rows / wells / columns: `14151` / `3` / `51`
  - learned feature decompressed SHA256: `27efc7c7ef776fc21a9792c8e1a587d4f9fc99a0b2e7945cd8d47d165c658fbb`
  - exp111 manifest SHA256: `178e8b3124b817a2b230080fc041aaaee1b06941e5a4223a68cc31bf26e68010`
  - loaded boosters: `15`
  - feature count: `375`
  - test rows / submission rows / predicted rows: `14151` / `14151` / `14151`
  - fallback rows: `0`
  - prediction min / max / mean / std: `11590.8623046875` / `12240.15234375` / `11905.147261360173` / `278.8603867428979`
  - prediction SHA256: `16ad86b3d400c3aa0bfd67e86e6340eda0d8293d919011694df81d6499b0b7da`
  - submission SHA256: `75c7374ae07314e996d69968cee3743f4119e6d6229ac5339195ee0107777571`
  - elapsed seconds: `155.072`
- Downloaded v4 output:
  - command: `kaggle kernels output kentookumura/exp162-ll-rank-slot-exp148-infer -p experiments/exp162_learned_likelihood_rank_slot_on_exp148/kaggle/output/inference_v4`
  - submission: `experiments/exp162_learned_likelihood_rank_slot_on_exp148/kaggle/output/inference_v4/submission.csv`
  - summary: `experiments/exp162_learned_likelihood_rank_slot_on_exp148/kaggle/output/inference_v4/artifacts/exp162_learned_likelihood_rank_slot_on_exp148_inference_summary.json`
- Submit-check:
  - command: `python3 .agents/skills/kaggle-submit-check/scripts/check_submission.py experiments/exp162_learned_likelihood_rank_slot_on_exp148/kaggle/output/inference_v4/submission.csv --sample data/raw/sample_submission.csv`
  - result: PASS
  - rows / columns: `14151` / `2`
  - no duplicate IDs, no empty/NaN/Inf-like values
  - header and row count match `data/raw/sample_submission.csv`
- This validates the hidden-safe inference path on the public rerun. Competition submission has not been re-run in this step.

## 2026-07-02 code submission result

- User reported scoring completed; CLI confirmed latest completed submission:
  - ref: `54247043`
  - file: `submission.csv`
  - date: `2026-07-02 00:02:06.833000`
  - status: `SubmissionStatus.COMPLETE`
  - Public LB: `8.100`
  - Private LB: blank
- Submission artifact:
  - kernel: `kentookumura/exp162-ll-rank-slot-exp148-infer` v4
  - submission path: `experiments/exp162_learned_likelihood_rank_slot_on_exp148/kaggle/output/inference_v4/submission.csv`
  - submission SHA256: `75c7374ae07314e996d69968cee3743f4119e6d6229ac5339195ee0107777571`
- Interpretation:
  - Worse than exp148 Public LB `7.960` by +0.140.
  - Worse than exp160 Public LB `8.061` by +0.039.
  - Hidden-safe fix worked, but the learned-likelihood rank-slot add-only features did not improve LB.
  - Do not adopt exp162 as the ML route anchor; keep exp148 as the anchor.
