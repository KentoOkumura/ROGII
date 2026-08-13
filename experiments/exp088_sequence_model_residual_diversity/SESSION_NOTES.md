# exp088_sequence_model_residual_diversity セッションノート

## 目的

`backlog/KAGGLE_DIRECTION.md` の backlog `sequence_model_residual_diversity` を、古い exp063 前提ではなく
現行の ML route raw deterministic anchor `exp073` に対する軽量 sequence residual diversity 診断として実装する。

## 現在の状態

- status: `completed`
- route: `ml_model`
- parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- Kaggle train: v4 completed on Kaggle
- Kaggle inference: no-op
- submission: なし

## 参照メモ

- `docs/discussions/...699289...`: pure tabular では sequence / spatial context が不足し、PF / Beam / spatial imputation が重要。
- `docs/discussions/...699853...`: CNN/MTP や mixture trajectory は hard well の multi-mode 性に合うが、learned matcher は一般化が難しい。
- `docs/discussions/...707613...`: PF を NN 化するなら、まず truth trajectory 近傍を candidate set が含むかを測るべき。
- `docs/discussions/...703344...`: transformer bfloat16 で copy task failure。exp088 は PyTorch float32 固定、AMP 無効。
- `docs/notebooks/...public_notebook_catchup_inventory_2026-06-11.csv`: CNN/seq 系は低優先度、現行上位は PF/Beam/TabICL/physical stack 中心。

## 実装内容

- `docs/legacy/steering/20260620-exp088-sequence-model-residual-diversity/` を作成。
- `experiments/exp088_sequence_model_residual_diversity/` を exp086 から作成し、diagnostic input resolver を流用。
- `sequence_model_residual_diversity.py` を追加。
  - exp073 OOF prediction から selected mode/model (`gpu_repro_guard_dp_threads8` / `lgb_mean`) を読む。
  - exp072 full replay feature cache を `id` / `well` で join する。
  - stable well-hash fold で GRU / TCN residual correction を fold-out 学習する。
  - correction target は `target_tvt - exp073_pred_tvt`。
  - OOF prediction、overall metrics、distance bucket metrics、diversity metrics、train history、RMSE plot、summary JSON を保存する。
  - alpha blend と ridge blend を OOF diagnostic として出力する。
- train notebook を exp088 実行用に更新。
- inference notebook は no-op に更新。
- `config.yaml` を exp088 用に更新し、Kaggle source と runtime metadata を設定。

## 再現性メモ

- seed policy: `validation.seed`、variant 名、valid fold から `blake2b` stable seed を作る。
- stochastic components: PyTorch weight initialization、DataLoader shuffle、train window subsampling。
- PF/Beam: 新規生成なし。exp072 deterministic train feature cache を読むだけ。
- GPU: Kaggle GPU 有効だが、torch float32 固定、AMP/bfloat16 無効、CuDNN deterministic flag を設定。
- deterministic anchor ではない。sequence NN の OOF diagnostic として扱う。
- 入力 cache / prediction と出力 OOF prediction は decompressed content SHA を summary に記録する。

## 実装後の検証

```bash
uv run ruff check experiments/exp088_sequence_model_residual_diversity/sequence_model_residual_diversity.py experiments/exp088_sequence_model_residual_diversity/settings.py
uv run python -m py_compile experiments/exp088_sequence_model_residual_diversity/sequence_model_residual_diversity.py experiments/exp088_sequence_model_residual_diversity/settings.py
uv run python -m json.tool experiments/exp088_sequence_model_residual_diversity/exp088_sequence_model_residual_diversity_train.ipynb
uv run python -m json.tool experiments/exp088_sequence_model_residual_diversity/exp088_sequence_model_residual_diversity_inference.ipynb
uv run python scripts/validate_experiment.py --experiment exp088_sequence_model_residual_diversity
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp088_sequence_model_residual_diversity --notebook train --kernel-id kentookumura/exp088-sequence-model-residual-diversity-train --title "exp088 sequence model residual diversity train" --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp088_sequence_model_residual_diversity --notebook inference --kernel-id kentookumura/exp088-sequence-model-residual-diversity-infer --title "exp088 sequence model residual diversity infer" --run-on-push --strict
uv run python -m py_compile experiments/exp088_sequence_model_residual_diversity/kaggle/train/sequence_model_residual_diversity.py experiments/exp088_sequence_model_residual_diversity/kaggle/train/settings.py experiments/exp088_sequence_model_residual_diversity/kaggle/inference/settings.py
uv run python -m json.tool experiments/exp088_sequence_model_residual_diversity/kaggle/train/exp088_sequence_model_residual_diversity_train.ipynb
uv run python -m json.tool experiments/exp088_sequence_model_residual_diversity/kaggle/inference/exp088_sequence_model_residual_diversity_inference.ipynb
```

結果:

- `ruff check`: pass
- `py_compile`: pass
- train notebook JSON: pass
- inference notebook JSON: pass
- `validate_experiment`: pass
- `prepare_kaggle_notebooks` train: pass
- `prepare_kaggle_notebooks` inference: pass
- Kaggle train package `py_compile`: pass
- Kaggle train/inference package notebook JSON: pass
- train metadata: `enable_gpu=true`, `enable_internet=false`, kernel sources `exp073-full-replay-repro-guard-train` / `exp072-exp063-full-replay-feature-cache-train`
- inference metadata: no-op notebook。共通 runtime のため `enable_gpu=true` だが、実行・提出対象ではない。

ローカル smoke:

- ローカル環境に `torch` が無く、in-memory smoke は import 時点で `ModuleNotFoundError: No module named 'torch'`。
- Kaggle image での実行を前提にし、ローカルでは静的検証までに留めた。

## 次のアクション

Kaggle train v3 の GPU 種別と実行ログを確認する。P100 のままなら GPU batch slot 解放後に
`machine_shape=NvidiaTeslaT4` metadata と `kaggle kernels push --accelerator NvidiaTeslaT4` を併用して再 push する。

## Kaggle train v1 / v2

```bash
kaggle kernels push -p experiments/exp088_sequence_model_residual_diversity/kaggle/train
kaggle kernels pull kentookumura/exp088-sequence-model-residual-diversity-train -p /tmp/kaggle-pull/exp088-sequence-model-residual-diversity-train -m
kaggle kernels logs kentookumura/exp088-sequence-model-residual-diversity-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp088-sequence-model-residual-diversity-train
```

- v1 push: success
- kernel id: `kentookumura/exp088-sequence-model-residual-diversity-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp088-sequence-model-residual-diversity-train`
- existence check: PASS
- v1 result: failed at run cell.
- failure: exp073 OOF predictions had 3,783,989 rows but exp072 feature cache joined 3,759,413 rows; strict `id,well` join check raised `ValueError`.
- fix: allow feature join missing rows up to `audit.max_feature_join_missing_fraction=0.02`, drop missing rows for the diagnostic, and record `feature_join_missing_rows` in summary.

Validation after fix:

- `ruff check`: pass
- `py_compile`: pass
- `validate_experiment`: pass
- `prepare_kaggle_notebooks` train/inference: pass
- Kaggle train package `py_compile`: pass
- Kaggle train notebook JSON: pass

```bash
kaggle kernels push -p experiments/exp088_sequence_model_residual_diversity/kaggle/train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp088-sequence-model-residual-diversity-train
```

- v2 push: success
- monitoring: started, no log body in the initial polling window.
- monitoring stopped locally by user request. Kaggle v2 execution remains on Kaggle.

## Kaggle train v2 failure / v3 GPU metadata

- v2 result: failed at run cell.
- failure: Kaggle GUI assigned P100. Kaggle image torch `2.10.0+cu128` does not support P100 `sm_60`, causing CUDA `no kernel image is available for execution on the device`.
- correction after v3 observation: "P100 だと必ず即落ちる" ではなく、CUDA execution に入ると落ちる。v3 は P100 を検出して CPU fallback して走り続ける可能性があるため、T4 必須という運用方針とはずれる。
- mitigation update: `select_torch_device()` now raises when `enable_gpu=true` but CUDA is unavailable or capability is below `runtime.min_cuda_capability_major=7`, unless `runtime.allow_cpu_fallback=true` is explicitly set.
- exp088 config update: `runtime.allow_cpu_fallback: false`.
- Kaggle discussion check: `nvidia-nemotron-model-reasoning-challenge/discussion/682197` was retrieved via `kaggle competitions topics show nvidia-nemotron-model-reasoning-challenge/682197 --page-size 200`. The resolved answer points to Kaggle CLI accelerator support.
- Kaggle CLI docs check: `kaggle kernels push --help` supports `--accelerator ACC`; public docs list `NvidiaTeslaT4`.
- local Kaggle CLI source check: current installed CLI reads metadata key `machine_shape` rather than camelCase `machineShape`.
- generator fix: `scripts/prepare_kaggle_notebooks.py` now emits `machine_shape` and also accepts legacy config key `machineShape` as fallback.
- exp088 config fix: `runtime.kaggle.machine_shape: NvidiaTeslaT4`.
- generated train metadata: contains `"machine_shape": "NvidiaTeslaT4"`.
- v3 status: `KernelWorkerStatus.RUNNING`; CLI logs are still empty at the latest check.
- attempted re-push after metadata fix: blocked by Kaggle with `Maximum batch GPU session count of 2 reached`.

## Kaggle train v4

```bash
kaggle kernels status kentookumura/exp088-sequence-model-residual-diversity-train
kaggle kernels push -p experiments/exp088_sequence_model_residual_diversity/kaggle/train --accelerator NvidiaTeslaT4
kaggle kernels pull kentookumura/exp088-sequence-model-residual-diversity-train -p /tmp/kaggle-pull/exp088-sequence-model-residual-diversity-train-v4 -m
kaggle kernels logs kentookumura/exp088-sequence-model-residual-diversity-train
```

- pre-push status: `KernelWorkerStatus.CANCEL_ACKNOWLEDGED` for v3.
- v4 push: success.
- URL: `https://www.kaggle.com/code/kentookumura/exp088-sequence-model-residual-diversity-train`
- Kaggle-pulled metadata: `machine_shape=NvidiaTeslaT4`, `enable_gpu=true`, `enable_internet=false`.
- post-push status: `KernelWorkerStatus.RUNNING`.
- initial CLI logs: empty.

## Kaggle train v4 completed

```bash
kaggle kernels status kentookumura/exp088-sequence-model-residual-diversity-train
kaggle kernels logs kentookumura/exp088-sequence-model-residual-diversity-train
kaggle kernels output kentookumura/exp088-sequence-model-residual-diversity-train -p experiments/exp088_sequence_model_residual_diversity/kaggle/output/train
```

- final status: `KernelWorkerStatus.COMPLETE`.
- device: `cuda`
- torch: `2.10.0+cu128`
- elapsed: `943.469` seconds
- rows: `3,759,413`
- wells: `773`
- feature join missing rows: `24,576` (`0.00649`)
- missing configured features ignored: `GR`, `MD`, `X`, `Y`, `Z`, `md_from_ps`
- baseline RMSE: `9.524813476455735`
- GRU RMSE: `10.499027091390566`
- TCN RMSE: `10.377743828941945`
- best prediction: `ridge_blend_pred_tvt`
- best RMSE: `9.509990853500197`
- best delta vs baseline: `-0.014822622955538378`
- alpha blends selected alpha `0.0`; sequence NN predictions were not useful as direct residual corrections.
- ridge weights: baseline `1.1149482108889932`, GRU `-0.15024057993821774`, TCN `0.03529269456802314`
- distance bucket readout: ridge improves 2500+ bucket by `-0.0306751807460941`, is near-neutral or worse in shorter buckets.
- conclusion: diagnostic completed. Do not port sequence model to inference; prioritize confidence / sample-weight follow-up instead.
