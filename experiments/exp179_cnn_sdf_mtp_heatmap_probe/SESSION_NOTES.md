# exp179_cnn_sdf_mtp_heatmap_probe セッションノート

## 目的

`cnn_sdf_mtp_heatmap_probe` backlog を実装する。discussion 699853 の 5ch heatmap CNN/SDF/MTP を、まず Kaggle GPU 上の 1 fold / small wells / fixed target-free window probe として再現し、real GR が shuffled-GR / no-GR control より topK coverage を改善するか確認する。

## 現在の状態

- Route: ml_model
- 状態: implemented_pending_kaggle_train
- CV: まだなし
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 予定

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp179_cnn_sdf_mtp_heatmap_probe/exp179_cnn_sdf_mtp_heatmap_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp179_cnn_sdf_mtp_heatmap_probe/exp179_cnn_sdf_mtp_heatmap_probe_train.py
.venv/bin/python -m py_compile experiments/exp179_cnn_sdf_mtp_heatmap_probe/exp179_cnn_sdf_mtp_heatmap_probe_train.py
.venv/bin/ruff check experiments/exp179_cnn_sdf_mtp_heatmap_probe/exp179_cnn_sdf_mtp_heatmap_probe_train.py --select F821
make validate-exp EXP=exp179_cnn_sdf_mtp_heatmap_probe
task prepare-kaggle-notebooks EXP=exp179_cnn_sdf_mtp_heatmap_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp179-cnn-sdf-mtp-heatmap-probe-train --title 'exp179 cnn sdf mtp heatmap probe train' --run-on-push --strict"
task push-kaggle-train EXP=exp179_cnn_sdf_mtp_heatmap_probe
kaggle kernels logs kentookumura/exp179-cnn-sdf-mtp-heatmap-probe-train
```

### 2026-07-03 JST 実装

```bash
make new-steering EXP=exp179_cnn_sdf_mtp_heatmap_probe
make new-exp EXP=exp179_cnn_sdf_mtp_heatmap_probe
```

実装方針:

- `mtp_heatmap_sdf_mdn_probe` umbrella は使わず、具体 backlog `cnn_sdf_mtp_heatmap_probe` として `exp179` を作成。
- 5ch heatmap は discussion 699853 の `t_gr`、`h_gr`、`t_gr-h_gr`、`history`、`mask` に合わせる。
- typewell window center は valid true TVT ではなく `last_known_tvt - (Z - last_known_z)` の target-free flat prior。
- `real_gr`、`shuffled_gr`、`no_gr` の 3 variants を同一 fold / sample schedule / epochs で比較する。
- Kaggle GPU 必須。CUDA がなければ RuntimeError にする。

## 変更点

- `config.yaml` を GPU train-side diagnostic 用に更新。
- `.steering/20260703-exp179-cnn-sdf-mtp-heatmap-probe/` に要件、設計、tasklist を記入。
- train notebook source は Jupytext percent 形式で実装する。

### 2026-07-03 JST validation / package

```bash
.venv/bin/python -m py_compile experiments/exp179_cnn_sdf_mtp_heatmap_probe/exp179_cnn_sdf_mtp_heatmap_probe_train.py experiments/exp179_cnn_sdf_mtp_heatmap_probe/exp179_cnn_sdf_mtp_heatmap_probe_inference.py
.venv/bin/ruff check experiments/exp179_cnn_sdf_mtp_heatmap_probe/exp179_cnn_sdf_mtp_heatmap_probe_train.py experiments/exp179_cnn_sdf_mtp_heatmap_probe/exp179_cnn_sdf_mtp_heatmap_probe_inference.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp179_cnn_sdf_mtp_heatmap_probe/exp179_cnn_sdf_mtp_heatmap_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp179_cnn_sdf_mtp_heatmap_probe/exp179_cnn_sdf_mtp_heatmap_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp179_cnn_sdf_mtp_heatmap_probe/exp179_cnn_sdf_mtp_heatmap_probe_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp179_cnn_sdf_mtp_heatmap_probe/exp179_cnn_sdf_mtp_heatmap_probe_inference.py
make validate-exp EXP=exp179_cnn_sdf_mtp_heatmap_probe
make prepare-kaggle-notebooks EXP=exp179_cnn_sdf_mtp_heatmap_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp179-cnn-sdf-mtp-heatmap-probe-train --title 'exp179 cnn sdf mtp heatmap probe train' --run-on-push --strict"
```

結果:

- `py_compile`: PASS
- `ruff --select F821`: PASS
- `jupytext --to ipynb --test`: train / inference とも PASS
- `validate-exp`: PASS
- Kaggle package: `experiments/exp179_cnn_sdf_mtp_heatmap_probe/kaggle/train`
- metadata: `enable_gpu=true`、`enable_internet=false`、`run_on_push=true`
- bootstrap config: `runtime.kaggle.enable_gpu=true`、`model.training.require_cuda=true`

### 2026-07-03 JST Kaggle train v1 push

```bash
kaggle kernels push -p experiments/exp179_cnn_sdf_mtp_heatmap_probe/kaggle/train
kaggle kernels pull kentookumura/exp179-cnn-sdf-mtp-heatmap-probe-train -p /tmp/kaggle-pull/exp179-cnn-sdf-mtp-heatmap-probe-train -m
kaggle kernels logs kentookumura/exp179-cnn-sdf-mtp-heatmap-probe-train
```

結果:

- Kernel version 1 push 成功。
- URL: https://www.kaggle.com/code/kentookumura/exp179-cnn-sdf-mtp-heatmap-probe-train
- Kaggle pull metadata: `id_no=125808745`、`enable_gpu=true`、`machine_shape=Gpu`。
- 初回 logs は warning 以外空。実行中 logs が空の既知挙動として扱い、同じ kernel id のまま監視継続。

v1 監視結果:

- Runtime は `Device: cuda`、`CUDA device: Tesla P100-PCIE-16GB`。
- Kaggle の PyTorch `2.10.0+cu128` は P100 の `sm_60` をサポートしておらず、`CUDA error: no kernel image is available for execution on the device` で失敗。
- 対応: `runtime.kaggle.machine_shape=NvidiaTeslaT4` と `model.training.min_cuda_capability_major=7` を追加し、T4 accelerator を明示して同じ kernel id へ v2 を再 push する。

### 2026-07-03 JST Kaggle train v2 T4 retry

```bash
make prepare-kaggle-notebooks EXP=exp179_cnn_sdf_mtp_heatmap_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp179-cnn-sdf-mtp-heatmap-probe-train --title 'exp179 cnn sdf mtp heatmap probe train' --run-on-push --strict"
kaggle kernels push -p experiments/exp179_cnn_sdf_mtp_heatmap_probe/kaggle/train --accelerator NvidiaTeslaT4
kaggle kernels pull kentookumura/exp179-cnn-sdf-mtp-heatmap-probe-train -p /tmp/kaggle-pull/exp179-cnn-sdf-mtp-heatmap-probe-train-v2 -m
kaggle kernels logs kentookumura/exp179-cnn-sdf-mtp-heatmap-probe-train
```

結果:

- Kernel version 2 push 成功。
- Kaggle pull metadata: `id_no=125808745`、`enable_gpu=true`、`machine_shape=NvidiaTeslaT4`。
- 初回 logs は warning 以外空。v2 は同じ kernel id のまま監視継続。

v2 監視結果:

- Runtime: `Device: cuda`、`CUDA device: Tesla T4`、Torch `2.10.0+cu128`。
- Input: train dir `/kaggle/input/competitions/rogii-wellbore-geology-prediction/train`。
- Active variants: `real_gr`, `shuffled_gr`, `no_gr`。
- Sample: train 2,304 / valid 512、valid wells 32、target-in-grid rate 1.0。
- v2 は完了し、summary を `/kaggle/working/artifacts/exp179_cnn_sdf_mtp_heatmap_probe_summary.json` に保存。

Metrics:

| variant | top1 within10 | top3 within10 | top5 within10 | top10 within10 | top10 oracle RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `real_gr` | 0.2109375 | 0.44921875 | 0.63671875 | 0.794921875 | 14.071005802624697 |
| `shuffled_gr` | 0.1015625 | 0.232421875 | 0.328125 | 0.541015625 | 21.39547633299744 |
| `no_gr` | 0.0625 | 0.0625 | 0.0625 | 0.0625 | 136.02539145612556 |

Interpretation:

- `real_gr` は top3 within10 で shuffled-GR を +0.216796875、no-GR を +0.38671875 上回った。
- `real_gr` は top10 coverage でも 0.794921875 まで上がり、GR を使えている smoke と判断する。
- ただし 1 fold / selected 160 wells / 128x64 fixed window なので、direct TVT replacement、inference port、submit には進めない。

### 2026-07-03 JST Kaggle train v2 output 取得

```bash
kaggle kernels output kentookumura/exp179-cnn-sdf-mtp-heatmap-probe-train -p experiments/exp179_cnn_sdf_mtp_heatmap_probe/kaggle/output/train_v2
```

結果:

- output: `experiments/exp179_cnn_sdf_mtp_heatmap_probe/kaggle/output/train_v2`
- log: `experiments/exp179_cnn_sdf_mtp_heatmap_probe/kaggle/output/train_v2/exp179-cnn-sdf-mtp-heatmap-probe-train.log`
- metrics: `artifacts/exp179_cnn_sdf_mtp_heatmap_probe_metrics.csv`
- summary: `artifacts/exp179_cnn_sdf_mtp_heatmap_probe_summary.json`
- model manifest: `artifacts/exp179_cnn_sdf_mtp_heatmap_probe_model_manifest.json`

SHA:

- sample index decompressed SHA: `4c23b5ca13a872cf6fd085f3d2c225357e2a73ecb5b47cf6ce423211127bfb92`
- validation prediction decompressed SHA: `2befa525c5922d3ac1cda7a38fd23e134e48b2618176e98b0c7bf4343a08d7ca`
- metrics CSV SHA: `900979d58ed4478985afcea48ca709932ab2d1950b78fc5adc13dbef6d6c5c64`
- model manifest SHA: `9870e05c6de6b244bdf75f66c01b3b405ea0a53ed08f38e7b1fa067feeebdef6`
- model SHA: `real_gr=c65ac7f5ac9f04d8b41477e0cff0fc79ca9b31f1fdcc35bde4a7b367eb9a9fb5`, `shuffled_gr=3729ef2325795af6a1f7b1362bf6341b575ad46232948f6c4b25f282f954fbc9`, `no_gr=1d322e3e283194fe11951c3521bed1c22f52c95e94c29234a41c054a37bb2e7d`

## GPU / 学習コストメモ

- active variant 数: 3 (`real_gr`, `shuffled_gr`, `no_gr`)
- fold 数: 1 selected fold (`fold_index=0` of 5 GroupKFold folds)
- PyTorch CNN model 数: 3
- LightGBM config 数: 0
- 合計 booster 数: 0
- 既存 baseline/control 再学習: なし
- Kaggle GPU: 使用。`runtime.kaggle.enable_gpu=true`。CPU fallback なし。

## 再現性メモ

- seed policy: global seed 42 + well keyed SHA256 for sample order and shuffled-GR roll。
- stochastic components: PyTorch CUDA conv training、AdamW、DataLoader shuffle。
- parallel RNG: DataLoader `num_workers=0`、global RNG を並列 worker から消費しない。
- CPU/GPU runtime: Kaggle GPU 必須。`torch.use_deterministic_algorithms(True, warn_only=True)`、CuDNN deterministic true、benchmark false。
- Kaggle kernel id / version: `kentookumura/exp179-cnn-sdf-mtp-heatmap-probe-train` v2。v1 は P100 非対応で失敗。
- input / feature schema SHA: feature schema SHA `a2c4fda671361eab9e876ff84e3c4600ac4d6cec1727aad1035dfd623b62e367`。
- feature content SHA: sample index decompressed SHA `4c23b5ca13a872cf6fd085f3d2c225357e2a73ecb5b47cf6ce423211127bfb92`。
- model manifest / model SHA: manifest SHA `9870e05c6de6b244bdf75f66c01b3b405ea0a53ed08f38e7b1fa067feeebdef6`。model SHA は v2 output section を参照。
- prediction SHA: validation prediction decompressed SHA `2befa525c5922d3ac1cda7a38fd23e134e48b2618176e98b0c7bf4343a08d7ca`。
- submission SHA: なし。submission は作らない。
- rerun check: TODO

## 次のアクション

1. `experiment_summary.md` と `KAGGLE_DIRECTION.md` を更新する。
2. 次候補は full-fold / larger-window / geometry-channel ablation。direct TVT replacement、inference port、submit はしない。
