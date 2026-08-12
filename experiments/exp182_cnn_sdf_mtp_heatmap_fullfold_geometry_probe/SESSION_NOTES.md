# exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe セッションノート

## 目的

`cnn_sdf_mtp_heatmap_fullfold_geometry_probe` backlog を実装する。exp179 の 5ch heatmap CNN/SDF/MTP smoke が positive だったため、full-fold control、geometry channel ablation、larger-window fold subset、worst-well / distance bucket readout を train-side GPU 診断として確認できる状態にする。

## 現在の状態

- Route: ml_model
- 状態: completed_train_side_gpu_probe
- CV: top3 within10 0.500000 (`base_real_w128_b64_fullfold`)
- LB: なし

## 実装方針

- 親: `exp179_cnn_sdf_mtp_heatmap_probe`
- target-free flat-prior window center、5ch base heatmap、closest-mode path loss、K=10 path head は維持する。
- `active_run_specs` で variant / channel set / fold / window / bins / history scale を制御する。
- geometry channel は `MD,X,Y,Z` と observed prefix だけから作る。
- true TVT は label / metric のみに使う。
- `submission.csv`、hidden-test inference、direct replacement、PF weight replacement、softmax weighted TVT は作らない。

## GPU / 学習コストメモ

- active run spec 数: 6
- fold 展開後の PyTorch CNN model 数: 24
- LightGBM config 数: 0
- 合計 booster 数: 0
- 既存 baseline/control 再学習: なし
- Kaggle GPU: 必須。`runtime.kaggle.enable_gpu=true`、`machine_shape=NvidiaTeslaT4`。
- Kaggle train push: 完了。T4 GPU v1 で 24 models を学習した。

## コマンドログ

### 2026-07-03 JST 実装

```bash
make new-steering EXP=exp180_cnn_sdf_mtp_heatmap_fullfold_geometry_probe
make new-exp EXP=exp180_cnn_sdf_mtp_heatmap_fullfold_geometry_probe SOURCE=experiments/exp179_cnn_sdf_mtp_heatmap_probe
mv experiments/exp180_cnn_sdf_mtp_heatmap_fullfold_geometry_probe experiments/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe
mv docs/legacy/steering/20260703-exp180-cnn-sdf-mtp-heatmap-fullfold-geometry-probe docs/legacy/steering/20260703-exp182-cnn-sdf-mtp-heatmap-fullfold-geometry-probe
```

変更点:

- `docs/legacy/steering/20260703-exp182-cnn-sdf-mtp-heatmap-fullfold-geometry-probe/` に要件、設計、tasklist を記入。
- `config.yaml` を exp180 の full-fold geometry diagnostic 用に更新。
- train notebook source を run spec 型に拡張。
- inference notebook source は diagnostic-only guard に更新。
- README / result / metrics を未実行状態へ初期化。
- 既存に `exp180_learned_gr_window_matcher_features_on_exp148` と `exp181_cluster_outlier_pfbeam_prior_gate` があったため、番号衝突を避けて `exp182` に改番。

### 2026-07-03 JST validation / package

```bash
.venv/bin/python -m py_compile experiments/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe_train.py experiments/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe_inference.py
.venv/bin/ruff check experiments/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe_train.py experiments/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe_inference.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe_inference.py
make validate-exp EXP=exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe
make prepare-kaggle-notebooks EXP=exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp182-cnn-sdf-mtp-heatmap-fullfold-geometry-probe-train --title 'exp182 cnn sdf mtp heatmap fullfold geometry probe train' --run-on-push --strict"
```

結果:

- `py_compile`: PASS
- `ruff --select F821`: PASS
- `jupytext --to ipynb --test`: train / inference とも PASS
- `validate-exp`: PASS
- Kaggle package: `experiments/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe/kaggle/train`
- metadata: `id=kentookumura/exp182-cnn-sdf-mtp-heatmap-fullfold-geometry-probe-train`、`enable_gpu=true`、`machine_shape=NvidiaTeslaT4`、`enable_internet=false`、`run_on_push=true`
- packaged config: `runtime.kaggle.enable_gpu=true`、`runtime.kaggle.machine_shape=NvidiaTeslaT4`、`model.active_run_specs` あり

### 2026-07-03 JST Kaggle train v1 push

```bash
kaggle kernels push -p experiments/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe/kaggle/train --accelerator NvidiaTeslaT4
kaggle kernels pull kentookumura/exp182-cnn-sdf-mtp-heatmap-fullfold-geometry-probe-train -p /tmp/kaggle-pull/exp182-cnn-sdf-mtp-heatmap-fullfold-geometry-probe-train -m
kaggle kernels list --mine --search exp182
make prepare-kaggle-notebooks EXP=exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp182-cnn-sdf-mtp-geometry-probe-train --title 'exp182 cnn sdf mtp geometry probe train' --run-on-push --strict"
kaggle kernels push -p experiments/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe/kaggle/train --accelerator NvidiaTeslaT4
kaggle kernels pull kentookumura/exp182-cnn-sdf-mtp-geometry-probe-train -p /tmp/kaggle-pull/exp182-cnn-sdf-mtp-geometry-probe-train-v1 -m
kaggle kernels logs kentookumura/exp182-cnn-sdf-mtp-geometry-probe-train
kaggle kernels status kentookumura/exp182-cnn-sdf-mtp-geometry-probe-train
```

結果:

- long slug `kentookumura/exp182-cnn-sdf-mtp-heatmap-fullfold-geometry-probe-train` は Kaggle `SaveKernel` 400 で失敗。`pull` は 403、`kernels list --mine --search exp182` は Not found だったため kernel は作成されていないと判断。
- Kaggle slug 制約の可能性が高いため、同じ exp のまま短縮 canonical `kentookumura/exp182-cnn-sdf-mtp-geometry-probe-train` / title `exp182 cnn sdf mtp geometry probe train` に再 prepare。
- v1 push 成功。
- URL: https://www.kaggle.com/code/kentookumura/exp182-cnn-sdf-mtp-geometry-probe-train
- Kaggle pull metadata: `id_no=125812716`、`enable_gpu=true`、`machine_shape=NvidiaTeslaT4`。
- 初回 logs は warning 以外空。約7分経過時点の status は `KernelWorkerStatus.RUNNING`。同じ kernel id のまま監視継続。

### 2026-07-03 JST Kaggle train v1 完了

```bash
kaggle kernels status kentookumura/exp182-cnn-sdf-mtp-geometry-probe-train
kaggle kernels logs kentookumura/exp182-cnn-sdf-mtp-geometry-probe-train
kaggle kernels output kentookumura/exp182-cnn-sdf-mtp-geometry-probe-train -p experiments/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe/kaggle/output/train_v1
```

結果:

- status: `KernelWorkerStatus.COMPLETE`
- runtime: Tesla T4 / PyTorch 2.10.0+cu128 / internet disabled
- output: `experiments/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe/kaggle/output/train_v1`
- artifacts: metrics、fold metrics、distance bucket metrics、well metrics、feature schema、run spec manifest、model manifest、24 model weights、sample index、validation predictions、summary を取得済み。

主要メトリック:

| run spec | folds | top3 within10 | top10 within10 | top10 oracle RMSE | worst-well top3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base_real_w128_b64_fullfold` | 5 | 0.500000 | 0.808908 | 13.296284 | 0.0 |
| `base_shuffled_w128_b64_fullfold` | 5 | 0.218536 | 0.545001 | 17.637821 | 0.0 |
| `base_no_gr_w128_b64_fullfold` | 5 | 0.071429 | 0.071429 | 134.767278 | 0.071429 |
| `geometry_real_w128_b64_fullfold` | 5 | 0.487710 | 0.809647 | 11.995428 | 0.0 |
| `geometry_shuffled_w128_b64_fold01` | 2 | 0.206682 | 0.549539 | 17.013071 | 0.0 |
| `geometry_real_w256_b96_fold01` | 2 | 0.417512 | 0.716129 | 18.643568 | 0.0 |

Margins:

- `base_fullfold_real_minus_shuffled_top3`: +0.281463685086
- `base_fullfold_real_minus_no_gr_top3`: +0.428571428571
- `geometry_fold01_real_minus_shuffled_top3`: +0.280875576037
- `geometry_real_minus_base_real_top3`: -0.012289780078

判断:

- full-fold でも real GR signal は支持される。
- geometry channel は top10 oracle RMSE を少し改善するが、primary top3 coverage を落とすため現設定では採用しない。
- 256x96 larger window fold01 は top3 0.417512 で弱く、拡大しない。
- worst-well top3 0.0 が残るため、full-length inference、direct TVT replacement、softmax weighted average、PF weight replacement、submission には進めない。
- 後続は `base_real_w128_b64_fullfold` の topK path / logit margin / entropy / path spread / shuffled gap を selector / confidence feature に変換する方向に限定する。

Artifact SHA:

- sample index decompressed: `f6ec40a1b89e70224127c355a6be31a56e5f38d02f0019505f9a9d45ba0b7d09`
- validation predictions decompressed: `d50f1ee515da7a68f142cae3d918902e111f1384d57dfa2d882093ad560945fa`
- metrics csv: `6224ff55edcfb4134eae3a706bd689c73aae5a35a90b27b26d704bd780311335`
- fold metrics csv: `3d8ada4c9f84157112dceca659a4f29e27c3c5a4493b97b7e635c5919cb036bd`
- distance bucket metrics csv: `d85f5447beab9c7a6dce14533eb110bf3cd22ace5ce0d1ca72519fe863b862cd`
- well metrics csv: `55b03944fee541a8378a1c388aa074f39cea276273d3ee02daa32b08537d102d`
- feature schema csv: `0216158ba90193566e2a148ab495bb0d8c5bd4b16e1425efcfc8bde8d2013b2d`
- run spec manifest json: `4e2a207174f6828c110efa8b503611a3478165be31d5fa2653d7b56186a76686`
- model manifest json: `7950bf80c4618198277a973ea82ea21c1c56a4cfc3a846949cc9f9af3eb404c3`
- training history csv: `f4e09e5971acdcf77c715e40a08593e48ea32d02a1e7b28edb6bb48537cc2c13`
- summary json: `4efeb705601bfd5cf6102e5e63a414c0f1f4402344c541350b6ab95f0f5e6c13`

## 再現性メモ

- seed policy: global seed 42 + run spec / fold / well keyed SHA256。
- stochastic components: PyTorch CUDA conv training、AdamW、DataLoader shuffle。
- parallel RNG: DataLoader `num_workers=0`。
- CPU/GPU runtime: Kaggle T4 GPU 必須。`torch.use_deterministic_algorithms(True, warn_only=True)`、CuDNN deterministic true、benchmark false。
- deterministic anchor: false。GPU train-side diagnostic として扱う。
- SHA 記録: sample index、validation predictions、metrics、fold metrics、well metrics、distance bucket metrics、feature schema、run spec manifest、model manifest、model weights。

## 次のアクション

1. `result.md`、`metrics.json`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` を完了結果で更新する。
2. 後続候補は selector / confidence feature 化に限定し、direct inference / submit には進めない。
