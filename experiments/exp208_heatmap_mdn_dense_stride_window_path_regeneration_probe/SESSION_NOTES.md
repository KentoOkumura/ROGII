# exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe セッションノート

## 2026-07-07 実装

目的: `heatmap_mdn_dense_stride_window_path_regeneration_probe` backlog を実験化する。exp207 は exp202 v2 sparse local path artifact の stitch diagnostic として完了したが、source overlap が 773 wells 中 3 wells / 39 pairs しかなかった。exp208 は exp202 saved fold model から dense stride local path を再生成し、overlap 付き stitch readout を再評価する。

実装方針:

- route: `pf_beam`
- 親: `exp202_heatmap_mdn_candidate_generator_probe`
- stitch 比較元: `exp207_heatmap_mdn_overlapping_window_path_stitch_probe`
- 比較 cache: `exp099_pf_multi_observation_likelihood_probe`
- dense generation: validation wells、stride 64、tail stop 追加、topK 10
- stitch: local topK 5 / 10、beam width 6、output top3
- score: center score、rank penalty、path smoothness、adjacent overlap disagreement、gap boundary continuity
- GPU: disabled
- LightGBM: 0 configs / 0 boosters
- CNN training: なし。exp202 saved model artifact を読むだけ
- parent/control retraining: なし
- inference / submit: なし

再現性:

- exp208 内では stochastic sampling / training は使わない。
- GroupKFold は sorted well id と seed 42 の設定を記録する。
- DataLoader は `num_workers=0`、shuffle なし。
- upstream exp202 は GPU-trained weights なので deterministic submission anchor ではない。
- exp202 model manifest / model file、exp099 candidate cache、dense path artifact、stitch outputs の SHA を記録する。

主な変更:

- `.steering/20260707-exp208-heatmap-mdn-dense-stride-window-path-regeneration-probe/` を作成した。
- `experiments/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe/` を exp207 から作成した。
- `config.yaml` を exp202 saved model dense stride regeneration 用に更新した。
- `heatmap_mdn_dense_stride_window_path_regeneration_probe.py` を実装した。
  - exp202 の fold split、5ch heatmap input builder、model architecture を必要部分だけ移植した。
  - exp202 saved fold model を読み、validation wells の dense row-center windows を CPU 推論する。
  - dense path samples / predictions / path npz / rank index を保存する。
  - exp207 の target-free beam stitch と candidate-union readout を local topK 5 / 10 で実行する。
  - `true_tvt_path` / `true_center_tvt` / `center_abs_error` は stitch score 入力に入れない。
- train notebook source を Jupytext percent 形式で追加した。
- inference notebook source を diagnostic-only no-submit guard にした。

実行前ガード:

- active dense spec: stride 64、topK 10、fold 0-4 validation wells
- active stitch specs: local topK 5 / 10、beam width 6、output top3
- GPU: 0
- LightGBM configs / boosters: 0 / 0
- parent/control retraining: なし
- inference / submit: なし

検証:

```bash
.venv/bin/python -m py_compile experiments/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe/heatmap_mdn_dense_stride_window_path_regeneration_probe.py experiments/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe_train.py experiments/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe_inference.py experiments/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe/settings.py
.venv/bin/ruff check experiments/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe/heatmap_mdn_dense_stride_window_path_regeneration_probe.py experiments/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe_train.py experiments/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe_inference.py experiments/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe/settings.py --select F821,E501
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe_inference.py
```

結果:

- `py_compile`: pass
- `ruff --select F821,E501`: pass
- Jupytext train / inference conversion and `--test`: pass
- train / inference source に `__file__` がないことを確認
- `make validate-exp`: pass
- local import smoke: `.venv` に `torch` がないため未実施。Kaggle runtime での torch 前提実行とする。

未実行:

- Kaggle prepare / push
- Kaggle train

## 2026-07-07 Kaggle train 実行計画

ユーザー依頼により Kaggle Notebook train を実行する。推論 notebook、提出は対象外。

実行前ガード:

- route: `pf_beam`
- active dense spec: stride 64、topK 10、fold 0-4 validation wells
- active stitch specs: local topK 5 / 10、beam width 6、output top3
- GPU: disabled
- CNN training models: 0。exp202 saved fold model artifact を読むだけ
- LightGBM configs / boosters: 0 / 0
- parent/control retraining: なし
- inference / submit: なし

予定 kernel:

- id: `kentookumura/exp208-hmdn-dense-stride-train`
- title: `exp208 hmdn dense stride train`

予定コマンド:

```bash
make validate-exp EXP=exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe
make prepare-kaggle-notebooks EXP=exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp208-hmdn-dense-stride-train --title 'exp208 hmdn dense stride train' --run-on-push --strict"
make push-kaggle-train EXP=exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe
kaggle kernels pull kentookumura/exp208-hmdn-dense-stride-train -p /tmp/kaggle-pull/exp208-hmdn-dense-stride-train -m
kaggle kernels logs kentookumura/exp208-hmdn-dense-stride-train
```

実行結果:

- `make validate-exp`: pass
- `make prepare-kaggle-notebooks`: pass
- generated metadata: `enable_gpu=false`、`enable_internet=false`、`run_on_push=true`
- kernel sources: `kentookumura/exp202-heatmap-mdn-candgen-train` / `kentookumura/exp099-pf-multiobs-likelihood-train`
- `make push-kaggle-train`: success、Kernel version 1
- URL: <https://www.kaggle.com/code/kentookumura/exp208-hmdn-dense-stride-train>
- `kaggle kernels pull`: success、pulled `id_no=126191369`
- pulled metadata: `enable_gpu=false`、`machine_shape=None`、`enable_internet=false`、competition source `rogii-wellbore-geology-prediction`
- initial status: `KernelWorkerStatus.RUNNING`
- initial / follow-up `kaggle kernels logs`: Kaggle CLI version warning のみで本文空。実行中 logs 空は既知挙動なので失敗扱いしない。
- 5分 logs follow は出力なしで timeout。追加 status も `KernelWorkerStatus.RUNNING`。
- ユーザー指示により監視を停止。完了後にユーザーから連絡を受けて logs / output / 結果記録を続ける。

## 2026-07-07 Kaggle train 完了確認

ユーザーから完了連絡を受けて status / logs / output を確認した。

確認コマンド:

```bash
kaggle kernels status kentookumura/exp208-hmdn-dense-stride-train
kaggle kernels logs kentookumura/exp208-hmdn-dense-stride-train
kaggle kernels output kentookumura/exp208-hmdn-dense-stride-train -p experiments/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe/kaggle/output/train_v1
```

実行結果:

- status: `KernelWorkerStatus.COMPLETE`
- output: `experiments/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe/kaggle/output/train_v1`
- runtime: 約 1,290 sec
- torch: `2.10.0+cpu`
- CUDA: false
- GPU: disabled
- LightGBM configs / boosters: 0 / 0
- parent/control retraining: なし
- inference / submit: なし

入力 / 生成物:

- exp099 candidate cache: `3,783,989` rows / `773` wells
- exp099 cache decompressed SHA: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- exp202 model manifest SHA: `480e341802dfe538243f8659cb25b11d349827450c7319b4a7b62a2c6340bc08`
- dense samples: `25,452`
- dense wells: `773`
- stride / horizon / topK: `64` / `128` / `10`
- dense path npz SHA: `0ad4b4046e8c5b1865684bbab131e5f91ff07c19d225733811524d3037d3f1c2`
- dense path samples decompressed SHA: `52812be9db9eaee830b667bd303f3dfe3c492f895821b4e038b89bfc71ed9238`
- dense predictions decompressed SHA: `ee85a23310e81eb03d6051cc7d0dc31c148e02efc322b9b7662b80324d85a2bc`

coverage / physicality:

- source windows per well mean: `32.926261320`
- source overlap wells / pairs: `773` / `24,679`
- source gap pair count: `0`
- merged rows: `1,627,462`
- row coverage vs exp099 cache: `0.430091631`
- stitched rows / wide rows: `5,030,802` / `1,676,934`
- assignment overlap abs mean: `7.196697638 ft`
- stitched path step abs mean: `0.247286874 ft`
- stitched curvature abs mean: `0.485335775 ft`
- stitched row gap count total: `0`

local topK10 metrics:

- existing union on stitched rows RMSE / within10: `5.139413349` / `0.947522584`
- stitched only top3 RMSE / within10: `47.188322489` / `0.285964895`
- existing + stitched top3 RMSE / delta / within10 / new-best rate: `4.420752853` / `-0.718660496` / `0.960702615` / `0.071626864`
- `1000_plus` existing / stitched / union RMSE: `6.352450934` / `47.975387107` / `5.422416550`
- by-well: `509 improved / 264 same / 0 worse`
- best well: `1b1eba53`、`37.534730853 -> 21.331056019`

local topK5 metrics:

- stitched only top3 RMSE: `49.388843882`
- existing + stitched top3 RMSE / delta / within10 / new-best rate: `4.426013080` / `-0.713400269` / `0.960899241` / `0.070470463`
- by-well: `513 improved / 260 same / 0 worse`

exp207 比較:

- source overlap wells / pairs: exp207 `3 / 39` -> exp208 `773 / 24,679`
- row coverage: exp207 `0.352337441` -> exp208 `0.430091631`
- stitched only top3 RMSE: exp207 `50.798377042` -> exp208 `47.188322489`
- existing + stitched top3 RMSE: exp207 `4.418699605` -> exp208 `4.420752853`

判断:

- dense stride regeneration は意図どおり各 well の overlapping local path を作れた。
- ただし oracle top3 は exp207 を更新せず、dense 化の追加価値は確認できなかった。
- stitched only は改善したが RMSE `47.188` で単体候補や直接置換には弱い。
- direct replacement、softmax average、PF weight replacement、inference、submit は行わない。
- 後続は dense stitch の stride 32 再実行より、exp204 selector candidate route で heatmap MDN 候補を target-free selection に渡す方向を優先する。
