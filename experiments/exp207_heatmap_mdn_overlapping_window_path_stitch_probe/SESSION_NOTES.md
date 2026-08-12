# exp207_heatmap_mdn_overlapping_window_path_stitch_probe セッションノート

## 2026-07-06 実装

目的: `heatmap_mdn_overlapping_window_path_stitch_probe` backlog を実験化する。exp202 の heatmap MDN local top10 path artifact を well 内で stitch し、full-well candidate としての coverage、overlap consistency、既存 PF/Beam candidate union への oracle headroom を train-side で診断する。

実装方針:

- route: `pf_beam`
- 親: `exp202_heatmap_mdn_candidate_generator_probe`
- 比較対象: `exp099_pf_multi_observation_likelihood_probe`
- 入力:
  - `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_paths_top10.npz`
  - `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_path_samples.csv.gz`
  - `exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz`
- stitch: local top10、beam width 6、output top3。
- score: center score、rank penalty、path smoothness、adjacent overlap disagreement、gap boundary continuity。
- LightGBM: 0 configs / 0 boosters。
- GPU: 0。
- parent/control retraining: なし。
- inference / submit: なし。

主な変更:

- `docs/legacy/steering/20260706-exp207-heatmap-mdn-overlapping-window-path-stitch-probe/` を作成した。
- `experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/` を作成した。
- `config.yaml` を exp202/exp099 cached artifact diagnostic 用に更新した。
- `heatmap_mdn_overlapping_window_path_stitch_probe.py` を追加した。
  - exp202 path npz と sample CSV を読み込む。
  - `true_tvt_path` / `true_center_tvt` / `center_abs_error` を stitch score 入力から除外する。
  - well 内 local path rank choice を beam stitch する。
  - row-level stitched path、window assignment、source coverage、candidate union metrics、distance bucket、by-well readout、summary JSON を保存する。
  - gzip output は raw SHA と decompressed SHA を記録する。
- train notebook source を Jupytext percent 形式で追加した。
- inference notebook source を diagnostic-only no-submit guard にした。

注意:

- 現行 exp202 v2 path artifact は 14 validation samples / well の sparse local window output で、dense overlapping full-well trajectory ではない。
- 事前確認では、window center gap は大半が 157/158 rows で、128-row window の overlap は 773 wells 中 3 wells のみ。exp207 はこの coverage / overlap 不足を source readout として記録する。
- positive でも direct TVT replacement、softmax weighted average、PF weight replacement、postprocess blend、submit はしない。
- Kaggle train push 前に、GPU 0、LightGBM 0 configs / 0 boosters、parent/control retraining なしを再確認する。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/heatmap_mdn_overlapping_window_path_stitch_probe.py experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/exp207_heatmap_mdn_overlapping_window_path_stitch_probe_train.py experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/exp207_heatmap_mdn_overlapping_window_path_stitch_probe_inference.py experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/settings.py
.venv/bin/ruff check experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/heatmap_mdn_overlapping_window_path_stitch_probe.py experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/exp207_heatmap_mdn_overlapping_window_path_stitch_probe_train.py experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/exp207_heatmap_mdn_overlapping_window_path_stitch_probe_inference.py experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/settings.py --select F821,E501
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/exp207_heatmap_mdn_overlapping_window_path_stitch_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/exp207_heatmap_mdn_overlapping_window_path_stitch_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/exp207_heatmap_mdn_overlapping_window_path_stitch_probe_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/exp207_heatmap_mdn_overlapping_window_path_stitch_probe_inference.py
make validate-exp EXP=exp207_heatmap_mdn_overlapping_window_path_stitch_probe
```

結果:

- `py_compile`: pass
- `ruff --select F821,E501`: pass
- Jupytext train / inference convert and `--test`: pass
- `make validate-exp`: pass
- local helper smoke: 書き出しなし、2 wells で `path_rows=10752`、`assignments=84`、`metrics_rows=5`、`merged_rows=3456` を確認した。

未実行:

- Kaggle prepare / push
- Kaggle output 取得

## 2026-07-06 Kaggle train 実行計画

ユーザー依頼により Kaggle Notebook train を実行する。推論 notebook、提出は対象外。

実行前ガード:

- route: `pf_beam`
- active stitch spec: local top10、beam width 6、output top3
- GPU: disabled
- LightGBM configs / boosters: 0 / 0
- CNN training: なし。exp202 v2 path artifact を読むだけ
- parent/control retraining: なし
- inference / submit: なし

予定 kernel:

- canonical long slug は `exp207-heatmap-mdn-overlapping-window-path-stitch-probe-train` で長く、exp204 の既知事例と同じ Kaggle slug 制限に当たりやすい。
- id: `kentookumura/exp207-hmdn-path-stitch-train`
- title: `exp207 hmdn path stitch train`

予定コマンド:

```bash
make validate-exp EXP=exp207_heatmap_mdn_overlapping_window_path_stitch_probe
make prepare-kaggle-notebooks EXP=exp207_heatmap_mdn_overlapping_window_path_stitch_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp207-hmdn-path-stitch-train --title 'exp207 hmdn path stitch train' --run-on-push --strict"
make push-kaggle-train EXP=exp207_heatmap_mdn_overlapping_window_path_stitch_probe
kaggle kernels pull kentookumura/exp207-hmdn-path-stitch-train -p /tmp/kaggle-pull/exp207-hmdn-path-stitch-train -m
kaggle kernels logs kentookumura/exp207-hmdn-path-stitch-train
kaggle kernels status kentookumura/exp207-hmdn-path-stitch-train
```

実行結果:

- `make validate-exp`: pass
- `make prepare-kaggle-notebooks`: pass
- generated metadata: `enable_gpu=false`、`enable_internet=false`、`run_on_push=true`、kernel sources は `kentookumura/exp202-heatmap-mdn-candgen-train` / `kentookumura/exp099-pf-multiobs-likelihood-train`
- `make push-kaggle-train`: success、Kernel version 1
- URL: <https://www.kaggle.com/code/kentookumura/exp207-hmdn-path-stitch-train>
- pulled metadata id_no: `126149730`
- pulled metadata: `enable_gpu=false`、`machine_shape=None`、`enable_internet=false`、competition source `rogii-wellbore-geology-prediction`
- initial `kaggle kernels logs`: Kaggle CLI version warning のみで本文空。実行中 logs 空は既知挙動なので失敗扱いしない。
- initial `kaggle kernels status`: `KernelWorkerStatus.RUNNING`

5分監視:

```bash
timeout 300 kaggle kernels logs -f --interval 30 kentookumura/exp207-hmdn-path-stitch-train
```

結果:

- v1 は `ValueError: No kernel name found in notebook and no override provided.` で即時失敗。
- 原因: Jupytext percent source に `kernelspec.name: python3` metadata がなく、Kaggle papermill が kernel を選べなかった。

修正:

- train / inference percent source に、既存 exp204 と同じ Jupytext + Python 3 kernelspec header を追加した。
- notebook を再変換し、train notebook metadata が `{'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}` になったことを確認した。

修正後チェック:

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/exp207_heatmap_mdn_overlapping_window_path_stitch_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/exp207_heatmap_mdn_overlapping_window_path_stitch_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/exp207_heatmap_mdn_overlapping_window_path_stitch_probe_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/exp207_heatmap_mdn_overlapping_window_path_stitch_probe_inference.py
.venv/bin/python -m py_compile experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/heatmap_mdn_overlapping_window_path_stitch_probe.py experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/exp207_heatmap_mdn_overlapping_window_path_stitch_probe_train.py experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/exp207_heatmap_mdn_overlapping_window_path_stitch_probe_inference.py experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/settings.py
.venv/bin/ruff check experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/heatmap_mdn_overlapping_window_path_stitch_probe.py experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/exp207_heatmap_mdn_overlapping_window_path_stitch_probe_train.py experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/exp207_heatmap_mdn_overlapping_window_path_stitch_probe_inference.py experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/settings.py --select F821,E501
make validate-exp EXP=exp207_heatmap_mdn_overlapping_window_path_stitch_probe
```

結果:

- Jupytext train / inference conversion and `--test`: pass
- `py_compile`: pass
- `ruff --select F821,E501`: pass
- `make validate-exp`: pass

次: 同じ kernel id に version 2 として再 prepare / push する。

## 2026-07-06 Kaggle train v2

修正版を同じ kernel id に version 2 として push。

```bash
make prepare-kaggle-notebooks EXP=exp207_heatmap_mdn_overlapping_window_path_stitch_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp207-hmdn-path-stitch-train --title 'exp207 hmdn path stitch train' --run-on-push --strict"
make push-kaggle-train EXP=exp207_heatmap_mdn_overlapping_window_path_stitch_probe
kaggle kernels pull kentookumura/exp207-hmdn-path-stitch-train -p /tmp/kaggle-pull/exp207-hmdn-path-stitch-train-v2 -m
kaggle kernels status kentookumura/exp207-hmdn-path-stitch-train
kaggle kernels logs kentookumura/exp207-hmdn-path-stitch-train
```

結果:

- `make prepare-kaggle-notebooks`: pass。packaged notebook の `kernelspec.name=python3` を確認。
- `make push-kaggle-train`: success、Kernel version 2
- URL: <https://www.kaggle.com/code/kentookumura/exp207-hmdn-path-stitch-train>
- `kaggle kernels pull`: success
- initial status: `KernelWorkerStatus.RUNNING`
- initial logs: Kaggle CLI version warning のみで本文空。実行中 logs 空は既知挙動なので失敗扱いしない。

5分監視と完了確認:

```bash
timeout 300 kaggle kernels logs -f --interval 30 kentookumura/exp207-hmdn-path-stitch-train
kaggle kernels status kentookumura/exp207-hmdn-path-stitch-train
kaggle kernels logs kentookumura/exp207-hmdn-path-stitch-train
kaggle kernels output kentookumura/exp207-hmdn-path-stitch-train -p experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/kaggle/output/train_v2
```

結果:

- status: `KernelWorkerStatus.COMPLETE`
- output: `experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/kaggle/output/train_v2`
- exp202 path artifact SHA: `e615f0d01a08fd37685fd1ac46335b99306f0bb0c9c43d37c1e1f620040839a3`
- exp202 path samples decompressed SHA: `cea6a29c716a1c5dedda1efec64a1b1f2371d1eadfd298084576f06170d0a7de`
- exp099 candidate cache rows / wells: `3,783,989` / `773`
- available existing candidates: `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb`
- missing optional candidates: `tvt_dense`、`tvt_densew`、`tvt_dense50`
- source windows: `10,822` samples、`14` windows / well
- source overlap: `3` wells、`39` center pairs
- source gap pairs: `10,010`
- stitched rows / wide rows: `4,148,139` / `1,382,713`
- merged covered rows: `1,333,241`
- row coverage vs exp099 cache: `0.352337441`

candidate union readout on covered rows:

- existing union oracle RMSE / within10: `5.154353660` / `0.947161091`
- stitched only top1 oracle RMSE / within10: `52.259030768` / `0.247782659`
- existing + stitched top1 oracle RMSE / within10 / new-best rate: `4.472998031` / `0.958441872` / `0.060972472`
- stitched only top3 oracle RMSE / within10: `50.798377042` / `0.275946359`
- existing + stitched top3 oracle RMSE / within10 / new-best rate: `4.418699605` / `0.959487445` / `0.069157039`
- existing + stitched top3 delta vs existing: RMSE `-0.735654055`、within10 `+0.012326354`

bucket / by-well:

- `1000_plus`: existing oracle RMSE `6.376418` -> union top3 `5.414525`、new-best `0.091269`
- near `0_50`: existing oracle RMSE `0.313605` -> union top3 `0.310983`
- by-well: 773 wells、461 improved、312 same、0 worse
- mean / median RMSE delta: `-0.338524069` / `-0.003918543`
- best improvement: `1b1eba53`, `37.761571 -> 17.418515`

判断:

- Covered-row oracle headroom は positive。
- ただし stitched only は RMSE 50.798 と弱く、現行 exp202 v2 artifact は sparse で overlap が 3 wells / 39 pairs しかない。
- したがって full-well overlapping stitch としては不十分。direct TVT replacement、softmax average、PF weight replacement、inference、submit はしない。
- この backlog は exp207 train-side diagnostic として完了。続ける場合は dense stride window path regeneration を別候補に切る。
