# exp202_heatmap_mdn_candidate_generator_probe セッションノート

## 2026-07-05 実装

目的: `heatmap_mdn_candidate_generator_probe` backlog を実験化する。exp182/184 の heatmap MTP 系を PF/Beam 候補生成器として再設計し、既存 PF/Beam/likPF/sc/hyb 候補 union に heatmap topK 候補を足した時の oracle headroom を train-side で確認する。

実装方針:

- Route: `pf_beam`
- 親: `exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe`、`exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`、`exp099_pf_multi_observation_likelihood_probe`
- active run spec: `candidate_real_w128_b64_fullfold`
- GPU cost before push: 1 active heatmap candidate-generator spec x 5 folds = 5 CNN models
- LightGBM: 0 configs / 0 boosters
- control / parent retraining: なし
- submit / inference port: なし

主な変更:

- `docs/legacy/steering/20260705-exp202-heatmap-mdn-candidate-generator-probe/` を作成。
- `experiments/exp202_heatmap_mdn_candidate_generator_probe/` を作成。
- exp182 の 5ch heatmap CNN/MTP train source を土台に、`id`、mode score margin / entropy、heatmap topK candidate CSV、existing-plus-heatmap union oracle readout を追加。
- `config.yaml` を `pf_beam` route、candidate generator probe、exp099 candidate cache readout 用に更新。
- inference source は diagnostic-only guard に更新。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp202_heatmap_mdn_candidate_generator_probe/exp202_heatmap_mdn_candidate_generator_probe_train.py experiments/exp202_heatmap_mdn_candidate_generator_probe/exp202_heatmap_mdn_candidate_generator_probe_inference.py
.venv/bin/ruff check experiments/exp202_heatmap_mdn_candidate_generator_probe/exp202_heatmap_mdn_candidate_generator_probe_train.py experiments/exp202_heatmap_mdn_candidate_generator_probe/exp202_heatmap_mdn_candidate_generator_probe_inference.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp202_heatmap_mdn_candidate_generator_probe/exp202_heatmap_mdn_candidate_generator_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp202_heatmap_mdn_candidate_generator_probe/exp202_heatmap_mdn_candidate_generator_probe_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp202_heatmap_mdn_candidate_generator_probe/exp202_heatmap_mdn_candidate_generator_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp202_heatmap_mdn_candidate_generator_probe/exp202_heatmap_mdn_candidate_generator_probe_inference.py
make validate-exp EXP=exp202_heatmap_mdn_candidate_generator_probe
```

結果:

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --test`: pass
- `validate-exp`: pass
- 古い exp182 notebook 内容が残っていないことを `rg` で確認済み

未実行:

- Kaggle prepare / push

注意:

- Kaggle train push 前に、上記 GPU cost と control 再学習なしを再確認する。
- positive でも direct TVT replacement、softmax weighted average、PF weight replacement、postprocess blend、submit はしない。

## 2026-07-05 Kaggle train push

ユーザーから「kaggleで実行してください」と指示あり。GPU cost guard を再確認した。

- active run spec: `candidate_real_w128_b64_fullfold`
- folds: 5
- CNN models: 5
- LightGBM configs / boosters: 0 / 0
- parent/control retraining: なし
- inference / submit: なし

予定 kernel:

- id: `kentookumura/exp202-heatmap-mdn-candgen-train`
- title: `exp202 heatmap mdn candgen train`

実行:

```bash
make prepare-kaggle-notebooks EXP=exp202_heatmap_mdn_candidate_generator_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp202-heatmap-mdn-candgen-train --title 'exp202 heatmap mdn candgen train' --run-on-push --strict"
kaggle kernels push -p experiments/exp202_heatmap_mdn_candidate_generator_probe/kaggle/train --accelerator NvidiaTeslaT4
kaggle kernels pull kentookumura/exp202-heatmap-mdn-candgen-train -p /tmp/kaggle-pull/exp202-heatmap-mdn-candgen-train-v2 -m
kaggle kernels logs kentookumura/exp202-heatmap-mdn-candgen-train
kaggle kernels status kentookumura/exp202-heatmap-mdn-candgen-train
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp202-heatmap-mdn-candgen-train
```

push / monitoring 結果:

- Kaggle train v2 pushed successfully.
- URL: https://www.kaggle.com/code/kentookumura/exp202-heatmap-mdn-candgen-train
- Kaggle kernel id_no: `126029283`
- pulled metadata: `enable_gpu=true`、`machine_shape=NvidiaTeslaT4`、`enable_internet=false`、`kernel_sources=["kentookumura/exp099-pf-multiobs-likelihood-train"]`
- initial `kaggle kernels logs`: empty. This is expected while running in this environment and is not treated as failure.
- initial `kaggle kernels status`: `KernelWorkerStatus.RUNNING`
- `logs -f`: 600秒で timeout、出力なし。
- status after monitoring: `KernelWorkerStatus.RUNNING`

完了後確認:

```bash
kaggle kernels status kentookumura/exp202-heatmap-mdn-candgen-train
kaggle kernels logs kentookumura/exp202-heatmap-mdn-candgen-train
kaggle kernels output kentookumura/exp202-heatmap-mdn-candgen-train -p experiments/exp202_heatmap_mdn_candidate_generator_probe/kaggle/output/train_v2
```

結果:

- status: `KernelWorkerStatus.COMPLETE`
- output: `experiments/exp202_heatmap_mdn_candidate_generator_probe/kaggle/output/train_v2`
- local path output status: `saved_for_plotting`
- local path output format: `npz_plus_index_csv`
- local path samples / paths / topK / horizon: `10822` / `108220` / `10` / `128`
- `pred_tvt_path` shape: `(10822, 10, 128)`
- `pred_bin_path` shape: `(10822, 10, 128)`
- `true_tvt_path`、`tvt_input_path`、`md_path`、`z_path`、`horizontal_row_index` shape: `(10822, 128)`
- `heatmap_candidate_path_samples.csv.gz` rows/columns: `10822` / `21`
- `heatmap_candidate_path_rank_index.csv.gz` rows/columns: `108220` / `15`
- `heatmap_candidate_paths_npz_sha256`: `e615f0d01a08fd37685fd1ac46335b99306f0bb0c9c43d37c1e1f620040839a3`
- `heatmap_candidate_path_samples_csv_decompressed_sha256`: `cea6a29c716a1c5dedda1efec64a1b1f2371d1eadfd298084576f06170d0a7de`
- `heatmap_candidate_path_rank_index_csv_decompressed_sha256`: `a21b7c9c056272c20d3c1b60f6b7224e78edeb156cb1af19b6a950bd1278f349`

key metrics:

- heatmap only top10 within10 / oracle RMSE: `0.808907780` / `13.352563025`
- existing union oracle RMSE / within10: `5.068679053` / `0.949639623`
- existing + heatmap top10 oracle RMSE / within10: `2.745528140` / `0.986970985`
- delta vs existing union: oracle RMSE `-2.323150913`、within10 `+0.037331362`
- new-best candidate rate: `0.252541120`

判断:

- v2 は path artifact materialization 目的を達成した。
- 保存された path は full well trajectory ではなく、validation sample ごとの local 128-row path。
- metrics は v1 と同じ判断で、direct replacement / softmax average / PF weight replacement / submit には進めない。

実行:

```bash
make prepare-kaggle-notebooks EXP=exp202_heatmap_mdn_candidate_generator_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp202-heatmap-mdn-candgen-train --title 'exp202 heatmap mdn candgen train' --run-on-push --strict"
kaggle kernels push -p experiments/exp202_heatmap_mdn_candidate_generator_probe/kaggle/train --accelerator NvidiaTeslaT4
kaggle kernels pull kentookumura/exp202-heatmap-mdn-candgen-train -p /tmp/kaggle-pull/exp202-heatmap-mdn-candgen-train-v2 -m
kaggle kernels logs kentookumura/exp202-heatmap-mdn-candgen-train
kaggle kernels status kentookumura/exp202-heatmap-mdn-candgen-train
```

結果:

- Kaggle train v2 pushed successfully.
- URL: https://www.kaggle.com/code/kentookumura/exp202-heatmap-mdn-candgen-train
- Kaggle kernel id_no: `126029283`
- pulled metadata: `enable_gpu=true`、`machine_shape=NvidiaTeslaT4`、`enable_internet=false`、`kernel_sources=["kentookumura/exp099-pf-multiobs-likelihood-train"]`
- initial `kaggle kernels logs`: empty. This is expected while running in this environment and is not treated as failure.
- `kaggle kernels status`: `KernelWorkerStatus.RUNNING`

10分監視:

```bash
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp202-heatmap-mdn-candgen-train
kaggle kernels status kentookumura/exp202-heatmap-mdn-candgen-train
kaggle kernels logs kentookumura/exp202-heatmap-mdn-candgen-train
```

結果:

- `logs -f`: 600秒で timeout、出力なし。
- status after monitoring: `KernelWorkerStatus.RUNNING`
- logs after monitoring: empty.
- 判断: Kaggle 側の実行は継続中。CLI logs 空は前回同様の既知挙動なので、失敗扱いにしない。

実行:

```bash
make prepare-kaggle-notebooks EXP=exp202_heatmap_mdn_candidate_generator_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp202-heatmap-mdn-candgen-train --title 'exp202 heatmap mdn candgen train' --run-on-push --strict"
kaggle kernels push -p experiments/exp202_heatmap_mdn_candidate_generator_probe/kaggle/train --accelerator NvidiaTeslaT4
kaggle kernels pull kentookumura/exp202-heatmap-mdn-candgen-train -p /tmp/kaggle-pull/exp202-heatmap-mdn-candgen-train-v1 -m
kaggle kernels logs kentookumura/exp202-heatmap-mdn-candgen-train
kaggle kernels status kentookumura/exp202-heatmap-mdn-candgen-train
```

結果:

- Kaggle train v1 pushed successfully.
- URL: https://www.kaggle.com/code/kentookumura/exp202-heatmap-mdn-candgen-train
- Kaggle kernel id_no: `126029283`
- pulled metadata: `enable_gpu=true`、`machine_shape=NvidiaTeslaT4`、`enable_internet=false`、`kernel_sources=["kentookumura/exp099-pf-multiobs-likelihood-train"]`
- initial `kaggle kernels logs`: empty. This is expected while running in this environment and is not treated as failure.
- `kaggle kernels status`: `KernelWorkerStatus.RUNNING`

## 2026-07-05 monitoring paused

ユーザー指示によりローカルの `kaggle kernels logs -f` 監視を停止した。Kaggle Notebook 実行自体は停止していない。

- latest known status before pausing: `KernelWorkerStatus.RUNNING`
- logs at that point: empty
- next action: ユーザーの完了連絡後に logs / output を取得し、metrics と result を記録する。

## 2026-07-05 Kaggle train complete

ユーザーから完了連絡あり。Kaggle 側の状態と output を取得した。

```bash
kaggle kernels logs kentookumura/exp202-heatmap-mdn-candgen-train
kaggle kernels status kentookumura/exp202-heatmap-mdn-candgen-train
kaggle kernels output kentookumura/exp202-heatmap-mdn-candgen-train -p experiments/exp202_heatmap_mdn_candidate_generator_probe/kaggle/output/train_v1
```

結果:

- status: `KernelWorkerStatus.COMPLETE`
- output: `experiments/exp202_heatmap_mdn_candidate_generator_probe/kaggle/output/train_v1`
- fold mean heatmap top10 within10: `0.808984978`
- fold mean heatmap top10 oracle RMSE: `13.293297981`
- existing union oracle RMSE / within10: `5.068679053` / `0.949639623`
- existing + heatmap top10 oracle RMSE / within10: `2.745528140` / `0.986970985`
- delta vs existing union: oracle RMSE `-2.323150913`、within10 `+0.037331362`
- new-best candidate rate: `0.252541120`
- distance `1000_plus`: oracle RMSE `6.413572416 -> 3.295946470`
- by-well: 773 wells、668 improved、105 same、0 worse

判断:

- heatmap top10 candidate は既存 PF/Beam candidate union の headroom を大きく増やす。
- heatmap only は existing union より弱いため、direct replacement / softmax average / PF weight replacement / submit はしない。
- 次は target-free selector / confidence feature として使えるかを別 backlog で評価する。

## 2026-07-06 plotting 用 local path 保存の実装

ユーザー指示により、現行 exp202 実装のまま validation sample ごとの heatmap candidate local path を output に保存する変更を入れた。これは plot / readout 用であり、学習、candidate union 評価、selector、推論、提出には使わない。

追加内容:

- `candidate_paths` config を追加。
- `evaluate_model(..., collect_paths=True)` で、deduplicate 済み center-row top10 candidate の mode index、center bin/TVT/score、対応する local 128-row predicted path を収集する。
- 同じ sample の true TVT path、observed `TVT_input` path、MD/Z、horizontal row index、horizontal offsets も保存する。
- 追加出力:
  - `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_paths_top10.npz`
  - `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_path_samples.csv.gz`
  - `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_path_rank_index.csv.gz`

注意:

- 保存されるのは full well trajectory ではなく、exp202 が学習・評価している 128-row local window path。
- topK は center row の TVT candidate を deduplicate した順位で、各 rank に対応する CNN mode の local path を保存する。
- overlapping windows を stitch して full-well / multi-trajectory path を再構成する案は、別 backlog `heatmap_mdn_overlapping_window_path_stitch_probe` として切り出した。
- v1 Kaggle output にはこの追加生成物は含まれない。必要なら同じ exp202 train notebook を再 push / rerun して materialize する。

## 2026-07-06 Kaggle train rerun for path artifact materialization

ユーザーから「kaggleで実行してください」と指示あり。目的は 2026-07-06 に追加した local path artifact を Kaggle output として materialize すること。

GPU cost guard:

- active run spec: `candidate_real_w128_b64_fullfold`
- folds: 5
- CNN models: 5
- LightGBM configs / boosters: 0 / 0
- parent/control retraining: なし
- inference / submit: なし
- expected new outputs:
  - `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_paths_top10.npz`
  - `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_path_samples.csv.gz`
  - `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_path_rank_index.csv.gz`

予定 kernel:

- id: `kentookumura/exp202-heatmap-mdn-candgen-train`
- title: `exp202 heatmap mdn candgen train`
