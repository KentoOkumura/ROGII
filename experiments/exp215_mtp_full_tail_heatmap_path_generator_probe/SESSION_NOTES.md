# exp215_mtp_full_tail_heatmap_path_generator_probe セッションノート

## 2026-07-07 実装

目的: `exp215_mtp_full_tail_heatmap_path_generator_probe` backlog を実験化する。exp212 は full-grid contract は通ったが、source coverage 0.430091631 / fallback unique row rate 0.569908369 で後半が endpoint hold 直線 tail になった。exp215 では learned `path_logit` を持つ MTP full-tail generator を train-side diagnostic として作る。

### 設計

- Route: `pf_beam`
- 親: `exp202_heatmap_mdn_candidate_generator_probe`
- dense / full-grid 参考: `exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe`、`exp212_heatmap_mdn_full_grid_path_generation_probe`
- 比較: `exp099_pf_multi_observation_likelihood_probe`
- 入力: exp202 と同じ 5ch heatmap
  - typewell GR
  - horizontal GR
  - typewell minus horizontal GR
  - observed `TVT_input` prefix history SDF
  - observed mask
- 出力: `path_pred [B,K,L]`、`path_logit [B,K]`
- Loss: closest-mode path regression + `cross_entropy(path_logit, best_k)`
- SDF output head / SDF target / `sdf_loss`: 使わない
- full-grid aggregation: rank path は `path_prob x triangular window weight`、weighted path は triangular window weight で row aggregation
- candidate union readout: exp099 cache に learned topK / weighted path を加えた oracle readout

### リークガード

- train labels は fold train wells の true TVT path のみ。
- OOF dense generation は fold valid wells のみ。
- valid true TVT は loss/eval/readout にのみ使う。
- aggregation weight、path_prob、path_logit、candidate_cost、coverage/fallback flag に oracle best、abs-error、within10、true-error rank を入れない。
- hidden/test true TVT は使わない。
- inference / submit は対象外。

### 予定コスト

Kaggle train push 前に再確認する。

- active variants: `1`
- folds: `5`
- CNN models: `5`
- LightGBM configs: `0`
- boosters: `0`
- parent/control retraining: なし
- Kaggle GPU: T4 想定
- inference / submit: なし

### 実装ファイル

- `.steering/20260707-exp215-mtp-full-tail-heatmap-path-generator-probe/`
- `experiments/exp215_mtp_full_tail_heatmap_path_generator_probe/config.yaml`
- `experiments/exp215_mtp_full_tail_heatmap_path_generator_probe/exp215_mtp_full_tail_heatmap_path_generator_probe_train.py`
- `experiments/exp215_mtp_full_tail_heatmap_path_generator_probe/exp215_mtp_full_tail_heatmap_path_generator_probe_inference.py`
- `experiments/exp215_mtp_full_tail_heatmap_path_generator_probe/settings.py`

### 検証ログ

実行:

```bash
.venv/bin/python -m py_compile experiments/exp215_mtp_full_tail_heatmap_path_generator_probe/exp215_mtp_full_tail_heatmap_path_generator_probe_train.py experiments/exp215_mtp_full_tail_heatmap_path_generator_probe/exp215_mtp_full_tail_heatmap_path_generator_probe_inference.py experiments/exp215_mtp_full_tail_heatmap_path_generator_probe/settings.py
.venv/bin/ruff check experiments/exp215_mtp_full_tail_heatmap_path_generator_probe/exp215_mtp_full_tail_heatmap_path_generator_probe_train.py experiments/exp215_mtp_full_tail_heatmap_path_generator_probe/exp215_mtp_full_tail_heatmap_path_generator_probe_inference.py experiments/exp215_mtp_full_tail_heatmap_path_generator_probe/settings.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp215_mtp_full_tail_heatmap_path_generator_probe/exp215_mtp_full_tail_heatmap_path_generator_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp215_mtp_full_tail_heatmap_path_generator_probe/exp215_mtp_full_tail_heatmap_path_generator_probe_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp215_mtp_full_tail_heatmap_path_generator_probe/exp215_mtp_full_tail_heatmap_path_generator_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp215_mtp_full_tail_heatmap_path_generator_probe/exp215_mtp_full_tail_heatmap_path_generator_probe_inference.py
make validate-exp EXP=exp215_mtp_full_tail_heatmap_path_generator_probe
make prepare-kaggle-notebooks EXP=exp215_mtp_full_tail_heatmap_path_generator_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp215-mtp-full-tail-heatmap-path-generator-train --title 'exp215 mtp full tail heatmap path generator train' --run-on-push --strict"
```

結果:

- `py_compile`: pass
- `ruff --select F821`: pass
- Jupytext train / inference conversion: pass
- Jupytext train / inference `--test`: pass
- `make validate-exp`: pass
- `make prepare-kaggle-notebooks`: pass

Kaggle train metadata:

- kernel id: `kentookumura/exp215-mtp-full-tail-heatmap-path-generator-train`
- code file: `exp215_mtp_full_tail_heatmap_path_generator_probe_train.ipynb`
- GPU: `NvidiaTeslaT4`
- internet: false
- run_on_push: true
- competition source: `rogii-wellbore-geology-prediction`
- kernel source: `kentookumura/exp099-pf-multiobs-likelihood-train`

未実行:

- Kaggle train push
- Kaggle output / metrics readout
- inference / submit

## 2026-07-07 Kaggle train 実行

ユーザー承認:

- Kaggle GPU 実行: OK

GPU cost guard:

- active variants: `1`
- folds: `5`
- CNN models: `5`
- LightGBM configs: `0`
- boosters: `0`
- parent/control retraining: なし
- accelerator: `NvidiaTeslaT4`
- inference / submit: なし

push command:

```bash
kaggle kernels push -p experiments/exp215_mtp_full_tail_heatmap_path_generator_probe/kaggle/train --accelerator NvidiaTeslaT4
```

push result:

- Kernel version 1 pushed successfully.
- URL: https://www.kaggle.com/code/kentookumura/exp215-mtp-full-tail-heatmap-path-generator-train
- Kaggle pulled metadata confirmed `machine_shape: NvidiaTeslaT4`.
- `kaggle kernels status`: `KernelWorkerStatus.RUNNING`
- `timeout 600 kaggle kernels logs -f --interval 30 ...`: timed out with no CLI log output while status remained `RUNNING`.
- After additional polling, status still remained `KernelWorkerStatus.RUNNING` and CLI logs remained empty.
- User requested to stop monitoring locally. Kaggle run is left running; no stop/re-push was performed.

completion readout:

```bash
kaggle kernels status kentookumura/exp215-mtp-full-tail-heatmap-path-generator-train
kaggle kernels logs kentookumura/exp215-mtp-full-tail-heatmap-path-generator-train
```

結果:

- status: `KernelWorkerStatus.COMPLETE`
- logs saved locally for parsing: `/tmp/exp215_kaggle_logs.json`
- output archive: not downloaded

Key metrics from Kaggle logs:

- full-grid rows: 18,919,945
- unique row ids: 3,783,989
- wells: 773
- path ranks: 5
- row coverage vs exp099 cache: 1.000000000
- fallback unique row rate: 0.000000000
- duplicate key rows: 0
- null required values: 0
- existing union oracle RMSE: 7.434029932
- existing union within10: 0.906525363
- learned MTP top5 only oracle RMSE: 32.333142886
- learned MTP top5 only within10: 0.477650966
- learned MTP weighted oracle RMSE: 59.272141581
- learned MTP weighted within10: 0.153523702
- existing + learned MTP top5 oracle RMSE: 5.113654814
- existing + learned MTP top5 within10: 0.945863743
- existing + learned MTP top5 RMSE delta vs existing: -2.320375117

MTP window aggregate:

- folds completed: 5
- train samples: 86,576
- valid samples: 7,730
- dense samples: 60,266
- dense top10 oracle center RMSE: 15.741723880
- dense weighted center RMSE: 60.093226732
- dense rank1 center RMSE: 64.792147625

Artifact SHA from logs:

- full_grid_candidate_paths_csv_decompressed_sha256: `2334e62225e0b35328886ef6a3ebfcbc02faf83cef57e43472a16f3b29549871`
- candidate_union_metrics_csv_sha256: `a8ca12409bea71e4378d9dbe68276d272a419b214d3f39fe03889f423530cadd`
- summary_json_sha256: `be1c0cab26c32da399ddd9c1acf022410c6b861c5b7f74b3ed39647778e8a828`

判断:

- exp212 の fallback-heavy / endpoint hold tail 問題は、coverage 1.0 / fallback 0.0 で解消。
- learned MTP weighted path は弱く、direct replacement / softmax weighted TVT / PF weight replacement / inference / submit はしない。
- existing + learned MTP top5 の oracle union headroom は positive。exp204 系で selector candidate / confidence feature として guarded に使う価値はある。

## 2026-07-07 hengck23 CNN MTP example 比較

参照:

- <https://www.kaggle.com/code/hengck23/cnn-mtp-example?scriptVersionId=320093395>
- Kaggle CLI pull: `kaggle kernels pull hengck23/cnn-mtp-example -p /tmp/hengck23-cnn-mtp-example -m`

参照 notebook の要点:

- `GeoStirringNet(K=10, L=24)`
- input: `heatmap [B,1,64,24]` と `history [B,1,64,24]`
- output: `path [B,K,L]` と `logit [B,K]`
- loss: per-mode trajectory MSE -> `best_k=argmin(error)`、best path regression loss + `cross_entropy(logit,best_k)`
- checkpoint visualization 例で、full-grid artifact / fold train / candidate union readout は含まない

exp215 との一致:

- continuous topK path prediction と learned path logit を使う。
- closest-mode loss + mode CE loss を使う。
- `softmax(path_logit)` を path probability として使う。
- SDF output head / SDF target / `sdf_loss` は使わない。

意図的な差分:

- exp215 は exp202 系 5ch heatmap input、`K=10,L=128`、5 folds、full-tail dense generation、full-grid candidate artifact、exp099 candidate union readout まで拡張。
- exp215 の history channel は observed `TVT_input` prefix 由来の連続 SDF-like channel。参照 notebook は prefix match path を描いた binary history image。

結論:

- 参照 notebook の MTP head / learned logit / closest-mode loss という中核には沿っている。
- 参照 notebook の checkpoint/crop 可視化をそのまま再現したものではなく、full-tail/full-grid 診断へ拡張した実験として正しく実行できている。

## 2026-07-07 plot parity 再確認

ユーザー指摘: 参照 notebook の plot と同じような結果には見えない。

確認:

- exp215 train notebook には `matplotlib` / `imshow` / `plot` の可視化セルはない。
- `kaggle kernels output ... --file-pattern "__(results|notebook)__\\.(html|ipynb)"` は log のみ取得。Kaggle CLI output から plot HTML / notebook output は取得できなかった。
- 参照 notebook の plot は y 軸 typewell bin index 0..63、x 軸 compressed horizontal segment 0..23、vertical line x=8 の local crop 可視化。exp215 の primary output は TVT feet の continuous path over 128 horizontal rows で、後段 artifact では nearest grid bin path も保存するが、描画座標は参照と同一ではない。
- 参照 notebook は flatten CNN -> MLP -> `Linear(K*L)`。exp215 は 2D conv feature から mode 別 map を作り、typewell 軸平均で `path_pred [B,K,128]` を出す。head 構造も完全一致ではない。
- 取得 artifact:
  - `/tmp/exp215_plot_check/artifacts/exp215_mtp_full_tail_heatmap_path_generator_probe_window_path_samples.csv.gz`
  - `/tmp/exp215_plot_check/artifacts/exp215_mtp_full_tail_heatmap_path_generator_probe_window_path_rank_index.csv.gz`

Artifact readout:

- dense samples: 60,266
- rank1 center RMSE: 65.567 ft / within10: 0.141
- top5 best center RMSE: 31.761 ft / within10: 0.472
- `000d7d20_1442`: true center TVT 11747.380 ft, rank1 11844.528 ft (97.148 ft error), rank2 11746.194 ft (1.186 ft error)

判断:

- ユーザーの違和感は妥当。exp215 は Kaggle 実行、full-grid coverage、artifact contract は成功しているが、hengck23 notebook の plot を再現したとは言えない。
- 現状は「候補集合には近い path が入ることがあるが、logit rank / probability weighted average が弱い」状態。
- 参照 notebook に近い見た目で検証するには、64x24 local crop、binary history、typewell bin target、flatten MLP `Linear(K*L)` head、plot cell を持つ専用 parity probe が必要。
