# exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector セッションノート

## 2026-07-03 実装

### 狙い

`cluster_outlier_prior_confidence_addonly_on_exp158_selector` backlog を実装する。exp181 では cluster-outlier gate 付き prior correction が PF/Beam/likPF 候補上では改善したが、direct posthoc correction としては worst-well regression が大きかった。そのため exp183 では correction を予測へ加算せず、exp157/158 selector の score feature として add-only する。

### 実装内容

- `docs/legacy/steering/20260703-exp183-cluster-outlier-prior-confidence-addonly-on-exp158-selector/` を作成。
- `experiments/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector/` を exp158 からコピーして作成。
- 実装本体を `cluster_outlier_prior_confidence_addonly_on_exp158_selector.py` として作成。
  - exp157 の train-side candidate ranker flow をベースにした。
  - exp157 と同じ 8候補、dense enrichment、GroupKFold by well を維持。
  - exp109 / exp114 prior、exp065 cluster assignment、exp114 geometry から cluster-outlier prior confidence features を生成。
  - candidate-long feature として prior-candidate delta、prior std/count/neighbor、gate x candidate family、alpha 0.2 clip 20/40 correction magnitude、clip hit を追加。
  - 新しい LightGBM score models の OOF predicted-error surface を exp158 と同じ Viterbi grid に渡す。
  - exp115 hidden-like role は subgroup metrics のみに使用し、feature には使わない。
- `config.yaml` を exp183 用に更新。
- Jupytext percent source の train / inference notebook を追加。
- README / result / metrics placeholder を exp183 用に更新。

### Kaggle train 実行前確認

- 実行対象: `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector`
- Route: `pf_beam`
- active selector variant: 1
- LightGBM configs: 3 (`lgb_multiclass`, `lgb_candidate_binary`, `lgb_candidate_error_ranker`)
- folds: 5
- planned boosters: 15
- Viterbi variants: 180
- control / parent retraining: なし
- GPU: disabled
- direct exp181 correction: なし
- inference port / submit: なし

### 再現性メモ

- exp183 自体の乱数は GroupKFold seed、LightGBM seed、candidate-long subsample seed に限定する。
- PF/Beam / likelihood-PF / typewell prior / spatial prior / cluster assignment は upstream fixed Kaggle output として読む。
- gzip 生成物は decompressed content SHA を主証拠として summary に記録する。
- deterministic submission anchor ではない。`submission.csv` は作らない。

### 検証予定

```bash
.venv/bin/python -m py_compile \
  experiments/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector/cluster_outlier_prior_confidence_addonly_on_exp158_selector.py \
  experiments/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_train.py \
  experiments/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_inference.py \
  experiments/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector/settings.py
.venv/bin/ruff check \
  experiments/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector/cluster_outlier_prior_confidence_addonly_on_exp158_selector.py \
  experiments/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_train.py \
  experiments/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_inference.py \
  --select F821,F401,E501
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector_inference.py
make validate-exp EXP=exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector
```

### 検証結果

- `py_compile`: pass
- `ruff --select F821,F401,E501`: pass
- Jupytext train / inference convert and `--test`: pass
- `make validate-exp`: pass
- `make prepare-kaggle-notebooks` train / inference: pass

### Kaggle package prepare

```bash
make prepare-kaggle-notebooks EXP=exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp183-cluster-outlier-prior-confidence-addonly-on-exp158-selector-train --title 'exp183 cluster outlier prior confidence addonly on exp158 selector train' --run-on-push --strict"
make prepare-kaggle-notebooks EXP=exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp183-cluster-outlier-prior-confidence-addonly-on-exp158-selector-infer --title 'exp183 cluster outlier prior confidence addonly on exp158 selector infer' --run-on-push --strict"
```

生成済み package:

- train: `experiments/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector/kaggle/train`
- inference: `experiments/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector/kaggle/inference`

metadata:

- train id: `kentookumura/exp183-cluster-outlier-prior-confidence-addonly-on-exp158-selector-train`
- inference id: `kentookumura/exp183-cluster-outlier-prior-confidence-addonly-on-exp158-selector-infer`
- GPU: false
- internet: false
- kernel sources: exp099 / exp072 / exp109 / exp114 / exp065 / exp115

bootstrap manifest に `cluster_outlier_prior_confidence_addonly_on_exp158_selector.py`、`config.yaml`、train / inference percent source、`settings.py`、`project.yml` が含まれることを確認した。

## 2026-07-04 Kaggle train push

### 初回 push 失敗

```bash
make push-kaggle-train EXP=exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector
```

`kaggle kernels push` が `SaveKernel` 400 を返した。詳細 message はなし。元の kernel id/title が長く、Kaggle 側の slug 制約に触れた可能性があるため、実験番号は変えず、kernel id/title だけを短い canonical 名へそろえて再 prepare する。

- old train id: `kentookumura/exp183-cluster-outlier-prior-confidence-addonly-on-exp158-selector-train`
- new train id: `kentookumura/exp183-cluster-outlier-prior-conf-addonly-exp158-train`
- new train title: `exp183 cluster outlier prior conf addonly exp158 train`

### 追加切り分け

短縮 id/title で再 prepare 後に push したが、同じく `SaveKernel` 400 で失敗した。6 個の `kernel_sources` はすべて `kaggle kernels pull -m` で存在確認できたため、Kaggle metadata 側の source 数または組み合わせ制約を疑う。

exp115 は学習特徴量ではなく subgroup metrics 用の fold assignment だけなので、CSV を `inputs/exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv` として exp183 package に bootstrap し、`kernel_sources` から `kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train` を外す。学習特徴量、LightGBM booster 数、Viterbi grid は不変。

5 source + exp115 bootstrap でも `SaveKernel` 400 が続いた。次に Kaggle slug/title の長さをさらに保守的にし、`kentookumura/exp183-copcf-train` / `exp183 copcf train` で push を試す。

### Kaggle train v1 実行開始

短い id/title では push 成功。

- kernel id: `kentookumura/exp183-copcf-train`
- URL: https://www.kaggle.com/code/kentookumura/exp183-copcf-train
- version: 1
- push result: `Kernel version 1 successfully pushed`
- initial status: `KernelWorkerStatus.RUNNING`
- initial logs: まだ出力なし
- status after ~2 min: `KernelWorkerStatus.RUNNING`
- logs after ~2 min: まだ出力なし

### Kaggle train v1 失敗

ユーザー報告後に status / logs を再確認した。

- status: `KernelWorkerStatus.ERROR`
- failure: `nbclient.exceptions.DeadKernelError: Kernel died`
- last successful progress: fold 0 の `lgb_multiclass` が best iteration 91 で完了。
- failure location: fold 0 の candidate-long prior feature 生成中。`cluster_outlier_prior_confidence_addonly_on_exp158_selector.py:1181-1219` 付近の DataFrame fragmentation warning が大量に出た後、kernel が死亡。

通常の Python 例外ではなく kernel process 死亡なので、CPU Kaggle RAM 上限による OOM と判断する。exp183 は exp157/158 の candidate-long 展開に cluster/prior/gate 特徴を多数追加しており、full fold の train/valid long frame を一括生成すると 5-6M 行 x 多数列になり過ぎる。

### Kaggle train v2 修正

同じ実験・同じ仮説のまま、OOM 対策だけを実装する。

- candidate-long feature frame は列ごとの `DataFrame.insert` をやめ、候補ごとの dict から一括構築する。
- long model の学習/eval row cap を明示:
  - `ranker.long_models.max_train_rows_per_fold: 120000`
  - `ranker.long_models.max_valid_rows_per_fold: 120000`
  - `ranker.long_models.predict_chunk_rows: 50000`
- early stopping は sampled eval long frame で行う。
- full valid OOF score は 50k row chunk ごとに long frame を作って予測し、score matrix へ詰める。
- LightGBM config 数、fold 数、booster 数は 3 x 5 = 15 のまま。

検証:

- `py_compile`: pass
- `ruff --select F821,F401,E501`: pass
- `make validate-exp`: pass

### Kaggle train v2 実行開始

```bash
make prepare-kaggle-notebooks EXP=exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp183-copcf-train --title 'exp183 copcf train' --run-on-push --strict"
make push-kaggle-train EXP=exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector
```

- kernel id: `kentookumura/exp183-copcf-train`
- URL: https://www.kaggle.com/code/kentookumura/exp183-copcf-train
- version: 2
- push result: `Kernel version 2 successfully pushed`
- initial status: `KernelWorkerStatus.RUNNING`
- initial logs: まだ出力なし

### Kaggle train v2 完了

ユーザー報告後に status / logs を確認した。

- status: `KernelWorkerStatus.COMPLETE`
- summary status: `completed_train_side_audit`
- runtime: 28,882.350 sec
- rows / wells: 3,783,989 / 773
- best viterbi: `viterbi_sw200_bias000_jw100_jf025_d0075_std999999_md0000_seg001`
- best viterbi RMSE: 10.601481774
- MAE: 6.386571251
- within10: 0.792418794
- oracle label accuracy: 0.266536716
- path switches: 5,650
- path switches / 1000 rows: 1.493133305
- default candidate rate: 0.459316346
- delta RMSE vs `likpf_mean`: -0.993415899
- delta RMSE vs exp157 row-wise: -0.194318063
- delta RMSE vs exp158 continuity: -0.187681479
- recommendation: `cluster_outlier_prior_addonly_selector_supported_for_review`

生成物 SHA:

- metrics: `c5fc041bcfa55b52580712d25762efdfd2439d7655eaffa46763923102456341`
- oof predictions: `d2a98e8212ff9fc06f46c5505f3dc870310453af989150921bc54ef42cedbf5d`
- oof predictions decompressed: `beddc97c04cdbddcd5d5756e90b66ff51dfc525c998f668a147bac540d0180a0`
- feature schema: `3b4c44e750e640066298542b70946b9a2d3733c71ba24ecc0a46d0e0f5b03ec4`
- score summary: `669b6fbb066d541442cbf91006bfd3fb578a0d6057defbee8376b087480ae515`
- subgroup metrics: `42aecfb0506e74c9b0a1376b74c02cf1635c928ffb21dc2da8abc53b11db587e`
- viterbi params: `002f5d2bf4be169842ec1e911ec545223b4d87a1ff4a586a388c18ffd801112b`

結論: exp181 cluster-outlier prior signal を direct correction ではなく selector confidence feature として使う方針は train-side で支持された。OOM 対策により long-model train/eval cap は 120k/fold になっているため、inference port / submit へ進める前に同じ exp183 内で raw-test parity、worst-well / bucket / exp115 subgroup 詳細、必要なら高メモリ再学習を確認する。

### 詳細解釈用 output 取得

結果解釈のため、Kaggle output を `/tmp/kaggle-output/exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector/train_v2` に取得した。OOF / model 本体も含まれるため、Git には保存しない。

追加確認:

- `lgb_candidate_error_ranker` row-wise は RMSE 10.640892。`likpf_mean` 11.594898 から -0.954006、best Viterbi 10.601482 から +0.039410。
- best Viterbi selection: `likpf_mean` 45.93%、`pf_ancc` 36.10%、dense family 14.02%、`beam_mean` 3.53%。
- distance bucket: `likpf_mean` 比では全 bucket 改善。row-wise 比では near `000_050` +0.018146 / `050_100` +0.007944 と微悪化、longtail `1000_plus` -0.044711 と改善。
- cluster-outlier subgroup: `copcf_gate_any_outlier_signal_k8` は `likpf_mean` RMSE 11.889295 -> 11.285943、`copcf_nearest_other_closer` は 11.335708 -> 10.855033、`copcf_nearby_majority_diff_k8` は 12.687996 -> 12.195280。
- exp115 subgroup: spatial valid 13.643808 -> 12.593127、typewell purged valid 13.506801 -> 12.479252。
- feature importance: `lgb_candidate_error_ranker` top40 中 15 個が `copcf_` features。`copcf_nearest_other_cluster_dist`、`copcf_own_cluster_dist_z`、`copcf_own_cluster_dist`、spatial/typewell prior minus candidate が上位。
- worst well は `86454a6f` RMSE 57.581365。row-wise 比の最大 regression は `7987f2f2` +1.545085 RMSE。
