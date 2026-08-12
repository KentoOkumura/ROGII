# exp158_segment_continuity_selector_on_exp157 セッションノート

## 2026-06-29 実装

ユーザーの「続けてください」を受け、exp157 の次手として `segment_continuity_selector_on_exp157` を実装する。

### 狙い

exp157 は `lgb_candidate_error_ranker` OOF RMSE 10.795800 で `likpf_mean_single` から -0.799098 改善したが、row-wise selector の path switch が大きく direct inference / submit は危険だった。今回は exp157 の 8 候補 predicted-error surface を保存済み booster から復元し、well-local Viterbi DP と minimum segment guard で candidate path continuity を強制する。

### 実装内容

- `docs/legacy/steering/20260629-exp158-segment-continuity-selector-on-exp157/` を作成。
- `experiments/exp158_segment_continuity_selector_on_exp157/` を exp155 からコピーして作成。
- 実装本体を `segment_continuity_selector_on_exp157.py` に差し替えた。
  - exp099 v2 train feature cache を読み込む。
  - exp072 dense feature cache を join し、`tvt_dense` / `tvt_densew` / `tvt_dense50` と exp157 の target-free enrichment feature を復元する。
  - exp157 feature schema / model manifest / booster を読み込む。
  - exp157 と同じ GroupKFold by `well`、同じ feature schema、同じ candidate-long sampled train rows で OOF score surface を復元する。
  - `lgb_candidate_error_ranker` の per-candidate predicted error を local cost とし、well ごとに Viterbi path を解く。
  - transition cost は switch penalty と candidate TVT jump penalty。
  - non-default candidate には `likpf_mean` からの delta cap、`pf_ancc_std` cap、`md_since` gate、minimum segment length pruning を適用する。
  - `likpf_mean_single`、`exp157_error_ranker_rowwise`、Viterbi variants、oracle を比較する。
  - metrics、保存対象 OOF predictions、selection distribution、by-well、bucket metrics、Viterbi params、score summary、summary JSON を保存する。
- `config.yaml` を no-new-model train-side posthoc audit として更新した。
- train notebook は設定、入力確認、score 復元・Viterbi 実行、結果 preview をセル単位で追える構成にする。
- inference notebook は train-side audit only と明記する。

### Kaggle GPU / runtime コスト

新規学習は行わない。追加 booster 数は 0。Kaggle train notebook は CPU posthoc audit として exp157 saved booster inference と 180 Viterbi variants の評価のみを行う。control / parent 再学習はない。

### 検証

```bash
python3 -m json.tool experiments/exp158_segment_continuity_selector_on_exp157/exp158_segment_continuity_selector_on_exp157_train.ipynb >/tmp/exp158_train_nb.jsoncheck
python3 -m json.tool experiments/exp158_segment_continuity_selector_on_exp157/exp158_segment_continuity_selector_on_exp157_inference.ipynb >/tmp/exp158_infer_nb.jsoncheck
python3 -m py_compile experiments/exp158_segment_continuity_selector_on_exp157/segment_continuity_selector_on_exp157.py experiments/exp158_segment_continuity_selector_on_exp157/settings.py
uv run ruff check experiments/exp158_segment_continuity_selector_on_exp157/segment_continuity_selector_on_exp157.py experiments/exp158_segment_continuity_selector_on_exp157/settings.py
uv run ruff format --check experiments/exp158_segment_continuity_selector_on_exp157/segment_continuity_selector_on_exp157.py experiments/exp158_segment_continuity_selector_on_exp157/settings.py
uv run python scripts/validate_experiment.py --experiment exp158_segment_continuity_selector_on_exp157
```

すべて成功。

### Kaggle package prepare

```bash
uv run python scripts/prepare_kaggle_notebooks.py \
  --experiment exp158_segment_continuity_selector_on_exp157 \
  --notebook train \
  --kernel-id kentookumura/exp158-segment-continuity-selector-on-exp157-train \
  --title 'exp158 segment continuity selector on exp157 train' \
  --run-on-push \
  --strict
```

成功。Kaggle train package:

- `experiments/exp158_segment_continuity_selector_on_exp157/kaggle/train/kernel-metadata.json`
- kernel id: `kentookumura/exp158-segment-continuity-selector-on-exp157-train`
- GPU: false
- internet: false
- competition source: `rogii-wellbore-geology-prediction`
- kernel sources:
  - `kentookumura/exp099-pf-multiobs-likelihood-train`
  - `kentookumura/exp072-exp063-full-replay-feature-cache-train`
  - `kentookumura/exp157-cand-ranker-enrich-train`

metadata と loose `config.yaml` は exp158 の設定を指している。生成 notebook の bootstrap manifest も `segment_continuity_selector_on_exp157.py` と exp158 `config.yaml` を含むことを確認した。

### Kaggle train v1 実行前確認

- 実行対象: `exp158_segment_continuity_selector_on_exp157`
- Kaggle runtime: CPU
- Viterbi variants: 180
- 新規 LightGBM booster: 0
- exp157 saved booster inference: 15 boosters
- control / parent 再学習: なし

## 2026-06-29 Kaggle train v1 running

### push

```bash
make push-kaggle-train EXP=exp158_segment_continuity_selector_on_exp157
```

成功。Kernel version 1 を push した。

- Kernel: `kentookumura/exp158-segment-continuity-selector-on-exp157-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp158-segment-continuity-selector-on-exp157-train`
- Version: 1

### 確認

```bash
kaggle kernels pull kentookumura/exp158-segment-continuity-selector-on-exp157-train -p /tmp/kaggle-pull/exp158-segment-continuity-selector-on-exp157-train -m
kaggle kernels status kentookumura/exp158-segment-continuity-selector-on-exp157-train
kaggle kernels logs kentookumura/exp158-segment-continuity-selector-on-exp157-train
```

metadata pull 成功。status は `KernelWorkerStatus.RUNNING`。通常 logs はこの時点では空。

監視はユーザーの前回指示に合わせて中止。Kaggle 側の実行は継続。

## 2026-06-30 Kaggle train v1 completed

### 状態確認

```bash
kaggle kernels status kentookumura/exp158-segment-continuity-selector-on-exp157-train
kaggle kernels logs kentookumura/exp158-segment-continuity-selector-on-exp157-train
kaggle kernels output kentookumura/exp158-segment-continuity-selector-on-exp157-train \
  -p experiments/exp158_segment_continuity_selector_on_exp157/kaggle/output/train_v1
```

status は `KernelWorkerStatus.COMPLETE`。logs と output を取得した。

### 結果

- rows / wells: 3,783,989 / 773
- runtime: 21,394.101 sec
- recommendation: `segment_viterbi_supported_for_continuity_audit`
- best Viterbi: `viterbi_sw050_bias000_jw050_jf025_d150_std999999_md0000_seg001`
- best Viterbi RMSE: 10.789163253
- best Viterbi MAE: 6.469585
- best Viterbi within10: 0.792647
- `likpf_mean_single` RMSE: 11.594897672
- delta vs `likpf_mean`: -0.805734419
- exp157 row-wise RMSE: 10.795753132
- delta vs exp157 row-wise: -0.006589879
- oracle RMSE: 4.564605

best Viterbi は exp157 row-wise をわずかに上回った。within10 も exp157 row-wise から +0.000142 改善。

### Continuity

- best Viterbi total path switches: 11,767
- exp157 row-wise total path switches: 277,110
- best Viterbi path switches / 1000 rows: 3.109681
- exp157 row-wise path switches / 1000 rows: 73.232242
- best Viterbi max well path switches / 1000 rows: 24.091920
- exp157 row-wise max well path switches / 1000 rows: 357.199056

by-well では exp157 row-wise 比で 428 wells 改善、345 wells 悪化。最大 regression は +1.906477 RMSE、最大 improvement は -3.919611 RMSE。

### Guardrail

- worst well: `86454a6f`
- best Viterbi worst well RMSE: 57.836738
- exp157 row-wise same well RMSE: 57.967201
- best Viterbi worst well within10: 0.049598
- best Viterbi worst well path switch / 1000 rows: 11.803114

global OOF と continuity は改善したが、worst well の絶対 RMSE はまだ重い。提出候補にする前に raw-test parity、hidden-like stress、worst-well guard を確認する。

### 生成物

- `kaggle/output/train_v1/artifacts/exp158_segment_continuity_selector_on_exp157_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp158_segment_continuity_selector_on_exp157_oof_predictions.csv.gz`
- `kaggle/output/train_v1/artifacts/exp158_segment_continuity_selector_on_exp157_selection_distribution.csv`
- `kaggle/output/train_v1/artifacts/exp158_segment_continuity_selector_on_exp157_by_well.csv`
- `kaggle/output/train_v1/artifacts/exp158_segment_continuity_selector_on_exp157_bucket_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp158_segment_continuity_selector_on_exp157_viterbi_params.csv`
- `kaggle/output/train_v1/artifacts/exp158_segment_continuity_selector_on_exp157_score_summary.csv`
- `kaggle/output/train_v1/artifacts/exp158_segment_continuity_selector_on_exp157_summary.json`

### SHA

- exp099 train feature decompressed SHA: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- exp072 auxiliary source decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- exp157 feature schema SHA: `891226fdf0f82c384e2fcca77f3c7d47b964d5837251ce9594249951d4e5b87c`
- exp157 model manifest SHA: `ab25fbfc0c8b92915bfbd11e62c8ffa6d84eadb3d8abf10e039927e2df7d4fb1`
- metrics SHA: `35828ea61896fae4b0cb463e7391cd62d59dc1abc6df1abd53863225fa172b2c`
- OOF predictions decompressed SHA: `7401d54395939100ca31fa131e74a16992d41936b86f3bc3f6142ed597f452a2`
- best Viterbi prediction SHA: `36f66b1547fbdac2d6bf3b3d8044d89ef0c0be0c4a6bd17e0c4874ec0f790b0f`

### 判定

exp158 は train-side continuity audit として supported。inference port / submit はまだしない。次に進む場合も新しい実験を切らず、同じ exp158 内で raw-test parity と inference candidate 化の是非を確認する。
