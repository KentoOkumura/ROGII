# exp136_gr_shape_descriptor_verifier_on_candidate_selector セッションノート

## 2026-06-26 実装

ユーザー依頼により `gr_shape_descriptor_verifier_on_candidate_selector` を実装する。

### 狙い

exp131 の `combo_descriptor_real` は candidate-long AUC 0.659206 で GR signal を示したが、direct top1 scorer は RMSE 84.919128 まで崩壊した。今回は descriptor score を candidate replacement には使わず、`likpf_mean` default の exp101/102 low-switch selector を承認・veto する verifier として評価する。

### 実装内容

- `docs/legacy/steering/20260626-exp136-gr-shape-descriptor-verifier-on-candidate-selector/` を作成。
- `experiments/exp136_gr_shape_descriptor_verifier_on_candidate_selector/` を exp102 から派生作成。
- `gr_shape_descriptor_verifier_on_candidate_selector.py` を追加。
  - exp099 v2 train feature cache を読み込む。
  - exp101 feature schema / model manifest / booster を読み込む。
  - exp101 と同じ GroupKFold by `well`、同じ feature schema、同じ candidate-long sampled train rows で OOF score surface を復元する。
  - raw train GR と visible `TVT_input` prefix から exp131 相当の descriptor score を再計算する。
  - `likpf_mean_single`、`exp101_error_ranker_rowwise`、`oracle`、descriptor verifier variants を比較する。
  - metrics、保存対象 OOF predictions、selection distribution、by-well、bucket metrics、gate params、score summary、descriptor well summary、summary JSON を保存する。
- train notebook は設定、入力確認、実行、結果 preview をセル単位で追える構成にした。
- inference notebook は train-side audit only と明記した。

### 未実行

- Kaggle train 実行。
- output 取得と結果記録。

### 検証

```bash
.venv/bin/python -m py_compile \
  experiments/exp136_gr_shape_descriptor_verifier_on_candidate_selector/gr_shape_descriptor_verifier_on_candidate_selector.py \
  experiments/exp136_gr_shape_descriptor_verifier_on_candidate_selector/settings.py
```

成功。

```bash
.venv/bin/python -m json.tool \
  experiments/exp136_gr_shape_descriptor_verifier_on_candidate_selector/exp136_gr_shape_descriptor_verifier_on_candidate_selector_train.ipynb
```

成功。

```bash
.venv/bin/python -m json.tool \
  experiments/exp136_gr_shape_descriptor_verifier_on_candidate_selector/exp136_gr_shape_descriptor_verifier_on_candidate_selector_inference.ipynb
```

成功。

```bash
.venv/bin/ruff check experiments/exp136_gr_shape_descriptor_verifier_on_candidate_selector
```

成功: `All checks passed!`

```bash
make validate-exp EXP=exp136_gr_shape_descriptor_verifier_on_candidate_selector
```

成功: `experiment validation passed (strict)`。

```bash
make prepare-kaggle-notebooks EXP=exp136_gr_shape_descriptor_verifier_on_candidate_selector \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp136-gr-shape-verifier-train --title 'exp136 gr shape verifier train' --run-on-push --strict"
```

成功。Kaggle train package:

- `experiments/exp136_gr_shape_descriptor_verifier_on_candidate_selector/kaggle/train/kernel-metadata.json`
- kernel id: `kentookumura/exp136-gr-shape-verifier-train`
- GPU: false
- internet: false
- competition source: `rogii-wellbore-geology-prediction`
- kernel sources:
  - `kentookumura/exp099-pf-multiobs-likelihood-train`
  - `kentookumura/exp101-pf-cand-ranker-train`

### 次

1. Kaggle train notebook を push する。
2. output を取得し、summary / metrics を `result.md`、`metrics.json`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` に反映する。

## 2026-06-26 Kaggle train v1 running

### 実行

```bash
kaggle kernels push -p experiments/exp136_gr_shape_descriptor_verifier_on_candidate_selector/kaggle/train
```

成功。Kaggle train v1 を push した。

- Kernel: `kentookumura/exp136-gr-shape-verifier-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp136-gr-shape-verifier-train`
- Version: 1

### 確認

```bash
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp136-gr-shape-verifier-train
```

180 秒の follow ではログ出力なし。

```bash
kaggle kernels pull kentookumura/exp136-gr-shape-verifier-train -p /tmp/kaggle-pull/exp136-gr-shape-verifier-train -m
```

成功。Kaggle 側の kernel 存在を確認した。

```bash
kaggle kernels status kentookumura/exp136-gr-shape-verifier-train
```

`KernelWorkerStatus.RUNNING` を確認した。

### 次

ユーザー指示により監視はしない。完了連絡後に output を取得し、結果を記録する。

## 2026-06-27 Kaggle train v1 failed

### 状態

```bash
kaggle kernels status kentookumura/exp136-gr-shape-verifier-train
```

`KernelWorkerStatus.ERROR`。

### ログ取得

```bash
kaggle kernels logs kentookumura/exp136-gr-shape-verifier-train
kaggle kernels output kentookumura/exp136-gr-shape-verifier-train \
  -p experiments/exp136_gr_shape_descriptor_verifier_on_candidate_selector/kaggle/output/train_v1
```

output と log を取得した。

### 失敗内容

descriptor score 計算は 773 wells 全て完了し、exp101 score surface 復元も fold 4 まで進んだ。

```text
[descriptor] processed 750/773 wells
[fold 0] reconstruct scores train=3026251 valid=757738
[fold 1] reconstruct scores train=3027339 valid=756650
[fold 2] reconstruct scores train=3027734 valid=756255
[fold 3] reconstruct scores train=3026888 valid=757101
[fold 4] reconstruct scores train=3027744 valid=756245
Kernel died while waiting for execute reply.
nbclient.exceptions.DeadKernelError: Kernel died
```

明示的な Python 例外ではなく、fold 復元後の gate variant 評価中に kernel が死んだ。v1 config は descriptor threshold / margin / segment / cap の積で 4,000 超の variants を作り、各 variant で 3.78M rows の metrics、by-well、bucket を作るため、CPU runtime または memory が過大だったと判断する。

### 修正

v2 用に grid を縮小した。

- confidence scores: `conservative_margin`、`descriptor_joint_margin`、`descriptor_conservative_margin`
- switch caps: `0.0025`、`0.005`、`0.01`
- delta caps: `20`、`35`
- PF std caps: `20`、unbounded
- descriptor floors: `0.25`、`0.35`
- descriptor margin mins: `0.0`、`0.05`
- segment lengths: `1`、`8`、`24`

これで variants は 216 程度になり、descriptor verifier の診断目的を残したまま Kaggle runtime を抑える。

### 次

v2 package を prepare して同じ kernel id に再 push する。

## 2026-06-27 Kaggle train v2 running

### 検証

```bash
.venv/bin/ruff check experiments/exp136_gr_shape_descriptor_verifier_on_candidate_selector
make validate-exp EXP=exp136_gr_shape_descriptor_verifier_on_candidate_selector
```

どちらも成功。

### 再 prepare

```bash
make prepare-kaggle-notebooks EXP=exp136_gr_shape_descriptor_verifier_on_candidate_selector \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp136-gr-shape-verifier-train --title 'exp136 gr shape verifier train' --run-on-push --strict"
```

成功。

### push

```bash
kaggle kernels push -p experiments/exp136_gr_shape_descriptor_verifier_on_candidate_selector/kaggle/train
```

成功。Kernel version 2 を push した。

- Kernel: `kentookumura/exp136-gr-shape-verifier-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp136-gr-shape-verifier-train`
- Version: 2

### 確認

```bash
kaggle kernels status kentookumura/exp136-gr-shape-verifier-train
kaggle kernels pull kentookumura/exp136-gr-shape-verifier-train -p /tmp/kaggle-pull/exp136-gr-shape-verifier-train-v2 -m
```

`KernelWorkerStatus.RUNNING` と Kaggle 側 metadata pull 成功を確認した。

### 次

監視はしない。完了連絡後に output を取得して結果を記録する。

## 2026-06-27 Kaggle train v2 complete

### 状態確認

```bash
kaggle kernels status kentookumura/exp136-gr-shape-verifier-train
```

`KernelWorkerStatus.COMPLETE`。

### output 取得

```bash
kaggle kernels logs kentookumura/exp136-gr-shape-verifier-train
kaggle kernels output kentookumura/exp136-gr-shape-verifier-train \
  -p experiments/exp136_gr_shape_descriptor_verifier_on_candidate_selector/kaggle/output/train_v2
```

成功。生成物は `experiments/exp136_gr_shape_descriptor_verifier_on_candidate_selector/kaggle/output/train_v2/artifacts/` に保存した。

### 結果

- rows / wells: 3,783,989 / 773
- runtime: 2,574.410 sec
- variants: 327
- best RMSE gate: `gate_descriptor_joint_margin_sr010_d035_std999999_df025_dm005_seg001`
- `likpf_mean_single`: RMSE 11.594898 / MAE 7.067633 / within10 0.772807
- best gate: RMSE 11.585115 / MAE 7.057835 / within10 0.772744 / switch rate 0.010000
- delta vs `likpf_mean`: RMSE -0.009782 / within10 -0.000063
- selection: `likpf_mean` 3,746,150 rows、`pf_ancc` 31,751 rows、`beam_mean` 6,088 rows
- best balanced gate: `gate_descriptor_conservative_margin_sr005_d035_std000020_df035_dm000_seg001`
- balanced gate: RMSE 11.585504 / within10 0.772849 / delta RMSE -0.009393 / delta within10 +0.000041

Guardrail:

- wells improved / worsened: 452 / 245
- max well regression: +3.542356 RMSE
- max well improvement: -4.799888 RMSE
- distance `000_050`: RMSE -0.065584
- distance `1000_plus`: RMSE -0.010945
- tail rank `1000_plus`: RMSE -0.009783

### 生成物 SHA

- metrics SHA: `1ff417b9b15ddee8e92c02f27ddfa97a32d5d95d97e102051abdb22570aa174b`
- OOF predictions raw SHA: `34014e3be12004f13ba12da3c1d677d30e9455e147e482c7704633e0f1c2650a`
- OOF predictions decompressed SHA: `ae45cfefe2c35fafd68529c3128b55b9bf36c17e1c90a9fa9a64f9c9b83bfdb0`
- best gate prediction SHA: `2bc210c26d445fe663a87ad90a92de03751c6555b9981dd37b34a3440940b952`
- by-well SHA: `91cf1082e371749d90a3c0847c90a2f240703cdb0b3c490261bb0476d634c1fd`
- bucket metrics SHA: `3a6068b2f1deb1a2a65a21d31172b3e2c2ad4a804e90253ea7daf02517963c07`
- descriptor well summary SHA: `3674dd83ffe66fbe1082db766105351dfb30f29ab6d36c19c02c889039a2e50c`

### 解釈

descriptor verifier は低 switch で `likpf_mean` を小さく改善したが、改善幅は exp102 best gate より小さい。best RMSE gate は within10 を微減させ、最大 well regression も残るため、direct inference port / submit はしない。descriptor は hard selector ではなく、ML add-only confidence feature または diagnostic 材料に留める。
