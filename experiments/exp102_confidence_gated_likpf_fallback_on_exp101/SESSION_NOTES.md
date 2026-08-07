# exp102_confidence_gated_likpf_fallback_on_exp101 セッションノート

## 2026-06-21 実装

ユーザー依頼により `confidence_gated_likpf_fallback_on_exp101` を実装する。

### 狙い

exp101 は `lgb_candidate_error_ranker` が `pf_ancc` を 35.98% 選べるようになったが、RMSE は `likpf_mean` 単体より +0.005199 悪く、path switch も多かった。このため row-wise selector 全体を使わず、`likpf_mean` default のまま高信頼行だけ `pf_ancc` / `beam_mean` へ切り替える conservative gate を監査する。

### 実装内容

- `.steering/20260621-exp102-confidence-gated-likpf-fallback-on-exp101/` を作成。
- `experiments/exp102_confidence_gated_likpf_fallback_on_exp101/` を exp101 から派生作成。
- `confidence_gated_likpf_fallback_on_exp101.py` を追加。
  - exp099 v2 train feature cache を読み込む。
  - exp101 feature schema / model manifest / booster を読み込む。
  - exp101 と同じ GroupKFold by `well`、同じ feature schema、同じ candidate-long sampled train rows で OOF score surface を復元する。
  - `likpf_mean_single`、`exp101_error_ranker_rowwise`、`oracle`、confidence-gated variants を比較する。
  - metrics、保存対象 OOF predictions、selection distribution、by-well、bucket metrics、score summary、summary JSON を保存する。
- train notebook は設定、入力確認、実行、結果 preview をセル単位で追える構成にした。
- inference notebook は train-side audit only と明記した。

### 未実行

- Kaggle train 実行。
- output 取得と結果記録。

### 検証

```bash
.venv/bin/python -m py_compile \
  experiments/exp102_confidence_gated_likpf_fallback_on_exp101/confidence_gated_likpf_fallback_on_exp101.py \
  experiments/exp102_confidence_gated_likpf_fallback_on_exp101/settings.py
```

成功。

```bash
.venv/bin/ruff check experiments/exp102_confidence_gated_likpf_fallback_on_exp101
```

成功: `All checks passed!`

```bash
make validate-exp EXP=exp102_confidence_gated_likpf_fallback_on_exp101
```

成功: `experiment validation passed (strict)`。

ローカル `.venv` には `lightgbm` がないため、保存済み booster のロード smoke は未実行。Kaggle runtime 前提で確認する。

```text
ModuleNotFoundError: No module named 'lightgbm'
```

```bash
make prepare-kaggle-notebooks EXP=exp102_confidence_gated_likpf_fallback_on_exp101 \
  EXTRA_ARGS="--notebook train --run-on-push --strict"
```

成功。Kaggle train package:

- `experiments/exp102_confidence_gated_likpf_fallback_on_exp101/kaggle/train/kernel-metadata.json`
- kernel id: `kentookumura/exp102-confidence-gated-likpf-fallback-on-exp101-train`
- GPU: false
- internet: false
- kernel sources:
  - `kentookumura/exp099-pf-multiobs-likelihood-train`
  - `kentookumura/exp101-pf-cand-ranker-train`

### 次

1. Kaggle train を実行する。
2. output を取得し、summary / metrics を `result.md`、`metrics.json`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` に反映する。

## 2026-06-21 Kaggle train v1 failed

### 実行

最初の長い kernel id は title slug mismatch / 400 で push できなかったため、短い canonical id で再作成して push した。

```bash
make prepare-kaggle-notebooks EXP=exp102_confidence_gated_likpf_fallback_on_exp101 \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp102-likpf-fallback-train --title 'exp102 likpf fallback train' --run-on-push --strict"
make push-kaggle-train EXP=exp102_confidence_gated_likpf_fallback_on_exp101
```

- Kernel: `kentookumura/exp102-likpf-fallback-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp102-likpf-fallback-train`
- version: 1
- status: `KernelWorkerStatus.ERROR`

### 失敗原因

fold 0-4 の exp101 score surface 復元は完了したが、bucket metrics 作成で失敗した。

```text
AttributeError: 'Series' object has no attribute 'codes'
```

`pd.cut` が Series を返すケースで `.codes` を直接読んでいたため。Kaggle runtime の pandas では Series に対して `.cat.codes` を使う必要がある。

### 修正

- `categorical_codes()` helper を追加し、`pd.Series` なら `.cat.codes`、`pd.Categorical` なら `.codes` を読むように変更。
- by-well path switch count で well 境界を switch として数えない guard を追加。

### 修正後検証

```bash
.venv/bin/python -m py_compile \
  experiments/exp102_confidence_gated_likpf_fallback_on_exp101/confidence_gated_likpf_fallback_on_exp101.py \
  experiments/exp102_confidence_gated_likpf_fallback_on_exp101/settings.py
.venv/bin/ruff check experiments/exp102_confidence_gated_likpf_fallback_on_exp101
make validate-exp EXP=exp102_confidence_gated_likpf_fallback_on_exp101
```

すべて成功。

### Kaggle train v2

```bash
make prepare-kaggle-notebooks EXP=exp102_confidence_gated_likpf_fallback_on_exp101 \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp102-likpf-fallback-train --title 'exp102 likpf fallback train' --run-on-push --strict"
make push-kaggle-train EXP=exp102_confidence_gated_likpf_fallback_on_exp101
```

push 成功。

- Kernel: `kentookumura/exp102-likpf-fallback-train`
- version: 2
- status: `KernelWorkerStatus.COMPLETE`
- output: `kaggle/output/train_v2`

### 結果

runtime 874.76 sec / rows 3,783,989 / wells 773。

| variant | mode | RMSE | MAE | within10 | oracle label accuracy | switch rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `oracle` | oracle | 7.434030 | 3.745228 | 0.906525 | 1.000000 | 0.614770 |
| `gate_error_margin_sr050_d020_std000020` | gated | 11.561206 | 7.053293 | 0.771966 | 0.385498 | 0.050000 |
| `likpf_mean_single` | baseline | 11.594898 | 7.067633 | 0.772807 | 0.385230 | 0.000000 |
| `exp101_error_ranker_rowwise` | oof | 11.600097 | 7.006912 | 0.771452 | 0.389880 | 0.452061 |

best gate は `gate_error_margin_sr050_d020_std000020`。`likpf_mean_single` から RMSE `-0.033692`、MAE `-0.014340` 改善したが、within10 は `-0.000842` 悪化した。switch は 189,199 rows / 5.0%。選択分布は `pf_ancc` 149,880 rows / 3.961%、`beam_mean` 39,319 rows / 1.039%、`likpf_mean` 3,594,790 rows / 95.000%。

bucket では distance bucket 全域の RMSE は改善したが、`1000_plus` within10 は `-0.001108`、eval_len q1 / q3 / q4 と likpf_delta q1 / q2 / q4 でも within10 が小さく悪化した。worst wells は `likpf_mean` と同じ well が上位に残り、改善と悪化が混在する。

### 生成物

- `kaggle/output/train_v2/artifacts/exp102_confidence_gated_likpf_fallback_on_exp101_metrics.csv`
- `kaggle/output/train_v2/artifacts/exp102_confidence_gated_likpf_fallback_on_exp101_summary.json`
- `kaggle/output/train_v2/artifacts/exp102_confidence_gated_likpf_fallback_on_exp101_selection_distribution.csv`
- `kaggle/output/train_v2/artifacts/exp102_confidence_gated_likpf_fallback_on_exp101_by_well.csv`
- `kaggle/output/train_v2/artifacts/exp102_confidence_gated_likpf_fallback_on_exp101_bucket_metrics.csv`
- `kaggle/output/train_v2/artifacts/exp102_confidence_gated_likpf_fallback_on_exp101_oof_predictions.csv.gz`

### SHA

- exp099 input raw SHA: `4bd9df60f5c09f7a3029dac399afef73aa45b0158a7fd06a62a56f85fd0fde38`
- exp099 input decompressed SHA: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- exp099 schema SHA: `203e4f9a280fe901f5f21d39b85c3e0e2a7fe10c466081c15015c7fb014a0413`
- exp101 feature schema SHA: `ea2819375dd025448c3e294b56fd92179b8b261e22f5a7fb37fbf3e8ddfac9c6`
- exp101 model manifest SHA: `4f453761f1cc09042767baa934f8a1c5a89036bfb1c244a5f3fc5ab0cc843cc5`
- metrics SHA: `e0d540e4e28c3ef43b62a29aeee7e9aca83f6fd49a6e083d7f3f2299a61904a0`
- OOF predictions raw SHA: `5517f938f14117ee802e85271ae3be66f771e8ea0cb766009c69278bd0bd9d47`
- OOF predictions decompressed SHA: `469e9fa137ffa7f3924711dbe1bb8f67f99d97678442c3d209426031a5f48330`
- best gate prediction SHA: `5181ebf6118fde1f78a5be8ea591fc8d9b05b8c15fa247d7e7bb711b12de8c79`

### 解釈

confidence-gated fallback は row-wise exp101 selector の崩れを抑え、`likpf_mean` default から小さく RMSE を改善した。ただし within10 がわずかに悪化し、worst-well guard も十分ではない。train-side では次の continuity / raw-test parity audit に進む価値はあるが、このまま inference port / submit する根拠はない。
