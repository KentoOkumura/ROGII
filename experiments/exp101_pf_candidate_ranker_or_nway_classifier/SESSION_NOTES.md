# exp101_pf_candidate_ranker_or_nway_classifier セッションノート

## 2026-06-21 実装

ユーザー依頼により `pf_candidate_ranker_or_nway_classifier` を実装する。

### 狙い

exp093 で PF/Beam 候補集合の oracle headroom は確認済みだが、target-free rank score は `pf_ancc` を選べていない。exp099 では multi-observation likelihood が oracle headroom を増やした一方で、そのまま top1 scorer にすると崩壊した。

このため exp099 v2 の wide cache を固定入力にし、5候補 (`pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`) から oracle best candidate index を選ぶ supervised ranker / N-way classifier を OOF で比較する。

### 実装内容

- `docs/legacy/steering/20260621-exp101-pf-candidate-ranker-or-nway-classifier/` を作成。
- `experiments/exp101_pf_candidate_ranker_or_nway_classifier/` を exp099 から派生作成。
- `pf_candidate_ranker_or_nway_classifier.py` を追加。
  - exp099 v2 train feature cache を読み込む。
  - true TVT は `last_known_tvt + target` として scoring / oracle label 作成にだけ使う。
  - GroupKFold by `well` で以下を比較する。
    - `likpf_mean_single`
    - `multiobs_score_top1`
    - `oracle`
    - `lgb_multiclass`
    - `lgb_candidate_binary`
    - `lgb_candidate_error_ranker`
  - metrics、OOF selected prediction、selection distribution、by-well path switch、bucket metrics、feature importance、model manifest、summary JSON を保存する。
- train notebook は設定、入力確認、実行、結果 preview をセル単位で追える構成にする。
- inference notebook は train-side audit only と明記する。

### 検証

```bash
.venv/bin/ruff check experiments/exp101_pf_candidate_ranker_or_nway_classifier
```

成功: `All checks passed!`

```bash
.venv/bin/python -m py_compile \
  experiments/exp101_pf_candidate_ranker_or_nway_classifier/pf_candidate_ranker_or_nway_classifier.py \
  experiments/exp101_pf_candidate_ranker_or_nway_classifier/settings.py
```

成功。

```bash
make validate-exp EXP=exp101_pf_candidate_ranker_or_nway_classifier
```

成功: `experiment validation passed (strict)`。

full LightGBM smoke は local `.venv` に `lightgbm` がないため停止した。

```text
ModuleNotFoundError: No module named 'lightgbm'
```

Kaggle runtime 前提の依存なので、ローカルでは入力ロード、oracle label、feature selection、candidate-long frame 生成までを dry smoke した。

```bash
.venv/bin/python - <<'PY'
# exp099 v2 cache を 20,000 rows 読み、candidate labels と long frame を作る
PY
```

結果:

```json
{"rows": 20000, "wells": 5, "features": 47, "long_rows": 5000, "long_error_mean": 20.73426628112793}
```

exp099 v2 cache の先頭 100,000 rows dry smoke では、22 wells、47 features、oracle label 分布は `pf_ancc` 33,392 / `beam_mean` 24,712 / `likpf_mean` 37,119 / `sc_ens` 2,078 / `hyb` 2,699。decompressed SHA prefix は `1939d536b1e5`。

```bash
make prepare-kaggle-notebooks EXP=exp101_pf_candidate_ranker_or_nway_classifier \
  EXTRA_ARGS="--notebook train --run-on-push --strict"
```

成功。Kaggle train package:

- `experiments/exp101_pf_candidate_ranker_or_nway_classifier/kaggle/train/kernel-metadata.json`
- kernel id: `kentookumura/exp101-pf-candidate-ranker-or-nway-classifier-train`
- GPU: false
- internet: false
- kernel source: `kentookumura/exp099-pf-multiobs-likelihood-train`

## 2026-06-21 Kaggle train v1

### 実行

最初の kernel id `kentookumura/exp101-pf-candidate-ranker-or-nway-classifier-train` は title slug mismatch / 400 で push できなかった。短い canonical id に変更して push した。

```bash
make prepare-kaggle-notebooks EXP=exp101_pf_candidate_ranker_or_nway_classifier \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp101-pf-cand-ranker-train --title 'exp101 pf cand ranker train' --run-on-push --strict"
make push-kaggle-train EXP=exp101_pf_candidate_ranker_or_nway_classifier
```

push 成功。

- Kernel: `kentookumura/exp101-pf-cand-ranker-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp101-pf-cand-ranker-train`
- version: 1
- CPU runtime (`enable_gpu=false`)
- internet: false
- kernel source: `kentookumura/exp099-pf-multiobs-likelihood-train`

### 出力取得

```bash
kaggle kernels status kentookumura/exp101-pf-cand-ranker-train
kaggle kernels logs kentookumura/exp101-pf-cand-ranker-train
kaggle kernels output kentookumura/exp101-pf-cand-ranker-train \
  -p experiments/exp101_pf_candidate_ranker_or_nway_classifier/kaggle/output/train_v1
```

status は `KernelWorkerStatus.COMPLETE`。output 取得済み。

### 結果

rows 3,783,989 / wells 773 / runtime 2,618.39 sec。

| variant | mode | RMSE | MAE | within10 | oracle label accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `oracle` | oracle | 7.434030 | 3.745228 | 0.906525 | 1.000000 |
| `likpf_mean_single` | baseline | 11.594898 | 7.067633 | 0.772807 | 0.385230 |
| `lgb_candidate_error_ranker` | oof | 11.600097 | 7.006913 | 0.771452 | 0.389880 |
| `lgb_candidate_binary` | oof | 11.814474 | 7.194226 | 0.759631 | 0.431650 |
| `lgb_multiclass` | oof | 12.106697 | 7.336443 | 0.756271 | 0.430136 |
| `multiobs_score_top1` | baseline | 89.994391 | 38.086730 | 0.523815 | 0.235524 |

best OOF は `lgb_candidate_error_ranker` だが、`likpf_mean_single` より RMSE が `+0.005199` 悪い。summary recommendation は `ranker_not_supported`。

selection distribution:

- `lgb_candidate_error_ranker`: `likpf_mean` 54.79%, `pf_ancc` 35.98%, `beam_mean` 5.25%, `hyb` 3.33%, `sc_ens` 0.65%
- oracle: `likpf_mean` 38.52%, `pf_ancc` 33.41%, `beam_mean` 21.94%, `hyb` 2.85%, `sc_ens` 3.28%

`pf_ancc` を選ぶようにはなったが、RMSE / within10 は改善しない。best OOF の最大 `path_switch_per_1000_rows` は 365.996 で、row-wise selector として切替が多い。

### 生成物

- `kaggle/output/train_v1/artifacts/exp101_pf_candidate_ranker_or_nway_classifier_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp101_pf_candidate_ranker_or_nway_classifier_summary.json`
- `kaggle/output/train_v1/artifacts/exp101_pf_candidate_ranker_or_nway_classifier_selection_distribution.csv`
- `kaggle/output/train_v1/artifacts/exp101_pf_candidate_ranker_or_nway_classifier_by_well.csv`
- `kaggle/output/train_v1/artifacts/exp101_pf_candidate_ranker_or_nway_classifier_bucket_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp101_pf_candidate_ranker_or_nway_classifier_feature_importance_mean.csv`
- `kaggle/output/train_v1/artifacts/exp101_pf_candidate_ranker_or_nway_classifier_model_manifest.json`

### SHA

- exp099 input raw SHA: `4bd9df60f5c09f7a3029dac399afef73aa45b0158a7fd06a62a56f85fd0fde38`
- exp099 input decompressed SHA: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- schema SHA: `203e4f9a280fe901f5f21d39b85c3e0e2a7fe10c466081c15015c7fb014a0413`
- OOF predictions decompressed SHA: `05cd6bc1658ab4e7c2958154bf9358582a6f1f38932ec7db8637613c00d6d09a`
- model manifest SHA: `4f453761f1cc09042767baa934f8a1c5a89036bfb1c244a5f3fc5ab0cc843cc5`

### 解釈

static supervised candidate selector は `likpf_mean` 単体を超えず、不採用。候補集合には oracle headroom があるが、row-wise supervised ranker では精度と path continuity を両立できなかった。
