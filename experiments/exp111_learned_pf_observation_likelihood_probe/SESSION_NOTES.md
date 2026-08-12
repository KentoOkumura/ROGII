# exp111_learned_pf_observation_likelihood_probe セッションノート

## 2026-06-22 実装

ユーザー依頼により、`pf_candidate_generation_likelihood_upgrade` の初手として learned PF observation likelihood smoke を実装する。

### 狙い

前段の確認では「likelihood 改善候補を 5 つ全部一度に入れる」のではなく、最初は `GR matching robust 化 + candidate-long likelihood calibration` に絞る方針とした。

exp099 は multi-observation likelihood で oracle headroom を増やしたが、`multiobs_top1` は崩壊した。exp101 は supervised selector としては `likpf_mean` 単体を超えなかった。そこで今回は selector ではなく、候補ごとの calibrated likelihood / confidence を作る。

### 実装内容

- `docs/legacy/steering/20260622-exp111-learned-pf-observation-likelihood-probe/` を作成。
- `experiments/exp111_learned_pf_observation_likelihood_probe/` を exp101 から派生作成。
- `learned_pf_observation_likelihood_probe.py` を追加。
  - exp099 v2 train feature cache を読み込む。
  - 5候補 (`pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`) を candidate-long に展開する。
  - candidate-specific `multiobs_score` / `multiobs_mae` / `multiobs_ncc`、row 内 gap/rank、candidate disagreement、prefix context を特徴量にする。
  - GroupKFold by `well` の first fold smoke で `within_10ft` classifier と expected error regressor を学習する。
  - metrics、topK coverage、calibration、bucket metrics、OOF likelihood long cache、feature importance、model manifest、summary JSON を保存する。
- train notebook は設定、入力 artifact、実行、結果 preview をセル単位で追える構成にする。
- inference notebook は train-side audit only と明記する。

### 未実行

Kaggle train smoke は実行済み。

### 検証

```bash
.venv/bin/python -m py_compile \
  experiments/exp111_learned_pf_observation_likelihood_probe/learned_pf_observation_likelihood_probe.py \
  experiments/exp111_learned_pf_observation_likelihood_probe/settings.py
```

成功。

```bash
.venv/bin/ruff check experiments/exp111_learned_pf_observation_likelihood_probe
```

成功: `All checks passed!`

```bash
make validate-exp EXP=exp111_learned_pf_observation_likelihood_probe
```

成功: `experiment validation passed (strict)`。

LightGBM 学習は Kaggle runtime 前提なので、ローカルでは exp099 v2 cache 読み込みと candidate-long 展開だけ dry smoke した。

```text
{'rows': 20000, 'wells': 5, 'required_columns': 42, 'row_features': 32, 'long_rows': 5000, 'long_columns': 54, 'within10_rate': 0.7496, 'source_sha_prefix': '1939d536b1e5'}
```

Kaggle train package 生成:

```bash
make prepare-kaggle-notebooks EXP=exp111_learned_pf_observation_likelihood_probe \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp111-learned-pf-likelihood-train --title 'exp111 learned pf likelihood train' --run-on-push --strict"
```

成功。metadata:

- kernel id: `kentookumura/exp111-learned-pf-likelihood-train`
- GPU: false
- internet: false
- kernel source: `kentookumura/exp099-pf-multiobs-likelihood-train`

## 2026-06-22 Kaggle train v1

### 実行

```bash
make push-kaggle-train EXP=exp111_learned_pf_observation_likelihood_probe
```

push 成功。

- Kernel: `kentookumura/exp111-learned-pf-likelihood-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp111-learned-pf-likelihood-train`
- version: 1
- CPU runtime (`enable_gpu=false`)
- internet: false
- kernel source: `kentookumura/exp099-pf-multiobs-likelihood-train`

`logs -f` は実行中に空で timeout したが、同じ kernel id のまま待機した。補助 status では `RUNNING` から `COMPLETE` へ遷移した。通常 `logs` と `output` は完了後に取得できた。

### 出力取得

```bash
kaggle kernels logs kentookumura/exp111-learned-pf-likelihood-train
kaggle kernels output kentookumura/exp111-learned-pf-likelihood-train \
  -p experiments/exp111_learned_pf_observation_likelihood_probe/kaggle/output/train_v1
```

status は `KernelWorkerStatus.COMPLETE`。output 取得済み。

### 結果

rows 3,783,989 / candidate rows 3,788,690 / wells 773 / runtime 402.23 sec。

| variant | AUC | logloss | brier | observed within10 |
| --- | ---: | ---: | ---: | ---: |
| `learned_within10_probability` | 0.913327 | 0.356157 | 0.115817 | 0.470412 |
| `baseline_multiobs_score` | 0.617355 | - | - | 0.470412 |
| `baseline_multiobs_ncc` | 0.500422 | - | - | 0.470412 |
| `baseline_negative_multiobs_mae` | 0.652704 | - | - | 0.470412 |

learned likelihood は exp099 hand-crafted `multiobs_score` より AUC +0.295972。summary recommendation は `likelihood_supported_for_pf_weight_or_feature_followup`。

diagnostic top1:

| variant | RMSE | MAE | within10 | pf_ancc selection |
| --- | ---: | ---: | ---: | ---: |
| `likpf_mean_single` | 11.604410 | 6.944251 | 0.784312 | 0.000000 |
| `learned_prob_top1` | 11.600926 | 6.968520 | 0.780423 | 0.323032 |
| `learned_error_top1` | 11.579703 | 6.915842 | 0.781725 | 0.469776 |
| `multiobs_score_top1` | 84.205911 | 35.776010 | 0.530026 | 0.210682 |

top1 replacement は within10 が悪いため不採用。topK coverage は learned likelihood が改善し、`learned_prob_top3` within10 coverage 0.892811、`learned_error_top3` 0.892955、`multiobs_score_top3` 0.832037。

### 生成物

- `kaggle/output/train_v1/artifacts/exp111_learned_pf_observation_likelihood_probe_summary.json`
- `kaggle/output/train_v1/artifacts/exp111_learned_pf_observation_likelihood_probe_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp111_learned_pf_observation_likelihood_probe_topk_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp111_learned_pf_observation_likelihood_probe_calibration.csv`
- `kaggle/output/train_v1/artifacts/exp111_learned_pf_observation_likelihood_probe_bucket_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp111_learned_pf_observation_likelihood_probe_oof_likelihood_long.csv.gz`
- `kaggle/output/train_v1/artifacts/exp111_learned_pf_observation_likelihood_probe_feature_importance_mean.csv`
- `kaggle/output/train_v1/artifacts/exp111_learned_pf_observation_likelihood_probe_model_manifest.json`

### SHA

- exp099 input raw SHA: `4bd9df60f5c09f7a3029dac399afef73aa45b0158a7fd06a62a56f85fd0fde38`
- exp099 input decompressed SHA: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- schema SHA: `203e4f9a280fe901f5f21d39b85c3e0e2a7fe10c466081c15015c7fb014a0413`
- OOF likelihood decompressed SHA: `3aa5e72e982417012a18f4172df1a233ef0f609cf91d48fb1250fc74fa9e89f8`
- model manifest SHA: `178e8b3124b817a2b230080fc041aaaee1b06941e5a4223a68cc31bf26e68010`
- OOF probability SHA: `f4fb66ffd42de8c8ab07c0bdfe1d935ca89a1b6df216b68e701495ec671cc7f3`

### 解釈

candidate-level likelihood calibration は支持。次は direct replacement ではなく、PF weight への弱い加算、topK verifier、または exp092 系 ML add-only confidence feature として検証する。
