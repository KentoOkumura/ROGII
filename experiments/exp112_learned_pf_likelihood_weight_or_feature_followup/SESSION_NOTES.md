# exp112_learned_pf_likelihood_weight_or_feature_followup セッションノート

## 2026-06-22 実装

ユーザー依頼により、`learned_pf_likelihood_weight_or_feature_followup` を実装する。

### 狙い

exp111 は learned within10 probability の candidate-level AUC が 0.913327 で、exp099 hand-crafted `multiobs_score` AUC 0.617355 を大きく上回った。一方で learned top1 は direct replacement として within10 が弱い。今回は learned likelihood を直接候補選択に使わず、PF weight / verifier gate / ML add-only feature の材料として評価する。

### 実装内容

- `docs/legacy/steering/20260622-exp112-learned-pf-likelihood-weight-or-feature-followup/` を作成。
- `experiments/exp112_learned_pf_likelihood_weight_or_feature_followup/` を exp111 から派生作成。
- `learned_pf_likelihood_weight_or_feature_followup.py` を追加。
  - exp111 OOF likelihood long cache を読み込む。
  - exp099 v2 wide cache から true TVT を評価用に復元する。
  - `likpf_mean_single`、learned top1、multiobs top1、PF weight alpha、conservative verifier gate、oracle を同一 surface で評価する。
  - target-free ML feature cache を出力する。true TVT、abs error、within labels は feature artifact に含めない。
  - metrics、by-well、bucket、selection distribution、OOF prediction、feature schema、summary JSON を保存する。
- train notebook は設定、入力 artifact、実行、結果 preview をセル単位で追える構成にする。
- inference notebook は train-side audit only と明記する。

### 検証

```bash
.venv/bin/python -m py_compile \
  experiments/exp112_learned_pf_likelihood_weight_or_feature_followup/learned_pf_likelihood_weight_or_feature_followup.py \
  experiments/exp112_learned_pf_likelihood_weight_or_feature_followup/settings.py
```

成功。

```bash
.venv/bin/ruff check experiments/exp112_learned_pf_likelihood_weight_or_feature_followup
```

成功: `All checks passed!`

```bash
.venv/bin/python experiments/exp112_learned_pf_likelihood_weight_or_feature_followup/learned_pf_likelihood_weight_or_feature_followup.py \
  --output-dir /tmp/exp112_smoke \
  --max-groups 1000
```

成功。exp111 long cache と exp099 wide cache の突き合わせ、metrics / by-well / bucket / selection distribution / OOF prediction / ML feature cache / summary JSON の保存まで確認した。debug smoke は rows 1000 / candidate rows 5000 / wells 1 のため、正式な結果としては扱わない。

```bash
make validate-exp EXP=exp112_learned_pf_likelihood_weight_or_feature_followup
```

成功: `experiment validation passed (strict)`。

Kaggle train package 生成:

```bash
make prepare-kaggle-notebooks EXP=exp112_learned_pf_likelihood_weight_or_feature_followup \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp112-learned-pf-likelihood-followup-train --title 'exp112 learned pf likelihood followup train' --run-on-push --strict"
```

成功。metadata:

- kernel id: `kentookumura/exp112-learned-pf-likelihood-followup-train`
- GPU: false
- internet: false
- run_on_push: true
- kernel sources:
  - `kentookumura/exp099-pf-multiobs-likelihood-train`
  - `kentookumura/exp111-learned-pf-likelihood-train`

Kaggle inference package 生成:

```bash
make prepare-kaggle-notebooks EXP=exp112_learned_pf_likelihood_weight_or_feature_followup \
  EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp112-learned-pf-likelihood-followup-infer --title 'exp112 learned pf likelihood followup inference' --strict"
```

成功。inference は `run_on_push=false`、submission は生成しない。

## 2026-06-22 Kaggle train v1

### 実行

```bash
make push-kaggle-train EXP=exp112_learned_pf_likelihood_weight_or_feature_followup
```

push 成功。

- Kernel: `kentookumura/exp112-learned-pf-likelihood-followup-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp112-learned-pf-likelihood-followup-train`
- version: 1
- CPU runtime (`enable_gpu=false`)
- internet: false
- kernel sources:
  - `kentookumura/exp099-pf-multiobs-likelihood-train`
  - `kentookumura/exp111-learned-pf-likelihood-train`

最初の `logs -f` は 180 秒 timeout で空だった。`kaggle kernels pull -m` で kernel 存在を確認し、`kaggle kernels status` は `KernelWorkerStatus.RUNNING` を返した。同じ kernel id のまま追加で `logs -f` し、完了ログを取得した。

### 出力取得

```bash
kaggle kernels output kentookumura/exp112-learned-pf-likelihood-followup-train \
  -p experiments/exp112_learned_pf_likelihood_weight_or_feature_followup/kaggle/output/train_v1
```

output 取得済み。Kaggle output に含まれた `__pycache__` は記録対象外として削除した。

### 結果

rows 757,738 / candidate rows 3,788,690 / wells 155 / runtime 310.78 sec。

| variant | RMSE | MAE | within10 | switch vs likPF |
| --- | ---: | ---: | ---: | ---: |
| `likpf_mean_single` | 11.604410 | 6.944251 | 0.784312 | 0.000000 |
| `learned_error_top1` | 11.579703 | 6.915842 | 0.781725 | 0.596719 |
| `learned_prob_top1` | 11.600926 | 6.968520 | 0.780423 | 0.477624 |
| `gate_expected_error_m2p0_d20p0` | 11.573266 | 6.926626 | 0.785064 | 0.004077 |
| `oracle_candidate` | 7.857730 | 3.852781 | 0.908454 | 0.587608 |

best non-oracle は `gate_expected_error_m2p0_d20p0`。`likpf_mean_single` から RMSE -0.031144、within10 +0.000752、MAE -0.017625 改善した。switch rate は 0.4077%。

PF weight alpha は全て悪化。best の `pf_weight_expected_error_alpha_0p4` でも RMSE 69.484358 / within10 0.584181 で不採用。

### 生成物

- `kaggle/output/train_v1/artifacts/exp112_learned_pf_likelihood_weight_or_feature_followup_summary.json`
- `kaggle/output/train_v1/artifacts/exp112_learned_pf_likelihood_weight_or_feature_followup_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp112_learned_pf_likelihood_weight_or_feature_followup_by_well.csv`
- `kaggle/output/train_v1/artifacts/exp112_learned_pf_likelihood_weight_or_feature_followup_bucket_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp112_learned_pf_likelihood_weight_or_feature_followup_selection_distribution.csv`
- `kaggle/output/train_v1/artifacts/exp112_learned_pf_likelihood_weight_or_feature_followup_oof_predictions.csv.gz`
- `kaggle/output/train_v1/artifacts/exp112_learned_pf_likelihood_weight_or_feature_followup_ml_features.csv.gz`
- `kaggle/output/train_v1/artifacts/exp112_learned_pf_likelihood_weight_or_feature_followup_feature_schema.csv`

### SHA

- exp111 OOF likelihood long decompressed SHA: `3aa5e72e982417012a18f4172df1a233ef0f609cf91d48fb1250fc74fa9e89f8`
- exp099 wide cache decompressed SHA: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- metrics SHA: `ac4068b86f4a1f43a5fb3e27c58692419ffd520d1b5570cd38256e8b993757ef`
- OOF predictions decompressed SHA: `e3df222a2bb11432f2474f51d672d6d961eb598c9a17e4905c41017e66bb15d7`
- ML features decompressed SHA: `56c0f62238abfc89f05e5700341344c15815bd3a5f93e5b0a6a079a661b9411e`
- prediction SHA: `642426a934e967310062411cdda05449291dd181a80c33965992d9caa90adfd9`

### 解釈

learned likelihood を PF weight alpha に足す方針は崩壊し、不採用。expected-error margin を使った low-switch verifier gate は train-side で小幅に支持。ただし raw-test parity と worst-well guard がないため提出候補にはしない。

ML feature cache は target-free columns のみで保存済み。次は exp092 系 ML add-only feature、または exp102/exp112 の low-switch gate をまとめた raw-test parity 診断に回す。
