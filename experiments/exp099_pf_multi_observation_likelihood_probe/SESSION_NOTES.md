# exp099_pf_multi_observation_likelihood_probe セッションノート

## 目的

PF/Beam/likelihood-PF 候補の target-free scorer を改善できるかを、複数 GR 観測点に基づく likelihood probe として監査する。exp093 の結論どおり、候補集合には headroom があるが rank score が弱いため、supervised ranker 前の scorer audit として扱う。

## 現在の状態

- Route: pf_beam
- 状態: completed_train_side_audit
- CV: candidate audit only
- LB: なし
- 提出: なし

## コマンドログ

### 2026-06-21 JST 実装

```bash
make new-steering EXP=exp099_pf_multi_observation_likelihood_probe
make new-exp EXP=exp099_pf_multi_observation_likelihood_probe SOURCE=experiments/exp093_pf_candidate_coverage_then_ranker_audit
```

実装内容:

- `.steering/20260621-exp099-pf-multi-observation-likelihood-probe/` を作成し、requirements / design / tasklist を記入する。
- `config.yaml` を train-side multi-observation likelihood audit 用に更新した。
- `pf_multi_observation_likelihood_probe.py` を追加し、既存候補を複数 GR 観測点で再採点する target-free likelihood と、top1 / softmax / likPF blend 診断候補を実装した。
- train notebook を、設定確認、入力前提、監査実行、出力 preview、metrics 保存のセル構成に更新した。
- inference notebook は診断専用 no-op として明記した。

### 予定

```bash
make validate-exp EXP=exp099_pf_multi_observation_likelihood_probe
make prepare-kaggle-notebooks EXP=exp099_pf_multi_observation_likelihood_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp099-pf-multiobs-likelihood-train --title 'exp099 pf multiobs likelihood train' --run-on-push --strict"
make push-kaggle-train EXP=exp099_pf_multi_observation_likelihood_probe
```

### 2026-06-21 JST validation / package

```bash
.venv/bin/python -m py_compile experiments/exp099_pf_multi_observation_likelihood_probe/pf_multi_observation_likelihood_probe.py experiments/exp099_pf_multi_observation_likelihood_probe/settings.py
make validate-exp EXP=exp099_pf_multi_observation_likelihood_probe
make prepare-kaggle-notebooks EXP=exp099_pf_multi_observation_likelihood_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp099-pf-multiobs-likelihood-train --title 'exp099 pf multiobs likelihood train' --run-on-push --strict"
.venv/bin/ruff check experiments/exp099_pf_multi_observation_likelihood_probe/pf_multi_observation_likelihood_probe.py experiments/exp099_pf_multi_observation_likelihood_probe/settings.py
```

結果:

- `py_compile`: PASS
- `validate-exp`: PASS
- `prepare-kaggle-notebooks`: PASS
- 生成先: `experiments/exp099_pf_multi_observation_likelihood_probe/kaggle/train`
- kernel id: `kentookumura/exp099-pf-multiobs-likelihood-train`
- title: `exp099 pf multiobs likelihood train`
- metadata: GPU false / internet false / run_on_push true / source `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- `ruff check`: 1 回目は行長 1 件で失敗。修正後 PASS。
- ruff 修正で helper SHA が変わったため、同じ `prepare-kaggle-notebooks` を再実行して bootstrap を更新した。
- 再生成後の support manifest では `pf_multi_observation_likelihood_probe.py` SHA は `6259ca8454806c38df03ee1500c76241a5369b73d760d052067b3ad86fd7e2db`。
- 再生成後の `validate-exp`: PASS。

### 2026-06-21 JST Kaggle train v1 push

```bash
kaggle kernels push -p experiments/exp099_pf_multi_observation_likelihood_probe/kaggle/train
kaggle kernels pull kentookumura/exp099-pf-multiobs-likelihood-train -p /tmp/kaggle-pull/exp099-pf-multiobs-likelihood-train -m
kaggle kernels logs kentookumura/exp099-pf-multiobs-likelihood-train
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp099-pf-multiobs-likelihood-train
```

結果:

- Kernel version 1 push 成功。
- URL: https://www.kaggle.com/code/kentookumura/exp099-pf-multiobs-likelihood-train
- `kaggle kernels pull ... -m` で存在確認済み。
- 初回通常 logs は warning 以外は空。
- `logs -f` も出力が返らない状態だったため、ユーザー指示によりこちら側の監視だけ停止した。Kaggle kernel 実行自体は停止していない。

### 2026-06-21 JST Kaggle train v1 output 取得

ユーザー完了連絡後に logs / output を取得した。

```bash
kaggle kernels logs kentookumura/exp099-pf-multiobs-likelihood-train
kaggle kernels output kentookumura/exp099-pf-multiobs-likelihood-train -p experiments/exp099_pf_multi_observation_likelihood_probe/kaggle/output/train_v1
```

結果:

- status: `completed_train_side_audit`
- rows: 3,783,989
- wells: 773
- runtime: 1,782.42 sec
- output: `experiments/exp099_pf_multi_observation_likelihood_probe/kaggle/output/train_v1`
- best single candidate: `likpf_mean` RMSE 11.594897 / within10 0.772807
- baseline primary oracle: RMSE 7.434030 / within10 0.906525
- baseline+multiobs oracle: RMSE 6.897510 / within10 0.922941 / selected multiobs rate 0.175083
- oracle gain from multiobs: RMSE -0.536520 / within10 +0.016415
- target-free rank score top1: RMSE 89.994392 / within10 0.523815、selected top は `beam_mean`
- multiobs single / blend candidates は単体では弱い。`likpf_multiobs_blend_w0p25` RMSE 25.110830、`multiobs_top1` RMSE 89.994392。

解釈:

- multiobs 候補は oracle headroom を増やすため、ranker feature / learned likelihood feature の材料としては残す。
- 現行 target-free rank score は `beam_mean` 偏重で壊れているため、直接 scorer / direct replacement / softmax blend としては不採用。

## 生成ファイル

- `exp099_pf_multi_observation_likelihood_probe_candidate_metrics.csv`
- `exp099_pf_multi_observation_likelihood_probe_rank_metrics.csv`
- `exp099_pf_multi_observation_likelihood_probe_bucket_metrics.csv`
- `exp099_pf_multi_observation_likelihood_probe_by_well.csv`
- `exp099_pf_multi_observation_likelihood_probe_multiobs_well_summary.csv`
- `exp099_pf_multi_observation_likelihood_probe_row_context.csv.gz`
- `exp099_pf_multi_observation_likelihood_probe_candidate_long.csv.gz`
- `exp099_pf_multi_observation_likelihood_probe_feature_schema.csv`
- `exp099_pf_multi_observation_likelihood_probe_summary.json`

### 2026-06-21 JST exp072-style train feature cache notebook 追加

ユーザー確認で、v1 output は exp072 と同じ wide feature cache 形式ではなく、`candidate_long.csv.gz` / `row_context.csv.gz` / 診断 schema だけであることを確認した。
下流 ranker が exp072 cache と同じ形で読めるように、train notebook 実行時に次を追加保存するよう更新した。

- `exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz`
- `exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_feature_schema.csv`

形式:

- train features: `id`, `well`, `target` をメタ列にし、source columns、既存 candidate absolute TVT、multiobs score / MAE / NCC / generated candidate を float32 feature として保存する。
- feature schema: exp072 と同じ `variant`, `feature_index`, `feature`。
- variant: `multiobs_likelihood_probe`

実行した確認:

```bash
python3 -m py_compile experiments/exp099_pf_multi_observation_likelihood_probe/pf_multi_observation_likelihood_probe.py
python3 -m json.tool experiments/exp099_pf_multi_observation_likelihood_probe/exp099_pf_multi_observation_likelihood_probe_train.ipynb >/tmp/exp099_train_nb.json
uv run ruff check experiments/exp099_pf_multi_observation_likelihood_probe/pf_multi_observation_likelihood_probe.py experiments/exp099_pf_multi_observation_likelihood_probe/settings.py
make validate-exp EXP=exp099_pf_multi_observation_likelihood_probe
make prepare-kaggle-notebooks EXP=exp099_pf_multi_observation_likelihood_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp099-pf-multiobs-likelihood-train --title 'exp099 pf multiobs likelihood train' --run-on-push --strict"
```

結果:

- py_compile: PASS
- notebook JSON: PASS
- ruff: PASS
- validate-exp: PASS
- Kaggle train package: `experiments/exp099_pf_multi_observation_likelihood_probe/kaggle/train` を再生成済み
- この後の v2 push で、上記 2 ファイルを Kaggle output として確認する方針にした。

### 2026-06-21 JST Kaggle train v2 push

cache 追加版を同じ canonical kernel id へ version 2 として push した。

```bash
kaggle kernels push -p experiments/exp099_pf_multi_observation_likelihood_probe/kaggle/train
kaggle kernels pull kentookumura/exp099-pf-multiobs-likelihood-train -p /tmp/kaggle-pull/exp099-pf-multiobs-likelihood-train -m
kaggle kernels logs kentookumura/exp099-pf-multiobs-likelihood-train
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp099-pf-multiobs-likelihood-train
kaggle kernels status kentookumura/exp099-pf-multiobs-likelihood-train
```

結果:

- Kernel version 2 push 成功。
- URL: https://www.kaggle.com/code/kentookumura/exp099-pf-multiobs-likelihood-train
- `kaggle kernels pull ... -m` 成功。
- 通常 logs と 180 秒 `logs -f` は無出力。v1 と同じく API log が空で返っている可能性がある。
- `kaggle kernels status` は `KernelWorkerStatus.RUNNING`。
- 完了後に output を取得し、`exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz` と schema の存在、SHA、行数、feature_count を確認する。

### 2026-06-21 JST Kaggle train v2 output 取得

ユーザー完了連絡後に logs / output を取得した。

```bash
kaggle kernels logs kentookumura/exp099-pf-multiobs-likelihood-train
kaggle kernels output kentookumura/exp099-pf-multiobs-likelihood-train -p experiments/exp099_pf_multi_observation_likelihood_probe/kaggle/output/train_v2
```

結果:

- status: `completed_train_side_audit`
- runtime: 2,531.32 sec
- rows: 3,783,989
- wells: 773
- output: `experiments/exp099_pf_multi_observation_likelihood_probe/kaggle/output/train_v2`
- exp072-style train cache: `artifacts/exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz`
- cache size: 563,047,217 bytes
- schema: `artifacts/exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_feature_schema.csv`
- feature_count: 40
- format: `id`, `well`, `target` + 40 feature columns
- feature schema columns: `variant`, `feature_index`, `feature`
- variant: `multiobs_likelihood_probe`
- raw SHA256: `4bd9df60f5c09f7a3029dac399afef73aa45b0158a7fd06a62a56f85fd0fde38`
- decompressed SHA256: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- schema SHA256: `203e4f9a280fe901f5f21d39b85c3e0e2a7fe10c466081c15015c7fb014a0413`

ローカル検査:

```bash
python3 - <<'PY'
# gzip stream で header / schema / SHA / line count を確認
PY
```

- header columns: 43
- meta columns: `id`, `well`, `target`
- data rows: 3,783,989
- line count including header: 3,783,990
- schema rows: 40
- local raw / decompressed / schema SHA は summary JSON と一致。

exp072 より実行時間が短い理由:

- exp072 は PF/Beam/likelihood-PF replay feature cache 自体を生成する実験。
- exp099 v2 は exp072 の生成済み train cache を Kaggle input から読み、raw train GR と既存候補を使って multiobs likelihood features を再計算・wide 保存するだけ。
- そのため、exp072 より短くても不自然ではない。

## 再現性メモ

- seed policy: exp099 内では新規乱数なし。
- stochastic components: upstream exp072 PF/Beam cache のみ。exp099 では再生成しない。
- CPU/GPU runtime: CPU train-side audit。GPU 不要。
- feature content SHA: source cache の gzip decompressed content SHA を主証拠にする。
- model manifest / model SHA: model なし。
- prediction SHA: prediction なし。
- submission SHA: submission なし。
- rerun check: deterministic submission anchor ではないため対象外。

## 次のアクション

1. `pf_candidate_ranker_or_nway_classifier` で multiobs score / MAE / NCC を feature として追加し、候補選択が改善するか確認する。
2. `learned_pf_observation_likelihood_probe` は、この hand-crafted likelihood を supervised scorer が上回れるかを見る follow-up として残す。
