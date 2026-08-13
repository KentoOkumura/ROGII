# exp096_projection_fadein_after_prefix セッションノート

## 目的

`projection_fadein_after_prefix` backlog を実装する。exp094 の global projection-only audit で確認した projection correction を、known prefix 直後では 0 にし、`md_since` とともに線形 fade-in して near-prefix 悪化を抑えられるか検証する。

## 現在の状態

- Route: ml_model
- 状態: submitted_public_lb_recorded
- CV: 9.397537231
- LB: Public 8.651 / Private 未確定
- inference: Kaggle v2 完了。submit-check PASS。ref `53896594` 提出済み。

## コマンドログ

### 2026-06-21 実装

```bash
make new-steering EXP=exp096_projection_fadein_after_prefix
make new-exp EXP=exp096_projection_fadein_after_prefix SOURCE=experiments/exp094_projection_only_on_exp073
.venv/bin/python -m py_compile experiments/exp096_projection_fadein_after_prefix/projection_fadein_after_prefix.py experiments/exp096_projection_fadein_after_prefix/settings.py
make validate-exp EXP=exp096_projection_fadein_after_prefix
make prepare-kaggle-notebooks EXP=exp096_projection_fadein_after_prefix EXTRA_ARGS="--notebook train --run-on-push --strict"
make prepare-kaggle-notebooks EXP=exp096_projection_fadein_after_prefix EXTRA_ARGS="--notebook inference --run-on-push --strict"
```

実装した内容:

- `docs/legacy/steering/20260621-exp096-projection-fadein-after-prefix/` を作成し、requirements/design/tasklist を記入。
- `config.yaml` に exp094 follow-up として fade-in projection grid を記入。
- `projection_fadein_after_prefix.py` に row-wise beta fade-in を実装。
- train/inference notebook 名と import を exp096 用に更新。
- README/result/metrics を未実行状態へ更新。
- `make validate-exp` は strict で PASS。
- Kaggle train package は `experiments/exp096_projection_fadein_after_prefix/kaggle/train` に生成済み。
- Kaggle inference package は `experiments/exp096_projection_fadein_after_prefix/kaggle/inference` に生成済み。`inference.selected_variant` は null のため submission は作らない。

確認した内容:

- synthetic frame で `md_since=[100,250,500,1000]` / `fade 250-750` / `beta 0.75` の effective beta が `[0.0, 0.0, 0.375, 0.75]` になることを確認。
- train metadata: `enable_gpu=false`、`enable_internet=false`、`kernel_sources=["kentookumura/exp073-full-replay-repro-guard-train"]`。
- inference metadata: `enable_gpu=false`、`enable_internet=false`、`kernel_sources=["kentookumura/exp073-full-replay-repro-guard-infer"]`。

### 2026-06-21 Kaggle train v1

最初の push は Kaggle API 400 になった。

```bash
make push-kaggle-train EXP=exp096_projection_fadein_after_prefix
```

理由:

- `kernel-metadata.json` の title が長く、Kaggle の slug が `kentookumura/exp096-projection-fadein-after-prefix-train` に解決されなかった。

同じ canonical kernel id のまま短い title で再生成して push した。

```bash
make prepare-kaggle-notebooks EXP=exp096_projection_fadein_after_prefix EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp096-projection-fadein-after-prefix-train --title 'exp096 projection fadein after prefix train' --run-on-push --strict"
make push-kaggle-train EXP=exp096_projection_fadein_after_prefix
kaggle kernels pull kentookumura/exp096-projection-fadein-after-prefix-train -p /tmp/kaggle-pull/exp096-projection-fadein-after-prefix-train-v1 -m
kaggle kernels status kentookumura/exp096-projection-fadein-after-prefix-train
kaggle kernels logs kentookumura/exp096-projection-fadein-after-prefix-train
kaggle kernels output kentookumura/exp096-projection-fadein-after-prefix-train -p /tmp/kaggle-output/exp096_projection_fadein_after_prefix/train_v1
```

結果:

- Kernel: `kentookumura/exp096-projection-fadein-after-prefix-train` v1
- id_no: 124064301
- URL: https://www.kaggle.com/code/kentookumura/exp096-projection-fadein-after-prefix-train
- Status: COMPLETE
- Output: `/tmp/kaggle-output/exp096_projection_fadein_after_prefix/train_v1`
- Runtime: 1137.359 秒
- Rows / wells: 3,783,989 / 773
- Baseline exp073 OOF RMSE: 9.526374817
- Best variant: `degree4_beta0.75_c2_fade250_750`
- Best RMSE: 9.397537231
- Delta vs exp073: -0.128837518
- Delta vs exp094 global best 9.399456024: -0.001918793
- Best prediction SHA: `0cdb4c7e0add9584f8847a8ab63fd02be09c3d573cc8ad776f76211f185634b3`

Guard:

- `passes_guard=true`
- `recommendation=port_to_inference_candidate`
- max fold regression: -0.092230
- near row regression: 0.0
- short tail regression: +0.001918
- distance 0-50 / 50-100 / 100-250 ft delta: 0.0 / 0.0 / 0.0
- tail rank 0-99 delta: 0.0
- correction p95: 3.350098

再現性 / 生成物:

- exp073 OOF raw gzip SHA: `986e26c5c6617ade714623d44433e9beacdb2b1027d46c4a4e70825bc8ab87fc`
- exp073 OOF decompressed content SHA: `fd6c68050058c40b4960f3ff2af9905bfcb1c12d540c71e331d0aa85ca9756a4`
- best predictions raw gzip SHA: `87a08c68a15abea9dd543d9e81800c2c6f208bc68f843f99af7ce62db259bdae`
- best predictions decompressed content SHA: `05c6787e19f5058b8d82bd1861072cb263976a70a35355969ae548ffc0d8d6a8`
- summary SHA: `0c15cc5a0dd69ac6b21dafbad3117b776d8e2c1e8095361c0452f39ca8700831`
- log SHA: `3417d7bddfbf8107f7d680b4cb7c0f7b0019d414d347732f40578306b8b99f18`

判断:

- train-side guard は通過。`degree4_beta0.75_c2_fade250_750` を inference candidate として固定する。
- 提出判断はまだ行わない。inference run と submit-check が次の必須確認。

### 2026-06-21 記録更新

```bash
make prepare-kaggle-notebooks EXP=exp096_projection_fadein_after_prefix EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp096-projection-fadein-after-prefix-inference --title 'exp096 projection fadein after prefix inference' --run-on-push --strict"
make validate-exp EXP=exp096_projection_fadein_after_prefix
```

- `metrics.json`、`result.md`、`README.md`、`experiment_summary.md`、`backlog/KAGGLE_DIRECTION.md` を更新。
- `config.yaml` の `inference.selected_variant` を `degree4_beta0.75_c2_fade250_750` に固定。
- inference package を再生成し、metadata は `enable_gpu=false`、`enable_internet=false`、`kernel_sources=["kentookumura/exp073-full-replay-repro-guard-infer"]`。
- `make validate-exp` は strict で PASS。

### 2026-06-21 Kaggle inference v1

```bash
make push-kaggle-infer EXP=exp096_projection_fadein_after_prefix
kaggle kernels pull kentookumura/exp096-projection-fadein-after-prefix-inference -p /tmp/kaggle-pull/exp096-projection-fadein-after-prefix-inference-v1 -m
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp096-projection-fadein-after-prefix-inference
kaggle kernels status kentookumura/exp096-projection-fadein-after-prefix-inference
kaggle kernels output kentookumura/exp096-projection-fadein-after-prefix-inference -p /tmp/kaggle-output/exp096_projection_fadein_after_prefix/inference_v1
.venv/bin/python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp096_projection_fadein_after_prefix/inference_v1/submission.csv --sample data/raw/sample_submission.csv
```

結果:

- Kernel: `kentookumura/exp096-projection-fadein-after-prefix-inference` v1
- id_no: 124117959
- URL: https://www.kaggle.com/code/kentookumura/exp096-projection-fadein-after-prefix-inference
- Status: COMPLETE
- Output: `/tmp/kaggle-output/exp096_projection_fadein_after_prefix/inference_v1`
- selected variant: `degree4_beta0.75_c2_fade250_750`
- rows / wells: 14,151 / 3
- fallback rows: 0
- submission SHA: `41b251c4ef29aa9daee62768890b406621310b7977164c24923a979e844dbaf5`
- test prediction gzip SHA: `e701f0c0f629a820e0e374695a4bfc5600ff41d9d3dc13edc396234eda5f2274`
- test prediction decompressed content SHA: `a99e3625ef886a67fc0731b943247ff508f5422d97ef438555b93a633885d089`
- exp073 inference decompressed content SHA: `f3f04e56f3035191d651e330d26ee48e819f42cf0497acecefc88fe985cdc219`
- prediction min / max / mean / std: 11591.730469 / 12239.677734 / 11905.696063 / 279.370888

Submit-check:

- PASS
- rows=14,151、sample row count 一致
- header は `sample_submission.csv` と一致
- id order は sample と一致
- duplicate IDs: 0
- missing / non-finite tvt: 0

exp073 inference v2 との差分:

- rows merged: 14,151
- id order match: true
- RMSE diff: 0.984292
- mean diff: 0.039805
- p95 abs diff: 1.953125
- p99 abs diff: 2.734375
- min / max diff: -2.744141 / 6.015625

判断:

- inference と submit-check は完了。まだ提出はしていない。
- 提出する場合はこの inference v1 の `submission.csv` を使う。

## 変更点

- exp073 `gpu_repro_guard_dp_threads8` / `lgb_mean` OOF prediction は固定入力のまま。
- projection fit は exp094 と同じく `U = pred_tvt + Z - (anchor_t0 + anchor_z0)` を well ごとに robust polynomial fit する。
- correction 適用だけを変更し、`md_since <= 250` は beta 0、`250-750` または `250-1000` で selected beta まで線形 fade、以降 selected beta 固定にする。
- 候補は `degree4/c2` と `degree5/c1.5`、beta `0.50/0.75`、fade window `250-750/250-1000` に限定する。
- variant metrics、fold metrics、bucket metrics、by-well metrics、best predictions、summary JSON を保存する。

## 再現性メモ

- seed policy: `no_new_rng_projection_postprocess`
- stochastic components: upstream exp073 GPU LightGBM OOF prediction のみ。exp096 自体に RNG はない。
- CPU/GPU runtime: CPU で十分。LightGBM/GPU 学習なし。
- gzip prediction は decompressed content SHA を主証拠として train 実行後に記録する。
- model manifest / model SHA: 対象外
- submission SHA: `41b251c4ef29aa9daee62768890b406621310b7977164c24923a979e844dbaf5`

## 次のアクション

1. 再提出するか判断する。
2. 再提出する場合は `kentookumura/exp096-projection-fadein-after-prefix-inference` v2 の `submission.csv` を使う。

### 2026-06-21 提出 v1 failure debug / hidden-compatible inference v2

ユーザー報告:

- Kaggle code submission が `Notebook Threw Exception` で失敗。

診断:

- inference v1 は public run では COMPLETE / submit-check PASS だったが、`data.exp073_inference_predictions` で `kentookumura/exp073-full-replay-repro-guard-infer` の public output prediction を読み込む構成だった。
- Kaggle code submission の hidden rerun では test row/well set が差し替わるため、public exp073 inference output の 14,151 rows / 3 wells をそのまま使う public-output-copy 型は hidden-compatible ではない。
- `kaggle competitions submissions` では最新 ref `53895968` が `SubmissionStatus.COMPLETE` / publicScore 空欄に見えたが、UI は `Notebook Threw Exception`。CLI の status 表示だけでは hidden rerun 失敗を十分に判定できない。

修正:

- `exp063_full_replay_reproducibility_guard.py` と `public_notebook_replay_audit.py` を exp096 package に同梱。
- inference v2 は exp073 train source `kentookumura/exp073-full-replay-repro-guard-train` から saved booster manifest を読み、current test files に対して exp073 base prediction を notebook 内で再生成してから fade-in projection を適用する。
- `runtime.kaggle.enable_gpu=true`、`inference_kernel_sources=["kentookumura/exp073-full-replay-repro-guard-train"]` に変更。
- `allow_submission_fallback=false` とし、submission id に prediction が欠けた場合は平均 fallback せず error にする。

実行:

```bash
make prepare-kaggle-notebooks EXP=exp096_projection_fadein_after_prefix EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp096-projection-fadein-after-prefix-inference --title 'exp096 projection fadein after prefix inference' --run-on-push --strict"
make push-kaggle-infer EXP=exp096_projection_fadein_after_prefix
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp096-projection-fadein-after-prefix-inference
kaggle kernels output kentookumura/exp096-projection-fadein-after-prefix-inference -p /tmp/kaggle-output/exp096_projection_fadein_after_prefix/inference_v2
.venv/bin/python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp096_projection_fadein_after_prefix/inference_v2/submission.csv --sample data/raw/sample_submission.csv
```

結果:

- Kernel: `kentookumura/exp096-projection-fadein-after-prefix-inference` v2
- Status: COMPLETE
- Output: `/tmp/kaggle-output/exp096_projection_fadein_after_prefix/inference_v2`
- public runtime: 約 182 秒
- exp073 base feature generation seconds: 107.386
- exp073 base rows / fallback rows: 14,151 / 0
- projection submission rows / fallback rows: 14,151 / 0
- submission SHA: `41b251c4ef29aa9daee62768890b406621310b7977164c24923a979e844dbaf5`
- exp073 regenerated base prediction SHA: `9b67c2540e93262a095f7f65f527e140a54c0c7e399b068ba85c437b61303822`
- exp073 regenerated base raw gzip SHA: `e9e7826fc4cfe3b0c56e34ac34ae9b8ebcefa8a8fe16bf44864e0d7ede397bab`
- exp073 regenerated base decompressed content SHA: `f3f04e56f3035191d651e330d26ee48e819f42cf0497acecefc88fe985cdc219`
- exp073 regenerated test feature SHA: `f778b7238ef333bf8a639435be4b924c97d0c3e1a685545991cfe9a3dd1b7623`
- exp096 inference summary SHA: `0408a410faba1567e04f71ced736e7e63e5d67082b576d58afa79aac9afe92fc`
- exp096 inference log SHA: `a04773890c5eac4171f765d2ff74e1231daa580cac9d8446c1b06762bda4b65e`

Submit-check:

- PASS
- rows=14,151、sample row count 一致
- header は `sample_submission.csv` と一致
- id order は sample と一致
- duplicate IDs: 0
- missing / non-finite tvt: 0

判断:

- public output copy 依存は解消。v2 は hidden test row/well set に追従する source-port 型。
- v2 は提出済み。

### 2026-06-21 提出結果

ユーザー確認:

- ref: `53896594`
- Public LB: 8.651
- Private LB: 未確定

解釈:

- exp073 raw deterministic anchor Public LB 8.780 より -0.129 改善。
- exp077 ML route submitted/postprocessed anchor Public LB 8.611 より +0.040 悪化。
- train-side OOF では exp073 raw から大きく改善したが、現行 ML route anchor の exp077 は更新できなかった。
- exp096 は submitted / recorded で完了し、anchor には昇格しない。
