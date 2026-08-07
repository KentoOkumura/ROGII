# exp103_pf_z_xy_likpf_ensemble_parity セッションノート

## 目的

exp100 の best `pf_z_xy_slope` は単発 PF 候補で、exp072 `likpf_mean` は 128 seed likelihood-PF ensemble だったため比較条件が揃っていなかった。exp103 では `pf_z_xy_slope` を exp072 `lik_pf` と同じ ensemble 形式に寄せ、exp072 保存済み `pf_z` / `likpf_*` と同じ candidate metrics で比較する。

## 現在の状態

- Route: `pf_beam`
- 状態: `implemented_not_run`
- CV: 未実行
- LB: なし
- 提出: なし

## 実装内容

- `exp103_pf_z_xy_likpf_ensemble_parity` を exp100 から作成した。
- `.steering/20260621-exp103-pf-z-xy-likpf-ensemble-parity/` に requirements / design / tasklist を記入した。
- `pf_z_xy_likpf_ensemble_parity.py` を実装した。
- `config.yaml` を parity audit 用に更新した。
- train / inference notebook を exp103 名で作り直した。

## 実装済み candidate

exp072 cache から読む baseline:

- `exp072_pf_z`
- `exp072_likpf_mean`
- `exp072_likpf_scale_3/5/8/12` は cache に存在する場合だけ読む

exp103 で生成する XY likelihood-PF:

- `xy_likpf_mean`
- `xy_likpf_scale_3`
- `xy_likpf_scale_5`
- `xy_likpf_scale_8`
- `xy_likpf_scale_12`

## コマンドログ

### 2026-06-21 JST 実装

```bash
make new-steering EXP=exp103_pf_z_xy_likpf_ensemble_parity
make new-exp EXP=exp103_pf_z_xy_likpf_ensemble_parity SOURCE=experiments/exp100_pf_z_unified_velocity_observation_prior
```

設計判断:

- `pos = TVT + Z` の state を使い、exp072 `lik_pf` と同じく prediction は `pos - Z` にする。
- prefix の `d(TVT_input + Z)/dMD ~ dZ/dMD + dXY/dMD` で XY rate prior を fitting する。
- rate likelihood は粒子重みと seed log likelihood の両方に入れる。
- exp072 baseline は再生成せず、Kaggle input の exp072 train cache から読む。

### 予定

```bash
python3 -m py_compile experiments/exp103_pf_z_xy_likpf_ensemble_parity/pf_z_xy_likpf_ensemble_parity.py experiments/exp103_pf_z_xy_likpf_ensemble_parity/settings.py
python3 -m json.tool experiments/exp103_pf_z_xy_likpf_ensemble_parity/exp103_pf_z_xy_likpf_ensemble_parity_train.ipynb
python3 -m json.tool experiments/exp103_pf_z_xy_likpf_ensemble_parity/exp103_pf_z_xy_likpf_ensemble_parity_inference.ipynb
uv run ruff check experiments/exp103_pf_z_xy_likpf_ensemble_parity/pf_z_xy_likpf_ensemble_parity.py experiments/exp103_pf_z_xy_likpf_ensemble_parity/settings.py
make validate-exp EXP=exp103_pf_z_xy_likpf_ensemble_parity
make prepare-kaggle-notebooks EXP=exp103_pf_z_xy_likpf_ensemble_parity EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp103-pf-z-xy-likpf-ensemble-train --title 'exp103 pf z xy likpf ensemble train' --run-on-push --strict"
```

Kaggle train 完了後に `candidate_metrics.csv`、summary、生成物 SHA、解釈を追記する。

### 2026-06-21 JST validation / package

```bash
python3 -m py_compile experiments/exp103_pf_z_xy_likpf_ensemble_parity/pf_z_xy_likpf_ensemble_parity.py experiments/exp103_pf_z_xy_likpf_ensemble_parity/settings.py
python3 -m json.tool experiments/exp103_pf_z_xy_likpf_ensemble_parity/exp103_pf_z_xy_likpf_ensemble_parity_train.ipynb
python3 -m json.tool experiments/exp103_pf_z_xy_likpf_ensemble_parity/exp103_pf_z_xy_likpf_ensemble_parity_inference.ipynb
uv run ruff check experiments/exp103_pf_z_xy_likpf_ensemble_parity/pf_z_xy_likpf_ensemble_parity.py experiments/exp103_pf_z_xy_likpf_ensemble_parity/settings.py
make validate-exp EXP=exp103_pf_z_xy_likpf_ensemble_parity
make prepare-kaggle-notebooks EXP=exp103_pf_z_xy_likpf_ensemble_parity EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp103-pf-z-xy-likpf-ensemble-train --title 'exp103 pf z xy likpf ensemble train' --run-on-push --strict"
make update-summary
```

結果:

- `py_compile`: PASS
- train notebook JSON: PASS
- inference notebook JSON: PASS
- `ruff check`: 行長修正後 PASS
- `validate-exp`: PASS
- Kaggle train package: `experiments/exp103_pf_z_xy_likpf_ensemble_parity/kaggle/train`
- kernel id: `kentookumura/exp103-pf-z-xy-likpf-ensemble-train`
- title: `exp103 pf z xy likpf ensemble train`
- metadata: GPU false / internet false / run_on_push true / competition source `rogii-wellbore-geology-prediction` / kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- `experiment_summary.md`: `implemented_not_run` として更新済み

補足:

- この環境では `.git` が見えず、`git status --short` は `fatal: not a git repository` で確認不可。

### 2026-06-21 JST Kaggle train v1 failed

```bash
make push-kaggle-train EXP=exp103_pf_z_xy_likpf_ensemble_parity
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp103-pf-z-xy-likpf-ensemble-train
```

結果:

- Kernel version 1 push 成功。
- URL: https://www.kaggle.com/code/kentookumura/exp103-pf-z-xy-likpf-ensemble-train
- status: `KernelWorkerStatus.RUNNING` から短時間で失敗。
- exp072 cache preview では baseline 候補列は `pf_z` と `likpf_mean_d` のみだった。
- `likpf_scale_3(_d)` を必須として読もうとして `ValueError`。

修正:

- exp072 baseline は `pf_z` と `likpf_mean` を必須にし、`likpf_scale_*` は cache に存在する場合だけ比較対象に含めるようにした。
- exp103 生成側の `xy_likpf_scale_3/5/8/12` は引き続き全て出力する。

### 2026-06-21 JST Kaggle train v2 push

```bash
python3 -m py_compile experiments/exp103_pf_z_xy_likpf_ensemble_parity/pf_z_xy_likpf_ensemble_parity.py experiments/exp103_pf_z_xy_likpf_ensemble_parity/settings.py
uv run ruff check experiments/exp103_pf_z_xy_likpf_ensemble_parity/pf_z_xy_likpf_ensemble_parity.py experiments/exp103_pf_z_xy_likpf_ensemble_parity/settings.py
make validate-exp EXP=exp103_pf_z_xy_likpf_ensemble_parity
make prepare-kaggle-notebooks EXP=exp103_pf_z_xy_likpf_ensemble_parity EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp103-pf-z-xy-likpf-ensemble-train --title 'exp103 pf z xy likpf ensemble train' --run-on-push --strict"
make push-kaggle-train EXP=exp103_pf_z_xy_likpf_ensemble_parity
kaggle kernels status kentookumura/exp103-pf-z-xy-likpf-ensemble-train
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp103-pf-z-xy-likpf-ensemble-train
```

結果:

- `py_compile`: PASS
- `ruff check`: PASS
- `validate-exp`: PASS
- Kaggle train package 再生成: PASS
- Kernel version 2 push 成功。
- URL: https://www.kaggle.com/code/kentookumura/exp103-pf-z-xy-likpf-ensemble-train
- status: `KernelWorkerStatus.RUNNING`
- 3 分の初期 tail ではログ出力なし。v1 の `likpf_scale_3(_d)` 即時エラーは再発していない。

### 2026-06-22 JST Kaggle train v2 timeout / cancelled

```bash
kaggle kernels status kentookumura/exp103-pf-z-xy-likpf-ensemble-train
kaggle kernels logs kentookumura/exp103-pf-z-xy-likpf-ensemble-train
kaggle kernels output kentookumura/exp103-pf-z-xy-likpf-ensemble-train -p experiments/exp103_pf_z_xy_likpf_ensemble_parity/kaggle/output/train_v2_timeout
```

結果:

- status: `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`
- logs: setup / input preview までは成功。PF 本体の出力はなし。
- output: Numba cache と support files のみ。`artifacts/` は未生成。
- 原因: 773 wells を `128 seeds x 500 particles` で逐次処理しており、Kaggle 制限時間内に完了できなかった。

修正:

- well 単位を `ThreadPoolExecutor` で `n_jobs=4` 並列化した。
- Numba JIT warmup を追加し、並列開始時の compile race を避ける。
- `progress_interval_wells=25` の進捗ログを追加した。
- seed は従来通り `stable_seed(exp103, xy_likpf, seed_root, well)` なので、well scheduling に依存しない。

### 2026-06-22 JST Kaggle train v3 failed

```bash
make push-kaggle-train EXP=exp103_pf_z_xy_likpf_ensemble_parity
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp103-pf-z-xy-likpf-ensemble-train
```

結果:

- Kernel version 3 push 成功。
- setup / input preview 成功。
- `xy_likpf audit: wells=773 n_jobs=4 n_seeds=128 n_particles=500` まで出力。
- 追加した Numba warmup が `_xy_likpf_allseeds` に 31 引数を渡しており、`TypeError: too many arguments: expected 30, got 31` で失敗。

修正:

- warmup 呼び出しから余分な `0.0` を 1 つ削除した。

### 2026-06-22 JST Kaggle train v4 push

```bash
python3 -m py_compile experiments/exp103_pf_z_xy_likpf_ensemble_parity/pf_z_xy_likpf_ensemble_parity.py experiments/exp103_pf_z_xy_likpf_ensemble_parity/settings.py
uv run ruff check experiments/exp103_pf_z_xy_likpf_ensemble_parity/pf_z_xy_likpf_ensemble_parity.py experiments/exp103_pf_z_xy_likpf_ensemble_parity/settings.py
make validate-exp EXP=exp103_pf_z_xy_likpf_ensemble_parity
make prepare-kaggle-notebooks EXP=exp103_pf_z_xy_likpf_ensemble_parity EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp103-pf-z-xy-likpf-ensemble-train --title 'exp103 pf z xy likpf ensemble train' --run-on-push --strict"
make push-kaggle-train EXP=exp103_pf_z_xy_likpf_ensemble_parity
kaggle kernels status kentookumura/exp103-pf-z-xy-likpf-ensemble-train
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp103-pf-z-xy-likpf-ensemble-train
```

結果:

- `py_compile`: PASS
- `ruff check`: PASS
- `validate-exp`: PASS
- Kaggle train package 再生成: PASS
- Kernel version 4 push 成功。
- URL: https://www.kaggle.com/code/kentookumura/exp103-pf-z-xy-likpf-ensemble-train
- status: `KernelWorkerStatus.RUNNING`
- `logs -f` と通常 `logs` は warning 以外空。Kaggle CLI が実行中ログを返していない可能性がある。

### 2026-06-22 JST Kaggle train v4 output 取得

ユーザー完了連絡後に logs / output を取得した。

```bash
kaggle kernels status kentookumura/exp103-pf-z-xy-likpf-ensemble-train
kaggle kernels logs kentookumura/exp103-pf-z-xy-likpf-ensemble-train
kaggle kernels output kentookumura/exp103-pf-z-xy-likpf-ensemble-train -p experiments/exp103_pf_z_xy_likpf_ensemble_parity/kaggle/output/train_v4
```

結果:

- status: `KernelWorkerStatus.COMPLETE`
- output: `experiments/exp103_pf_z_xy_likpf_ensemble_parity/kaggle/output/train_v4`
- runtime: 22252.17 sec
- rows: 3,783,989
- wells: 773
- best overall: `exp072_likpf_mean` RMSE 11.594898 / MAE 7.067633 / within10 0.772807
- best XY: `xy_likpf_scale_12` RMSE 13.916271 / MAE 8.554011 / within10 0.705260
- exp072 `pf_z`: RMSE 17.788171 / MAE 10.677487 / within10 0.647668

Candidate ranking:

| candidate | RMSE | within10 |
| --- | ---: | ---: |
| `exp072_likpf_mean` | 11.594898 | 0.772807 |
| `xy_likpf_scale_12` | 13.916271 | 0.705260 |
| `xy_likpf_scale_8` | 13.961015 | 0.701849 |
| `xy_likpf_scale_5` | 14.030092 | 0.700313 |
| `xy_likpf_scale_3` | 14.092584 | 0.698493 |
| `xy_likpf_mean` | 14.580554 | 0.650353 |
| `exp072_pf_z` | 17.788171 | 0.647668 |

判定:

- `xy_likpf_scale_12` は exp072 `pf_z` から RMSE -3.871901、within10 +0.057592 改善。
- `xy_likpf_scale_12` は exp072 `likpf_mean` に対して RMSE +2.321373、within10 -0.067548 で悪化。
- direct inference port / submit はしない。

生成物:

- `artifacts/exp103_pf_z_xy_likpf_ensemble_parity_candidate_metrics.csv`
- `artifacts/exp103_pf_z_xy_likpf_ensemble_parity_bucket_metrics.csv`
- `artifacts/exp103_pf_z_xy_likpf_ensemble_parity_by_well.csv`
- `artifacts/exp103_pf_z_xy_likpf_ensemble_parity_xy_likpf_quality.csv`
- `artifacts/exp103_pf_z_xy_likpf_ensemble_parity_candidate_wide.csv.gz`
- `artifacts/exp103_pf_z_xy_likpf_ensemble_parity_candidate_long.csv.gz`
- `artifacts/exp103_pf_z_xy_likpf_ensemble_parity_summary.json`

SHA:

- `candidate_metrics.csv`: `5570a058351aa460425bc3504159b225819ea163a6fd5a52669107470a140b9f`
- `candidate_wide.csv.gz`: `bd8ef95b6cdaf029c13bd843e50b552dc68921ed8a24d2775bb1f54a3bda8466`
- `candidate_wide.csv.gz` decompressed: `68974d20eaca854c0458ecb234c9c569daf4d66764f1a788083b74cc9359a012`
- `candidate_long.csv.gz`: `a7b9bca2c67ce8d3c122f642b4f2a99855a57f631eb86d38a125b4f71f949ad7`
- `candidate_long.csv.gz` decompressed: `b087071b619c3de820058920e2b985b22e7580205561677957c89485b03625a1`
