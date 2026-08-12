# exp140_z_slope_posthoc_correction_on_pfbeam_candidates セッションノート

## 目的

`z_slope_posthoc_correction_on_pfbeam_candidates` を実装する。固定済み PF/Beam 候補に対して、target-free な `dZ/dMD` 由来の小補正を gate 付きでかけ、Z-driven representative well と longtail で救済余地があるかを監査する。

## 現在の状態

- Route: pf_beam
- 状態: completed_train_side_rejected_no_submit
- CV: 11.594897672217703
- LB: なし
- 提出: なし

## コマンドログ

### 2026-06-27 JST 実装

```bash
make new-steering EXP=exp140_z_slope_posthoc_correction_on_pfbeam_candidates
make new-exp EXP=exp140_z_slope_posthoc_correction_on_pfbeam_candidates SOURCE=experiments/exp099_pf_multi_observation_likelihood_probe
```

実装内容:

- `docs/legacy/steering/20260627-exp140-z-slope-posthoc-correction-on-pfbeam-candidates/` を作成し、requirements / design / tasklist を記入した。
- `config.yaml` を PF/Beam route の Z-slope posthoc audit 用に更新した。
- `settings.py` の experiment name を exp140 に更新した。
- `z_slope_posthoc_correction_on_pfbeam_candidates.py` を追加した。
- train notebook を設定確認、入力契約、監査実行、出力 preview、metrics/SHA 表示のセル構成に更新した。
- inference notebook は no-op guard として更新した。

実装メモ:

- exp072 cache の `pf_z` は absolute TVT、`likpf_mean_d` / `beam_mean_d` / `sc_ens_d` / `hyb_d` は `last_known_tvt + delta` として復元する。
- fallback で absolute 列を拾った場合は delta 加算しない。
- variant は `likpf_mean` / `pf_ancc` / `beam_mean` を base に、alpha、clip、Z slope threshold、candidate disagreement、`pfz_agree` / `pfz_pull` を比較する。
- 行単位 OOF gzip は基準候補と上位 variant のみ保存し、全 variant は metrics CSV に保存する。

### 予定

```bash
.venv/bin/python -m py_compile experiments/exp140_z_slope_posthoc_correction_on_pfbeam_candidates/z_slope_posthoc_correction_on_pfbeam_candidates.py experiments/exp140_z_slope_posthoc_correction_on_pfbeam_candidates/settings.py
.venv/bin/ruff check experiments/exp140_z_slope_posthoc_correction_on_pfbeam_candidates/z_slope_posthoc_correction_on_pfbeam_candidates.py experiments/exp140_z_slope_posthoc_correction_on_pfbeam_candidates/settings.py
make validate-exp EXP=exp140_z_slope_posthoc_correction_on_pfbeam_candidates
make prepare-kaggle-notebooks EXP=exp140_z_slope_posthoc_correction_on_pfbeam_candidates EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp140-z-slope-pfbeam-train --title 'exp140 z slope pfbeam train' --run-on-push --strict"
```

### 2026-06-27 JST validation / package

```bash
.venv/bin/python -m py_compile experiments/exp140_z_slope_posthoc_correction_on_pfbeam_candidates/z_slope_posthoc_correction_on_pfbeam_candidates.py experiments/exp140_z_slope_posthoc_correction_on_pfbeam_candidates/settings.py
.venv/bin/ruff check experiments/exp140_z_slope_posthoc_correction_on_pfbeam_candidates/z_slope_posthoc_correction_on_pfbeam_candidates.py experiments/exp140_z_slope_posthoc_correction_on_pfbeam_candidates/settings.py
.venv/bin/python -m json.tool experiments/exp140_z_slope_posthoc_correction_on_pfbeam_candidates/exp140_z_slope_posthoc_correction_on_pfbeam_candidates_train.ipynb
.venv/bin/python -m json.tool experiments/exp140_z_slope_posthoc_correction_on_pfbeam_candidates/exp140_z_slope_posthoc_correction_on_pfbeam_candidates_inference.ipynb
make validate-exp EXP=exp140_z_slope_posthoc_correction_on_pfbeam_candidates
make prepare-kaggle-notebooks EXP=exp140_z_slope_posthoc_correction_on_pfbeam_candidates EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp140-z-slope-pfbeam-train --title 'exp140 z slope pfbeam train' --run-on-push --strict"
```

結果:

- `py_compile`: PASS
- `ruff check`: PASS
- notebook JSON: PASS
- `validate-exp`: PASS
- Kaggle train package: `experiments/exp140_z_slope_posthoc_correction_on_pfbeam_candidates/kaggle/train`
- kernel id: `kentookumura/exp140-z-slope-pfbeam-train`
- title: `exp140 z slope pfbeam train`
- metadata: GPU false / internet false / run_on_push true / source `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- support manifest に `z_slope_posthoc_correction_on_pfbeam_candidates.py` と exp140 `config.yaml` が含まれることを確認した。

## 生成予定ファイル

- `exp140_z_slope_posthoc_correction_on_pfbeam_candidates_candidate_metrics.csv`
- `exp140_z_slope_posthoc_correction_on_pfbeam_candidates_bucket_metrics.csv`
- `exp140_z_slope_posthoc_correction_on_pfbeam_candidates_by_well.csv`
- `exp140_z_slope_posthoc_correction_on_pfbeam_candidates_group_metrics.csv`
- `exp140_z_slope_posthoc_correction_on_pfbeam_candidates_oof_predictions.csv.gz`
- `exp140_z_slope_posthoc_correction_on_pfbeam_candidates_feature_schema.csv`
- `exp140_z_slope_posthoc_correction_on_pfbeam_candidates_summary.json`

## 次

1. `make push-kaggle-train EXP=exp140_z_slope_posthoc_correction_on_pfbeam_candidates` で Kaggle train を実行する。
2. Kaggle train 完了後に output を取得する。
3. result / metrics / summary / backlog を更新する。

### 2026-06-27 JST Kaggle train v1 push / immediate failure

```bash
make push-kaggle-train EXP=exp140_z_slope_posthoc_correction_on_pfbeam_candidates
kaggle kernels pull kentookumura/exp140-z-slope-pfbeam-train -p /tmp/kaggle-pull/exp140-z-slope-pfbeam-train -m
kaggle kernels logs kentookumura/exp140-z-slope-pfbeam-train
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp140-z-slope-pfbeam-train
```

結果:

- Kernel version 1 push 成功。
- URL: https://www.kaggle.com/code/kentookumura/exp140-z-slope-pfbeam-train
- metadata pull 成功。
- v1 は papermill の `ValueError: No kernel name found in notebook and no override provided.` で実行開始直後に失敗した。

対応:

- 正の train / inference notebook に `kernelspec.name=python3` と `language_info` metadata を追加した。
- `make validate-exp` を再実行して PASS。
- `make prepare-kaggle-notebooks ... --strict` を再実行し、同じ kernel id に v2 として再 push する。

### 2026-06-27 JST Kaggle train v2 push / running

```bash
make push-kaggle-train EXP=exp140_z_slope_posthoc_correction_on_pfbeam_candidates
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp140-z-slope-pfbeam-train
kaggle kernels status kentookumura/exp140-z-slope-pfbeam-train
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp140-z-slope-pfbeam-train
```

結果:

- Kernel version 2 push 成功。
- URL: https://www.kaggle.com/code/kentookumura/exp140-z-slope-pfbeam-train
- `status`: `KernelWorkerStatus.RUNNING`
- CLI logs / output は空のまま。v1 の papermill immediate failure は再発していない。
- ユーザー指示により、こちら側の `logs -f` 監視は停止した。Kaggle kernel 自体は停止していない。

### 2026-06-27 JST Kaggle train v2 complete / output retrieved

```bash
kaggle kernels status kentookumura/exp140-z-slope-pfbeam-train
kaggle kernels output kentookumura/exp140-z-slope-pfbeam-train -p experiments/exp140_z_slope_posthoc_correction_on_pfbeam_candidates/kaggle/output/train_v2
```

結果:

- `status`: `KernelWorkerStatus.COMPLETE`
- output: `experiments/exp140_z_slope_posthoc_correction_on_pfbeam_candidates/kaggle/output/train_v2`
- rows / wells: 3,783,989 / 773
- runtime: 1290.247607 sec
- variant count: 432
- primary baseline `likpf_mean`: RMSE 11.594897672 / MAE 7.067632584 / within10 0.772807479
- best overall: `likpf_mean`
- best Z-slope variant: `zsl_likpf_mean_a0p1_c10_z0p1_d5_pfz_agree`
- best Z-slope: RMSE 11.597150618 / MAE 7.068923098 / within10 0.772796115
- delta vs baseline: +0.002252946 RMSE
- max well regression vs baseline: +0.204513805 RMSE

解釈:

- Z-slope posthoc correction は `likpf_mean` を global に超えなかった。
- best Z-slope variant でも RMSE / within10 が悪化したため、inference port / submit はしない。
- 500-1000ft bucket には小改善 variant があるが、1000+ longtail と `z_abs_top_quartile` で baseline を超えず、採用条件を満たさない。
- `pf_z` や Z slope は hard correction ではなく、confidence feature / verifier / transition prior の補助材料に限定する。
