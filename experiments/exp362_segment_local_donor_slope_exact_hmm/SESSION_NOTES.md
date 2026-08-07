# exp362_segment_local_donor_slope_exact_hmm セッションノート

## 目的

exp226 の予測値を使わず、K16 区間分割と近傍 donor slope 補間の考えを exp209 exact HMM の
時変 rate-prior mean として最初から評価する。設計凍結後、compact self-contained train 候補まで
実装する。2026-07-23のユーザー指示「実行してください」により、compact train候補の正規採用と
Kaggle CPU train実行が承認された。

## 現在の状態

- Route: pf_beam
- 状態: `completed_postrun_support_audit_failed_closed`
- CV: `11.161677223 ft`。ただし全segmentがprefix-rate-onlyへ退化した参考値
- LB: まだなし
- 実装: compact self-contained train 候補と fail-closed inference 候補を実装済み
- 正規 notebook 採用: 承認済み。compact self-contained train候補を正規train notebookへ採用
- Kaggle package / push / run: version 1完了、branch close後の再実行は無効
- 推論 / 提出: 無効

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 2026-07-23 実行承認

- 承認元: ユーザー指示「実行してください」
- 実行対象: train-side HMM 1 variant
- reporting folds: 5
- HMM well-runs: 773
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- 親control再学習: 0。保存済みexp209 predictionのみ参照
- runtime: Kaggle CPU、GPU off、internet off
- 対象外: inference、submission、同一OOF parameter rescue、blend rescue
- kernel: `kentookumura/exp362-segment-local-donor-slope-exact-hmm-train`

正規採用・事前検証:

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  --output experiments/exp362_segment_local_donor_slope_exact_hmm/exp362_segment_local_donor_slope_exact_hmm_train.ipynb \
  experiments/exp362_segment_local_donor_slope_exact_hmm/exp362_segment_local_donor_slope_exact_hmm_compact_selfcontained_train.py
.venv/bin/pytest -q tests/test_exp362_segment_local_donor_slope_exact_hmm.py
make validate-exp EXP=exp362_segment_local_donor_slope_exact_hmm
make prepare-kaggle-notebooks EXP=exp362_segment_local_donor_slope_exact_hmm \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp362-segment-local-donor-slope-exact-hmm-train \
  --title 'exp362 segment local donor slope exact hmm train' --run-on-push --strict"
kaggle kernels list --mine --search exp362-segment-local-donor-slope-exact-hmm-train --page-size 20
```

- 正規notebook: 26 cells（code 12、markdown 14）、109,643 bytes、`__file__`依存なし
- 専用テスト: `10 passed`
- strict experiment validation: PASS
- package metadata: private、CPU、internet off、run-on-push
- package input: exp209 control kernel + exp115 hidden-like audit kernel
- 同名kernel検索: `Not found`。新規canonical slugとしてpushする

push・canonical照合:

```bash
make push-kaggle-train EXP=exp362_segment_local_donor_slope_exact_hmm
kaggle kernels pull kentookumura/exp362-segment-local-donor-slope-exact-hmm-train \
  -p /tmp/exp362-prepush-8Ng0vT -m
kaggle kernels status kentookumura/exp362-segment-local-donor-slope-exact-hmm-train
```

- push結果: kernel version 1
- canonical id_no: `128368310`
- pull照合: id/title、private、CPU、internet off、2 kernel sourcesが一致
- 初回status: `KernelWorkerStatus.RUNNING`

### 2026-07-24 完了確認・成果物監査

```bash
.venv/bin/python .agents/skills/kaggle-platform/shared/check_all_credentials.py
kaggle kernels status kentookumura/exp362-segment-local-donor-slope-exact-hmm-train
kaggle kernels logs kentookumura/exp362-segment-local-donor-slope-exact-hmm-train
kaggle kernels files kentookumura/exp362-segment-local-donor-slope-exact-hmm-train \
  --page-size 200
kaggle kernels output kentookumura/exp362-segment-local-donor-slope-exact-hmm-train \
  -p /tmp/exp362-output-PYmq5W \
  --file-pattern '.*(\.json|_metrics\.csv|_sha_manifest\.csv)$' --page-size 200
kaggle kernels output kentookumura/exp362-segment-local-donor-slope-exact-hmm-train \
  -p /tmp/exp362-output-PYmq5W \
  --file-pattern '.*target_segment_prior\.csv\.gz$' --page-size 200
```

実行結果:

- Kernel status: `COMPLETE`
- Kernel / version / id_no:
  `kentookumura/exp362-segment-local-donor-slope-exact-hmm-train / 1 / 128368310`
- Runtime: `19,777.653140535 sec`
- 実行量: 1 variant / 5 reporting folds / 773 HMM well-runs /
  model config・trained fold・booster・GPU・control再実行各0
- Rows / wells: `3,783,989 / 773`
- Inference / submission: 0

Notebook gate:

- technical: PASS
- scientific: FAIL
- pooled: exp209 `11.938287235` -> candidate `11.161677223 ft`
  （`+0.776610012 ft`改善）
- fold: 0/2/3改善、1/4悪化、合計3/5。固定下限4/5をFAIL
- 1000+: `+0.858876684 ft`改善
- hidden-like spatial / typewell-purged:
  `+0.273186542 / +0.351965543 ft`改善
- by-well p95差: `-2.896173574 ft`
- worst: `86454a6f`、`+52.741425793 ft`。固定上限`+0.25 ft`をFAIL
- improved / worsened wells: `462 / 311`

Post-run support audit:

- target segment prior: 12,368 rows / 773 wells
- local gradient採用: `0 / 12,368`
- `mu_rate == prefix_rate`: `12,368 / 12,368`
- finite gradient: 0
- effective donors `>=10`: 0
- fallback reason: `effective_donors=11,596`、`nearest_distance=772`
- 保存`fallback`列は全行Falseだが無効。target rowの
  `{**target_row, **estimate, **prefix}`で`estimate["fallback"]`が
  `prefix["fallback"]`に上書きされた記録バグ
- `fallback_reason`と`mu_rate`は上書きされず、全segmentのprefix-rate退化を実ファイルで確認

このため、version 1のRMSEは意図したlocal donor-slope scheduleの科学的評価ではなく、
prefix-rate-only residual exact HMMの参考値として扱う。notebook scientific gateも不通過なので、
記録バグ修正、support/bandwidth/近傍/ridge/fallback/HMM parameter変更を理由とするversion 2は実行しない。

再現性:

- raw identity SHA:
  `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`
- scientific contract SHA:
  `2e80fd0573acc601e3b5fe28e6673725c3e7038a1393d0ae6110ed46dd4b2128`
- donor ledger logical / decompressed:
  `92a6f7e9...f33 / cafa6009...14c`
- target prior logical / decompressed:
  `b84487c4...a91 / 84318915...0c9`
- schedule logical / decompressed:
  `d6cbf1a7...cb7 / 5cabe7f3...8e8`
- prediction logical / decompressed:
  `bdf616e0...5cb / e1d672ff...7ef`
- SHA manifest raw:
  `5ac384b2153cf962132dff9bec12a46e95a5a549e190ac6a594632625adf5fc3`
- 取得したmanifest対象10成果物: raw SHA 10/10一致
- Kaggle log raw SHA:
  `839a08237196d2eeb7d6d2fd49e7c9eeb33cdf66c1f8b6f8fac720233bfebdbd`
- rerun parityなし、deterministic anchorにはしない

最終判断:

- `completed_postrun_support_audit_failed_closed`
- parameter/fallback/HMM/blend/selector救済、再実行、inference、submissionなし
- 同じK16 donor support前提を持つexp356は非退化supportの独立証拠までblocked/demoted

記録更新後の検証:

```bash
.venv/bin/python -m json.tool \
  experiments/exp362_segment_local_donor_slope_exact_hmm/metrics.json
make validate-exp EXP=exp362_segment_local_donor_slope_exact_hmm
make validate-exp EXP=exp356_exp226_donor_covariance_sig_r_on_exp209
.venv/bin/pytest -q \
  tests/test_exp362_segment_local_donor_slope_exact_hmm.py \
  tests/test_kaggle_notebooks.py
make update-summary
make validate-template
```

- JSON / exp356 strict / exp362 strict / template validation: PASS
- exp362専用 + notebook tests: `14 passed`
- `experiment_summary.md`: 367 experimentsで更新

### 2026-07-23 実行済み

```bash
make new-steering EXP=exp362_segment_local_donor_slope_exact_hmm
make new-exp EXP=exp362_segment_local_donor_slope_exact_hmm

.venv/bin/python -m py_compile \
  experiments/exp362_segment_local_donor_slope_exact_hmm/exp362_segment_local_donor_slope_exact_hmm_compact_selfcontained_train.py \
  experiments/exp362_segment_local_donor_slope_exact_hmm/exp362_segment_local_donor_slope_exact_hmm_compact_selfcontained_inference.py
.venv/bin/ruff check \
  experiments/exp362_segment_local_donor_slope_exact_hmm/exp362_segment_local_donor_slope_exact_hmm_compact_selfcontained_train.py \
  experiments/exp362_segment_local_donor_slope_exact_hmm/exp362_segment_local_donor_slope_exact_hmm_compact_selfcontained_inference.py \
  --select F821
.venv/bin/pytest -q tests/test_exp362_segment_local_donor_slope_exact_hmm.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp362_segment_local_donor_slope_exact_hmm/exp362_segment_local_donor_slope_exact_hmm_compact_selfcontained_train.py \
  experiments/exp362_segment_local_donor_slope_exact_hmm/exp362_segment_local_donor_slope_exact_hmm_compact_selfcontained_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp362_segment_local_donor_slope_exact_hmm/exp362_segment_local_donor_slope_exact_hmm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp362_segment_local_donor_slope_exact_hmm/exp362_segment_local_donor_slope_exact_hmm_compact_selfcontained_inference.py
make validate-exp EXP=exp362_segment_local_donor_slope_exact_hmm
task update-summary
make update-summary
make validate-template
make test
```

検証結果:

- 専用テスト: `10 passed`
- exp209 exact forward-backward kernel: synthetic inputでbitwise parity PASS
- raw train identity: 773 wells、
  `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32` 一致
- stable fold counts: `155 / 155 / 155 / 154 / 154`
- Jupytext train / inference: 変換と `--test` PASS
- `py_compile` / Ruff F821 / strict experiment validation: PASS
- `task update-summary`: ローカルに `task` executable がなく
  `/bin/bash: task: command not found` で失敗。Makefile同等の `make update-summary` はPASSし、
  最終再実行時に `experiment_summary.md` を366実験で更新
- `make validate-template`: PASS
- `make test`: `737 passed, 5 skipped, 2 failed`。全体実行時点のexp362専用9件と
  `tests/test_kaggle_notebooks.py` はPASS。2 FAILは既存
  `tests/test_exp296_exp223_self_gr_known_tvt_support_gate.py` のみで、
  exp296 config statusが`completed_train_side_guard_failed_closed`なのにtestが
  `kaggle_cpu_*`を要求する不一致と、同configの`execution.run_variant=false`により
  push approval検査より先に停止する期待順序不一致。exp362と無関係のため変更しない。

## 変更点

- 親を exp209 exact HMM、route を `pf_beam` に固定した。
- `U=TVT+Z` の donor segment rate と坑跡方向から局所 2D gradient を推定し、target 方向へ射影する。
- K16、近傍 50 wells、bandwidth 500 ft、relative ridge 0.001、support/fallback 条件を固定した。
- exp209 の emission、`sig_r=0.002`、`sig_p=0.02`、41 rate states、span 0.10、
  momentum 0.998、初期分布、posterior mean を固定した。
- Stage 0 なし、HMM 1 variant / 773 well-runs / 0 booster / control 再実行 0 とした。
- exp226 artifact 全般を禁止入力にした。exp355 は今回の入力・親ではない。
- raw well file SHA manifest、`SHA256("42|well_id")` 5-fold、outer-valid donor 除外、
  donor K16 OLS、target K16 geometry、nearest-segment-per-donor、weighted ridge、
  support/fallback、rowwise linear interpolationを実装した。
- exp209 exact kernelはそのまま保ち、`effective_dz = dz - mu_t * delta_MD` と residual-rate gridにより
  `U_t = U_(t-1) + (mu_t + q_t) * delta_MD + eta_t` を実装した。
- candidate prediction SHAをfreezeしてから raw suffix truth、保存済みexp209 control、
  hidden-like roleをjoinする境界を実装した。
- pooled/fold/distance/1000+/hidden-like 2面/by-well p95/worst/support/fallbackと、
  raw/logical/decompressed SHA manifestを保存する。
- compact trainは12章・2,152行で、比較参照のexp338 compact 10章・2,002行に対し、
  donor field、target schedule、freeze、gateの役割を欠かさない構成とした。
- inference compactは常にRuntimeErrorで停止し、sample submission copyを生成しない。

## 再現性メモ

- seed policy: RNG なし。`SHA256("42|well_id")` fold 順と、fold/well/segment/distance/donor/row の stable sort。
- stochastic components: なし。PF、Beam、likelihood-PF、seed bagging なし。
- CPU/GPU runtime: Kaggle CPU、GPU off、internet off、`19,777.653141 sec`、上限30,600秒内。
- Kaggle kernel id / version:
  `kentookumura/exp362-segment-local-donor-slope-exact-hmm-train / 1`。
- input / feature schema SHA: raw identity
  `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`。
  ローカルとKaggleの両方で一致。
- parent control decompressed SHA:
  `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`。
- feature content SHA: donor ledger、target segment prior、rowwise schedule のlogical /
  decompressed SHAをversion 1成果物から記録済み。
- model manifest / model SHA: estimator がないため非該当。
- prediction logical / decompressed SHA:
  `bdf616e0...5cb / e1d672ff...7ef`。
- submission SHA: inference / submission 無効のため非該当。
- rerun check: 未実施。branch closeのためversion 2は実行せず、deterministic anchorとはしない。

## 次のアクション

1. exp362は完了済み・fail closedとして維持する。
2. version 2、parameter/fallback/HMM/blend/selector救済、inference、submissionを行わない。
3. exp356は非退化donor supportの独立証拠が得られるまでblocked/demotedとする。
