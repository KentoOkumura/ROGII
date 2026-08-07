# exp156_test_batch_covariate_context_audit セッションノート

## 現在の状態

- status: `completed_train_side_rejected_no_submit`
- route: `ml_model`
- parent: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- baseline: exp148 `lgb_mean` CV 8.501281182 / Public LB 7.960
- LightGBM training: なし
- inference: disabled diagnostic only
- blocked: なし

## 実装内容

- `.steering/20260628-exp156-test-batch-covariate-context-audit/` を作成。
- `experiments/exp156_test_batch_covariate_context_audit/` を exp154 からコピーして作成。
- 実装本体を `test_batch_covariate_context_audit.py` に変更した。
- exp148 `lgb_mean` OOF、exp073 reference OOF、exp072 PF/Beam/dense feature cache を読む posthoc audit にした。
- raw train covariates から X/Y/Z/MD/GR、prefix/eval length、GR coverage、prefix `TVT_input` range/std を target-free context として集計する。
- well centroid の XY quantile bin から pseudo test batch context を作る。
- `context_risk_score` と PF-dense / base-dense disagreement を使い、target-free gate を通った segment だけ fallback candidate へ clipped blend する。

## 設計メモ

- LightGBM の新規学習は行わない。
- `target_tvt` は scoring、oracle coverage readout、posthoc error 集計だけに使う。
- 他 well の `TVT_input` 値を label / residual / correction target として使わない。
- 合否は exp148 に対する overall RMSE だけでなく、PF worst50、common PF+ML worst26、near-row、context risk bucket、worst-well regression、raw-test feature parity で見る。

## 実行予定

```bash
uv run python -m py_compile experiments/exp156_test_batch_covariate_context_audit/test_batch_covariate_context_audit.py experiments/exp156_test_batch_covariate_context_audit/settings.py
uv run ruff check experiments/exp156_test_batch_covariate_context_audit/test_batch_covariate_context_audit.py experiments/exp156_test_batch_covariate_context_audit/settings.py
uv run ruff format --check experiments/exp156_test_batch_covariate_context_audit/test_batch_covariate_context_audit.py experiments/exp156_test_batch_covariate_context_audit/settings.py
make validate-exp EXP=exp156_test_batch_covariate_context_audit
```

Kaggle train push 前の booster count:

- active variant 数: 0 train variants
- LightGBM config 数: 0
- folds: 0
- total boosters: 0
- control retraining: なし

Kaggle で audit を走らせる場合:

```bash
make prepare-kaggle-notebooks EXP=exp156_test_batch_covariate_context_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp156-test-batch-context-audit-train --title 'exp156 test batch context audit train' --run-on-push --strict"
make push-kaggle-train EXP=exp156_test_batch_covariate_context_audit
```

## Kaggle train v1

```bash
make push-kaggle-train EXP=exp156_test_batch_covariate_context_audit
kaggle kernels status kentookumura/exp156-test-batch-context-audit-train
kaggle kernels logs kentookumura/exp156-test-batch-context-audit-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp156-test-batch-context-audit-train
```

- kernel: `kentookumura/exp156-test-batch-context-audit-train`
- version: 1
- push: success
- URL: `https://www.kaggle.com/code/kentookumura/exp156-test-batch-context-audit-train`
- status after push: `KernelWorkerStatus.RUNNING`
- CLI logs: initially empty; `logs -f` for 300 seconds also returned no log payload.
- user reported completion; rechecked status as `KernelWorkerStatus.COMPLETE`.
- logs were retrieved successfully after completion.
- output: `experiments/exp156_test_batch_covariate_context_audit/kaggle/output/train_v1/`

Result:

- rows / wells: 3,783,989 / 773
- base exp148 `lgb_mean`: RMSE 8.501281182 / MAE 5.335650921 / within10 0.856332035
- best non-oracle: base exp148 `lgb_mean`
- best gate: `context_densew_tail1500_q85_min4_clip10_a025`, RMSE 8.502466, delta +0.001185, gate rate 0.050990, gate wells 138, max well regression +2.321141
- common PF+ML worst26: best gate `context_densew` delta -0.399641
- PF worst50: best gate `context_densew` delta -0.304031
- `1000_plus + pf_dense_diff_q4`: `context_densew` delta -0.006011
- raw-test parity checklist: required columns / target-free gate / no LightGBM training pass

Decision:

- train-side rejected
- inference port / submit は行わない
- batch context は risk bucket として外れやすい領域を捉えたが、fallback 選択精度が足りず、global OOF と worst-well guard を満たさない。
