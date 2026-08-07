# exp156_test_batch_covariate_context_audit

## 状態

- status: `implemented_not_run`
- route: `ml_model`
- parent: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- baseline: exp148 `lgb_mean` CV 8.501281182 / Public LB 7.960
- LightGBM training: なし
- inference: disabled diagnostic only

## 仮説

test batch 内で同時に見える target-free covariate context から、exp148 を信用しにくい high-drift / high-disagreement regime を識別できる可能性がある。識別できる場合だけ、exp073 / likPF / dense 系 candidate への低頻度 clipped fallback を train-side pseudo-tail 上で診断する。

## 実装

- exp148 `lgb_mean` OOF、exp073 `lgb_mean` OOF、exp072 PF/Beam/dense feature cache を結合する。
- raw train covariates から X/Y/Z/MD/GR、prefix/eval length、GR coverage、prefix `TVT_input` range/std を読む。
- well centroid の XY quantile bin で pseudo test batch context を作り、小さい batch は global context に落とす。
- batch 内平均との差と batch-level risk から `context_risk_score` を作る。
- target-free gate を通った segment だけ、exp148 から fallback candidate へ clipped blend する。

## 検証方針

- 検証: train-side pseudo-tail posthoc audit
- scoring rows: exp072 full replay train pseudo-tail rows
- metric: RMSE / MAE / within10
- 比較: exp148 `lgb_mean`、exp073 `lgb_mean`、`likpf_mean`、dense candidates、context gate variants
- guard: near-row、common PF+ML worst26、PF worst50、exp148 worst50、context risk bucket、worst-well regression、raw-test parity

## 所見

未実行。実装済みだが Kaggle train audit はまだ走らせていない。実行前の計画では active train variants 0、LightGBM configs 0、folds 0、total boosters 0。

## 注意

他 well の `TVT_input` は label、residual、correction target として使わない。`target_tvt` は scoring と oracle readout のみに使う。

## 実行

```bash
make prepare-kaggle-notebooks EXP=exp156_test_batch_covariate_context_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp156-test-batch-context-audit-train --title 'exp156 test batch context audit train' --run-on-push --strict"
make push-kaggle-train EXP=exp156_test_batch_covariate_context_audit
```

Kaggle push 前の booster count:

- active variants: 0 train variants
- LightGBM configs: 0
- folds: 0
- total boosters: 0
- control retraining: なし
