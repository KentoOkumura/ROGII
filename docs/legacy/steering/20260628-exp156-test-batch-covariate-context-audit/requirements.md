# 要件

## 依頼

`test_batch_covariate_context_audit` を実装する。test batch 内で同時に見える target-free covariate context を使い、exp148 `lgb_mean` を信用できる regime と、exp073 / likPF / dense 系へ fallback すべき可能性がある high-drift / high-disagreement regime を train-side pseudo-tail 上で診断する。

## 制約

- Route: `ml_model`
- LightGBM の新規学習は行わない。保存済み exp148 / exp073 OOF prediction と exp072 feature cache を読む posthoc audit に限定する。
- 他 well の `TVT_input` 値を label / residual / correction target として使わない。
- raw context は inference 時に見える X/Y/Z/MD/GR、prefix/eval length、GR coverage、candidate disagreement、tail-drift proxy に限定する。
- `target_tvt` は scoring、oracle readout、posthoc 集計だけに使う。
- 改善判断は global RMSE だけでなく、near-row、common worst 26 wells、PF worst50、exp148 worst50、worst-well regression、raw-test parity を見る。
- 再現性: `docs/06_reproducibility.md` に従い、保存済み gzip input は decompressed SHA を記録する。

## 受け入れ基準

- `experiments/exp156_test_batch_covariate_context_audit/` に config、train/inference notebook、実装 helper、README、SESSION_NOTES、result、metrics がある。
- `config.yaml` に route、親実験、no-new-model 方針、leakage policy、gate variants、expected outputs が書かれている。
- train notebook が設定確認、入力確認、audit 実行、metrics/生成物 preview をセル単位で追える。
- 実装が `py_compile` と `validate-exp` を通過する。
- Kaggle push 前の booster count が 0 で記録されている。
