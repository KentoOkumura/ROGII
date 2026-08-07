# exp236_exact_hmm_posterior_bimodality_audit セッションノート

## 2026-07-12 実装開始

- `exact_hmm_posterior_bimodality_audit` backlog を `exp236` として実験化した。
- Route は `ensemble`。主対象は exp221 の completed single variant
  `hmm_lgb_exp148_lgb_mean_s2000_l0500` で、保存済み exp148 `lgb_mean` OOFを
  Gaussian emission center として読む。
- HMM設定は `step=0.35`、41 rate、`sig_r=0.002`、`sig_p=0.02`、`mom=0.998`、
  `sigma=20.0`、`lambda=0.50` に固定する。emission、transition、grid、predictionを
  変更しない。
- 実行予定は active HMM variant 1、LightGBM config 0、fold 0、booster 0。親/controlの
  再学習はない。CPU-only、GPU/internet disabled、`outer_workers=1`、
  `numba_num_threads=1`。
- `return_post=True` のwell単位posteriorを解析後に破棄し、full posterior tensorは保存しない。
  row / segment / well summaryと最大12件の代表plotだけを生成する。
- true TVT/errorはdecoder metrics、error lift、oracle top2 coverage、train plot overlayにのみ
  使用する。peak・二峰flag・mode追跡・plot対象のposterior特徴量順位には使用しない。
- raw-test inference、submit、mixture emission、mode-state追加、midpoint correctionは対象外。

## 2026-07-12 実装検証

```bash
.venv/bin/python -m py_compile experiments/exp236_exact_hmm_posterior_bimodality_audit/*.py
.venv/bin/ruff check experiments/exp236_exact_hmm_posterior_bimodality_audit --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp236_exact_hmm_posterior_bimodality_audit/exp236_exact_hmm_posterior_bimodality_audit_train.py
make validate-exp EXP=exp236_exact_hmm_posterior_bimodality_audit
```

結果:

- `posterior_bimodality_audit.py` を追加し、exp221互換 `run_hmm2(..., return_post=True)` の
  well単位実行、target-free peak / valley / mode segment集計、decoder / oracle readout、
  代表plotのwell再実行を実装した。
- synthetic二峰posteriorで、固定thresholdによる `bimodal_flag`、`mean_in_valley_flag`、
  dominant-mode conditional meanを確認し、synthetic mode segmentでも連続segmentと
  mass-dominance switchの集計を確認した。
- `py_compile`、`ruff --select F821`、train / disabled inferenceのJupytext round-trip test、
  strict `make validate-exp` はすべてPASS。
- canonical Kaggle train package
  `kentookumura/exp236-exact-hmm-posterior-bimodality-audit-train` をstrict生成した。
  metadataはCPU (`enable_gpu=false`)、internet disabled、competition sourceと
  `exp148-train` / `exp115-hidden-like-spatial-holdout-from-ppt-train` kernel sourceを確認した。
- Kaggle train push / auditは未実行。実装依頼の範囲ではCPU runtimeを消費しない。

## 2026-07-12 Kaggle CPU train v1

- canonical kernel `kentookumura/exp236-exact-hmm-posterior-bimodality-audit-train` を v1 としてpushした。
- push後の `kaggle kernels pull -m` で同一IDのsource / metadataを取得できた。
- CPU、internet disabled、competition source、`exp148-train`、`exp115-hidden-like-spatial-holdout-from-ppt-train` sourceを確認した。
- 実行数は HMM variant 1、LightGBM config 0、fold 0、booster 0、parent/control再学習なし。
- push直後は `KernelWorkerStatus.RUNNING`。実行中の通常ログは空であり、同じcanonical IDを監視対象とした。

## 2026-07-12 Kaggle CPU train v1 完了

- `kaggle kernels status kentookumura/exp236-exact-hmm-posterior-bimodality-audit-train` は
  `KernelWorkerStatus.COMPLETE`。v1 は 3,783,989 rows / 773 wells を 27,168.333 sec
  （約7時間33分）で完走し、notebook error はなかった。
- HMM input の exp148 `lgb_mean` OOF は 3,783,989 rows を coverage し、raw SHA は
  `12f2980972c19ef72a88b198efa0f5329ee3614a21b269f1bebc5a37b3ac21b5`、decompressed
  SHA は `ec28d89641b74c67482aff7a1ebc925db536716f1a024467ae0339dd2326e14d`。
- posterior mean は RMSE 8.327728486 / MAE 4.811963870 で最良。親 exp221 記録値
  8.327736951との差 -0.000008465 は fixed replay の許容差内である。marginal MAP は
  RMSE 8.365160435（+0.037431949）、dominant-mode conditional mean は 8.331754352
  （+0.004025866）で、いずれも置換根拠にならない。
- 二峰 row は 35,399（0.9355%）、138 wells / 317 segments、mean-in-valley row は
  6,781（0.1792%）。mode mass switch / track break は 17 / 17 で頻繁なmode slipはない。
  二峰 subsetでも posterior mean RMSE 11.053503351 が dominant mode 11.373182708 と
  MAP 11.438155606 より良かった。
- oracle top2 は二峰 rowだけで MAE 4.400838542 / within10 0.878329896 の診断的 headroom を
  示したが、target-free な選択規則ではない。decoder、threshold、plot選択には使っていない。
- MAP は `|step delta| > 0.2` が 3.2659% と posterior mean の 0.0219% より大きく、spike
  risk が明確だった。dominant modeも 0.0394%で posterior meanを上回る。
- Kaggle output は summary / row / segment / well / status / step artifactだけを必要範囲で取得した。
  summary SHA は `d40109cd499e8f38f9ffe32dfe7a47a3083f9bde1b4fd590e6db89cb2e127f1d`、row summary
  decompressed SHA は `bf124fda2aed9fd1309ef9c8e608537cd16dfc116e920fd2387904888410e9a0`。
- 結論として、posterior mean を維持する。midpoint correction、mode-state、mixture emission、
  raw-test HMM再生成、inference、submitは実施しない。`two_regime_rate_noise_pf` を開始する
  transition-mismatch 根拠も得られなかった。

## 次のアクション

1. exp236 からの直接 decoder / HMM generation follow-up は行わない。
2. 将来の利用は、raw-test parity と fold-safe 生成を先に満たす add-only confidence 特徴量だけに限定する。
