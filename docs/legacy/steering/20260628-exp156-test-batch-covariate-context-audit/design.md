# 設計

## アプローチ

exp148 `lgb_mean` の保存済み OOF prediction を base とする。exp073 `lgb_mean` OOF、exp072 PF/Beam/dense feature cache、raw train covariates を結合し、pseudo test batch context を target-free に作る。batch context は well centroid の XY quantile bin を基本にし、小さすぎる batch は global context に落とす。

各 row/well には、batch 内平均との差や batch-level risk から `context_risk_score` を付ける。gate variants は `context_risk_score`、row-level context risk、PF-dense disagreement、base-dense disagreement、tail rank、near guard、candidate path continuity を使い、連続 segment だけに clipped blend を適用する。

補正式:

```text
pred = exp148_lgb_mean
pred[gate] = exp148_lgb_mean + alpha * clip(candidate - exp148_lgb_mean, -clip_abs, clip_abs)
```

## 実験範囲

- 対象実験: `exp156_test_batch_covariate_context_audit`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 比較基準: exp148 `lgb_mean`、exp073 `lgb_mean`、likPF、dense candidates
- 変更する変数: context batch size、context/risk quantile、fallback candidate、alpha、clip、minimum segment length、near guard
- 固定する変数: exp148 OOF、exp073 OOF、exp072 PF/Beam/dense cache、score rows

## 再現性設計

- seed policy: 新規乱数なし。保存済み OOF / cache と deterministic quantile bins を読む。
- stochastic 処理の有無: exp156 内にはなし。upstream exp148 / exp073 / exp072 の stochastic component は source meta として記録する。
- PF/Beam / likelihood-PF / seed bagging の有無: 再生成せず、exp072 cache の保存済み列を使用する。
- 並列処理と乱数の関係: 乱数なし、single process audit。
- CPU/GPU runtime と deterministic flags: Kaggle CPU audit。GPU 不要。
- train cache / test feature regeneration の SHA 記録方針: input file SHA と gzip decompressed SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: 新規 model / submission は作らない。posthoc prediction sample と summary は audit output として保存する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` と `validate-exp` を通してから push する。

## リスク

- リークリスク: context/gate 条件に true TVT、true error、oracle best を混ぜると漏洩になる。コード上は scoring 後の readout に限定する。
- CV/LB 不一致リスク: batch context は pseudo batch の作り方に依存する。global RMSE だけで採用せず、near、worst-well、raw-test parity を必須 guard にする。
- ランタイム/メモリリスク: 3.78M rows と raw well csv を読むため Kaggle CPU memory に注意する。LightGBM 学習はしない。
- 再現性リスク: exp148 / exp073 / exp072 output がローカル未取得の場合、Kaggle input mount 依存になる。実行時の source path / SHA を summary に残す。
