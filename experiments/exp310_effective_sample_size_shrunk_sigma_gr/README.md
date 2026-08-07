# exp310_effective_sample_size_shrunk_sigma_gr

## 状態

- ルート: `pf_beam`
- 状態: exp307 promotion gate FAILにより未実装・未実行のまま閉鎖
- CV / LB / Submit: 未実行
- 作成日: 2026-07-21
- 親実験: `exp307_finite_only_robust_sigma_gr`

## 仮説

有限GR残差数が多くても強い自己相関があれば、独立な情報量は見かけより少ない。exp307のwell別finite-MAD `σ_GR`を、自己相関から求める有効標本数`n_eff`に応じてleave-one-well-out priorへ縮約すれば、少数・高相関wellのscale推定分散を下げられる。

## 確定した差分

- raw scale、20 pair未満のfallback `30`、clip `[10, 60]`はexp307と同一。
- 欠損をまたがず、raw行番号が連続するfinite residual run内だけでlag 1--20の自己相関を計算する。
- `tau=max(1,1+2*sum(max(rho_k,0)))`、`n_eff=clip(n/tau,1,n)`とする。
- priorは対象wellを除くraw scaleの中央値、重みは`n_eff/(n_eff+50)`、log scale上で縮約する。
- exp308のGR downweightとexp309のtransition noiseは混ぜない。

## 実行条件

exp307のscale auditだけで、median `n_eff/n <= 0.5`または`n_eff < 50`のwell比率`>= 0.20`を満たす場合だけ1 variant、最大773 HMM well-runsを許可する。満たさなければHMMを実行せず閉じる。実装、Kaggle push、推論、提出は未承認である。

## 検証方針

- exp307比RMSE改善`>=0.03 ft`かつ4/5 folds改善。
- 1000+、spatial hidden-like、typewell-purged hidden-like、by-well p95、worst well、fixed LikPF blendのguardをすべて通過。
- trigger判定後のlag、prior、threshold変更は禁止。

## 所見

設計時点では精度所見はない。自己相関はscaleを直接膨らませず、well別raw scaleをpriorへ縮約する信頼度だけに使う。

詳細は[steering design](../../.steering/20260721-exp310-effective-sample-size-shrunk-sigma-gr/design.md)を正とする。

## 結果

未実行。support audit、trigger判定、HMM prediction、metrics、submissionは存在しない。

## 次

exp307 PASSの必須条件が成立しないためtriggerを評価せず、実装・実行へ進まない。
