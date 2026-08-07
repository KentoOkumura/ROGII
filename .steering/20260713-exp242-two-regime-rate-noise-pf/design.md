# 設計

## アプローチ

exp232で再構成済みのexp072 pseudo-tail評価面とraw horizontal/typewell GRを使う。
各particleに整数regimeを追加し、各rowのpropagation前に固定Markov transitionをsampleする。
`smooth`では`velocity_noise=0.002`、`turn`では`0.008`を使う。それ以外のtransition、
Gaussian observation likelihood、ESS resampling、seed meanはexp072-compatible設定を固定する。

初期regimeは定常turn比率約0.99%に対応する`495:5`を各seedで必ず確保し、stable seedで
particle indexを決定的にshuffleする。resamplingではancestorのregimeもコピーし、regime自体へ
別のjitterやtarget-derived gateを加えない。

## 仮説

全粒子のprocess noiseを広げず、約1%のstickyな`turn`粒子だけに4倍のrate noiseを与えると、
exp072の通常軌道を維持しながら一部の急なrate変化を追跡できる。

## 実験範囲

- 対象実験: `exp242_two_regime_rate_noise_pf`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`
- 実装参照: `exp232_adaptive_robust_likelihood_pf`
- 変更する変数: latent 2-regime stateとregime別rate process noiseだけ。
- 固定する変数: position/rateの定義、momentum、position noise、Gaussian GR likelihood、
  GR sigma、particle/seed数、ESS threshold、resampling jitter、seed mean、score rows。

## 再現性設計

- seed policy: `sha256(exp242, well, two_regime_k4)`のbaseにseed indexを加える。
- stochastic 処理の有無: particle初期化、regime transition、rate/position propagation、
  conditional resamplingに乱数を使う。
- PF/Beam / likelihood-PF / seed bagging の有無: likelihood-PF 1 variant、128 seed mean。
- 並列処理と乱数の関係: Numba kernelをsingle workerで実行し、well間のglobal shared RNGを使わない。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU false、internet false。
- train cache / test feature regeneration の SHA 記録方針: input cacheとrow prediction gzipの
  decompressed content SHAを記録する。testは生成しない。
- model manifest / prediction / submission SHA 記録方針: model/submissionなし。row prediction、
  regime diagnostics、summaryのSHAを記録する。
- Kaggle package bootstrap 確認方針: push前に生成notebook内のconfig、helper、kernel source、
  CPU/internet metadataを確認する。

## リスク

- リークリスク: true TVTはtrain-side scoreにのみ使用し、regime transitionやparticle weightに使わない。
- CV/LB 不一致リスク: pseudo-tail train-side auditであり、raw-testやPublic LBへ直接進めない。
- ランタイム/メモリリスク: 500 particles x 128 seeds x 773 wells。新規variant 1件のみ実行し、
  seed-level predictionはwellごとにaggregateして解放する。
- 再現性リスク: regime transitionで乱数消費が増える。stable per-well seed、variant固定、single worker、
  input/output SHAで追跡する。exp072 saved controlとのseed-paired因果比較ではない点を明記する。

## 次のアクション

静的検証後、Kaggle CPU push前に単一variant、500 particles、128 seeds、control再生成なしを
ユーザーと再確認する。実行後はoverall/1000_plus/hidden-like/worst-well guardで採否を決める。
