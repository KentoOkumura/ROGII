# 要件

## 依頼

`two_regime_rate_noise_pf` を `exp242` として実装する。exp072-compatible likelihood-PF の
particle stateを `(position, rate, regime)` に拡張し、stickyな `smooth / turn` の2状態だけを
持たせる。`smooth` は exp072 の rate process noise を保ち、`turn` だけ4倍にする。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- transition matrix は行を現在状態、列を次状態として
  `[[0.9998, 0.0002], [0.02, 0.98]]` に固定する。
- 初期500粒子は `smooth=495 / turn=5`、turn multiplierは`4.0`に固定する。
- exp072-compatible Gaussian likelihood、momentum、position noise、resampling threshold、
  resampling jitter、500 particles、128 seeds、seed mean aggregationを変更しない。
- continuous acceleration、target/error/oracleによるregime切替、adaptive likelihood、
  position-noise変更、particle増加を禁止する。
- 保存済みexp072 `likpf_mean`を比較controlとし、control PFは再生成しない。
- train-side auditのみを実装し、raw-test inferenceとsubmissionは無効化する。

## 受け入れ基準

- 新規variantは`two_regime_k4`の1件だけ、LightGBM config / fold / boosterは`0 / 0 / 0`である。
- regimeをresampling時にposition/rateと一緒に継承する。
- overall、distance bucket、1000_plus、hidden-like、by-well、worst-wellを保存する。
- regime occupancy、entry/exit、switch回数、turn particle mass、ESS、resamplingを保存する。
- stable per-well/variant seedとsingle-worker executionを実装する。
- notebook上で入力、固定contract、実行variant、主要metrics、生成物を追える。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
