# 要件

## 依頼

`fixed_lag_particle_smoother_pf` backlog を exp235 として実装する。exp072 likelihood-PF の transition、Gaussian GR likelihood、resampling、500 particles、128 seeds は固定し、particle state と ancestor を bounded ring buffer に保存する。`lag=64/128/256` の後続 GR を使って直前 state を再推定する。

## 制約

- Route: `pf_beam`
- future GR は同一 well の evaluation tail の raw GR / typewell GR のみ。
- future TVT、target/error、oracle、well 横断情報は使用しない。
- tail 末端は forward PF estimate に fallback する。
- inference / submission は train-side guard 通過後まで無効。

## 受け入れ基準

- lag 64/128/256 と forward control を同じ rows / wells で比較できる。
- ancestor trace が実際に lag 行前の state を返し、row order と tail fallback が検証される。
- overall、step-delta、particle coverage、first-loss、1000+、hidden-like、worst-well、memory/runtime を保存する。
- stochastic seed、input SHA、生成物の decompressed SHA、Kaggle kernel version を記録する。
