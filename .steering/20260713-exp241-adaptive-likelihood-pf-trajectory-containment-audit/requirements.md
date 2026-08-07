# 要件

## 依頼

`adaptive_likelihood_pf_trajectory_containment_audit` を exp241 として実装し、exp232 の
target-free gate 発火後に PF の resampling と seed aggregate が長期 path divergence を
増幅したかを train-side で診断する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- exp072-compatible transition、500 particles、128 seeds、resampling threshold、noise、gate を固定する。
- paired T=1 control と gated T=2 treatment は同一 well/seed の stable seed base を使う。
- gate event の選択に true TVT、error、oracle candidate を使わない。
- true TVT は event 後の診断指標にだけ使う。
- LightGBM 0 config、fold 0、booster 0、GPU なし、inference/submission なしとする。

## 受け入れ基準

- well×seed の最初の gate event を起点に、事前固定 horizon
  `8/32/64/128/256/512/1024/end` の cumulative RMSE delta、path divergence、ESS、
  resampling、seed disagreement を出力する。
- seed aggregate の row-level prediction と control/treatment 診断を出力する。
- overall、1000_plus、hidden-like、worst-well readout を保存する。
- Kaggle CPU push 前に active treatment 1、control replay 1、500 particles、128 seeds、
  合計 PF replay 2 を `SESSION_NOTES.md` で再確認する。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
