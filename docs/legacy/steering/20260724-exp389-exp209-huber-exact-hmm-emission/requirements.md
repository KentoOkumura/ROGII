# 要件

## 目的

exp209のGaussian exact HMMに対するfixed Huber emissionの単独効果を、
exp281 residual-offset系と混ぜずに検証できるdesign-only契約を確定する。

## 依頼

本来の依頼を、`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`の
absolute-TVT exact HMMでGaussian row emissionだけを固定Huberへ置換する
独立実験として設計確定する。

誤ってexp281 residual-offset HMMを親にしたexp357は履歴として残すが、
本実験の親、control、入力、性能根拠には使わない。新番号exp389でbacklog、
実験scaffold、steering、固定configを作成する。2026-07-24の追加ユーザー指示
「exp389を実装してください」により、compact self-contained実装と専用テストまで
承認範囲を拡張する。

## 制約

- Route: `pf_beam`
- 親/control:
  `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- saved Gaussian direct RMSE:
  `11.938287234887435`
- 変更はexp209の行別emission
  `-0.5*min(z^2,600)`からfixed Huber `delta=1.345`への置換だけとする。
- Huberは
  `-0.5*z^2` if `|z|<=1.345`、
  `-(1.345*|z|-0.5*1.345^2)` otherwise とし、追加clip、temperature、
  mixture、scale変更を入れない。
- absolute TVT座標、grid、41 rate states、transition、prior、sigma、
  missing-GR処理、Type Well GR、momentum、likelihood weight、
  posterior-mean outputはexp209から変更しない。
- exp226はwell / row / suffix-offset / reporting fold identityだけに使い、
  `tvt_geop`、prediction、`gr_delta`をdecoderへ渡さない。
- exp357のshift-rank Stage 0、exp281 residual-offset座標、exp374 Student-tを
  candidateや解禁条件へ持ち込まない。
- 0-HMM proxyは置かず、将来別承認された場合だけactual exact-HMM 1 variant /
  773 wellsを直接評価する。
- exp209 Gaussian HMMは保存済み予測とSHAをload-onlyで使い、再実行しない。
- 将来実行量はscientific variant 1、HMM well-run 773、
  model / LightGBM config / trained fold / booster / PF / Beam 0、
  parent control再実行0とする。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 現在の承認範囲はcompact self-contained実装と専用テストまで。
  2026-07-24の追加ユーザー指示「実行してください」により、正規train
  Notebook採用とKaggle private CPU package/push/runまで承認範囲を拡張する。
  inference、submissionは引き続き別承認とする。

## 受け入れ基準

- technical gate:
  input SHA、saved control parity `atol=1e-5`、3,783,989 rows / 773 wells、
  ID/order/fold identity、finite coverage 1.0、posterior normalization、
  all-well status、truth-before-freeze 0をすべて要求する。
- scientific gate:
  exp209 Gaussian比direct RMSE `>=0.05 ft`改善、4/5 folds改善、
  raw-GR observed `>=0.05 ft`改善を要求する。
- safety gate:
  raw-GR missing、高missing-fraction wells、1000+、hidden-like spatial /
  typewell-purgedを非悪化、by-well delta p95 `<=0`、worst well
  `<=+0.25 ft`とする。
- secondary gate:
  fixed LikPF/HMM 50:50 control `10.269696146642758`に対する同じ50:50
  candidate blendの非悪化を要求する。blend weightは探索しない。
- 全gateのANDのみPASSとし、FAIL時は
  `huber_exp209_failed_close_without_rescue`で閉じる。
- delta、scale、clip、temperature、sigma、missing weight、transition、
  grid、prior、blend weightの同一OOF救済を禁止する。
- PASSしてもinference実装、raw-test再生成、submissionは自動承認しない。
- gzip predictionを作る場合はraw gzip SHAとdecompressed/logical content SHAを
  分け、主証拠にはdecompressed/logical SHAを使う。
