# 要件

## 依頼

- `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`のabsolute-TV​T
  exact HMMを科学的親とし、Gaussian GR emissionだけを固定`df=4` Student-tへ
  置換する独立実験を新番号で設計する。
- バックログ、steering、実験ディレクトリを作成して設計を確定する。
- コード、Jupytext source、Notebook、test、Kaggle packageはまだ実装しない。

## 仮説

exp209のGaussian二乗誤差は、一部の大きなGR残差を過大評価し、absolute-TV​T
posteriorを誤ったmodeへ固定している可能性がある。同じsigmaとexact HMMを維持し、
heavy-tailなStudent-t尤度だけへ置換すれば、large-residual rowの影響を抑えて
exp209 direct pathを改善できる。

## 制約

- Route: `pf_beam`。
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`。
- 変更は行別GR emissionだけとする。
  - Gaussian control:
    `-0.5 * min(z^2, 600)`
  - Student-t candidate:
    `-0.5 * (df + 1) * log1p(z^2 / df)`、`df=4`
- `z`のsigma、known-prefix calibration、raw GR missing補間、Type Well GR、
  absolute-TV​T grid、rate states、transition、prior、momentum、likelihood weight、
  posterior-mean出力をexp209から変更しない。
- exp226 `tvt_geop`、residual-offset座標、segment/rate prior、GR affine、
  missing-distance weight、sigma変更、ACF temperingを導入しない。
- Gaussian controlはSHA固定済みexp209 OOFを保存値として比較し、再実行しない。
- exp342のshift-rank Stage 0は使用しない。将来の科学実行はStudent-t 1 variant /
  773 HMM well-runsを直接評価する。
- 実装、正規Notebook採用、Kaggle実行、inference、submissionはそれぞれ別承認を必要とする。
- 再現性は`docs/06_reproducibility.md`に従い、gzipはdecompressed content SHAを
  主証拠とする。

## 将来実行時の固定評価

- 期待行数 / well数 / fold: `3,783,989 / 773 / 5`。
- primary control: exp209 Gaussian exact HMM
  `11.938287234887435`。
- secondary control: exp209 fixed LikPF/HMM 50:50
  `10.269696146642758`。同じ保存LikPFと固定weightだけを使う。
- reporting foldsはexp226保存OOFのgroup-safe foldを使うが、exp226 predictionや
  `tvt_geop`をdecoderへ渡さない。
- candidate predictionとlogical content SHAをfreezeした後だけ、
  unknown-suffix truth、error、hidden-like role、scope readoutをjoinする。

## 受け入れ基準

technical gateはすべて必須とする。

- exp209 HMM cache、exp072 LikPF cache、fold assignment、hidden-like assignmentの
  SHAが固定値と一致する。
- parent Gaussian direct RMSEとfixed 50:50 RMSEが保存値へ`1e-5 ft`以内で一致する。
- 773 wells / 3,783,989 rows、ID/order/row identity、fold identityが一致する。
- candidate prediction finite coverage、posterior normalization、well statusが
  すべてPASSする。
- truth/error access before candidate freezeが0行である。

scientific gateは次のANDとする。

- Student-t direct RMSEがexp209 Gaussian directより`>=0.05 ft`改善する。
- direct RMSEを4/5 folds以上で改善する。
- raw-GR observed rowで`>=0.05 ft`改善し、raw-GR missing rowと
  high-missing-fraction wellsを悪化させない。
- 1000+、hidden-like spatial、hidden-like typewell-purgedを悪化させない。
- by-well RMSE delta p95が`<=0 ft`、worst-well regressionが`<=+0.25 ft`。
- fixed LikPF/HMM 50:50が保存Gaussian 50:50を悪化させない。

いずれかがFAILなら`student_t_exp209_failed_close_without_rescue`として閉じる。
df、scale、temperature、clip、mixture、Huber、sigma、missing weight、transition、
grid、blend weightの救済や同一OOF再実行は行わない。

## 2026-07-24 実装承認

- ユーザー指示「exp374を実装してください」により、compact self-contained
  train候補、fail-closed inference候補、専用contract testの実装を承認済みとする。
- 正規Notebook採用、Kaggle package/push/run、raw-test inference、submissionは
  この承認に含めない。

## 2026-07-24 実行承認

- ユーザー指示「実行してください」により、正規train Notebook採用と
  Kaggle package/push/runを承認済みとする。
- 実行量はStudent-t 1 variant / 773 HMM well-runs / model・trained fold・
  booster・Gaussian control再実行各0から変更しない。
- raw-test inferenceとsubmissionは承認対象外とする。

## 次のアクション

Kaggle version 1は完了し、technical gate PASS、direct`+0.217809 ft`、
4/5 folds改善だった。一方、by-well p95`+0.982661 ft`とworst
`+35.015963 ft`が固定gateをFAILしたため、事前登録どおり救済・再実行・
inference・submissionなしでterminal closeする。
