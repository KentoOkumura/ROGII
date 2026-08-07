# 要件

## 依頼

- exp226 residual-offset座標でGaussian GR emissionをStudent-tへ置換する高リスク案を、0-HMM Stage 0と条件付きfull HMM Stage 1に分けて設計確定する。
- 実装・実行はまだ行わない。

## 制約

- Route: `pf_beam`。
- Stage 0親は`exp280_exp226_shift_likelihood_separability_readout`、Stage 1 decoder controlは`exp281_exp226_residual_offset_exact_hmm_transition_probe`。
- Student-tは`df=4`、`ell=-0.5*(df+1)*log1p(z^2/df)`、`z=residual/same_exp281_sigma`に固定する。
- Stage 0はexp280と同じ非重複512-row block、13 shifts、fold、missing処理、sigmaでGaussianとStudent-tのrankだけを比較する。
- Stage 1はStudent-t 1 variant × 773 HMM runs。exp226 shape、offset grid/rates、transition、prior、missing補間、sigma、posterior meanを固定する。
- df/scale/likelihood weight/grid、Huber/cap、missing uncertainty、ACF temperingを混ぜない。

## 受け入れ基準

- Stage 0 technical parity、coverage、shift identityがPASSする。
- Student-tがGaussian比でpooled MRRとtop3を各`>=0.01`改善し、両指標を4/5 foldsで改善する。
- 1000+、hidden-like 2面、persistent-offset scopeでMRR/top3を悪化させない。
- real-vs-circular-shuffle gapがGaussian以上で、`|z|>=3` residual contribution blockのtop3/regretを改善する。
- Stage 0全gate PASSと別承認時だけStage 1を実装・実行する。
- Stage 1はexp281比RMSE`>=0.05 ft`、4/5 folds、1000+・hidden-like・p95非悪化、worst`<=+0.25 ft`。direct promotionにはexp226 `9.427110`更新も必要。
- FAIL後のdf/scale/temperature/grid救済、inference、submissionは禁止。

## 次のアクション

固定gate FAILとして閉鎖済み。救済、再実行、inference、submissionへ進めない。

## 2026-07-23 実装承認

- ユーザー依頼により Stage 0 implementation と正規 notebook 採用を承認済みとする。
- Kaggle package / push / run、Stage 1、inference、submission の承認には拡張しない。
- `|z|>=3` residual contribution block は、truth-nearest shift に該当 residual が
  1行以上ある block と固定し、結果を見て share threshold を追加しない。

## 2026-07-23 Stage 1探索実行override

- Stage 0は固定gateをFAILしたが、ユーザーの明示依頼「Stage1に進んでください」により、
  Stage 1の実装・Kaggle private CPU実行を探索的overrideとして承認する。
- Stage 0 FAILをPASSへ読み替えず、Stage 1結果の信頼度と事前条件違反を
  `SESSION_NOTES.md`、`result.md`、`metrics.json`へ明記する。
- 実行量はStudent-t `df=4` 1 variant / 773 HMM well-runs。
  LightGBM config / trained fold / boosterは`0 / 0 / 0`。
- Gaussian parentはSHA固定済みexp281 OOFを読み、parent/control HMMを再実行しない。
- exp281のoffset grid、rate states、transition、prior、sigma、missing補間、
  posterior meanを固定し、emission familyだけをStudent-tへ置換する。
- Stage 1 gateは既存予約どおり、exp281比RMSE`>=0.05 ft`、4/5 folds、
  1000+、hidden-like 2面、well別p95非悪化、worst`<=+0.25 ft`とする。
- exp226 `9.427110`更新はdirect promotionの追加条件として分離して記録する。
- inference、submission、df/scale/temperature/grid、Huber/cap、missing/ACF変更、
  Gaussian control再実行は承認対象外。

## 2026-07-24 Stage 1結果

- Kaggle private CPU version 2で1 variant / 773 HMM well-runsを完了した。
- exp281比RMSE gainは`0.047648 ft`、改善foldは3/5で固定gateをFAILした。
- hidden-like 2面、by-well p95、worst-well safetyもFAILし、
  direct exp226 ceilingも満たさなかった。
- `stage_1_failed_close_without_rescue`として閉じ、承認対象外だった救済、
  inference、submissionは実施しない。
