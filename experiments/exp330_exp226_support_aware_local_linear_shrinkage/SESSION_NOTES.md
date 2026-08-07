# exp330 セッションノート

## 目的

exp226本体のlocal-linear donor driftを、exp329で検証済みのsupport riskに応じて同一donorのweighted constantへbounded shrinkする。

## 現在の状態

- 2026-07-21: steering、scaffold、式、固定親parameter、gateを確定。
- 2026-07-21: exp329 Stage 0がtechnical/coverage全PASS、scientific gate FAILでclose。必須依存不成立のためdesign only / 未実装 / 未実行のままclose。
- Route: `pf_beam`。
- CV/LB: まだなし。

## 固定事項

- exp329と同じ6-feature risk、q80から連続発火、最大50% shrink。
- fallbackは同じk50/同じweightのweighted constant。zero driftではない。
- raw/smoothed fieldだけを変更する。
- fold別parent kappaをSHA固定し、再fitしない。
- K16、donor選択、distance bucket、ANCC、GR、U projectionは固定。
- HMM、K12/K24 selector、error/bias transferを使わない。

## 実行量

- Stage 0: fixed 32 parity wells、scientific variant 0。
- Stage 1最大: 1 scientific + 1 circular control、5 field builds、1,546 prediction well-runs。
- kappa fit/model/booster/HMM/PF/Beam/control再学習0。

## 再現性

RNGなし。raw identity、parent OOF/kappa、exp329 contract、fold/well/segment/donor、support primitive、risk、raw/smoothed parent/constant/regularized field、real/control predictionのSHAを保存する。

## 次

dependency policyどおりparity preflight、full OOF、inference、submissionを実装・実行しない。同じexp329 riskの救済gridは追加しない。
