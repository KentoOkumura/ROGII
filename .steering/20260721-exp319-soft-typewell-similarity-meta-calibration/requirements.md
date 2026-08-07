# 要件

## 依頼

exact同一Type Well群にpeerがないwellへ、Type Well content similarityだけでGR calibration/noise priorをsoft transferできるか監査する。設計のみで実装しない。

## 制約

- Route: `pf_beam`。exp311/313 PASSが先行条件。
- similarity inputはcanonical Type Well GR contentだけ。horizontal suffix、truth、error、well IDは禁止。
- descriptor、distance、k=3、kernel、temperature、max distanceを固定する。
- exp313 fallbackへ採用するのは本exp全gate PASS後だけ。

## 受け入れ基準

- leave-one-typewell-group-outを5 well folds内にnestedする。
- robust descriptors 9種、diagonal Mahalanobis、exp kernel、outer-train p90 cutoffを用いる。
- group-out gain 0.03 ft、4/5 folds、permuted差 0.03 ft、far fallback非悪化、worst +0.25 ft以下を要求する。
- exact group pathを置換せずunseen/singleton fallbackだけを対象にする。
