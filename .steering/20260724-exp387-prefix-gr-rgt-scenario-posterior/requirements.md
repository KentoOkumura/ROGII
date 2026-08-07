# 要件

## 依頼

exp386が生成した8〜32個のRGT物理pathを一切変更せず、target坑井の観測GRと既知TVT prefixから
各scenarioの尤度を計算する。物理的に互換なscenario間だけをexact forward-backwardで周辺化し、
posterior mean TVTを生成する。今回はバックログ、実験scaffold、steeringと設計だけを確定し、
実装・実行は行わない。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md`に従い、parent SHA、window/state順、posterior/prediction SHAを記録する。
- exp386 Stage 0/1/2全PASSとscenario-bank manifest SHA pinまで実装しない。
- exp386のcandidate値、順序、prior cost、reference-GR templateを変更・再生成しない。
- targetは`MD/X/Y/Z/GR/TVT_input`だけをpre-freezeに使い、生Formationとsuffix truthを読まない。
- GR levelとfirst difference、256-row/stride64、Student-t `df=4`、sigma `[5,60]`、
  circular shift 512、stay 0.995 / refresh 0.005を固定する。
- scenario switchは共通RGT control nodeかつboundary差2 ft以下だけに許可する。
- hard top1、likelihood/transition/temperature grid、candidate改変、ML/HMM/PF/Beam、
  current-test、inference、submissionは禁止する。

## 受け入れ基準

- exp386 manifest SHA、scenario count 8--32、candidate value/orderをfit前に完全照合する。
- eligible GR window率`>=0.25`、eligible well率`>=0.50`、posterior正規化誤差`<=1e-12`、
  ineligible rowのprior-only parity差`<=1e-8 ft`を満たす。
- 512-row known-prefix rolling-originでprior meanより`>=0.25 ft`改善し、4/5 folds以上で正である。
- real GRのMRRがcircular controlより`>=0.02`改善し、entropyも4/5 folds以上で良い。
- Stage 1はpooled RMSE`<=7.20 ft`かつexp226比`>=2.0 ft`改善、4/5 folds以上正、
  1000+で`>=2.0 ft`、hidden-like 2面で各`>=1.5 ft`改善し、near悪化`<=0.05 ft`である。
- exp386 scenario oracle `<=5.50 ft`をparity確認する。
- promotion時は別承認とし、by-well p95悪化0、worst悪化`<=0.25 ft`、
  deterministic rerun SHA一致を追加で要求する。
- parent manifest、real/circular score、transition、posterior、OOF predictionのlogical/decompressed SHAを保存する。
- gzip生成物を比較する場合は、raw `.csv.gz` SHAではなくdecompressed content SHAを主証拠とする。
