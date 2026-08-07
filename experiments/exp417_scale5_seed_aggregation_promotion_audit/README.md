# exp417_scale5_seed_aggregation_promotion_audit

## 状態

Kaggle private CPU version 1でStage Aを完了した。technical gateはPASSしたが、
by-well p95 / worst-well guardがFAILしたため、fixed scale-5 direct PF候補を
`stage_a_failed_closed_no_rescue`で閉じた。推論と提出は実行していない。

## 仮説

exp072の128 seed算術平均を、同じseed bankの固定temperature-5
likelihood-weighted meanへ置き換えると、exp410で確認したacross-seed basin平均の
offsetを安全に減らせる。

## 検証方針

- Route: `pf_beam`
- artifact parent: exp404、scientific control: exp072
- Stage A: 保存済みx1.0 control / scale5 candidateの0-PF audit
- temperature: 5.0固定
- PF / model / booster / GPU: 0
- primary: pooled RMSE gain `>=0.05 ft`、4/5 folds
- missing / 1000+ / hidden-like / by-well / fixed blendをAND gateにする
- PASS後のraw-test batch inferenceも別承認

## 実装

- exp404 frozen predictionのraw / decompressed / logical / schema SHAを検証する。
- exp404 scientific contractとwell auditから、x1.0の算術平均とscale5が同じ
  500 particles ×128 seeds、同じseed label、同じGR scaleのreadoutであることを確認する。
- identity freeze後にのみexp226 truth / fold、exp115 hidden-like roleを読む。
- exp072算術平均、exp209 exact-HMM、固定HMM/LikPF 50:50を保存生成物から読み、
  control再実行なしでparityとscientific gateを判定する。
- Stage AのPF / HMM / model / booster / GPU実行はすべて0。

## 所見

scale5は算術平均よりRMSEを`11.594897884 → 10.914522073`へ
`0.680375810 ft`改善し、5/5 folds、raw/missing/high-missing/1000+/
hidden-like 2面、固定HMM/LikPF 50:50も通過した。しかしdirect by-well delta p95は
`+2.941688483 ft`、worst well `70925e23`は`+25.311274575 ft`で、事前上限
`0.0 / 0.25 ft`を破った。平均改善を少数wellの安全性へ一般化できないため棄却する。

実行量は保存readout 1、PF / HMM / Beam / model / booster / GPU 0。input SHA、
same-bank、truth-late、control parity、artifact manifestは全PASSした。

詳細値は`result.md`と`metrics.json`、固定契約はsteeringの`design.md`と
`requirements.md`を正とする。
