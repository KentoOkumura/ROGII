# 要件

## 依頼

exp374でexact HMMへ適用したfixed Student-t GR emissionを、現行
temperature-5 likelihood-PFのparticle filtering尤度へ移植するdesign-only
実験を作成する。

## 根拠

- exp374のdf=4 Student-tはexp209 Gaussian比でdirect RMSEを
  `0.217808533 ft`改善し、4/5 foldsで改善した。
- 一方、by-well delta p95 `+0.982661 ft`、worst `+35.015963 ft`で
  tail safetyを大幅にFAILした。
- Student-tをPF particle likelihoodとして使い、resampling軌道まで変える実験はない。

## 制約

- Routeは`pf_beam`。科学的親exp417、実装親・保存control exp404。
- `df=4`、x1.0 GR scale、temperature `5.0`を固定する。
- Gaussian/Student-t mixture、df/scale/temperature/clip探索は禁止。
- Stage 0はfixed32 technical preflight。Stage 1は全PASS・別承認時のみ773 wells。
- 保存controlを使い、control PFは再実行しない。
- 2026-07-30に実装、Stage 0実行を承認済み。Stage 0はKaggle CPU v2で
  16/16 technical gateをPASSした。
- 追加依頼`Stage1へ進んでください`により、同じ科学契約の全773 wells
  Stage 1実装、canonical package、push/runを別承認済み。
- Stage 1 version 3は18/18 technical checksをPASSしたが、pooled改善
  `+0.017424990 ft`、2/5 folds、well-tailで科学gateをFAILしterminal close。
- inference、submissionは実行しない。

## 受け入れ基準

- Student-t式と変更しないPF契約が一意である。
- Stage 0/1の実行量、truth-late、seed、SHA、promotion gateが固定されている。
- exp374の平均signalとtail failureの双方を設計根拠として明記する。
- 一項目FAIL時の救済禁止とbranch closeが事前登録されている。
