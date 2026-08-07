# タスクリスト

## 目的

旧exp328のcausal affine仮説をexp209直系の独立実験として固定し、exp338 chainに混入させず、事前定義したStage 0 gateで採否を確定する。

## 完了

- 2026-07-22: exp209直系の独立実験`exp345_exp209_time_varying_gr_affine_calibration_hmm`を作成し、steering、単一変更、固定項目、leakage boundary、stage別実行量、promotion gate、再現性方針を記録した。
- 2026-07-22: compact self-contained Jupytext train、last-640 mask、outer-fold process noise、causal EKF、schedule freeze、exp209 exact-HMM再実行、late truth readout、段階gate、SHA出力を実装した。
- 2026-07-22: identity scheduleとexp209親HMMの数値一致を含む専用test、Jupytext round-trip、構文、ruff、strict experiment validationを通した。
- 2026-07-22: Kaggle CPU version 1でstable SHA順32-well microbenchmarkを完了し、64/64 HMM runs、外挿1.2867時間でruntime gateをPASSした。
- 2026-07-22: negative scientific preview提示後の別承認により、Kaggle CPU version 2でStage 0 fullを実行した。
- 2026-07-22: parent 773 + variant 773 = 1,546/1,546 HMM runs、494,720 finite rows、fallback 0%、runtime 1.3531時間でtechnical gateをPASSした。
- 2026-07-22: pooled `+0.169505 ft`、4/5 folds、GR NLL、boundaryはPASSしたが、hidden-like 2 scopeの証拠欠落とworst well `+9.354827 ft`によりscientific AND gateをFAILした。
- 2026-07-22: `stage_failed_close_without_rescue`として全run flagとpush承認を無効化し、Stage 1を禁止してbranchを閉じた。
- 2026-07-22: promotion gate、paired metrics、各SHA、result、metrics、session notes、summary、戦略判断を記録し、完了済みbacklogを削除した。

## 最終判定

- 状態: `stage_0_full_failed_closed`
- technical gate: PASS
- scientific gate: FAIL
- decision: `stage_failed_close_without_rescue`
- Stage 1 eligible: false
- inference / submission: 未実行

## 禁止

- Stage 1 full suffix audit。
- version 3または別slugへの再push。
- affine parameter / process noise / transition / sigma / missing weight / grid / blendのpost-hoc救済。
- inference、submission。
- exp338との組合せ、exp338依存化、exp338 successor chainへの合流。
- 旧exp328のreopen、reparent、実装再開。

## 次のアクション

本familyでは追加作業なし。同じ仮説を再訪する場合は、独立した新しい根拠、別実験の事前設計、ユーザー確認を必須とする。
