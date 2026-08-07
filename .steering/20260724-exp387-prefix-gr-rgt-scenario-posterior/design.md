# 設計

## アプローチ

exp386の各scenarioには、固定TVT path、RGT path、graph prior cost、同じouter-train対応から作った
reference-GR templateを持たせる。exp387はmanifest logical SHAを照合してbankをfreezeした後にだけ、
targetの観測GRを開く。GR levelとfirst differenceをwell内robust標準化し、256-row window /
stride64でscenario templateとの差をStudent-t `df=4`尤度へ変換する。sigmaはouter-train LOOの
template残差MAD、clipは5--60 GR APIとする。

scenario stateはexact forward-backwardで周辺化する。stay 0.995、refresh 0.005とし、
共通RGT control nodeを持ち、window境界のTVT差が2 ft以下のscenario間だけswitch可能とする。
incompatible transitionは0。行posteriorはoverlap-addし、primaryをposterior weighted TVTとする。
hard top1は禁止し、GR不足windowではexp386 graph-cost prior meanへexact fallbackする。

## 実験範囲

- 対象実験: `exp387_prefix_gr_rgt_scenario_posterior`
- Route: `pf_beam`
- 親実験: `exp386_cycle_consistent_rgt_scenario_bank`
- 変更する変数: 固定scenario bankへtarget GR Student-t likelihoodとexact posteriorを追加する。
- 固定する変数: exp386 path/value/order/prior、fold、window、df、sigma、transition、評価scope。
- 出力: posterior mean physical TVT path、window posterior、real/circular score。
- 実行量: Stage 0 likelihood audit 1 / 5 folds / full decoder 0。
  Stage 1は別承認時だけ1 variant / 773 exact decoder well-runs。
  fitted model・HMM・PF・Beam・booster・parent regeneration各0。

## 段階と停止条件

1. exp386全gate PASSとmanifest SHA pinまでblocked。
2. Stage 0でparent parity、GR coverage、posterior normalization、known-prefix rolling-origin、
   real-vs-circularをtarget-freeに監査する。
3. 全AND PASS・別承認時だけ773-well Stage 1を実行する。
4. scenario bankとposteriorをSHA freezeした後だけsuffix truthをlate joinする。
5. CV `<=7.20 ft`とfold/scope gateを満たす場合だけ、別のpromotion safety判定を検討する。
6. FAIL時はwindow/df/sigma/transition/temperature/hard top1を救済せず閉じる。

## 再現性設計

- seed policy: RNGなし。fold/well/scenario/window/state順をstable sortする。
- stochastic処理の有無: なし。
- PF/Beam / likelihood-PF / seed baggingの有無: exact forward-backwardのみ。samplingなし。
- 並列処理と乱数の関係: RNGなし。well別並列結果をimmutable keyで再sortする。
- CPU/GPU runtimeとdeterministic flags: CPUのみ、GPUなし。
- train cache / test feature regenerationのSHA記録方針:
  exp386 manifest、reference-GR、real/circular window score、transition、posteriorのlogical SHAを保存する。
- model manifest / prediction / submission SHA記録方針:
  fitted modelなし。decoder contractとOOF prediction SHAを保存し、submission SHAは対象外。
- Kaggle package bootstrap確認方針:
  将来pushする場合はprivate / CPU / internet off、parent kernel sourceとembedded manifest SHAを照合する。
- deterministic anchor:
  初回成功runでは主張せず、promotion前にposterior/prediction content SHA一致rerunを要求する。

## リスク

- リークリスク: suffix truthやFormationがscenario likelihood/transitionへ混入すること。
  parent freeze後のrole read ledgerと0-read hard gateで防ぐ。
- CV/LB不一致リスク: scenario oracleが良くてもGR posteriorがhidden testへ転送しない可能性がある。
  known-prefix rolling-origin、circular control、hidden-likeを全て必須にする。
- ランタイム/メモリリスク: 32 states × overlapping windows × 773 wells。
  16-well resource projectionをStage 0で先行する。
- 再現性リスク: scenario availabilityとequal posteriorの集約順。
  parent orderとstable state/window orderをmanifestへ固定する。
- 過信リスク:長系列GR尤度の積でposteriorが過度に尖る可能性がある。
  valid-pair mean loglik、fixed clip、outer-train LOO sigma、circular controlで監査する。
