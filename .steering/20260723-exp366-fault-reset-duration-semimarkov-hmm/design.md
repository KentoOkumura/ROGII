# 設計

## アプローチ

visible prefixで求めたrobust分布に対して、raw GR change zと保存済みexp209 pathのemission surpriseが
ともに99.5 percentile以上の行だけをtriggerにする。trigger後512行は次triggerを無視する。

Stage 0ではtriggerごとにbase pathと、positionを`±6.3/±12.6 ft`だけshiftしたbranchを作る。
各jumpを`128/256/512`行保持する12候補とbaseを、exp209 GR log emission累積値で順位付けする。
trigger、path、score、selectionのcontent SHAをfreeze後にtruthをjoinする。triggerはexp209 horizon
RMSE 10 ft以上のAUC、branchはalternative within-10 coverageとselected MRRで評価する。

Stage 1ではtrigger時のexp209 posteriorをposition gridで`±18/±36 cells` shiftし、rate分布はcopyする。
baseを必ず保持し、branch内はexp209 exact transition/emissionを使う。固定horizon終了時、
base比5 log units以上ならreset branchをcommitし、そうでなければbaseを採る。これは
`(position, rate, reset_jump, remaining_duration)`のexplicit-duration semi-Markov branchである。

## 実験範囲

- 対象実験: `exp366_fault_reset_duration_semimarkov_hmm`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更する変数: trigger時だけのreset jumpとremaining duration。
- 固定する変数: branch内exp209 dynamics、grid、emission、sigma、rate state、posterior output。
- Stage 0 gate: trigger AUC`>=0.60`、circular差`>=0.05`、event率`[0.001,0.10]`、
  alternative within10 coverage`>=0.60`、selected MRR gain`>=0.01`、4/5 folds、
  hidden-like 2面正方向。
- Stage 1 gate: exp209比`>=0.05 ft`、4/5 folds、1000+/hidden-like/p95回帰
  `<=0.02 ft`、worst回帰`<=0.25 ft`。

## 再現性設計

- seed policy: RNGなし。base、jump magnitude、durationの順を固定。
- stochastic処理、PF、seed bagging: なし。
- CPU single worker、GPU off、上限30,600秒 / 25GB。
- trigger ledger、branch path/score、selection、predictionのcontent SHAを保存する。
- truth/controlはfreeze後にjoinし、gzipはdecompressed SHAを使う。

## リスク

- リーク: errorでtrigger/commitを決める危険。両方ともGR evidenceだけでfreezeする。
- CV/LB不一致: validationのfault頻度がtestと異なる可能性。
- runtime/memory: triggerが多いとbranchが増える。refractoryと上限でfail closedする。
- 再現性: branch orderとsingle workerを固定する。
- 科学リスク: exp289/290/231が先行条件を支持しておらず、現優先度は保留。
