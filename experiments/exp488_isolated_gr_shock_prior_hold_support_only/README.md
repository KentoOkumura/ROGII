# exp488_isolated_gr_shock_prior_hold_support_only

## 状態

- Route: `pf_beam`
- 状態: `stage0_failed_closed`
- 親実験: `exp482_isolated_gr_shock_prior_hold`
- 科学的親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Kaggle: version 2 / `id_no=129170127` / COMPLETE
- Stage 1、inference、submission: 実施しない

## 仮説

exp482と同じisolated raw-GR shock triggerを使い、current observationだけを
除いたrow-local posterior meanへ発火行だけ置き換える。親HMM stateと後続予測は
変更しない。

exp482との唯一の科学的差分は、zero-shock control wellを要求せず、raw-only
censusのshock count降順、suffix rows降順、well ID昇順で選んだsupport top32
だけをStage A1で評価することである。

## 検証方針

support32上で保存exp209 parentとcandidateを同じ行で比較する。manifest、
message、trigger、predictionをfreezeするまでtruth / fold / errorを読まず、
事前固定したtechnical / scientific gateをAND判定する。support32は
mechanism sampleであり、CVやpromotion evidenceとは扱わない。

## 所見

raw censusではisolated shockが17,047行、763/773 wellsにあった。しかし、
support top32の183,093行に対して、raw shock・message agreement・current
emission conflictの最終AND triggerは0行だった。

保存exp209 parentとcandidateは全行で同一だった。

- parent / candidate RMSE: `7.668975975 / 7.668975975 ft`
- 改善: `0.0 ft`
- trigger: `0 rows / 0 wells / 0 folds`
- improving folds: `0 / 5`
- technical / scientific gate: `FAIL / FAIL`

full runtime投影も`39,059.748 > 30,600 sec`で、saved-parent replay parityも
FAILした。詳細は`result.md`と`metrics.json`に記録する。

## 結論

zero-shock対照群の有無ではなく、固定した完全triggerが非発火だった。
事前契約どおりthreshold、window、outputの救済は行わず、
`stage0_failed_close_without_trigger_threshold_or_output_rescue`で閉じる。
