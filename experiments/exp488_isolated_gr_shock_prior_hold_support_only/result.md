# exp488_isolated_gr_shock_prior_hold_support_only 結果

## 結論

Kaggle private CPU version 2でsupport-only Stage A0/A1を完走したが、
`stage0_failed_closed`とする。

zero-shock対照群を使わず、raw shockが多い上位32 wellだけを評価しても、
事前固定した最終AND triggerは183,093行中0行だった。candidateは全行で保存
exp209 parentと同一となり、性能差も0だった。対照群不足ではなく、現条件では
介入機構が発火しないことが分かった。

## 実行内容

- Kaggle kernel: `kentookumura/exp488-gr-shock-support-only-train`
- version / id_no: `2 / 129170127`
- 完了: `2026-07-30 13:42:12 UTC`
- scientific candidate: 1
- raw census: 773 wells
- support-only評価: 32 wells / 183,093 rows
- unchanged exp209 parent-message HMM replay: 32 wells
- candidate state HMM / parent予測rerun / model / booster / PF / Beam / GPU: 全て0

## 結果

| 指標 | 結果 |
| --- | ---: |
| isolated raw-shock rows | 17,047 |
| raw-shock support wells | 763 / 773 |
| support32 final trigger | 0 rows / 0 wells / 0 folds |
| saved parent RMSE | 7.668975975 ft |
| candidate RMSE | 7.668975975 ft |
| 改善 | 0.000000000 ft |
| 改善fold | 0 / 5 |
| by-well delta p95 / worst | 0.0 / 0.0 ft |
| Stage 0 elapsed | 1,899.470 sec |
| full実行時間投影 | 39,059.748 sec（上限30,600 sec） |
| peak RSS | 1.417904 GB |

全foldでcandidateとparentは同一だった。

- fold 0: 8.627291 vs 8.627291 ft
- fold 1: 5.770497 vs 5.770497 ft
- fold 2: 3.308634 vs 3.308634 ft
- fold 3: 6.237724 vs 6.237724 ft
- fold 4: 10.298702 vs 10.298702 ft

## Gate判定

- eligibility: PASS。isolated shock 17,047行、support 763 wellsでsupport32は構成可能。
- technical: FAIL。trigger最小行/well/fold、full runtime投影、
  saved-parent replay parityが不合格。
- scientific: FAIL。trigger行がなく、trigger行改善率とSSE削減は評価不能。
  pooled改善は0 ft、改善foldは0/5。
- decision:
  `stage0_failed_close_without_trigger_threshold_or_output_rescue`

親rerun parityの実測差は最終log summaryに表示されていないため、PASS/FAILだけを
記録する。

## 解釈

raw GRの孤立shock自体は広く存在する。しかし、それと
「past predictiveとleave-one-outの一致」「current emission conflict」を同時に
要求すると、shock集中sampleでも発火行がなくなる。zero-shock対照群を外しても
仮説は支持されなかった。

support32はtarget-freeに選んだmechanism sampleであり、CVやpromotion evidence
ではない。ただし今回はcandidateが一度も発火していないため、全OOFへ拡張して
性能を測る根拠もない。

## 終了判断

事前契約どおり、threshold、window、outputを同じデータで調整する救済は行わない。
Stage 1、inference、submissionへ進まず、このbranchを閉じる。

Kaggle logにgate、metrics、runtime、SHAが全て含まれていたため、output archiveは
取得していない。詳細なSHAは`metrics.json`と`SESSION_NOTES.md`を正とする。
