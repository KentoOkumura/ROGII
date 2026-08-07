# exp299_base_posterior_self_gr_boundary_handoff 結果

## 状態

Kaggle private CPU version 2（id_no `127957958`）を同じcanonical slugで完了した。最終statusは`completed_train_side_guard_failed`。version 1のexp209 CSV parity bugは解消したが、候補RMSEと事前固定performance gateが不合格のためbranchを閉じる。実推論とsubmissionは行わない。

## 仮説

outside self-GRをexact 0にしながら、base-only posteriorが境界へ近づくとrow全体をneutralへhandoffし、range内ではsupport総massを保持する条件付きlikelihoodとしてself-GRを使えば、exp296のboundary wallを作らずexp223のinside signalを利用できる。

## 設定

- 親: `exp223_joint_typewell_self_gr_hmm_likelihood_probe`
- negative/reference: `exp296_exp223_self_gr_known_tvt_support_gate`
- base parity: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `ensemble`
- validation: train-side no-training / official unknown suffix / stable SHA256 reporting 5 folds
- planned: 1 scientific variant / Pass A+B 1,546 HMM well-runs / 0 booster / CPU

## 結果

| メトリック | 値 |
| --- | --- |
| 専用tests | 12 passed |
| repository tests | 354 passed / 1 skipped / exp296既知2 failed |
| py_compile / Ruff F821 / Jupytext round-trip | PASS / PASS / PASS |
| strict experiment / project template validation | PASS / PASS |
| 実行count | Pass A/B `773/773`、合計1,546 HMM well-runs |
| Kaggle runtime | `22,481.454 sec`（約6.245時間） |
| exp209 parity | ordered 3,783,989 rows / 773 wells、max/mean abs `0.0 / 0.0 ft`、PASS |
| candidate RMSE | `11.789577561` |
| saved exp223 RMSE | `11.349942946` |
| delta vs exp223 | `+0.439634615 ft`、改善fold `0/5` |
| delta vs exp296 | `-0.370171579 ft` |
| technical gate | `24/25` PASS。唯一のFAILはrow gate max `1.0000000000000029 > 1.0` |
| performance gate | `2/11` PASS、総合FAIL |
| inside / outside delta vs exp223 | `+0.415018842 / +0.478309948 ft` |
| upper-boundary 0--12 / distance 1000+ delta | `+0.971109945 / +0.508852469 ft` |
| hidden-like spatial / typewell-purged delta | `-0.014390249 / -0.013615362 ft` |
| by-well p95 / worst-well delta | `+1.454561921 / +35.990274405 ft` |
| Public / Private LB | 未提出 |

## 再現性

- deterministic anchor: false
- seed policy: no RNG + stable SHA256 reporting folds
- kernel: `kentookumura/exp299-base-post-self-gr-boundary-handoff-train` version 2 / id_no `127957958`
- source / pushed config / canonical Notebook SHA: `d033f3e4...f66f` / `ebaad8ee...b6c4` / `3133d91c...27c`
- prediction decompressed SHA: `6d354abc32df1989ed2a74da16f7e2dbbf7a99e2110a8ec216dad7ad2611a28e`
- OOF readout decompressed SHA: `2738281aa92f5d54c8c0f5172b9e3b262945d055c8b6db5ea5b4c9af3cac7266`
- summary SHA: `c5e98734355fbec17b3fccb0e45cfa84034f1e0a6203ee8dd2b6b0e8df1efeae`
- CV、gate、生成物path、SHAは完了ログで確認できたため、Kaggle output archiveは取得していない。
- model / submission SHA: 対象外

## 解釈

version 2ではexp209 parityがexact 0となり、version 1のserialization contract bugは解消した。outside contribution exact 0、boundary neutral、conditional mass preservation、late truth/control joinも満たした。technical gateの唯一のFAILはrow gate上限の`2.9e-15`超過で、浮動小数丸めの範囲である。ただしperformanceは9/11項目が明確にFAILしており、この微小technical超過を許容しても結論は変わらない。

exp296からは`0.370172 ft`回復したため、base-posterior handoffはstrict state-wise wallの悪化を一部緩和した。しかしexp223比ではoverall `+0.439635 ft`、0/5 folds、inside/outsideの双方が悪化し、boundary、1000+、p95、worst-wellも大幅に悪化した。hidden-like 2面の約`-0.014 ft`だけでは相殺できない。outside exact-zeroを維持したままconditional normalizationでsupport massを保存する今回の一体policyは、exp223のself-GR signalを安全に利用する方法として棄却する。

全repository testsの2 failureはexp296の完了後statusと`run_variant=false`に対する既存test期待の不一致であり、今回変更していない既知failureである。exp299専用testsは全PASSした。

## 次

事前固定fail actionどおり、handoff/fade/normalizer/alpha/clip/support/threshold救済を行わずbranchを閉じる。version 3 repush、inference、submissionへ進めない。本結果だけを根拠にした救済backlogは追加しない。
