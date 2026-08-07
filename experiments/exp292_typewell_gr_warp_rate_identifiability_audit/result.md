# exp292_typewell_gr_warp_rate_identifiability_audit 結果

## 状態

Kaggle private CPU version 1を完了し、事前登録した判定は `FAIL_CLOSE_NO_RESCUE_GRID` となった。
technical guardは通過したが、coverage、識別性、選択改善guardを満たさなかった。inference、
selected row prediction、submissionは生成していない。

## 仮説

known prefixだけでType Well GRをhorizontal GRへrobust calibrationした後、exp268の固定rate
candidateが作るforward GRのGaussian residual、NCC、derivative residualを読むと、truth-best
candidateをstable circular-shuffleより良く識別できる。

## 設定

- 親: `exp268_multi_scale_initial_rate_candidates`
- Route: `pf_beam`
- 候補: `tail30 / w32 / w64 / w128 / w256`
- horizon: `H128 / H256 / H512`、primary `H256`
- control: stable within-well circular-shuffle、always-tail30 safe
- 実行量: 1 audit variant / 0 configs / 0 trained folds / 0 boosters / 0 HMM/PF runs
- evaluation folds: well単位5 folds（fitなし）
- inference / submission: disabled

## 結果

| guard / メトリック | 実測 | 条件 | 判定 |
| --- | ---: | ---: | --- |
| technical validation | PASS | PASS | PASS |
| H256 eligible wells | 29/773 = 3.7516% | 90%以上 | FAIL |
| H256 eligible rows | 3.6178% | 90%以上 | FAIL |
| candidate-best AUC real | 0.484190 | - | - |
| candidate-best AUC shuffled | 0.531181 | - | - |
| AUC lift real - shuffled | -0.046991 | 0.02以上 | FAIL |
| AUC lift正のfold | 0/5 | 4/5以上 | FAIL |
| top1 RMSE / safe RMSE | 11.938287 / 11.938287 | - | - |
| pooled RMSE gain | 0.000000 ft | 0.10 ft以上 | FAIL |
| RMSE改善fold | 0/5 | 4/5以上 | FAIL |
| 1000+ / hidden-like非悪化 | 3/3 | 3/3 | PASS（safe fallback） |

H256 realでeligibleだった29 wellsもtop1はすべて`hmm_ir_tail30`で、全773 wellsの最終選択がsafeと
同一だった。したがってsubgroup非悪化は追加GR識別性の正の証拠ではなく、safe fallbackの結果である。

主なH256 real fallback reasonは `common_finite_pairs_below_minimum` 219 wells、tail30の
`candidate_forward_gr_std_below_minimum` 159、`common_derivative_pairs_below_minimum` 152、tail30の
`candidate_forward_derivative_below_minimum` 108、prefix Type Well GR std不足69、calibration slope範囲外36だった。

## 再現性

- Kaggle kernel: `kentookumura/exp292-typewell-gr-warp-identifiability-train`, id `127888550`
- runtime: 122.469秒
- package run config SHA: `d3bb0f040f77a3c057744b44665b7f6f7f207c882fc212a4186c32fdee74ca44`
- train source SHA: `4488aece9cf0dd998c4b25ba6a2d159d32db2b29686119e94a7cc3f41f31c5ab`
- fold manifest SHA: `2c4d67f2d47cc44e215e16b7c4312631ed2c3ed51953b7d3667d3610b108493e`
- target-free score decompressed content SHA: `9165d52fb24152ea17c2a620247177ab4e4e223306869fba9bf5e9f59ca0ed01`
- target-free score schema SHA: `62181c0f628ef61c791918b1a3bb4813f36873f4679d3f72e21ba88adfdd7d9c`
- target-free selection content SHA: `5cebdca26a1a33f4d15fa141d465c7e2d187f1d42f555b9fc95edc61da03fdc1`
- target-free selection schema SHA: `2be8f05d0fc878faf8534688c3e6f0bd8424183e12fe786bfd84a6dd6136bd8d`
- summary raw SHA: `4562c33b53bcf180874ef8962de618b56b50d4ce9e86688bb69d6c133522c557`

## 解釈

固定したType Well forward-GR scoreは、厳格なcoverage/variance/derivative契約の下でほとんどのwellを
評価できず、評価可能な小集団でもshuffleを上回る識別性を示さなかった。exp268のrate candidateに
約0.10 ftのoracle headroomはあるが、このfrequency-warp readoutではdeployableに回収できない。

## 次

事前登録どおりfrequency-warp rate branchを閉じる。同一truthを見た後のrate window、horizon、affine、
coverage、variance/gradient threshold、component weight、shuffle、tie、guardの救済gridは行わない。
top1 replacement、raw-test inference、submissionへ進めない。
