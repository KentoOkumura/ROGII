# exp507_exp504_nested_rank_compact_addonly_on_exp413 結果

## 状態

Stage N technical PASS。Stage D private T4 version 1はtechnical PASS / scientific FAILで終端閉鎖。

## 仮説

exp504のpairwise rank面をhard winnerではなくstrict nestedなcompact特徴としてexp413 TVT
LightGBMへ渡すと、rank情報を弱く条件付き利用でき、保存exp413 OOFを改善できる可能性がある。

## 設定

- 親 / matched control: exp413、OOF RMSE `7.884802794404715`
- rank source: exp504 Kaggle CPU version 1
- feature: `clean273 + compact74 + signed23 + rank_compact45 = final415`
- Stage N: outer 5 × inner 4 = 20 CPU rank models（version 1で完了・technical PASS）
- Stage D: 1 treatment × 3 configs × 5 folds = 15 GPU TVT boosters（完了）
- control再学習: 0
- metric: suffix-row unweighted TVT RMSE
- seed: 42

## 実装結果

- exp504 v1 frozen surfaceのfile / logical SHA preflight: PASS
- coverage: `3,783,989 rows / 773 wells / 7,787 H512 blocks / 1,986 pair features`
- 保存outer surface parity: Borda max abs `0.0`、provisional / fallback exact match
- rank compact schema: 45列、一意、float32。block constant 42列、row varying 3列
- nested split: 20新規CPU rank models / 25 compact partitions / outer model retrain 0
- downstream: final415 / 15 GPU TVT boosters / exp413 control retrain 0
- tests: 7 passed
- Jupytext round-trip / py_compile / Ruff / strict experiment validation: PASS
- Stage N version 1: 20 CPU models / 25 partitionsを完走しtechnical gate 8/8 PASS
- Stage D: 15 models完走、technical PASS / scientific FAIL。推論、提出: 未実施

## Stage N結果

| 項目 | 値 |
| --- | --- |
| Kaggle kernel | `kentookumura/exp507-exp504-nested-rank-compact-stage-n` v1 |
| id_no | `129565024` |
| status | `COMPLETE / stage_n_technical_pass` |
| runtime to PASS | `10,263.367916 sec` |
| models / trees | `20 / 16,000` |
| partitions | `25`（train 20 / outer-valid 5） |
| row-role | `18,919,945` |
| rank features | `45` |
| leakage overlap / forbidden features | `0 / 0` |
| technical gate | `8 / 8 PASS` |
| Stage N manifest SHA | `9a126024f0a67ab571e053038aa4a36e8b6773b6f0ff839d1fdf9ec63bcb7735` |

## 結果

| メトリック | 値 |
| --- | --- |
| exp507 CV | 7.889515565580203 |
| matched exp413 CV | 7.884802794404715 |
| gain (exp413-exp507) | -0.004712771175488406 ft |
| nonworse folds | 2 / 5 |
| maximum fixed-scope delta | +0.036938806684603476 ft |
| primary gate | FAIL |
| Public LB | 未提出 |
| Private LB | 未提出 |

### Fold別

| fold | exp413 | exp507 | delta (exp507-exp413) |
| ---: | ---: | ---: | ---: |
| 0 | 7.919988324 | 7.951788922 | +0.031800597 |
| 1 | 8.377381333 | 8.379757856 | +0.002376523 |
| 2 | 7.539713352 | 7.511558171 | -0.028155181 |
| 3 | 7.574331167 | 7.531054386 | -0.043276780 |
| 4 | 7.982868393 | 8.039329513 | +0.056461120 |

### Scope別

| scope | exp413 | exp507 | delta |
| --- | ---: | ---: | ---: |
| md_since 0--250 | 1.472437865 | 1.509376671 | +0.036938807 |
| md_since 250--1000 | 3.890857670 | 3.903219674 | +0.012362003 |
| md_since 1000+ | 8.663017156 | 8.666930539 | +0.003913383 |
| hidden-like spatial | 8.364712530 | 8.389046495 | +0.024333965 |
| hidden-like typewell-purged | 8.307714847 | 8.329497297 | +0.021782450 |

### By-well tail

- delta median / p90 / p95 / p99: `-0.008439353 / +0.359385171 / +0.514122918 / +0.988264358 ft`
- worst: `fd8f77fa +2.038361827 ft`
- `+1 / +3 / +5 ft`悪化well: `8 / 0 / 0`

## 再現性

- deterministic anchor: false
- input file / logical SHA: configとSESSION_NOTESに固定済み
- feature/model/prediction SHA: Stage N / Dとも記録済み
- Stage D kernel: `kentookumura/exp507-exp504-nested-rank-compact-stage-d` v1、id_no `129584313`
- submission SHA: 対象外

## 解釈

technical gateはPASSしたが、pooled gain、fold、固定scopeの科学3条件がすべてFAILした。
fold 2 / 3は改善した一方、fold 0 / 1 / 4が悪化し、short scopeとhidden-like 2面も上限を超えた。
45 rank特徴は全て少なくとも1 modelで利用されたが、重要度は予測改善を意味しない。

## 次

`FAIL_CLOSE_WITHOUT_PAIR_FEATURE_SUBSET_TEMPERATURE_OR_GATE_RESCUE`として閉じる。same-OOF rescue、
inference、submissionへ進まない。必要なら別承認のsaved-artifact-only原因readoutだけを検討する。
