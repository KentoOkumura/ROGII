# exp007_hard_well_router 結果

## 状態

- 状態: completed
- CV: 13.921559
- Public LB: 12.675
- Private LB: -

## 実装内容

`exp006_hard_well_router_diagnostic` の診断結果をもとに、rule-based hard router を追加した。selected variant の `hard_router_low_gr_guarded` は、strong-GR wells を all-GR default に残し、低 GR coverage かつ hard 条件の wells だけ no-GR へ倒す。

## 検証

Kaggle train kernel `kentookumura/exp007-hard-well-router-train` version 1 が完了。output は `/tmp/kaggle-output/exp007_hard_well_router/train` に取得し、`experiments/exp007_hard_well_router/` に反映済み。

| Variant | CV | exp002 差分 |
| --- | ---: | ---: |
| `control_exp003_no_gr` | 13.882944 | -0.241625 |
| `hard_router_low_gr_guarded` | 13.921559 | -0.203010 |
| `hard_router_low_gr_any` | 13.932968 | -0.191601 |
| `control_exp005_guarded` | 13.936732 | -0.187837 |
| `control_exp002_all` | 14.124569 | 0.000000 |

Selected `hard_router_low_gr_guarded` route counts:

| Route | Wells | Eval Rows |
| --- | ---: | ---: |
| `all_gr` | 476 | 2,277,855 |
| `no_gr` | 247 | 1,300,064 |
| `guarded` | 50 | 206,070 |

## 解釈

selected router は exp005 guarded より CV を 0.015173 改善し、exp002 all-GR からは 0.203010 改善した。一方で pure no-GR control の 13.882944 には届かない。

Inference kernel `kentookumura/exp007-hard-well-router-inference` version 1 は完了し、`submission.csv` は submit-check PASS。submission ref `53254030` の Public LB は 12.675。exp004 12.730 と exp003 no-GR 12.852 は上回ったが、exp005 12.579 と exp002 12.533 には届かない。CV 改善は LB 基準 更新にはつながらなかった。
