# exp178_supervised_gr_window_matcher_from_known_tvt_prefix 結果

## 結論

Kaggle train v1 は positive。real GR learned matcher は shuffled/no-GR control を明確に上回り、`learned_gr_match_prob_*` / expected-error 系を後続の confidence feature として評価する価値がある。

ただしこれは known-prefix pair-level smoke であり、TVT path の direct replacement、PF weight 直接置換、softmax weighted average、candidate midpoint には使わない。

## 実行結果

- Kernel: `kentookumura/exp178-supervised-gr-window-matcher-train` v1
- Output: `kaggle/output/train_v1`
- rows: 102,400 pair rows
- anchors: 10,240
- wells: 160
- runtime: 38.58 sec

| 指標 | 値 |
| --- | ---: |
| real GR logistic pair AUC | 0.765413549 |
| shuffled GR logistic pair AUC | 0.662345939 |
| AUC margin vs shuffled | +0.103067610 |
| real GR logistic top1 within10 | 0.355957031 |
| no-GR logistic top1 within10 | 0.252929688 |
| top1 margin vs no-GR | +0.103027344 |
| real GR expected-error AUC | 0.827294 |
| real GR expected-error top1 within10 | 0.513672 |
| real GR expected-error top3 within10 coverage | 0.808594 |
| real GR expected-error top5 within10 coverage | 0.959961 |

## 評価計画の判定

- primary: validation fold の real GR logistic pair AUC
- negative control: shuffled GR logistic AUC、no-GR logistic top1 within10
- rank metric: top1/top2/top3/top5 の true alignment within10 coverage
- stress: by-well top1 rate と worst well

## 採用条件

real GR が shuffled GR に pair AUC で +0.02 以上、no-GR に top1 within10 で +0.03 以上勝つ場合だけ、後続の observation likelihood feature または exp148/exp092 add-only confidence feature 候補として扱う。

v1 は AUC margin +0.103067610、top1 margin +0.103027344 で採用条件を満たした。

## 不採用条件

shuffled/no-GR と差がない、または by-well worst が大きく崩れる場合は diagnostic で閉じ、direct TVT replacement や PF weight 置換には進めない。

## 次

1. real GR expected-error と logistic probability を、既存 PF/Beam candidate 別 feature として hidden-safe に生成できるか確認する。
2. 最初の downstream は direct selector ではなく、exp148/exp092 への add-only confidence feature または PF/Beam observation likelihood feature に限定する。
