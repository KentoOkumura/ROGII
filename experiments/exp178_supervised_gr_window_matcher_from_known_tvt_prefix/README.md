# exp178_supervised_gr_window_matcher_from_known_tvt_prefix

## 状態

- ルート: pf_beam
- 状態: completed_train_side_smoke_supported
- CV: pair AUC 0.765413549
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-03
- 親実験: supervised_gr_window_matcher_from_known_tvt_prefix backlog

## 仮説

既知 `TVT_input` prefix から作った GR window の positive/negative pair を教師ありで学習すれば、exp131/132 の手作り descriptor よりも「候補 window が正しい深度に対応する確率」を分離できる可能性がある。

## 変更点

- raw train の known `TVT_input` prefix row だけを使い、positive は observed `TVT_input`、negative は +/-15/25/50/100ft decoy と hard local decoy で pair dataset を作る。
- real GR、shuffled GR、no-GR context の logistic control と、real GR の expected-error regressor を同じ 1 fold smoke で比較する。
- 出力は pair probability、expected error、topK true-alignment coverage、negative-control gap に限定する。

## 検証方針

- Fold: GroupKFold by well の fold 0
- Group: well
- Stratification: なし
- Leakage Check: `TVT_input` が finite な prefix row のみを label/window center に使い、tail true TVT や `TVT_input` NaN 評価区間は特徴生成、正例、threshold 選択に使わない。

## 実行入口

- 学習 notebook: `exp178_supervised_gr_window_matcher_from_known_tvt_prefix_train.ipynb`
- 推論 notebook: `exp178_supervised_gr_window_matcher_from_known_tvt_prefix_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp178_supervised_gr_window_matcher_from_known_tvt_prefix`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| real GR logistic pair AUC | 0.765413549 |
| shuffled GR logistic pair AUC | 0.662345939 |
| AUC margin vs shuffled | +0.103067610 |
| real GR logistic top1 within10 | 0.355957031 |
| no-GR logistic top1 within10 | 0.252929688 |
| real GR expected-error AUC | 0.827294 |
| real GR expected-error top1 within10 | 0.513672 |
| real GR expected-error top5 within10 coverage | 0.959961 |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- 実装は GPU なし、LightGBM booster なしの小さな supervised pair smoke に収めた。
- real GR / shuffled GR / no-GR を同じ split で比較できるため、GR 波形そのものの寄与を分けて読める。
- Kaggle train v1 では real GR が shuffled/no-GR controls を明確に上回った。

### 悪かった点

- pair-level smoke なので、PF/Beam 候補や exp148/exp092 への add-only feature として有効かは次段の別実験が必要。
- worst wells は残り、real GR logistic の worst `1b6ba517` は top1 within10 0.09375 だった。

### リスク / 注意

- direct TVT regression、hard path replacement、softmax weighted average、candidate midpoint、PF weight 直接置換には使わない。
- real GR が shuffled/no-GR control を明確に上回らない場合は diagnostic で閉じる。

## 次

- Direct replacement ではなく、`learned_gr_match_prob_*` / expected-error / margin / entropy を PF/Beam candidate feature または exp148/exp092 add-only confidence feature として小さく評価する。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
