# exp371_exp333_fixed13_dual_selector_on_exp264

## 状態

- ルート: `ml_model`
- 状態: Stage D version 1完了・平均改善 / by-well tail gate FAILで終了
- CV: Stage D fixed13 compact add-only OOF RMSE `8.369996`
- Public LB: 未実行
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-24
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- 追加候補: `exp333_segment_offset`

## 仮説

exp361でfixed12へのadd-one noveltyを示したexp333を、corrected exp264と同じ
candidate-long dual selectorへ13本目として追加すると、target-freeなcandidate value、
近傍shape、bank disagreementからexp333が有効な区間を学習でき、保存済みfixed12 selectorを
安全面で悪化させずに改善できる。

## 変更点

- exp263 deployable12の順序・値・fixed fallback 7本は固定する。
- exp333 Stage 1 OOFを13本目として追加し、primary domainを11本から12本へ拡張する。
- selectorの2目的、outer 5 / inner 4、sample上限、LightGBM設定はexp264から変えない。
- compact metaは74列から77列になる。追加はexp333の2 scoreとprimary top1 one-hotの3列。
- corrected exp264 Stage C v6は保存済みscoreを比較基準にし、再学習しない。
- downstream TVT、current-test inference、submissionは今回の実行範囲外。

## 検証方針

- Fold: outer 5-fold、各outer-train内inner 4-fold
- Group: well単位
- 評価行: `TVT_input`欠損のunknown suffix
- Leakage check: outer-valid wellをinner assignment、fit、early stoppingから除外する。
- exp333入力: `well_id,row_idx,outer_fold,tvt_pred_stage1`だけをpre-freezeに読む。
- exp333 source foldはprovenanceだけに使い、global key join後にexp263 selector foldへ
  再partitionする。source foldはselector特徴にしない。
- 比較: 同じ行の保存済みexp264 fixed12 outer-valid selector score。
- 安全性: pooled / 5 folds / near / 1000+ / hidden-like 2面 / by-well p95・worst。

## 実行量

| 項目 | 数 |
| --- | ---: |
| active variant | 1 |
| selector objective | 2 |
| outer folds | 5 |
| inner folds | 4 |
| CPU boosters | 40 |
| parent/control retraining | 0 |
| GPU boosters | 15 |

## 実行入口

- 候補train source:
  `exp371_exp333_fixed13_dual_selector_on_exp264_compact_selfcontained_train.py`
- 候補train notebook:
  `exp371_exp333_fixed13_dual_selector_on_exp264_compact_selfcontained_train.ipynb`
- 正規train notebookへの採用: 2026-07-24承認済み
- inference: fail-closed
- Kaggle実行: version 1はpath resolver、version 2はexp333 / exp263 fold不一致で
  fit前停止。version 3はglobal-key repartition後に40 selector boostersを完走した。
- Kernel:
  `kentookumura/exp371-exp333-fixed13-selector-train` version 3
- Stage D:
  `clean273 + fixed13 compact77 = 350`特徴、3 configs × 5 folds = 15 GPU boosters。
  ユーザー明示例外によりKaggle T4 version 1で15/15 boostersを完走した。
- Stage D kernel:
  `kentookumura/exp371-exp333-fixed13-selector-tvt-train` version 1、
  id_no `128524177`

## 所見

Stage A / Stage Cのtechnical、score、leakage guardはPASSした。fixed13は親fixed12
selectorをpooledで`0.232535 ft`改善し、4/5 folds、near、1000+、hidden-likeも改善した。
一方、by-well p95は`+0.861529 ft`、worst wellは`+10.757997 ft`で安全gateをFAILした。
同じOOFでcandidate weight、使用率threshold、domainを救済調整しない。

このStage C FAILは再分類しない。一方、ユーザーの「平均で改善しているのなら次に進む」
という明示判断により実行したStage Dは、親fixed12 compact add-only
`8.460811`に対して`8.369996`、`-0.090815 ft`改善し、3/5 folds、
near / 1000+ / hidden-like 2面も改善した。ただしby-well p95は
`+1.179312 ft`、worst wellは`+4.637599 ft`で固定安全gateをFAILした。

## 次

Stage CとStage Dのtail gate FAILを保持し、このbranchを閉じる。同一OOFでの
weight / threshold / gate救済、inference、submissionは行わない。原因確認を再訪する
場合は、既存の0-booster attribution案を独立承認して実施する。
