# タスクリスト

## 未着手

- なし

## 進行中

- なし

## ブロック中

- なし

## 次のアクション

- branch closeを維持し、閾値・family・shift grid・pair/std条件・sentinel・
  supporting familyで救済しない。
- add-only特徴化、prediction変更、再実行、inference、submissionへ進まない。
- この結果だけを根拠とする同familyの新規backlogを追加しない。

## 完了

- exp360 の仮説を「raw Gaussian score から raw-finite ZNCC への単一変更」として確定。
- Route を `ensemble`、親を exp340、比較対象を exp280、prediction readout を exp264 に固定。
- shift の定義、13-shift bank、512-row block、finite-pair/std gate、tie policy を確定。
- safe input と freeze-before-truth leakage barrier を確定。
- core 6 families、primary 1 family、historical raw control、stable permutation control を確定。
- technical / scientific gate、primary-only fail-closed、pass 後も prediction unchanged の停止条件を確定。
- 0 model、0 booster、0 control retraining、CPU-only の計算予算と再現性設計を記録。
- Jupytext percent形式のcompact self-contained train/inference sourceと正規Notebookを実装。
- safe input loader、Type Well interpolation、raw-finite ZNCC、valid/support maskを実装。
- exact tie policyとstable SHA256 shift-label permutation controlを実装。
- exp280 historical raw scoreを同じblock/family式で再集約するmatched controlを実装。
- score/mask/control/feature/quantile/manifest freezeとtruth-access ledgerを実装。
- late join後のpooled/fold/1000+/hidden-like readoutとprimary-only gateを実装。
- synthetic fixtureで`GR_typewell(T+δ)`の正符号と`+10 ft` top1を確認。
- 専用pytest 10件、構文、Ruff F821/E9、Jupytext roundtripをPASS。
- Kaggle push前にvariant 1、permutation control 1、保存済みraw baseline 1、
  LightGBM 0、trained fold 0、booster 0、親control再学習0を再確認。
- canonical metadataとbootstrap内configを検証し、private CPU version 1をpush。
- version 1のcompetition mount候補不足をscore生成前のtechnical errorとして記録し、
  scientific契約を変えずcanonical competition pathだけを追加。
- 同じkernel IDのversion 2で773 wells、5 folds、7,787 blocksを完走。
- input / score / mask / control / feature / quantile / readout SHAを記録し、
  SHA manifest 13件のローカル一致を確認。
- primary technical / scientific gateのFAILを確認し、
  `close_zncc_confidence_branch_without_rescue`としてbranchを閉鎖。
