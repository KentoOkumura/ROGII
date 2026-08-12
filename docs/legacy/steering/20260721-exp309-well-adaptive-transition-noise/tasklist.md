# タスクリスト

## 未着手（exp308 PASS・実行承認後）

- exp308 promotion status、prediction SHA、parent/direct/blend metricsをconfigへ固定する。
- 1 variant、773 HMM runs、0 booster、control再実行0を再確認する。
- Kaggle package/push/runは別途承認後にだけ行う。

## 進行中

- なし

## ブロック中

- Kaggle実行はexp308 primaryの全gate PASSとdependency SHA固定待ち。

## 完了

- `exp309_well_adaptive_transition_noise`として採番し、fixed formula、support shrinkage、clip、sig_p固定を確定した。
- compact self-contained trainにexp307 finite-MAD GR scale、exp308 missing-distance confidence、adaptive `sig_r`、exact-HMM、prediction freeze、late readoutを実装した。
- parent status/prediction SHA/metrics mismatchで実行開始前に停止するdependency guardを実装した。
- fail-closed inferenceを実装し、raw-test predictionとsubmission生成を禁止した。
- q/a式、support/fallback/clip、missing confidence、dependency、truth late-join、gate、inferenceのcontract testを作成した。
- Jupytext形式のcompact sourceからcompact/正規train・inference Notebookを生成した。
- 親exp308にはcompact実装がないため、実行可能な科学祖先exp307と章立て・行数を比較した。exp307 10章/1,676行に対しexp309 10章/1,991行で、helper importだけの薄いNotebookではない。
- Jupytext往復、構文、ruff、9 contract tests、strict experiment validation、template validationをPASSした。
- Kaggle package/push/run、inference、submissionは行っていない。
