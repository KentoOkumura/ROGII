# タスクリスト

## 未着手

- なし。

## 進行中

- なし

## ブロック中

- なし。

## 完了

- `exp308_imputed_gr_confidence_downweight`として採番し、design-only scaffoldを作成した。
- weight `max(0.25,2^(-d/8))`の1 variant、exp307 dependency、禁止gridを固定した。
- exp269 blanket-neutralityとの違いと厳格なtail guardを固定した。
- compact self-contained trainを実装し、raw mask/distance/weight freeze、parent scale固定、weighted HMM、late readoutをNotebookへ展開した。
- fail-closed inferenceを実装し、prediction/submission生成を禁止した。
- observed weight 1、missing formula/floor、no-finite fallback、parent interpolation parity、dependency fail-closeのcontract testを追加した。
- 1 variant、773 HMM runs、0 booster、control再実行0をconfig/testで確認した。
- Jupytext/構文/ruff/対象test/experiment/template validationを実施した。
- Kaggle package/push/run、Notebook実行、inference、submissionは行っていない。
- exp307 version 2のpromotion gate FAILを確認し、dependency SHA/metricsを凍結せず、未実行のまま閉鎖した。
