# タスクリスト

## 未着手（依存・trigger PASS・実装承認後）

- exp307の全promotion gate PASS後にdependency SHAとtarget-free trigger inputをconfigへ固定する。
- compact self-contained trainへcontiguous ACF、n_eff、LOO prior、shrinkage、trigger、条件付きHMM、late readoutを実装する。
- fail-closed inferenceを実装し、prediction/submission生成を禁止する。
- gapをまたがないACF、pair fallback、LOO prior、log shrink、trigger fail-close、truth late-joinのtestを作る。
- 実装後に最大1 variant、最大773 HMM runs、0 booster、control再実行0を確認する。
- Jupytext/構文/ruff/test/experiment/template validationを行う。
- Kaggle package/push/runは別途承認後にだけ行う。

## 進行中

- なし

## ブロック中

- exp307の全promotion gate PASS、target-free trigger、実装承認待ち。

## 完了

- `exp310_effective_sample_size_shrunk_sigma_gr`として採番し、design-only scaffoldを作成した。
- contiguous-run ACF、lag 20、k=50、LOO median prior、log shrinkage、triggerを固定した。
- exp308/309とのcombined変更とFAIL後のgrid救済を禁止した。
- 実装、Kaggle package/push/run、inference、submissionは行っていない。
