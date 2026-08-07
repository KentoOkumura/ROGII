# タスクリスト

## 未着手

- なし。

## ブロック中

- current-test生成、inference、submissionは未承認。
- scientific guard FAILのためprefix-calibrated correction、parameter rescue、exp281 blend/selector救済は恒久停止。

## 完了

- 未使用の実験番号`exp285`を確認した。
- 640行maskでvisible 512行以上を満たすwellが766/773あることをread-onlyに確認した。
- steering 3文書を作成した。
- pseudo cut、group-safe donor replay、truth freeze順序、3 summary、256 permutation、guardを固定した。
- `experiments/exp285_exp226_prefix_masked_offset_predictability_readout/`をtemplateから作成した。
- 実験docs / planned metrics / configを設計確定後、compact実装完了・未実行へ更新した。
- `experiment_summary.md`と`KAGGLE_DIRECTION.md`へexp285 compact実装完了・未実行として登録した。
- strict `make validate-exp EXP=exp285_exp226_prefix_masked_offset_predictability_readout`を通した。
- `review_exp_docs.py`でcore evidence categoriesが揃っていることを確認した。
- compact self-contained trainを9章・20 cells、inferenceを4章・10 cellsで実装した。
- fold-safe exp226 replay、well-end pseudo geometry、2段freeze、3 summary、256 permutation、guardを実装した。
- 専用合成test 8件、Jupytext round-trip、`py_compile`、ruffを通した。
- repository test 219件、strict experiment validationを通した。
- 正規trainへcompact実装を採用し、正規inference notebookはtemplate stubのまま維持した。
- 1 variant / 0 config / 0 fold training / 0 boosterを再提示し、Kaggle CPU実行のユーザー承認を得た。
- Kaggle version 1のraw `id`列契約不一致を修正し、同一kernel version 2を完走した。
- 766 wells / 5 foldsのtechnical guardは全PASS、primary/supporting/scope guardはFAILした。
- result / metrics / experiment summary / KAGGLE_DIRECTIONへnegative resultを記録し、backlogから削除した。
