# タスクリスト

## 未着手

- なし。

## 実行中

- なし。

## 完了

- 未使用の実験番号`exp284`を確認した。
- steeringを作成し、640-row mask、128-row wrong-mode observation、K=3、H=256、5 policiesを固定した。
- `experiments/exp284_prefix_masked_wrong_mode_branch_recovery_backtest/`をtemplateから作成した。
- planned config / README / SESSION_NOTES / result / metricsを設計確定状態へ更新した。
- backlogとexperiment summaryへ設計済み未実装として登録した。
- ユーザー追加依頼により、exp283の生成物に依存しないcompact self-contained train 10章 / 2,407行と
  fail-closed inference 4章 / 127行を別名で実装した。
- exp226 fold-safe geometry replay、640-row mask、visible shift injection、causal self-GR top-3、stable
  shuffled control、8 branch path、3 checkpoint / 5 policy evidence、post-freeze readout / guardを実装した。
- mask leakage、wrong shift、proposal causal boundary、dedup/shuffle、future evidence/policy、freeze、
  fail-closed inferenceの専用tests 7件を追加し、7/7 PASSした。
- Jupytext変換 / sync test、`py_compile`、ruff F821をPASSした。正規stub notebookは上書きしていない。
- short canonical Kaggle packageを監査し、private CPU version 1をpushした。推論・提出は行っていない。
- 1 variant / 5 policies / 0 config / 0 fold / 0 boosterを再確認した。
- exp283依存が実験順序だけであることを説明後、ユーザーからstandalone実行の明示承認を得た。
- version 1はhorizontal CSVにない`id`列を要求して評価前にtechnical failureとなった。
- well名と行番号から監査専用IDを決定的に生成する修正を入れ、実CSV smoke、専用tests 8/8、ruff、
  Jupytext sync、strict validationをPASSした。
- 固定contractのversion 2 packageを監査し、同じprivate CPU kernelへpushした。
- version 2は766 eligible wells / 5 foldsを11,717.244秒で完走し、全technical guardをPASSした。
- pairwise、safe+wrong比incremental recovery、H512 persistence、real-vs-shuffled、false-switch guardのFAILを
  確認し、`close_without_parameter_rescue`としてbranchを閉じた。
- 小規模metrics/manifestだけを対象指定で取得し、Kaggle summary記録SHAと全件一致を確認した。
- decoder、current-test生成、inference、submissionへ進めず、新規救済backlogも追加しないと判断した。
