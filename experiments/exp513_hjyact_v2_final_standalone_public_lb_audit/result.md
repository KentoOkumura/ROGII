# exp513_hjyact_v2_final_standalone_public_lb_audit 結果

## 結論

同一packageをKaggle GPU / internet-offで2回実行し、どちらも完走した。14,151行のvisible finalは
公開sourceとbyte-for-byte一致し、exp512 v1/v2のmount failureは解消した。一方、後段override前の
PF/Ridge blend統計は2 runで異なり、未知wellを含むhidden code executionの決定性は確認できなかった。
このためcompetition submitは行っておらず、exp513のPublic LB実績はまだない。

## 固定設定

- Route: `ensemble`
- 親: `exp512_hjyact_v2_final_10pct_hedge_on_exp413`
- final boundary: `after_complete_hjyact_v2_final_stack_and_pf_seed_branch_hedge`
- exp413 / downstream blend / cross-consumer reuse: 除外
- scientific variant / new booster / parent-control retraining: `1 / 0 / 0`
- runtime Ridge: 1 config × 5 folds
- saved model files / contained estimators: `13 / 33`
- package: 437,126 bytes / SHA `9d7efcca...8e6d4b`
- kernel: `kentookumura/exp513-hjyact-v2-standalone-lb-audit-inference`、id_no `129735820`

## Kaggle実行結果

| 項目 | version 1 | version 2 |
| --- | ---: | ---: |
| status | COMPLETE | COMPLETE |
| 科学runtime | 819.939秒 | 816.881秒 |
| rows / wells | 14,151 / 3 | 14,151 / 3 |
| sample ID-order match | PASS | PASS |
| source final SHA match | PASS | PASS |
| submission SHA | `b192d3f3...9ded4a` | `b192d3f3...9ded4a` |
| external submit | false | false |

最終`submission.csv`と`hjyact_v2_final_component.csv`は各run内でも完全一致した。input manifest SHA
`194d9df7...88b182`、model manifest SHA`491da496...71509`も固定された。Kaggle readbackでは
49/49 cell source、private、GPU、internet off、7 dataset、0 kernel source、competition inputを確認した。

## exp512失敗の回帰確認

| exp512 failure | exp513結果 |
| --- | --- |
| v1: competition mount固定から空well / `KeyError: wid` | required-content resolverで3 test wellsを解決し、全工程完走 |
| v2: Ridgeだけ旧dataset mountを直接参照 | SHA監査済みRidge rootをCFGへ注入し、5 fold fit完走 |
| 初回SaveKernel 400: 1 MiB超過 | package 437,126 bytesでPASS |

## 再現性の判定

visible finalの2-run再現性はPASSした。ただしlogs上のpre-override
`submission_sp45_learned_w0.60.csv`統計は次のように異なる。

| 統計 | version 1 | version 2 |
| --- | ---: | ---: |
| mean | 11905.160092 | 11904.617062 |
| std | 277.945461 | 277.397902 |
| RMSE vs SP45 | 2.214184 | 2.302374 |
| p95 abs vs SP45 | 4.379985 | 4.613829 |

source由来の`_pf_ancc` / `_pf_z`はNumba内乱数を明示seedせず、well-levelをthread並列生成する。
visible 3 wellsは後段のguarded/gold overrideで全行置換されるため最終SHAは一致するが、未知wellでは
この差が残る可能性がある。したがって`deterministic_anchor=false`、`hidden_code_submission_ready=false`
としてfail-closeした。これはsource parityの失敗ではなく、hidden一般化時の再現性リスクである。

## 検証と生成物

- 専用contract tests: 7 passed、親込み13 passed
- 構文 / Ruff F821 / Jupytext / strict validate-exp: PASS
- version 1最終成果物: `kaggle/output/inference_v1/`
- version 2成果物と中間診断: `kaggle/output/inference_v2/`
- Kaggle CLIの旧version指定不具合でlatestを取得した複製は
  `kaggle/output/cli_latest_alias_diagnostic/`へ隔離し、比較証拠から除外した。

## LBと次の判断

公開source `6.568`、exp413 / exp510 `7.201`は参照値であり、exp513の結果ではない。
competition submitは未承認・未実行なので、Public LBは未評価である。

次は複数の妥当な選択肢があるため自動決定しない。

1. source RNG semanticsをそのまま維持して単独code submissionへ進む。
2. well別明示seedを入れた別candidateを作り、visible parityと未知well相当の中間2-run一致を再検証する。

どちらの場合も、正規Notebook採用、submit-check、competition submissionは別承認とする。
