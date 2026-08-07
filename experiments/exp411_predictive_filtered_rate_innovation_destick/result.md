# exp411_predictive_filtered_rate_innovation_destick 結果

## 状態

Stage 0 Kaggle CPU Version 5完了。technical gateは全PASSしたが、mechanism gateは
2 / 6 PASSに留まり、`stage0_fail_closed`で終了した。Stage 1、inference、
submissionは実行していない。

## 仮説

predictive→filtered rate innovationの同符号累積をtriggerにして、rate transitionの
stay massだけを方向付きで弱めれば、exp209の主因であるrate追従遅れを減らせる。

## 固定設計

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- evidence: `exp408_hmm_message_rate_basin_audit`
- trigger: rate-cell正規化two-sided CUSUM
- threshold / activation / transfer / refractory:
  `1.0 cell / 32 transitions / 10% / 128 rows`
- Stage 0: 1 treatment × 32 HMM well-runs
- Stage 1: 1 treatment × 773 HMM well-runs
- parent control rerun: 0
- model / booster / PF / Beam / GPU: 0

## 結果

| メトリック | 値 |
| --- | --- |
| fixed32 manifest | 32 wells（persistent 16 / control 16） |
| manifest SHA256 | `fbbc62b7cb79e16a7fb436f3a9d11f8975e935ad2475a17e2dec4fd7b142e4d6` |
| contract tests | 18 passed（exp408回帰込み26 passed） |
| Jupytext / py_compile / Ruff F821 / strict experiment validation | PASS |
| Kaggle kernel | Version 5 `COMPLETE`、id_no `128773391` |
| Version 5 HMM | 32 / 32 wells、`1,133.133秒`、peak RSS `1.020561 GB` |
| technical gate | 13 / 13 PASS |
| future-rate方向一致 | `0.225397` / 必須`>=0.60`、FAIL |
| fold方向一致 | `0 / 5` / 必須`>=4 / 5`、FAIL |
| pre-onset trigger coverage | `1.0` / 必須`>=0.50`、PASS |
| eligible lead-time episodes | `25` / 必須`>=8`、PASS |
| control active-row fraction | `0.136119` / 上限`0.10`、FAIL |
| persistent-control active-well差 | `0.0` / 必須`>=0.20`、FAIL |
| active-row fraction | `0.129645`、technical範囲内 |
| full runtime projection | `27,372.239秒`（約7.60時間）、PASS |
| fixed32 parent / treatment RMSE | `9.968803 / 9.972554 ft`（差`+0.003752 ft`、診断値） |
| CV | 未実行 |
| Public LB | 未提出 |
| Private LB | 未提出 |

主要生成物はKaggle outputから取得し、ログ記録と実ファイルを照合した。

| 生成物 | 行数 | 主SHA256 |
| --- | ---: | --- |
| activation schedule | 156,088 | decompressed `fd27809c...34a7` |
| predictions | 156,088 | decompressed `5f6e469c...8c0d` |
| well metrics | 32 | `4a8e9842...5e91` |
| trigger truth-late readout | 633 | `a0981988...0d48` |
| episode lead readout | 25 | `3d2a6b8b...c963` |
| summary | 1 | `5c952e7e...42b3` |

## 解釈

triggerはpersistent episodeより十分早く発火したが、必要なrate方向を示さなかった。
630件のfuture-direction eligible triggerに対する方向一致は22.54%で、全foldが
事前固定した50%超条件を下回った。さらにpersistent / controlとも16 / 16 wellsで
activeとなり、control active-row fractionも上限を超えた。したがって、このCUSUMは
早いものの非特異的で、方向付きde-stickの因果triggerとして採用できない。

小標本RMSEはpromotion gateではないが、treatmentはparentより`0.003752 ft`悪化しており、
失敗判断を覆す補助証拠もない。technical / truth-late / parity / SHA契約はすべて通ったため、
結果は実装不良ではなく固定した科学仮説のnegative resultとして信頼できる。

## 次

exp411のStage 1は実行せずbranchを閉じる。exp412は「exp411のcausal trigger /
future evidence不足」という先行条件を満たしたが、設計確定・未実装のままであり、
実装と各Kaggle実行には別指示・別承認が必要である。

同じexp411 scheduleと同じdirection / control gateを使うexp420は、現行契約のままでは
Stage 0 prerequisiteを既にFAILしている。高コストPF runは行わず、実装参照として保持する。
