# exp342 結果

## 状態

Kaggle private CPUでStage 0 version 1と、ユーザー明示overrideによるStage 1
version 2を完了した。Stage 1ではexp281 exact HMMのGaussian emissionだけを固定
`df=4` Student-tへ置換した。

Student-t HMMはGaussian親より全体RMSEを`0.047648 ft`改善したが、固定gateの
`0.05 ft`に届かず、改善fold、hidden-like 2面、well別tail safetyもFAILした。
判定は`stage_1_failed_close_without_rescue`。inference、submission、LBはない。

## 実行

- Kernel: `kentookumura/exp342-student-t-residual-offset-emission-train`
- Version / id_no: `2 / 128356155`
- 完了: `2026-07-23T17:01:51.609982+00:00`
- scientific runtime: `14,789.392992 sec`（約4時間6分29秒）
- 773 wells / 3,783,989 rows
- Student-t variant 1 / HMM well-run 773
- model config / trained fold / booster / Gaussian control再実行:
  `0 / 0 / 0 / 0`
- GPU / inference / submission: `false / false / false`
- Stage 0 prerequisiteはFAILのままで、実行根拠は
  `explicit_user_override_after_stage_0_fail`

完了判定と数値はKaggle logsの最終summaryを根拠とした。必要なmetrics、fold、
scope、gate、SHAがログに揃っているため、Kaggle output archiveは取得していない。
fatal errorはなく、Kaggle statusは`COMPLETE`だった。

## Stage 1主結果

| 指標 | Student-t HMM | Gaussian HMM（exp281保存OOF） | 差 / 判定 |
| --- | ---: | ---: | ---: |
| RMSE | 9.779772 | 9.827420 | `-0.047648 ft`、FAIL（必要`-0.05`以下） |
| MAE | 5.261317 | 5.290694 | `-0.029377 ft` |
| 改善fold | 3/5 | - | FAIL（必要4/5） |
| exp226との差 | 9.779772 | 9.427110 | `+0.352662 ft`、direct promotion FAIL |

fold 0/1はそれぞれ`+0.168351 / +0.172175 ft`悪化し、fold 2/3/4は
`-0.054308 / -0.138495 / -0.336756 ft`改善した。平均では少し良いが、
fold横断で一貫していない。

## Scopeとtail safety

Student-t minus GaussianのRMSE差は次のとおり。

| scope | RMSE差 | gate |
| --- | ---: | --- |
| 1000+ | `-0.049643 ft` | PASS |
| hidden-like spatial | `+0.014174 ft` | FAIL |
| hidden-like typewell-purged | `+0.220136 ft` | FAIL |
| by-well delta p95 | `+1.063793 ft` | FAIL |
| worst well `77b0d905` | `+12.893602 ft` | FAIL |

finite coverage、row identity、exp281 parent RMSE parity、exp226 RMSE parityは
すべてPASSした。したがって失敗は実装・入力・数値異常ではなく、Student-t化の
科学的な改善量と安定性が不足したためである。

## Stage 0との関係

Stage 0のshift-rank proxyはFAILしたが、実際のHMMでは全体RMSEがわずかに改善した。
つまりStage 0とfull HMMは同じ指標ではなく、Stage 0 FAILから「HMMが絶対に改善
しない」とは言えない。一方、full HMMの固定gateでも改善量は
`0.05 ft`に`0.002352 ft`届かず、3/5 folds、hidden-like悪化、well-tail悪化だった。
実HMMまで確認した結果としても採用条件は満たさない。

## 判定

- scientific promotion: FAIL
- direct promotion: FAIL
- decision: `stage_1_failed_close_without_rescue`
- df、scale、temperature、grid、Huber、cap、missing、ACF、blendの救済なし
- 再実行、inference、submissionなし
- exp344 dependency patternはStage 0で不成立のまま

固定`df=4` Student-tへの単純置換には小さい平均改善があるが、Gaussian exp281を
安全に置き換える根拠にはならない。exp226よりも`0.352662 ft`悪いため、直接採用
候補でもない。

## 再現性

- candidate logical content SHA:
  `9af93eecb7bfcf0b43bbfbe9a0d759abf6031b0744da637147b05cbac72c38b7`
- decoder manifest content SHA:
  `17453228ad41ede225ff2b6b51e35c2908d20fbf4a58ef78d0b31cfb6246ff4f`
- prediction decompressed SHA:
  `767bf0726e696fc291923438ae2f87fafc1642b857f5c26bfaa9361505e9820b`
- Stage 1 gate SHA:
  `fe217ee638993ed3f7977ee5b479f56909520c22cb9eb0077bfadc70a3aaa790`
- Student-t全pathとlogical content SHAのfreeze後にのみ、
  SHA固定済みexp281 OOFのGaussian親・truth・foldをjoinした。

Stage 0の結果とSHAは`metrics.json`に履歴として保持する。
