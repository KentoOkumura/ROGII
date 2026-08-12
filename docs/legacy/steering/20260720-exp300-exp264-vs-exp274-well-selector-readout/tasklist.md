# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 2026-07-20: 要件、route、比較surface、再現性、non-use contractを固定した。
- 2026-07-20: selector primary distributionとmanifestをStage C v6 strict nested surface / SHAで確認した。
- 2026-07-20: well、threshold、row、selector readoutを実験配下のコードで再生成した。
- 2026-07-20: Jupytext sourceと診断専用train/inference notebookを作成した。
- 2026-07-20: 実験文書、metrics、strategyを更新し、strict experiment validationを通過した。
- 2026-07-20: `/tmp/exp264-well-analysis`、`/tmp/exp264-analysis-exp274`、`/tmp/exp264_row_analysis.py`を削除し、残存0件を確認した。
- 2026-07-20: 再生成済みの重複`artifacts/preliminary`を削除した。final artifactsとsource inputsはexp300配下に保持した。
- 2026-07-20: 追加依頼を「candidate prevalence」ではなく「row-level candidate switchの悪化寄与」として再定義した。
- 2026-07-20: Kaggleから`nested_outer_valid_candidate_score.parquet` 1ファイルだけを選択取得し、45,407,868 rows、SHA `a10b7848...abc`を確認した。
- 2026-07-20: row-level switch±0/1/5/25/100とprevious-candidate hold run counterfactualを実装・実行した。
- 2026-07-20: `>3 ft`群の正のSSE悪化の85.74%がswitch±5外であり、switch suppressionを支持しないnegative attributionを確認した。
- 2026-07-20: 追加結果を実験文書、metrics、summary、direction、canonical notebookへ反映し、strict experiment validationを再通過した。
- 2026-07-20: actual-error oracle候補、selector top1、Stage D final、exp274をrow/well/distanceで加法分解した。
- 2026-07-20: `>3 ft`群のselection regret `+266.9864 MSE`が主因、Stage D effect `-17.3527 MSE`は集約上緩和と確認した。
- 2026-07-20: well別mechanismを67 selector failure / 41 Stage D mitigation / 32 Stage D additional worsening / 6 Stage D-only failureとして保存した。
- 2026-07-20: 結論・metrics・summary・direction・canonical notebookを再訂正し、strict experiment validationとexperiment reviewerを通過した。
- 2026-07-20: confusionへselected/oracle MAE・RMSEを追加し、Self-GR/LikPF→K16誤rankingとBeam誤選択のrate/excessを記録した。
