# タスクリスト

## 設計（今回）

- [x] exp432 の experiment/steering 雛形を作成
- [x] exp410原因証拠とexp412 trigger/negative resultを系譜化
- [x] event、±datum、0.80/0.10/0.10 mixture、importance correctionを固定
- [x] RNG/common-random、truth-late、no-event parity契約を固定
- [x] Stage 0/full の実行量と AND gateを固定
- [x] README、SESSION_NOTES、result、metricsをdesign-onlyに更新
- [x] `KAGGLE_DIRECTION.md` と `experiment_summary.md` に登録
- [x] repository validationを通す

## Stage 0実装

- [x] 実装の明示承認を得る
- [x] Jupytext起点のcompact self-contained実装とproposal contract testを作成
- [x] fixed32 package前にbaseline 1 + treatment 1、HMM 32、PF 64、seed trajectories 8,192、particle starts 4,096,000を再確認
- [x] exp209/exp404 parity、log-domain importance、RNG分離、truth-late guardを専用testで確認
- [x] compact候補を正規train notebookへ採用する明示承認を得る
- [x] 親PF control再実行を含むKaggle Stage 0 pushの明示承認を得る
- [x] technical AND gateを監査
- [x] mechanism AND gateをlate truth joinで判定
- [x] Stage 0 FAILを記録し、full / inference / submissionを再ロック

## full以降（Stage 0 PASS時のみ）

- [ ] exp410 baseline support artifact parityを確認
- [ ] HMM trigger cache 773 + treatment PF 773 のruntime/RSSを再見積り
- [ ] full実行の明示承認を得る
- [ ] trigger cacheと4 PF shardを実行・strict merge
- [ ] promotion AND gateを判定
- [ ] inference/submissionは別途承認を得る
