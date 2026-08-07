# exp315 セッションノート

## 現在の状態

- 2026-07-21: 2-stage設計確定。exp312/313待ち。実装・実行なし。
- Route: `ml_model`
- Stage A: 1 rank readout / 0 model。
- Stage B: 1 variant / 2 objectives / outer 5 × inner 4 / 40 selector models。

## 設計契約

- deployable12 candidate value/ID/orderを固定する。
- add-onlyはrank percentile、top1 margin、entropy、availabilityの4列。
- candidate/table/feature/schema/content SHA、Stage B model/OOF SHAを記録する。
- hard top1、新candidate、control再学習、inference、submissionは禁止する。

## 次

Stage A実装承認を得てrank qualityを確認し、PASS時だけ40 modelsの承認を得る。
