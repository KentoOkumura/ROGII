# exp054_pseudo_tail_seed_bagging_inference_submit 結果

## 状態

Kaggle inference version 1 と code submit が完了した。

## 評価計画

- CV reference: `exp053` seed bag3 fixed 12.633797
- LB reference: `exp052` single-seed capacity fixed Public LB 12.076
- 確認項目: submit-check、予測範囲、exp052 submission との差分、Public LB

## 提出前確認

- submit-check: PASS
- rows: 14,151
- missing values: 0
- duplicate ids: 0
- SHA256: `73c978e3bff87fe6eb195d10adf318916cdc554f92870704e8b51efc5a3428bc`
- prediction range: 11590.045172 - 12236.916569
- diff vs exp052 submission RMSE: 1.113177

## 提出

- ref: `53526321`
- kernel: `kentookumura/exp054-seed-bag-infer` v1
- first status check: pending
- final status: complete
- Public LB: 11.856

## 解釈

3-seed bagging は Public LB 11.856 で、exp052 single-seed capacity fixed 12.076 から -0.220 改善した。CV では exp051/053 がほぼ同等だったが、LB では seed averaging が効いた。

一方、ML route 全体基準の exp039 11.740 には +0.116 届かない。pseudo-tail 自前系の Public LB 基準は exp054 に更新するが、ML route 全体基準は exp039 のまま維持する。
