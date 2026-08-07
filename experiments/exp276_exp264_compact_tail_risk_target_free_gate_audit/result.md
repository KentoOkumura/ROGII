# exp276_exp264_compact_tail_risk_target_free_gate_audit 結果

## 結論

corrected exp264 Stage C v6 / Stage D v3を入力にしたKaggle private CPU version 3は
104.017秒で完了した。technical contractは全PASSしたが、事前固定したq70/q80/q90 guardは
すべてFAILした。このtarget-free fallback gateは不採用とし、branchを閉じる。

- 実行量: 1 audit / 5 evaluation folds / model 0 / trained fold 0 / booster 0
- 入力: Stage C v6 25 partitions、Stage D v3 OOF 3,783,989 rows / 773 wells
- anchor: matched clean-273 control RMSE 10.476169、compact-74 add-only RMSE 8.460811
- tail: 255 worsened wells / 220 over-0.25 wells
- inference / submission: なし

| Gate | risk wells | `delta>0` lift | `delta>0.25` lift | gated RMSE | 改善保持 | worst-well delta | lift folds `>0 / >0.25` | guard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| q70 | 223 | 0.914943 | 0.946217 | 9.770623 | 35.01% | +7.989016 | 1/5 / 1/5 | FAIL |
| q80 | 148 | 0.979240 | 1.055743 | 9.651452 | 40.92% | +13.441268 | 2/5 / 2/5 | FAIL |
| q90 | 74 | 1.165139 | 1.211019 | 9.271525 | 59.77% | +13.441268 | 2/5 / 4/5 | FAIL |

3 quantileともgated predictionはcontrolを5/5 foldsで改善したが、全quantileでpositive-lift
5/5とworst-well `<= +0.25 ft`を満たさなかった。q70/q80は改善保持50%も下回った。
q90のpooled liftと改善保持だけを事後採用せず、feature/family/weight/quantileの救済gridは行わない。

この有効なFAILにより、事前登録したexp303の`exp276_completed`と
`exp276_promotion_guard_fail`は両方成立した。

## 再現性

- deterministic anchor: いいえ。成功runは1回で、byte-stable rerunは未実施。
- seed policy: no RNG、well/row stable sort、outer-train empirical quantile。
- Stage C manifest / partition manifest SHA: `f4855726de446b8308a8acf80d6ff6cd6a789f18ef90e165b98fa05d12aecf1c` / `17930b7b50da7c783bffb8db8e34a0f69e5e583e028bde5b356d50a63bfacf66`。
- Stage C schema file / logical SHA: `e3a677610899cb33bf58262f4cf02f650300c8c2207c46b53588d3418162ea74` / `23614916c99edbbd513bcefee958d26cdfae5b83fb05c232c19736f2708dd725`。
- Stage D OOF SHA: `b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2`。
- input manifest SHA: `b6db86362e1d5f6c7a4fa19d66b27760a47d53d142f5903f5d0c2fc17f49a255`。
- risk feature schema logical SHA: `3e178e78e05620f610ea4fc99e8ca3ca205f6fece8013f040bc4b9a94ddb772d`。
- risk feature / score content SHA: `7e8c3ccac6a1573651e24bd43baab756d0adde7b433e7f87b4ff4f681a54199d` / `c09b74fd939545fa9a28d1e71982995be04af3a5588e2609c80b4f08a7a5f470`。
- gated OOF logical prediction SHA: `ee370eb443d2d65a80b9aabcfc28e65f72dc208c7d7b69d273c8b09735eb8843`。
- model / submission SHA: 新規modelとsubmissionを生成していないため対象外。

## 旧version 2の無効履歴

2026-07-18のversion 2は計算上完走したが、入力の旧exp264 Stage C compact / Stage D OOFに
feature availability leakageが判明した。そのrisk lift、gated RMSE、guard FAILはすべて無効であり、
現行判定・性能比較・backlog判断には使用しない。version 3では入力SHAだけをcorrected parentへ
差し替え、166 features、5 family、prefix128/early512/full、q70/q80/q90、全guardを変更せず再検証した。
