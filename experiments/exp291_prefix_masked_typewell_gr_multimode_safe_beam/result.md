# exp291_prefix_masked_typewell_gr_multimode_safe_beam 結果

## 結論

Kaggle CPU version 1は技術ガードを全て通過したが、性能・安全性ガードを失敗した。
Type Well GRの局所modeを全保持する固定policyはsafe-onlyを大幅に悪化させたため、
`close_without_parameter_rescue` としてbranchを閉じる。decoder、推論、提出へ進めない。

## 変更点

exp284のsame-well self-GR候補とvisible top1早期確定を削除し、safe baseを絶対保持したまま
固定bankのType Well GR局所modeを全件保持するpersistent checkpoint policyへ変更した。

## 実行

- kernel: `kentookumura/exp291-typewell-multimode-safe-beam-backtest-train` version 1
- id_no: `127882960`
- status: `COMPLETE`
- 完了: 2026-07-19 23:00:58 JST
- runtime: 6,805.497秒（約1時間53分25秒）
- runtime contract: CPU single process、GPU/TPU/internet off
- eligible / ineligible: 766 / 7 wells
- model / booster / HMM / PF再生成: 0 / 0 / 0 / 0

## 主結果

| H256 policy | RMSE | all-modeとの差 |
| --- | ---: | ---: |
| safe-only | 4.827483 | all-modeは17.372335 ft悪化 |
| top1 Type Well mode | 18.713110 | all-modeは3.486709 ft悪化 |
| all Type Well modes | 22.199818 | — |
| matched-count shuffle | 17.360718 | all-modeは4.839101 ft悪化 |

- safe比改善fold: 0/5
- top1比改善fold: 1/5
- matched shuffle比非悪化fold: 0/5
- H512 all-mode gain vs safe: -11.497241 ft
- safe unique-best false switch: 34.9462%（上限5%）
- pairwise evidence AUC pooled: 0.672737
- balanced choice accuracy pooled: 0.576907（下限0.60）
- alternative better rate: 1.2128%

pairwise AUCはpooledでは0.67あるが、fold AUCは
`0.869081 / 0.636179 / 0.563686 / 0.689306 / 0.467049`で2 foldが0.60未満だった。
balanced accuracyもfold 2が0.398374まで低下し、fold安定性を満たさない。

## Guard判定

| 区分 | 判定 | 根拠 |
| --- | --- | --- |
| technical | PASS | 766 wells、5 folds、mask/candidate/branch/evidence coverage 1.0、pre-freeze truth access 0、self-GR候補0 |
| pairwise | FAIL | fold AUC、fold balanced accuracy、pooled balanced accuracyが固定閾値未達 |
| safe改善 | FAIL | H256 gain -17.372335 ft、改善0/5 folds |
| top1超え | FAIL | H256 gain -3.486709 ft、改善1/5 folds |
| H512持続性 | PASS | -11.497241 ftはH256の-17.372335 ftを下回らない |
| false switch | FAIL | 34.9462% > 5% |
| shuffle超え | FAIL | pooledで悪化、非悪化0/5 folds |
| overall | FAIL | 1条件でも失敗ならcloseする固定contract |

## 再現性

post-cut truthは全target-free table保存・content SHA凍結後にだけ接続され、
`heldout_post_cut_truth_access_before_freeze_count=0`だった。

- executed config SHA: `e6f995a79b8802c0ae49033598d42ca1ce64f0e587b475e24eb1dc66c8bed2ef`
- candidate content SHA: `07bea4d3dceb6fa4404c4ca79df68c6ed8cdee46f275e74650ff8471a842cf3a`
- branch content SHA: `2fd84727ed795628c942703aa6ab897f13565855005e753d19f36932f62a8b4c`
- evidence content SHA: `84d4d5e40e1c0a57dc02357b5d46f2c6cfb476571613980982e189d0833ba0e5`
- policy content SHA: `d914824e96159eeeab11028b54392507d2c4dbbca8b17656a2986d04b934bb17`
- overall / fold / pairwise metrics SHA:
  `bd44eec7...11da0 / de683f95...d6d5 / d5d57a90...9bf`

限定取得したcontract、mask、input、overall、fold、pairwiseのbyte SHAはKaggle summaryと一致した。
巨大なbranch path archiveは取得していない。

## 解釈

same-well self-GRを除外しても、visible Type Well GR evidenceによるpersistent commitはsafeを守れなかった。
pooled AUCだけでは、実際にはほとんどない良いalternativeを安全に選ぶには不十分である。
all-modeがscore-blind shuffleにも負けたため、候補数やcheckpointを微調整する根拠もない。
この結果はmulti-mode decoderの支持ではなく、固定選択仮説の明確な反証として扱う。

## 次

同じbacktest truthでK、top-K、shift bank、window、horizon、margin、likelihood、vetoを救済調整しない。
exp291を推論化・提出せず、既存の独立した高優先backlogを維持する。
