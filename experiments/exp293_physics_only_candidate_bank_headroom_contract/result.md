# exp293_physics_only_candidate_bank_headroom_contract 結果

## 状態

Kaggle private CPU version 2完了。technical/scientific supportはPASSし、固定分岐どおり次は
`stage2_prefix_calibrated_latent_registration_gr_evidence`とする。inferenceとsubmissionは実施していない。

## 仮説

exp263でcurrent-test再生成済みの12物理candidateだけでも、H512 block oracle RMSEは5.5以下となり、
物理モデル単体Public LB 6.5を狙うStage 2/3に十分なcandidate supportが存在する。

## 親実験との差分

exp263のcandidate生成・値・formula・fold・evaluation suffixは変更していない。deployable12を固定したまま、
row/H128/H256/H512/whole-well oracle、truth freeze、support判定とSHA evidenceだけを追加した。

## 実行契約

- Route: `pf_beam`
- candidate: 6 primitive + 5 fixed pair + `exp226_w500_50_50`
- rows / wells / folds: `3,783,989 / 773 / 5`
- audit / model config / trained fold / booster / HMM-PF再生成: `1 / 0 / 0 / 0 / 0`
- runtime: Kaggle private CPU、GPU/TPU/internet off
- kernel: `kentookumura/exp293-physics-bank-headroom-audit-train`
- version: v1はpackage不足で計算前停止、v2が約200秒で完了

## 結果

| 粒度 | Oracle RMSE | 6.5への必要headroom回収率 |
| --- | ---: | ---: |
| row | 3.446407 | 0.457564 |
| H128 | 3.492440 | 0.460189 |
| H256 | 3.552829 | 0.463733 |
| H512（primary） | **3.683763** | **0.471825** |
| whole-well | 4.784904 | 0.569655 |

H512 fold RMSEは`3.998262 / 3.745013 / 3.317686 / 4.117908 / 3.141067`で、最大は
fold 3の`4.117908`。全foldで6.5未満かつanchorを改善した。

| H512 risk面 | Anchor RMSE | Oracle RMSE | 必要回収率 |
| --- | ---: | ---: | ---: |
| 1000+ | 9.042324 | 4.009694 | 0.601553 |
| hidden-like spatial | 8.748108 | 3.540513 | 0.535664 |
| hidden-like typewell-purged | 8.694132 | 3.531630 | 0.528205 |

- H512 by-well p95 / worst: `7.080306 / 25.698447`、worst wellは`91b301ce`。
- technical checks 10件とscientific checks 5件はすべてPASS。
- `support_passed = true`。
- oracle/selected TVT predictionは保存していない。

## 再現性

- executed config SHA: `bb75990e6a144c27d87e6da37db17babe645922ed809a035e2b6c6f02770222d`
- candidate bank content SHA: `29477141685662bae7417e788ec5dbe914c2220b7f4cc45ab01befa4e5e3b474`
- truth content SHA: `e9067327058431278a0fd994e8e6005b76ab99acbd3942118974599afb69a8d0`
- block assignment decompressed SHA: `b0755c22aa8d791012d3f605e2f1b66063ce9bb6ba46ddd4b48dca77cce032d7`
- oracle readout content SHA: `69d14a236205eaa1aaafa09abf9bb9b1984797fec54f2fb6533f8243f0a97003`
- SHA manifest 11件を取得物に対して再計算し、不一致0。

## 解釈

deployable12 bankには、row-wise switchだけでなくH512 blockで候補を固定してもRMSE 3.68の強い上限がある。
したがって現時点の主問題はcandidate不足ではなく、truthを使わずに良いcandidateへ約47.2%のSSE headroomを
配分できる観測モデルと時間統合である。whole-well oracleでも4.78なので、候補切替を極端に細かくしなくても
6.5を下回る余地がある。一方、これはoracle上限であり、実運用可能なselector性能を示すものではない。

## 分岐

固定契約に従いStage 4 candidate birthは開始しない。次はStage 2
`prefix_calibrated_latent_registration_gr_evidence`だけを新しいsteering/実験として設計する。
Stage 2がFAILした場合もStage 4へ自動分岐しない。
