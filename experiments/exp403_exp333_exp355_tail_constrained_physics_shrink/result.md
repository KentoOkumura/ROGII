# exp403 結果

## 状態

Kaggle train version 4で完走し、scientific promotionはFAILした。
固定contractどおり、λ・component weight・gate・routerを救済せずbranchを閉じる。
inferenceとsubmissionは実施しない。

## 仮説

exp263のK16成分をexp333、exact-HMM成分をexp355へ置換したfull候補を、
outer-trainだけで選ぶtail制約付きscalar λでexp263へ縮約すれば、
平均RMSE gainとwell-tail safetyを同時に満たせる。

## 固定設定

- 親: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- Route: `ensemble`
- reporting: exp226 outer 5 group folds
- candidate:
  `exp263 + lambda_fold * (0.50*exp333 + 0.25*LikPF +
  0.25*exp355 - exp263)`
- λ: 固定9値、outer-trainで最大eligible、no-positive時0
- model / booster / PF / HMM / Beam / parent rerun: すべて0

## Kaggle実行

- Kernel:
  `kentookumura/exp403-exp333-355-tail-physics-shrink-train`
- version / id_no / status: `4 / 128628482 / COMPLETE`
- accelerator / internet: `CPU / off`
- scientific runtime: `172.418 sec`
- final log: `327.001 sec`
- peak RSS: `1.921 GB`
- rows / wells: `3,783,989 / 773`

version 1--3はそれぞれ非portable upstream hash、float32 1 ULP guard、
schema未確認のtruth path選択でtruthまたはmetric前に停止した。科学設定は変えず、
portable SHA、ULP-aware parity、公式train schema優先の技術修正だけを行った。

## 結果

| 対象 | RMSE / 値 |
| --- | ---: |
| exp263 control | 8.238331667 |
| full両置換 reference | 8.159425494 |
| full reference gain | 0.078906173 ft |
| cross-fit candidate | 8.238331667 |
| cross-fit gain | 0.000000000 ft |
| positive λ folds | 0 / 5 |
| improved folds | 0 / 5 |
| current-test λ | 0 |

outer-trainで選ばれたλは`[0, 0, 0, 0, 0]`だった。最小positive λ
`1/64`でも、fold別pooled gainは`0.005785--0.007919 ft`で固定下限
`0.01 ft`に届かず、by-well delta p95も`+0.023577--+0.026743 ft`で
非悪化制約`<=0`を全foldで破った。したがってpositive eligible候補は5 folds
すべて0件だった。

λ=0 fallbackにより、scope、by-well p95、worst well、persistent episode、
512-row recoveryはcontrolと同一になった。一方、promotionに必要なpositive λ、
pooled gain、improved foldsはFAILした。

fold RMSEは`7.233137 / 8.251973 / 8.660235 / 8.364633 / 8.581319`。

## Technical gate

全項目PASS:

- input SHA / rows / wells / fold support
- exp226 reporting foldとexp263 generation foldの独立性
  （label mismatch `631 / 773 wells`）
- truth pre-freeze access `0`
- finite coverage `1.0`
- control reference差 `7.8781e-08 ft`
- full reference差 `0`
- runtime / peak RSS / execution count

## SHA

- source content:
  `6c2ec0157d3a397992fdaa9678c6ef79de63fba58aa4ed85c7a3682f95b41cbd`
- formula:
  `f33bf94f4960dbf1634a0144fc2a058ce3f879482312af9e2ddbddf3b972af4a`
- prediction raw / decompressed / content:
  `dfbf21ceba7aab7713b835257045666651db467bf29d38055b709944a3aad350` /
  `7b49a8c2ff392ffbea03ad20ee35d5e277ca77b5cdcd190d1adc16021da27b35` /
  `17a7bae695c56500e55d1b724b7ca29908d70b4a2cb39e03d618016f02d1618a`
- promotion gate:
  `094c59ee5f6dbd1335332c950c00cab68bf84449f67403693ec9848c2d838592`
- summary:
  `d8a79932fbfe2fe18471e805a36d1f1da963bd8a3069532a464f5e96b59f372b`

## 解釈

full両置換には平均gainがあるが、単一scalarを1/64まで縮めてもouter-trainの
平均gain下限とwell-tail非悪化を同時に満たせない。scalar shrinkで平均改善と
tail safetyを両立できる、というexp403の仮説は反証された。

## 次

exp403をterminal closeし、同じOOFでλ grid、component weight、gate、routerを
変更しない。原因を追う場合も、予測選択や救済を行わない低優先度の
component別tail attribution readoutとして別設計にする。
