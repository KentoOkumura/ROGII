# exp322_gr_likelihood_weak_exp226_soft_shrink_readout 結果

## 状態

Kaggle private CPU train version 2を完了した。固定判定は`INCONCLUSIVE_COVERAGE`で、同時に主要な科学guardも悪化したため、救済gridなしでbranchを閉じる。inferenceとsubmissionは行わない。

## 仮説

GR emission likelihoodがshiftを識別できず、同時にexp226 K16を表すshift 0が棄却されていないH512 blockだけ、exp263固定blendからexp226へ小さく戻せば、常時GR decoderのtail riskを避けながらRMSEを改善できる。

## 実行

- kernel: `kentookumura/exp322-gr-weak-exp226-shrink-readout-train`
- version / id_no: `2 / 128089589`
- runtime: `195.331601 sec`
- runtime設定: private CPU、GPU/TPU/internet off
- 実行量: `1 candidate / 1 matched control / 5 exp263 strata / 0 model / 0 booster / 0 decoder / 0 parent rerun`
- rows / wells / H512 blocks / shifts: `3,783,989 / 773 / 7,787 / 13`

version 1はraw-well scoring前に、exp226元OOF foldとexp263 readout foldが一致しないという誤ったguardで停止した。version 2では親exp263の保存`outer_fold`をreadout strataにし、exp226元foldを各well一意なsource-fold identityとして別監査した。予測、split、threshold、gate、科学guard、実行量は変更していない。

## 結果

| メトリック | 値 |
| --- | --- |
| Technical hard checks | 全PASS |
| Fixed decision | `INCONCLUSIVE_COVERAGE` |
| exp263 base RMSE | `8.238331715` |
| exp322 RMSE | `8.239202313` |
| Overall RMSE delta | `+0.000870598 ft`（悪化） |
| Changed rows | `4,870 / 3,783,989 = 0.128700%` |
| Changed wells / folds | `10 / 5` |
| Coverage下限 | row `1%`とwell `50`をFAIL |
| Improved folds | `1 / 5` |
| Activated subset | `7.744743179 -> 8.433567710`、`+0.688824530 ft`悪化 |
| Circular control RMSE | `8.237948157` |
| Real gain minus control gain | `-0.001254155 ft` |
| Near 0--250 ft | bitwise parity PASS |
| 1000+ RMSE delta | `+0.000966632 ft`（FAIL） |
| Hidden-like spatial / typewell-purged | `0.0 / 0.0 ft`（発火なし） |
| by-well p95 / worst delta | `0.0 / +0.261431339 ft` |
| Improved / same / worse wells | `2 / 763 / 8` |
| Public LB / Private LB | 未実行 / 未実行 |

fold別ではfold 4だけ`-0.000564897 ft`改善し、fold 0/1/2/3はそれぞれ`+0.000514717 / +0.000169465 / +0.000090227 / +0.004085765 ft`悪化した。worst wellは`8c167025`で`+0.261431339 ft`、best wellは`94467f50`で`-0.158189147 ft`だった。

## 再現性

- target-free contract SHA: `e0b23f2d5852202ab4b4e5aca98cedccc5bcc331888233d0e4af8de0ee389b5f`
- shift-score decompressed SHA: `adfa974e14a2bcbf481d98784c31e9959fbeee86dc91945007e162adfc8581ce`
- block-gate decompressed SHA: `8d42a676cf314bf2fb055f6997cd3f243bf128651ec2a253ea447f7a393f2f9a`
- prediction decompressed SHA: `8e335ef58235c44a0cbaae893ee4447054067021f14d3f5858e61d82785ad2c1`
- exp263 formula parity / cached exp226 anchor parity: `0.0 / 0.0 ft`
- truth attachment: target-free score/gate/prediction freeze後のみ
- download後のraw SHAとgzip decompressed SHAはKaggle summary記録と一致した。

## 実装検証

- compact self-contained Jupytext source / canonical train Notebook: 作成・採用済み
- dedicated unit test: `13 passed`
- 関連横断テスト: `37 passed`
- ruff / py_compile / Jupytext round-trip: PASS
- strict `make validate-exp`: PASS
- repository full test: `442 passed / 1 skipped / 2 existing exp296 contract failures`（exp322とは独立）

## 解釈

gateは10 wells、0.1287%の行にしか届かず、事前coverageを大きく下回った。しかも発火行ではexp226方向へのshrinkが`+0.6888 ft`悪化し、同じ発火数をずらしたcircular controlよりも悪い。GR likelihoodの弱さとshift 0 admissibilityは「exp263よりexp226へ戻すべき場所」を識別していない。

coverage不足だけなら`INCONCLUSIVE`だが、本実行ではoverall、4/5 folds、activated subset、1000+、worst well、negative controlも同時に不支持である。固定停止条件どおりalpha、quantile、block、clip、emission、selectorによるsame-OOF救済は行わない。

## 次

exp322 branchを完了・不採用として閉じる。inference / submissionは行わず、新しい救済backlogも追加しない。
