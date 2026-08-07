# exp392 結果

## 結論

`huber_exact_hmm`はselectorから利用されましたが、fixed13 hard selectorは
親fixed12より`0.117260 ft`悪化しました。科学gateをFAILし、
`FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`として閉じます。

downstream TVT、current-test候補生成、inference、submission、same-OOF rescueは
行いません。

## 実行

- kernel: `kentookumura/exp392-exp389-huber-fixed13-selector-train`
- version / id_no: `1 / 128523057`
- status: `KernelWorkerStatus.COMPLETE`
- runtime: `3666.541645 sec`
- active variant / objectives / outer / inner: `1 / 2 / 5 / 4`
- CPU selector models: `40 / 40`
- parent/control再学習 / GPU / downstream TVT / inference / submission:
  `0 / 0 / 0 / 0 / 0`

## Technical / selector score

- 3,783,989 rows / 773 wells、key欠損0、truth/error読込0
- exp389 source foldなし、source-fold特徴利用0
- Stage A: 153候補特徴から90特徴、compact 77特徴
- Stage C: 40 models / 25 partitions / 18,919,945 compact rows /
  49,191,857 outer-valid candidate-score rows
- leakage audit: PASS
- expected-error MAE: `5.854091 -> 3.845602`
- within10 logloss: `0.510919 -> 0.359695`
- within10 Brier: `0.165364 -> 0.111966`
- 3 score指標すべてpooled・5/5 folds改善、score guard PASS
- fixed fallback error parity max abs: `0.0 ft`

## Hard selector

| 指標 | fixed13 | 親fixed12 | 差 |
|---|---:|---:|---:|
| pooled RMSE | 8.769792 | 8.652532 | +0.117260 |
| near 0--250 | 1.681550 | 1.663645 | +0.017905 |
| 1000+ | 9.629834 | 9.503799 | +0.126035 |
| hidden-like spatial | 9.697008 | 9.536496 | +0.160512 |
| hidden-like typewell-purged | 9.566854 | 9.412065 | +0.154789 |

- fold差:
  `-0.035900 / +0.243169 / +0.385042 / -0.131227 / +0.111192 ft`
- 改善fold: `2 / 5`
- Huber top1: `91,035 rows / 2.405795%`
- Huber利用fold: `5 / 5`
- improved / regressed wells: `343 / 430`
- by-well delta median / p95: `+0.014665 / +0.774302 ft`
- worst well `8902c3f6`: `+7.875188 ft`

利用率、near、selector scoreだけをPASSし、pooled、改善fold数、1000+、
hidden-like 2面、by-well p95、worst-wellをFAILしました。

## Reranking診断

worst `8902c3f6`のHuber top1率は`0%`でした。全773 wellsでもHuber利用率と
fixed13 deltaのPearson / Spearman相関は`0.004539 / -0.010965`です。
Huberを一度も選ばない285 wellsのうち158 wellsが悪化しています。

これはHuberの直接誤選択だけでは説明できず、13候補でselectorを再学習した結果、
既存12候補の順位まで変わるincumbent reranking不安定性を支持するpost-hoc診断です。
この診断は科学gateの判定には使っていません。

## 非gating oracle診断

- H512 oracle: `3.700320 -> 3.696657`、`0.003663 ft`改善
- whole-well oracle: `4.801786 -> 4.791666`、`0.010120 ft`改善
- Huber strict unique-best: H512 `275/7787`、whole-well `29/773`

候補自体の局所補完性も小さく、hard candidateとして再訪する根拠はありません。

## 再現性

- exp389 decompressed SHA:
  `f5d44d9d9ee380bb7ea408006030363efbe8fcdb3573cfa18031b2d31c617f90`
- post-read prediction SHA:
  `b16e91d3493f168b6d4a527d157febb9940120d80e729eb09d657f8d5d9445ad`
- feature schema SHA:
  `9fa5f2373a7fbfa566f880b1985a3f0f1689807ba36c33299e35f50d2e236baa`
- model manifest SHA:
  `e9b03df33755f1be15bd11e76254d0b3592144202d8de1c39b670ca9dcf5b625`
- compact manifest SHA:
  `9a818679ef21f3f3481590d5448a8339cc9a9c58bc08fc0e9a2d014a667f9bf0`
- outer-valid score SHA:
  `b4de2552ef4806f2ee644c201669e14ad78ca4e47f2df922029dded27e5472a0`
- summary SHA:
  `d244a26eab5fd1a3961b956058561a503269fe46b74b14dcbf9595c209c6b7ba`

## 次アクション

fixed13 Huber hard-selector枝は閉じます。weight、usage threshold、candidate
domain、gateを同じOOFで調整しません。

既存の`fixed13_selector_incumbent_reranking_instability_readout`には、exp392の
「worst wellでHuber利用0%なのに大幅悪化」という独立根拠を追加します。
実行は別承認がある場合だけ0 model / 0 boosterで行います。
