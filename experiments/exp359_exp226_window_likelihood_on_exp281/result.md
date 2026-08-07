# exp359 結果

## 結論

Kaggle private CPU Stage 0は完了したが、固定科学gateはFAILした。
exp226 500-row window scoreは非ランダムなshift signalを持つ一方、
保存exp280 row-Gaussian controlよりpooled、全5 folds、全stress scopeで弱かった。
Stage 1の773 HMM runs、inference、submissionへは進まず、救済なしで閉じる。

## 実行

- Kernel:
  `kentookumura/exp359-exp226-window-likelihood-on-exp281-train`
- Version / id_no: `1 / 128528648`
- Status: `KernelWorkerStatus.COMPLETE`
- Runtime: `4523.211267 sec`（約75.39分）
- Rows / wells: `3,783,989 / 773`
- Scientific score / saved control / shifts / reporting folds: `1 / 1 / 13 / 5`
- HMM well-run / model config / trained fold / booster: `0 / 0 / 0 / 0`
- Parent/control再実行: 0

## 技術gate

- candidate / eligible windows: `27,561 / 10,628`
- eligible fraction: `0.385617`（必要値0.25以上）
- eligible wells: `729 / 773`
- score finite / row identity / saved-control rank parity / quantization coverage:
  `1.0 / 1.0 / 1.0 / 1.0`
- real > shuffle: `5 / 5 folds`
- target-free score content SHA:
  `8a4c5623c5836a734dcd6bd44ff5b214546afaa7316ae1fdb882f0ce6f344c4a`

技術面とnegative controlは成立しているため、FAILは入力欠損やrandom scoreではなく、
保存Gaussian controlに対する科学的な順位付け不足として扱う。

## 科学結果

| Scope | window MRR − control | window top3 − control |
| --- | ---: | ---: |
| overall | -0.022264 | -0.033496 |
| long-tail 1000+ | -0.015922 | -0.024830 |
| hidden-like spatial | -0.018404 | -0.019949 |
| hidden-like typewell-purged | -0.017774 | -0.015251 |

overallの絶対値:

- window / control MRR: `0.372904 / 0.395168`
- window / control top3: `0.414471 / 0.447968`
- window − shuffle MRR / top3: `+0.124636 / +0.180467`

fold別差:

| Fold | MRR差 | top3差 |
| ---: | ---: | ---: |
| 0 | -0.014113 | -0.014743 |
| 1 | -0.025702 | -0.043925 |
| 2 | -0.005178 | -0.014001 |
| 3 | -0.031855 | -0.047371 |
| 4 | -0.030024 | -0.041473 |

MRR/top3の改善foldはいずれも`0 / 5`で、pooled `+0.01`、4/5 folds、
stress 3面正方向の全条件を満たさない。

## 失敗原因の考察

平均posterior SDは`26.511584 ft`で、全10,628 eligible windowsのlambdaが
固定下限`0.075`へ飽和した。固定式はwindow間のconfidence差を表現できなかった。
ただし各window内では正のlambdaがshift rankを変えないため、lambda飽和だけを
control比悪化の原因とはみなせない。主な証拠は、長いprofile score自体が
saved row-Gaussian aggregateより全fold・全stress scopeで弱いことにある。

同じOOFを見たwindow/stride/weight/lambda調整は事前禁止した救済gridになる。
このStage 0結果から同familyを調整して再実行せず、500-row window potential案を閉じる。

## 再現性

- scientific contract SHA:
  `43aa952498f2fd1474bcca8c7bf651a854b2fc6f671384936ab67c795afd671a`
- window readout content SHA:
  `617cfd7ee7239f6df71a6effbf50f8fc1fcf10943ffed495385aef81520dd1d9`
- gate file SHA:
  `956d194661828ab9766f9d632288130579c39df2e9b1fe8d3cb287bd9b1a9d05`
- fold / scope metrics SHA:
  `5f3eaa7aee9bab139a21988701a56985a73171647e4520e151902fb2ad747f2e` /
  `4f29d2e6c61835767fd9957523b1df52449b30a0f1e81df024b7dbaedb412b30`

## 次

exp359を完了・閉鎖し、Stage 1、inference、submissionは行わない。
同じwindow familyの救済実験は追加しない。関連するregistration-offset候補は、
exp359の負結果を根拠にfull unknown-suffix window scoreを再利用せず、
既存のknown-prefix rolling-origin gateを独立に通す場合だけ検討する。
