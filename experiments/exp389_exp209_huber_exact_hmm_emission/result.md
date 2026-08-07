# exp389_exp209_huber_exact_hmm_emission 結果

## 状態

Kaggle private CPU train version 1（id_no `128466838`）を完了した。
technical gateはPASSしたが、事前登録したby-well tail gateをFAILしたため、
decisionは`huber_exp209_failed_close_without_rescue`。inference、submission、
同一OOFでの救済は行わずterminal closeする。

## 仮説

exp209 absolute-TVT exact HMMのGaussian row emissionだけをfixed Huber
`delta=1.345`へ置換し、大きなGR残差への過大な罰を弱めることでdirect pathを
安全に改善できるか検証した。

## 固定設計

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更: capped Gaussianからfixed Huber `delta=1.345`への行emission置換のみ
- control: 保存済みexp209 Gaussian exact HMM
  `RMSE 11.938287234887435`
- 実行量: 1 variant、773 HMM well-runs、control再実行0
- metric: unknown-suffix row RMSE
- seed: RNGなし、well・row・grid・rate順を固定
- runtime: `19,417.246 sec`（約5時間23分37秒）

## 結果

| 比較 | Huber | Gaussian control | 改善 |
| --- | ---: | ---: | ---: |
| direct | 11.852741130 | 11.938287235 | +0.085546105 ft |
| fixed LikPF/HMM 50:50 | 10.227661781 | 10.269692505 | +0.042030724 ft |

directは必要`+0.05 ft`を超え、fold改善も5/5だった。

| Fold | Huber | Gaussian | 改善 |
| --- | ---: | ---: | ---: |
| 0 | 10.918954 | 10.923776 | +0.004822 ft |
| 1 | 12.251975 | 12.302481 | +0.050506 ft |
| 2 | 11.452443 | 11.570050 | +0.117608 ft |
| 3 | 12.483583 | 12.723861 | +0.240278 ft |
| 4 | 12.060889 | 12.067702 | +0.006813 ft |

主要stress scopeもすべて非悪化だった。

| Scope | 改善 |
| --- | ---: |
| raw GR observed | +0.111286 ft |
| raw GR missing | +0.030431 ft |
| high missing-fraction | +0.023166 ft |
| MD 1000+ | +0.083031 ft |
| hidden-like spatial | +0.079037 ft |
| hidden-like typewell-purged | +0.014020 ft |

一方、well単位では411/773 wells改善、362/773 wells悪化だった。
delta中央値は`-0.000001 ft`だが、p95は`+0.002234 ft`で必要`<=0`をFAILした。
worst well `00bbac68`は`4.224995 → 5.975244`、`+1.750248 ft`悪化し、
上限`+0.25 ft`を超えた。

## Technical gate

- 773 HMM runs、3,783,989 rows、773 wells: PASS
- input preflight、row identity、ID mismatch 0: PASS
- finite coverage 1.0、posterior normalization誤差最大`3.55e-15`: PASS
- Huber delta / exp209 sigma-clip contract: PASS
- exp209 Gaussian control parity: 完全一致
- fixed blend control parity: 差`3.64e-6 ft`、許容内
- truth access before prediction freeze: 0 rows
- raw observed / missing partition: `2,583,152 / 1,200,837` rows、PASS
- runtime上限30,600秒: PASS

## 再現性

- scientific contract SHA:
  `d685276820e999818aed316b6a67dc9c290f0c5b54b7f0bdbbc67fd9b430b165`
- prediction raw gzip SHA:
  `95302d547e8c49cdf67dabe6200e08e5c83f01ea158cf2fbd4f25b2fd1f74d75`
- prediction decompressed / logical content SHA:
  `f5d44d9d9ee380bb7ea408006030363efbe8fcdb3573cfa18031b2d31c617f90`
- raw-GR emission contract content SHA:
  `660d72c9f67e04af6641ea7bde43057379169be9608a5149847e0f0c9befca63`
- observation audit content SHA:
  `d0a9fffcb8e16aacf4b03ee01bbc4fb1fd07bef4fb59df2135e2158ed5193c75`
- input/control manifest SHA:
  `89747fc5004a3b28d0d947981a0c54925518a16f221fc2bf8197bd73f15b7728`
- promotion gate SHA:
  `fe9dc5467120747847508eb60fe6e5bf45c3fb98b0070072d2c3e39a7e83271a`
- overall/fold/scope metrics SHA:
  `16831c2b5cf0c6dd74c7eb1619aaeb6b72445eeeb0db8138b86b346800c6c7f2`
- by-well metrics SHA:
  `b40199bd0b09a5c2a27a2b828f46b7fd7962a7ba2b79cbf152e39e4e6bab7bab`

Kaggle logsを一次証拠とし、不足したfold/scope/tail確認のため評価JSON/CSV
7ファイル、合計約0.38 MBだけを一時取得した。86 MBのpredictionと20 MBの
raw-GR emission contractは取得していない。

## 解釈

fixed Huber emissionはoverall、5/5 folds、missing、long-tail、hidden-like、
fixed blendを一貫して改善した。Student-tのexp374よりtail悪化は大幅に小さいが、
Huberでもwrong stateへの罰を弱めた一部wellのmode固定悪化を完全には防げない。
事前登録したp95/worst gateがこの少数wellリスクを検出したため、平均改善だけで
採用しない。

## 次

delta、scale、temperature、clip、mixture、Student-t、sigma、missing weight、
transition、grid、prior、blend weightによる救済や再実行は行わない。
inference、submissionも行わず閉じる。同familyの新規backlogは追加せず、
既存P1/P2を維持する。
