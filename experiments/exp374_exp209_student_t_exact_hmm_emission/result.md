# exp374_exp209_student_t_exact_hmm_emission 結果

## 状態

Kaggle private CPU train version 1（id_no `128436182`）を完了した。
technical gateはPASSしたが、事前登録したby-well tail gateをFAILしたため、
decisionは`student_t_exp209_failed_close_without_rescue`。inference、submission、
同一OOFでの救済は行わずterminal closeする。

## 仮説

exp209 absolute-TV​T exact HMMのGaussian emissionだけを固定`df=4`
Student-tへ置換し、大きなGR残差への過大な罰を弱めることでdirect pathを改善する。

## 固定設計

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更: Gaussianから`df=4` Student-tへの行emission置換のみ
- control: 保存済みexp209 Gaussian exact HMM
  `RMSE 11.938287234887435`
- 実行量: 1 variant、773 HMM well-runs、control再実行0
- metric: unknown-suffix row RMSE
- seed: RNGなし、well・row・grid・rate順を固定
- runtime: `19,662.082 sec`（約5時間27分42秒）

## 結果

| 比較 | Student-t | Gaussian control | 改善 |
| --- | ---: | ---: | ---: |
| direct | 11.720478702 | 11.938287235 | +0.217808533 ft |
| fixed LikPF/HMM 50:50 | 10.125385545 | 10.269692505 | +0.144306960 ft |

directは必要`+0.05 ft`を超え、fold改善も4/5だった。

| Fold | Student-t | Gaussian | 改善 |
| --- | ---: | ---: | ---: |
| 0 | 10.018466 | 10.923776 | +0.905310 ft |
| 1 | 12.093373 | 12.302481 | +0.209108 ft |
| 2 | 11.357167 | 11.570050 | +0.212883 ft |
| 3 | 12.715238 | 12.723861 | +0.008624 ft |
| 4 | 12.190340 | 12.067702 | -0.122638 ft |

主要stress scopeもすべて非悪化だった。

| Scope | 改善 |
| --- | ---: |
| raw GR observed | +0.131927 ft |
| raw GR missing | +0.404486 ft |
| high missing-fraction | +0.503873 ft |
| MD 1000+ | +0.236105 ft |
| hidden-like spatial | +0.488873 ft |
| hidden-like typewell-purged | +0.415856 ft |

一方、well単位では430/773 wells改善、343/773 wells悪化だった。
delta中央値は`-0.022888 ft`だが、p95は`+0.982661 ft`で必要`<=0`をFAILした。
worst well `a6f967fb`は`12.785496 → 47.801459`、`+35.015963 ft`悪化し、
上限`+0.25 ft`を大幅に超えた。

## Technical gate

- 773 HMM runs、3,783,989 rows、773 wells: PASS
- input SHA、row identity、ID mismatch 0: PASS
- finite coverage 1.0、posterior normalization誤差最大`4.22e-15`: PASS
- exp209 Gaussian control parity: 完全一致
- fixed blend control parity: 差`3.64e-6 ft`、許容内
- truth access before prediction freeze: 0 rows
- runtime上限30,600秒: PASS

## 再現性

- scientific contract SHA:
  `1425655ef89d0b7f887480a28a74f747115df9104a158cc821aa27b58b5ba0e5`
- prediction content SHA:
  `668fe87da902955acee742c72d30724abb53f32050bb5d0a5c1b3dee0cbd626e`
- raw-GR emission contract content SHA:
  `64703b79937aff0c68af809274f70de7813c24eddaaf042e97f9531c1057da2b`
- observation audit content SHA:
  `d0a9fffcb8e16aacf4b03ee01bbc4fb1fd07bef4fb59df2135e2158ed5193c75`
- promotion gate SHA:
  `d8334237a3da5e3e8deee159971dfc7fbe50a2793332ba93c10311c879298d4a`

logsを一次証拠とし、不足したfold/scope/tail確認のためmetrics/gate/summaryの
小規模4ファイルだけを取得した。86 MBのprediction archiveは取得していない。

## 解釈

固定Student-t emissionはoverall、4/5 folds、missing、long-tail、hidden-like、
fixed blendを一貫して改善し、exp209のGaussian emissionが平均性能上は過度に
外れ値へ反応していたことを支持する。ただし少数wellでwrong modeへの罰も弱まり、
極端な誤mode固定を発生させた。平均改善だけでは安全に採用できず、
事前登録したp95/worst gateが意図どおりこの失敗を検出した。

## 次

df、scale、temperature、clip、mixture、Huber、sigma、transition、grid、
blend weightによる救済や再実行は行わない。inference、submissionも行わず閉じる。
同familyの新規backlogは追加せず、既存P1/P2を維持する。
