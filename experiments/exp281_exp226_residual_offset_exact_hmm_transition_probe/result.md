# exp281_exp226_residual_offset_exact_hmm_transition_probe 結果

## 状態

Kaggle private CPU version 1を完了し、negative resultとして閉じた。train-side promotion guardはFAIL。
raw-test inferenceとsubmissionは実行しない。

## 仮説

exp226の局所形状を固定し、GR likelihoodにはslow offsetだけを選ばせることで、exp279の
absolute geometry unaryより安全にmode offsetを抑え、exp263 fixedを改善できる。

## 設定

- shape親: exp226
- decoder親: exp209
- 先行条件: exp280 guard PASS
- 失敗参照: exp279
- 検証: exp226保存5-fold OOF、773 wells / 3,783,989 rows
- variant / HMM well-runs: 1 / 773
- LightGBM config / trained fold / booster: 0 / 0 / 0
- offset grid / step: `[-80,80] / 0.35 ft`
- offset-rate: 41 states / `+-0.10`
- runtime: 15,042.787秒（4時間10分42.8秒）
- Kaggle kernel version 1 / id_no `127831519`

## 結果

| candidate | RMSE | MAE | within5 | within10 |
| --- | ---: | ---: | ---: | ---: |
| exp226 prediction | 9.427110 | 6.148528 | 0.582034 | 0.807710 |
| exp209 exact HMM | 11.938287 | 6.769555 | 0.624890 | 0.784387 |
| exp263 fixed | 8.238332 | 5.398485 | 0.634146 | 0.845884 |
| residual-offset HMM | 9.827420 | 5.290694 | 0.689860 | 0.837068 |

residual-offset HMMはexp263 fixedよりRMSEが`+1.589088 ft`悪化し、overall gain guardをFAILした。
exp279 absolute-unary版10.035987よりは`-0.208567 ft`改善したが、exp226単体よりも
`+0.400310 ft`悪い。

## Fold・scope guard

| fold | exp263 fixed | residual-offset HMM | delta |
| ---: | ---: | ---: | ---: |
| 0 | 7.233137 | 7.987927 | +0.754790 |
| 1 | 8.251972 | 9.654037 | +1.402064 |
| 2 | 8.660236 | 10.124565 | +1.464329 |
| 3 | 8.364634 | 11.406236 | +3.041602 |
| 4 | 8.581319 | 9.662696 | +1.081377 |

改善foldは0/5。near、1000+、hidden-like 2面も全て悪化した。

| scope | delta RMSE vs exp263 fixed |
| --- | ---: |
| near 0–250 ft | +0.280916 |
| 1000+ ft | +1.792419 |
| hidden-like spatial | +1.808499 |
| hidden-like typewell-purged | +1.610008 |

773 wells中408 wellsは改善し、well別delta中央値も`-0.221848 ft`だった。しかし365 wellsが悪化し、
p95 deltaは`+10.982960 ft`、worst well `8a3da6d1`は`+30.961675 ft`だったためtail safetyをFAILした。

## Persistent-offset recovery

| candidate | episodes | 256行以内復帰率 | 512行以内復帰率 |
| --- | ---: | ---: | ---: |
| exp263 fixed | 551 | 0.021779 | 0.090744 |
| residual-offset HMM | 530 | 0.022642 | 0.122642 |

episode数は21減り、256 / 512行復帰率も`+0.000863 / +0.031897`改善したため、recovery guardはPASS。
一方でoverall RMSE、全fold、全scope、worst-wellを救えず、採用条件にはならない。

## Guard判定

| guard | 判定 |
| --- | --- |
| exp263 parity | PASS（差 `7.45e-7 ft`） |
| delta-grid / finite coverage | PASS（1.0 / 1.0） |
| overall gain | FAIL |
| 3/5 fold改善 | FAIL（0/5） |
| scope非悪化 | FAIL |
| worst-well +0.25 ft以下 | FAIL |
| persistent episode非増加 | PASS |
| 256/512 recovery非悪化 | PASS |
| 総合 | FAIL |

## 再現性

- full run: 3,783,989 rows / 773 wells / 773 status `ok`
- input decompressed SHA: exp226 / exp209 / exp072はconfig hard guardと一致
- hidden-like assignment SHA: config hard guardと一致
- OOF raw gzip SHA: `57d18866...872f1`
- OOF decompressed SHA: `3a99b1d9...7386`
- OOF logical content SHA: `d7f902b8...e440`
- decoder manifest file SHA: `a5fa3c15...ede8`
- decoder scientific mapping SHA: `876a6d57...069`
- input / well manifest SHA: `d6faac0b...2125` / `18cbfc50...4d0`
- 取得OOFからraw / decompressed / logical SHAを再計算し、Kaggle summaryと一致確認済み
- deterministic anchor: いいえ。negative branchの証拠として保存する
- submission SHA: 対象外

## 解釈

GR likelihoodをslow offsetへ限定するとMAE、within5、改善well数、persistent recoveryは良くなるが、
一部wellで大きな誤offsetを維持し、RMSE tailを悪化させる。exp280で確認した局所的なshift識別力は
常時稼働のglobal offset decoderを安全にするほど強くなかった。exp279より探索自由度を狭める効果は
あったものの、exp263 fixedを置き換える根拠にはならない。

## 次

本branchは閉じる。offset grid、process noise、rate span、likelihood weightの救済探索、PF、blend、
selector、raw-test inference、submissionは行わない。独立仮説の
`exp226_prefix_masked_offset_predictability_readout`と、既存のtarget-free future-evidence回復監査を優先する。
