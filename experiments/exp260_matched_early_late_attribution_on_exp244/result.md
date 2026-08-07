# exp260_matched_early_late_attribution_on_exp244 結果

## 仮説

exp244 mixed augmentationのhidden-like改善とworst-well崩壊を、同一条件のearly-only / late-onlyへ分離する。

## 設定

- 親: `exp244_bidirectional_prediction_start_pseudotail_augmentation`
- Route: `ml_model`
- variants: `early_only (-1000/-250)`、`late_only (+250/+1000)`
- official / pseudo weight: 1.0 / 0.5
- 学習: 2 variants / 3 configs / 5 folds / 30 boosters
- parent/control再学習: なし
- validation: official-start 3,783,989 rows
- seed/mode: exp218 `gpu_repro_guard_dp_threads8`

## 結果

Kaggle GPU train v1は30 boostersを完走した。比較値はすべて同じofficial-start 3,783,989 rows上のOOF RMSE。

| surface | raw exp218 | exp244 mixed | early-only | late-only |
| --- | ---: | ---: | ---: | ---: |
| overall | 8.475794 | 8.472380 | 8.513934 | 8.489116 |
| 000-050 | 0.957638 | 0.952745 | 0.975785 | 0.957114 |
| 050-100 | 1.310177 | 1.304418 | 1.324606 | 1.299091 |
| 100-250 | 2.094429 | 2.101476 | 2.114391 | 2.084652 |
| 250-500 | 3.315459 | 3.358468 | 3.389732 | 3.344811 |
| 500-1000 | 4.800747 | 4.863575 | 4.880185 | 4.815243 |
| 1000+ | 9.295198 | 9.286063 | 9.331677 | 9.308933 |
| hidden-like spatial | 9.661607 | 9.245771 | 9.328626 | 9.714391 |
| hidden-like typewell-purged | 9.636010 | 9.230900 | 9.310899 | 9.694470 |

overall差はearly-onlyがraw比`+0.038140063`、late-onlyが`+0.013322404`。late-onlyはearly-onlyより
`-0.024817659`良いが、rawにもmixedにも届かなかった。

| fold | raw exp218 | exp244 mixed | early-only | late-only |
| --- | ---: | ---: | ---: | ---: |
| 0 | 8.716010 | 8.245739 | 8.328439 | 8.548386 |
| 1 | 8.578534 | 9.488172 | 9.456993 | 8.801912 |
| 2 | 7.672251 | 7.639240 | 7.690065 | 7.642750 |
| 3 | 8.351301 | 8.483999 | 8.567360 | 8.362215 |
| 4 | 9.001172 | 8.399399 | 8.431915 | 9.024209 |

rawからの改善foldはearly-only / late-onlyとも2/5で、最低条件3/5に届かなかった。

### by-well

- early-only: 394 wells改善 / 379悪化、+2 ft超悪化17 wells。worst `059c8f24`は
  `7.655552 -> 26.278711`、`+18.623158`。
- late-only: 388 wells改善 / 385悪化、+2 ft超悪化2 wells。worst `7850c72e`は
  `18.002610 -> 21.411061`、`+3.408451`。

## 採用条件

late-onlyはoverall、1000+、hidden-like 2面、worst-well、3/5 foldsの全条件を失敗した。
`late_independent_compensation_supported=false`とする。early-onlyもoverall、1000+、fold、worst-wellを失敗した。

mixed exp244だけがoverallとhidden-likeを同時改善しており、方向単独の効果の加算では説明できない。
両方向同時学習の非加法的相互作用は示唆されるが、mixed自身もworst-well guardを失敗しているため採用根拠にはしない。
hidden-like改善と`059c8f24`の崩壊はearly-onlyで再現される一方、同wellのlate-onlyはraw比
`-0.434122`なので、両者はearly方向へ帰属する。lateはmixed内で崩壊を部分緩和している可能性はあるが、
late-onlyの全体guardが不通過なので独立補償とは呼ばない。

## 再現性

- deterministic anchor: false（GPU rerun SHA未確認）
- input cache / exp218 OOF / exp244 mixed OOF SHAをhard assertionする。
- model manifest SHA: `cdfbda0d...e00b`
- OOF decompressed SHA: `3e55541d...e28a`
- metrics / by-well / importance SHA: `104c021e...aa3` / `1ee0ef90...e989` / `93618218...391e`
- runtime: 26,501.519秒、peak RSS 20,371.730 MiB。
- inference / submissionは生成しない。

## 判断

early-only / late-onlyをともに不採用とする。lateの独立補償を否定し、weight / offset grid、risk gate、
inference、submissionへ進まない。prediction-start augmentation branchを終了する。

## 次

なし。このbranchの救済実験は追加しない。
