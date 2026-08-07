# exp260_matched_early_late_attribution_on_exp244

## 状態

- Route: `ml_model`
- 状態: train-side attribution完了・不採用
- CV / Public / Private LB: late-only 8.489116155 / 未提出 / 未提出
- 親実験: `exp244_bidirectional_prediction_start_pseudotail_augmentation`
- 比較control: 保存済みexp218 OOF、保存済みexp244 mixed OOF
- 作成日: 2026-07-16

## 仮説

exp244 mixedで見えたhidden-like改善とworst-well崩壊が、early viewとlate viewのどちらに由来するかを
matched条件で分離する。late-onlyが独立にoverall / 1000+ / hidden-likeを改善しworst-well +2 ft以内を
通るなら、early-only悪化をlateが補償した可能性を支持する。

## 変更点

- early-only: official 3,783,989 rows + `-1000/-250` 1,537 views / 384,250 rows。
- late-only: official 3,783,989 rows + `+250/+1000` 1,544 views / 385,907 rows。
- official weight 1.0、pseudo weight 0.5、380 features、3 configs、5 foldsを両variantで固定する。
- exp244 mixedと同じ4 cacheを1回だけmemmapへstreamし、direction maskだけを変える。
- outer-valid source well由来pseudo rowsをtrainから除外し、validationはofficial-start rowsだけにする。
- raw exp218とexp244 mixedは保存済みOOFを使い再学習しない。

## 学習量

- active variants: 2
- LightGBM configs: 3 / variant
- folds: 5
- 合計boosters: 30
- parent/control再学習: なし
- inference / submission: 無効

この計算量は2026-07-16にユーザーへ提示済みで、「これに進んでください」を実行承認として記録する。

## 検証方針

- primary: official-start OOF RMSE。
- stress: 6 distance buckets、1000+、hidden-like 2面、5 folds、by-well、worst-well。
- late独立補償guard: overall改善、1000+非悪化、hidden-like 2面非悪化、worst-well +2 ft以内、
  3 / 5 folds以上改善。
- exp239 early-onlyはcutoff distributionとsamplingが異なるためmatched比較に使わない。

## 実行入口

- 学習notebook: `exp260_matched_early_late_attribution_on_exp244_train.ipynb`
- Jupytext source: `exp260_matched_early_late_attribution_on_exp244_train.py`
- 推論notebookは今回使用しない。

## 結果

| 候補 | overall RMSE | raw exp218差 | 1000+差 | hidden spatial差 | hidden typewell差 | 改善fold | worst-well差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exp218 raw | 8.475793752 | - | - | - | - | - | - |
| exp244 mixed | 8.472379731 | -0.003414021 | -0.009134943 | -0.415836329 | -0.405110136 | 3/5 | +16.650567 |
| early-only | 8.513933814 | +0.038140063 | +0.036479760 | -0.332981056 | -0.325110885 | 2/5 | +18.623158 |
| late-only | 8.489116155 | +0.013322404 | +0.013735077 | +0.052783536 | +0.058460626 | 2/5 | +3.408451 |

## 所見

late-onlyはearly-onlyよりoverallで0.024817659 ft良く、+2 ft超悪化wellも17から2へ減った。しかしraw exp218より
0.013322404悪く、1000+、hidden-like 2面、3/5 folds、worst-well +2 ft以内の全guardを通らなかった。
したがってlate viewの独立補償は支持しない。

early-onlyはhidden-like 2面を改善したが、overallと1000+を悪化させ、worst `059c8f24`が
+18.623158 ft崩れた。mixedだけがraw overallを改善し、方向単独では再現できないため、exp244の小gainは
late-onlyの独立効果ではなく、両方向同時学習の非加法的相互作用として扱う。ただしmixedもworst-well guardを
失敗しており採用しない。

方向別の帰属は明瞭で、hidden-like改善と`059c8f24`の崩壊はいずれもearly側に由来する。同wellのlate-onlyは
raw比`-0.434122`で、late側が崩壊源ではない。mixedでは同wellが`+16.650567`まで部分的に緩和されるものの、
許容範囲には戻らない。

## 判断

両variantを不採用とし、weight / offset grid、risk gate、current-test inference、submissionへ進まない。
prediction-start augmentation branchは本実験で終了する。

## 次

なし。このbranchから新規実験、inference、submissionは行わない。
