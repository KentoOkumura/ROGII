# exp144_learned_likelihood_hidden_stress_and_rawtest_parity 結果

## 状態

Kaggle train v1 完了。提出なし。

- Kernel: `kentookumura/exp144-train`
- URL: https://www.kaggle.com/code/kentookumura/exp144-train
- Output: `kaggle/output/train_v1`

## 目的

exp127 の learned likelihood add-only features は shared-row control を改善したが、subset 評価に限られる。exp144 では exp127 の保存済み OOF predictions を exp115 hidden-like split で再集計し、raw-test/full-train parity の未充足条件を checklist として残す。

## 評価設計

- 新規 LightGBM 学習なし。
- `exp092_shared_row_control` vs `learned_likelihood_confidence_addonly` を同じ saved OOF rows で比較する。
- split は `all_shared_rows`、`verification_like_spatial`、`verification_like_typewell_purged`。
- bucket は eval rank、md_since、spatial/eval/prefix/GR/TVT/typewell、learned likelihood confidence bucket。
- raw-test parity は pass/fail checklist で、exp112 raw-test feature regeneration がない場合は fail とする。

## 結果

`lgb_mean` の add-only は all shared rows と exp115 hidden-like split の両方で control を改善した。

| split | rows | wells | control RMSE | add-only RMSE | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| all shared rows | 757,738 | 155 | 9.847053 | 9.727318 | -0.119735 |
| verification_like_spatial | 169,691 | 34 | 13.037491 | 12.760311 | -0.277180 |
| verification_like_typewell_purged | 166,972 | 33 | 13.082838 | 12.787921 | -0.294917 |

hidden-like split の eval rank bucket も全 bucket で改善した。`verification_like_spatial` では near `000_050` が -0.141747、`1000_plus` が -0.295132。`verification_like_typewell_purged` では near `000_050` が -0.129077、`1000_plus` が -0.311980。

by-well の最大改善は `1b1eba53` の -2.155680 RMSE。最大悪化は `aed44918` の +1.071000 RMSE で、worst-well regression は残る。

raw-test parity checklist は pass 5 / fail 3。fail は次の通り。

- full-train coverage: exp112 feature cache は 155/773 wells。
- raw-test feature regeneration: exp112 inference feature generator / raw-test `ml_features` artifact がない。
- hidden submission candidate: not selected。

## 解釈

exp127 learned likelihood add-only feature は hidden-like stress でも支持された。特に near と longtail の両方で改善が残るため、confidence feature としての信号はある。

ただし full-train coverage と raw-test feature regeneration が未充足なので、direct inference port / submit はしない。次に進めるなら、exp112 learned likelihood feature を raw test に target-free に再生成する generator と schema parity audit を先に作る。
