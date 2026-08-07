# exp258 結果

## 状態

Kaggle selector train v1は完了し、selector guard不通過で終了しました。final TVT LightGBM、
inference、submissionは実行していません。

## 親実験と変更点

親は`exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`です。候補生成と後段構成を固定し、
selectorのinner-trainだけへ実測GR残差block augmentationを追加しました。比較にはhistorical exp238を
使い、親/controlは再学習していません。

## 実行

- Kernel: `kentookumura/exp258-gr-residual-noise-transplant-selector-train` v1
- Kaggle id_no: `127392032`
- Runtime: 約4時間36分52秒、CPU、internet disabled
- Data: 3,783,989 rows / 773 wells
- Model: `real_residual_block` 1 variant、outer 5 × inner 4 = 20 selector boosters
- Candidate/context: 11 / 184
- Parent/control再学習: なし

## Selector結果

| 指標 | exp258 | historical exp238 | 差 | 判定 |
| --- | ---: | ---: | ---: | --- |
| global delta RMSE vs likpf | -3.085357 | -3.089911 | +0.004554 | fail |
| near 000_050 delta RMSE | -0.607888 | -0.609540 | +0.001652 | fail |
| 1000+ delta RMSE | -3.365354 | -3.372225 | +0.006871 | fail |
| worst-well regression | +38.002960 | +37.680897 | +0.322063 | fail |
| expected-error MAE | 4.523354 | 4.532978 | -0.009625 | pass |
| candidate AUC within 10 ft | 0.919159 | 0.919334 | -0.000175 | fail |

residual auditは773 wells、20 nested splitでdonor/validation overlap 0でした。20 selector modelも
outer 5 × inner 4を完全被覆しており、validationはcleanのままです。したがって、guard不通過は
明白なfold leakageやmodel欠落ではなく、実測GR残差augmentationの品質結果として扱えます。

## 判断

不採用です。expected-error calibrationの小改善だけでは、candidate選別・near・longtail・worst-wellの
悪化を補えません。selector guard通過時だけ実行する契約だったため、後段TVT LightGBM 15 GPU
boostersは実行しません。white-noise / shuffled controlやaugmentation比率gridによる救済も行いません。

## 主要SHA

- selector summary: `9e6575577ace80054073d28281885c80a3ed6b266116b400df6ffcafcbcae2b7`
- model manifest: `8effd9232b895dd776ae29f85be1170224f98cf0b91f8d1786649dc9e7489fb4`
- residual audit: `83668b9b0a4b18653eb8d11c6e21eba015a0411d518b6f114862e8426c71ae3d`
- augmentation inventory decompressed: `630dac60d16f87ce08bb75289730e1f7544f29610b3d1fcdcad381afa9455af1`

## 次アクション

なし。この分岐はnegative resultとして閉じ、final学習、救済grid、inference、submissionへは進みません。
