# exp165_coordinate_frame_normalization_features_on_exp148 結果

## 状態

Kaggle CPU split train 完了。train-side OOF は exp148 historical `lgb_mean` より悪化したため、推論化・提出はしない。

## 評価方針

exp148 の既存 feature surface と learned likelihood confidence features を固定し、coordinate-frame normalization features だけを add-only する。control 再学習は行わず、保存済み exp148 `lgb_mean` CV 8.501281182 / Public LB 7.960 を比較基準にする。

## 実行

- `coordinate_frame_addonly`
- CPU deterministic threads8
- `lgb0` / `lgb1` / `lgb2` split train
- 3 configs x 5 folds = 15 boosters

## 結果

| model | RMSE |
| --- | ---: |
| lgb0 | 8.623039477 |
| lgb1 | 8.586673413 |
| lgb2 | 8.616753590 |
| lgb_mean | 8.549931602 |

exp148 historical `lgb_mean` 8.501281182 に対して、exp165 `lgb_mean` は +0.048650420 悪化。単体 split も 3-model mean も exp148 を上回れなかった。

3-model mean prediction SHA proxy (`id,pred_tvt` rounded 8 decimals): `e58c971423e3972bd29a8bd3cfd328835964df93e95315c1c07528849909e535`

worst wells 上位は `86454a6f` 47.182148、`1b1eba53` 46.856087、`fb03ae90` 45.687022、`91b301ce` 37.431130、`81bf5923` 32.283569。

## 判定

coordinate-frame normalization features は importance / diagnostic には残せるが、exp148 add-only の train-side 改善にはならなかった。direct TVT candidate / hard correction / selector 化は当初方針どおり行わず、この実験は rejected として閉じる。
