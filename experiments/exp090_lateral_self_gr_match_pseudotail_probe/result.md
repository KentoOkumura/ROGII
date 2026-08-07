# exp090_lateral_self_gr_match_pseudotail_probe 結果

## 状態

Kaggle train v1 完了。弱い positive 結果だが、推論化は未選択。

## 仮説

同一 horizontal well の既知 prefix GR tail と評価区間 GR を照合し、match confidence と matched prefix TVT offset を特徴量として追加すれば、exp073 の deterministic full replay ML surface が苦手な中距離/遠距離 tail を少し改善できる可能性がある。

## 評価方針

exp072 deterministic full replay train cache と exp073 LightGBM config family を固定し、GroupKFold by well で以下を比較する。

- `control_exp073_base196`
- `self_gr_core`
- `self_gr_core_multiscale`
- `self_gr_core_context`

exp008 typewell NCC、exp017 DTW/DWT、exp042 Ravaghi NCC/GR match の失敗を前提に、pooled RMSE だけではなく near rows、distance bucket、tail rank bucket、worst-well 悪化を確認する。

## 結果

| variant | lgb_mean CV | control 差分 | feature 数 |
| --- | ---: | ---: | ---: |
| `self_gr_core_multiscale` | 9.516732864806912 | -0.009557442830422147 | 210 |
| `control_exp073_base196` | 9.526290307637334 | 0.000000000000000 | 196 |
| `self_gr_core` | 9.541383726295855 | +0.01509341865852143 | 201 |
| `self_gr_core_context` | 9.599141986796033 | +0.07285167915869906 | 208 |

`self_gr_core_multiscale` が best。改善は exp073 control から -0.00956 RMSE と小さい。distance bucket では `1000_plus` が -0.01060、`500_1000` が -0.00525、`250_500` が -0.00118 改善した一方、`100_250` は +0.00325、`050_100` は +0.00075 悪化した。near `000_050` は -0.00299 と壊れていない。

well 単位では 402 wells 改善、371 wells 悪化。最悪悪化は `8f201368` +0.97509、`4c2208f5` +0.95594、`89f1085d` +0.90185。最大改善は `fdfd57da` -0.98479、`70925e23` -0.96946、`9d3ec64c` -0.81335。

best variant の self-GR 重要度上位は `self_gr_sc25_delta_tvt`、`self_gr_sc25_score`、`self_gr_sc25_l2`、`self_gr_delta_tvt_ens`。短窓より half window 25 の一致が効いた。

証拠:

- kernel: `kentookumura/exp090-self-gr-train` v1
- output: `/tmp/kaggle-output/exp090_lateral_self_gr_match_pseudotail_probe/train_v1`
- copied readout: `artifacts/exp090_lateral_self_gr_match_pseudotail_probe_result_readout.json`
- source feature SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- best OOF prediction SHA: `6ffd17d023d1b0db0d85e4782d5b5cc75effb094635d030721b095a1616fc3d9`

## 次の判断

この結果だけでは推論化しない。CV 改善は小さく、well 単位の悪化も大きい。次に進めるなら `self_gr_core_multiscale` をそのまま submit するのではなく、long-tail / high-confidence 条件での gate、または悪化 well 条件の診断を先に行う。
