# exp404_scale5_sigma_gr_likelihood_pf_ablation 結果

## 結論

Kaggle private CPU train version 4（id_no `128628818`）が完了した。
technical gateはPASSしたがscientific gateはFAILしたため、
likelihood temperature 5でも全well一律`gs×1.3`を棄却し、同じOOFで救済せず
終了する。inferenceとsubmissionは行わない。

## Primary結果

| 指標 | control: scale5 x1.0 | candidate: scale5 x1.3 | x1.0 - x1.3 |
| --- | ---: | ---: | ---: |
| pooled RMSE | 10.914522 | 11.174615 | -0.260093 ft |
| MAE | 6.429207 | 6.839276 | -0.410069 ft |
| within 10 ft | 0.801094 | 0.780547 | -0.020547 |

候補x1.3はpooled RMSEを`0.260093 ft`悪化させた。nonworseだったfoldは
fold 3だけで`1/5`、事前条件の`4/5`を満たさなかった。

| scope | x1.0 RMSE | x1.3 RMSE | gain |
| --- | ---: | ---: | ---: |
| raw GR observed | 10.912496 | 11.092128 | -0.179632 ft |
| raw GR missing | 10.918879 | 11.350024 | -0.431145 ft |
| high missing fraction | 10.326934 | 11.020866 | -0.693932 ft |
| suffix 1000 ft以上 | 11.939478 | 12.225576 | -0.286098 ft |
| hidden-like spatial | 13.038363 | 12.970037 | +0.068326 ft |
| hidden-like typewell-purged | 12.777980 | 12.764206 | +0.013773 ft |

hidden-like 2面は小幅改善したが、pooled、fold再現性、raw GR observed/missing、
high-missing、long-tailを同時に満たさない。by-well RMSE差のp95は
`+4.826467 ft`、worst-well regressionは`+37.333851 ft`で、tail gateもFAILした。
改善wellは`286/773`に留まり、worst well `60e37807`はmissing fraction
`0.025936`でも`+37.333851 ft`悪化した。high-missingだけでは失敗を説明できず、
単純なmissingness gateを同じOOFから作る根拠にもならない。

## Technical gate

technical gateと4つのparity checkはすべてPASSした。

- 3,783,989 rows / 773 wells / finite coverage 1.0
- 5 reporting folds、全well fallbackなし
- common seed labels、実行量、scale ratio、post clip 0を確認
- truth / error / fold / hidden-likeのpre-freeze read countはすべて0
- x1.0 mean対exp072 RMSE差: `3.4881e-06 ft`
- x1.3 mean対exp400 RMSE差: `1.8500e-07 ft`
- x1.3 scale5対exp400 RMSE差: `1.6152e-07 ft`
- scientific contract SHA:
  `41c1f95ca8bd7d20eef00f244ced7d4dbc4b3571cc9fd4189c08d6831ef15b57`
- prediction logical SHA:
  `5f4b6e715081b598b0a34607ad0c81339d0ecd5882ea3a45dd79f33123959a00`
- prediction raw gzip SHA:
  `b3699432a691229da5a6562ce74e0b84f1bee3021bd80d650526906f5aa390f8`
- prediction decompressed SHA:
  `00fe1b90fce84bd601b4b91442d9fc698200aafadd48658f7d8c26ec1fbe0d00`
- artifact manifest SHA:
  `131a65c36acafc8d3cac9bdc18b2b5e296ff9aceb93cbf2702b1a79e675b58f3`
- metrics SHA:
  `9b69317a8979cb29e899df336f218d2008504844a27af53a3de0b4ff34e3b83d`

version 4のlate-readout runtimeは`270.988 sec`。新規PFは0件で、version 1で
freeze済みの1,546 PF well-runs / 197,888 seed-well trajectories /
98,944,000 particle startsをSHA固定して再利用した。

## 実行履歴

- version 1: PFとprediction freeze完了後、hidden-like role countの設定欠落でERROR
- version 2: `.csv.gz.bin`をplain CSVと推論してERROR
- version 3: pandas 2/3の文字列dtype表記差によるschema SHA不一致でERROR
- version 4: semantic dtype正規化後に全technical checkとlate readoutを完了

3回のERRORはいずれも科学設定や予測値の失敗ではない。version 4はversion 1の
同一prediction bytes、logical SHA、scientific contractを保持して判定した。

## 判断

`scale5_rejects_global_gs_x1p3_close_without_rescue`。
temperature、multiplier、clip、particle、seed、well gate、HMM、Beam、ML、
blendによる同一OOF救済は行わない。LB、推論、提出はなし。

実行URL:
<https://www.kaggle.com/code/kentookumura/exp404-scale5-sigma-gr-likpf-ablation-train>
