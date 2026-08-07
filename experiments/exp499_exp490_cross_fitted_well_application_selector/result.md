# exp499_exp490_cross_fitted_well_application_selector 結果

## 状態

Kaggle private CPU version 2完了。technical PASS / predictability FAIL /
safe-router FAIL、terminal close。

## 仮説

target-free well特徴からexp490のsigned benefitを予測し、exp357へのfallbackを
strict-nestedに選ぶことで、always-exp490より平均とtailを安全に改善できる。

## 実行

- kernel: `kentookumura/exp499-exp490-cross-fitted-well-selector-train`
- Kaggle id_no / version: `129362815` / `2`
- runtime: private CPU、internet off、`57.051717 sec`、peak RSS `1.454178 GiB`
- scope: 3,783,989 rows / 773 wells / target-free 32 features
- validation: outer 5 × inner 4 strict nested well selector
- execution: 1 variant / 2 learned configs / 45/45 CPU model fits
- LightGBM / control retraining / new prediction / HMM / PF / Beam / GPU: すべて0
- inference / submission: 0 / 0

version 1はexp498のローカル仮実行contract SHAをpinしていたため入力SHAで停止した。
feature CSV SHAと科学条件は変えず、Kaggle version 2の正本contract SHAへ修正して
version 2を完走した。

## Technical gate

9項目を全てPASSした。

- input SHA、3,783,989 rows、773 wells、5 folds、32 featuresが一致。
- finite coverage、outer prediction coverage、45 model fitsが完全。
- feature freeze前のoutcome readは0、forbidden loaded columnsは0。
- feature content SHA: `54c7e1dac064f929edd57dc03bf00d1a15b47340d5c40d7e6e8afc3e707bb0d4`
- feature contract logical SHA: `f8d510b83344eccebc101f94e0df8145e77fd5c6411da77e81c7b2483e14ad71`

## Predictability gate

FAIL。

| 指標 | 結果 | gate |
| --- | ---: | ---: |
| pooled beneficial-well AUC | 0.521151 | >= 0.60 |
| AUC >= 0.55 folds | 1 / 5 | >= 4 / 5 |
| Spearman正方向folds | 5 / 5 | >= 4 / 5 |
| pooled Spearman | 0.122250 | descriptive |

単一の`parent_exp226_abs_mean`はpooled AUC `0.591912`、fold最小
`0.564386`、正方向5/5で最強だった。ただし複数target-free特徴からsigned SSEを学習した
cross-fitted scoreでは識別力が低下し、0.60 gateへ届かなかった。

## Safe-router gate

FAIL。

| 指標 | always exp490 | selector / 結果 |
| --- | ---: | ---: |
| pooled RMSE | 8.480155260 | 8.514310626 |
| gain vs always | - | -0.034155367 ft |
| applied wells | 773 | 716（92.626%） |
| beneficial precision | 58.085% | 58.101% |
| catastrophic applied wells | 51 | 48 |
| selected-minus-parent p95 | +7.257814 ft | +7.098191 ft |
| selected-minus-parent worst | +49.602560 ft | +49.602560 ft |

4 foldsではinner OOFが学習selectorを棄却してalways-exp490を選んだ。唯一HGBを選んだ
fold 1は`8.659383 -> 8.822361 ft`へ`0.162978 ft`悪化し、pooled悪化の原因になった。
tailもほぼ除去できず、324 harmful wells中300 wellsへexp490を適用した。

## 参考上限

truthを使うreport-only well oracleは`6.560582422 ft`で、always-exp490より
`1.919573 ft`良い。したがって選択余地そのものは大きいが、現在観測できるtarget-free
32特徴とこのfold構造から、未知wellへ一般化する安全な判断規則は得られなかった。

## 再現性

- output: `kaggle/output/train_v2`
- feature table SHA: `54c7e1dac064f929edd57dc03bf00d1a15b47340d5c40d7e6e8afc3e707bb0d4`
- selector OOF SHA: `8b9a44d3bfd4b62203c2ac85598bb3c5970914cf769ad53f93e50041343d6610`
- inner score SHA: `5293c8b93cbc04899aeb095d296bb775bd70b6de9168f8984a39500bb87f0d62`
- fold metrics SHA: `f6e4cc7351f60272c70c9ab8b7832f6d81f63947b56c65a508f034e16ffe538a`
- model manifest SHA: `72577e6da61cfef4ec67679d1be1b635e645e652ec89fa37a03e1500bda55ac1`
- summary SHA: `d1ce774dff2fe113490616318291538e120714aa9a98fe2dccff34cfc0054413`
- metrics SHA: `05be2063b36940585e1aa4b73a6077ce3af923df46291a75b89550fc39ddf824`
- submission SHA: not applicable

## 判断

`close_safe_target_free_exp490_router_without_same_oof_rescue`。

現状では、exp490を適用すべきwell／避けるべきwellを事前に安全に見極められない。
弱いranking signalはあるがhard routingへ昇格させず、threshold・特徴・model追加による
same-OOF救済、inference、submissionは行わない。exp490のterminal closeを維持する。
exp500の固定PF機構移植は別仮説・別承認のままとし、本結果をadaptive gateへ使わない。

