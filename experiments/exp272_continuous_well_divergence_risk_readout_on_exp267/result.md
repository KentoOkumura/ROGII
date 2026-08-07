# exp272_continuous_well_divergence_risk_readout_on_exp267 結果

> **部分無効化:** exp263候補のactual MAEとtarget-free divergenceの関係だけを保持する。exp264 Stage B
> predicted errorを使うcalibration biasと、それを含む総合guardは親のfeature availability leakageにより無効。

## 仮説

exp267 の離散 K=3 structure が失敗しても、target-free divergence は連続 risk 軸として
candidate actual MAE と calibration bias の単調変化を保持する可能性がある。

## 設定

- primary: 12 range/gap features の outer-train robust-scaled 等重み平均
- sensitivity: 全18特徴 outer-train PCA1、report-only
- outcome: 6 primitive の well 別 actual MAE / calibration bias
- bootstrap: outer-fold stratified well resampling、5,000回、95% interval
- 実行量: 0 variant / 0 config / 0 trained fold / 0 booster

## 結果

Kaggle CPU train version 1を完了した。773 wells / 5 folds / 18 signature featuresを読み、
6 primitive各3,783,989 score rowsをstreaming集約した。0 variant / 0 config / 0 trained fold /
0 boosterで、モデル・予測・submissionは作成していない。

| 主判定 | fold Spearman / pooled | 95% bootstrap interval | 判定 |
| --- | --- | --- | --- |
| actual MAE | `0.710570 / 0.773823 / 0.814784 / 0.801434 / 0.826475` / `0.785818` | `[0.749473, 0.817829]` | PASS |
| calibration bias | `0.123895 / 0.099191 / 0.054332 / 0.045484 / -0.112217` / `0.040968` | `[-0.032918, 0.115824]` | FAIL |
| continuous-risk総合guard | actual正5/5、calibration負1/5 | 必須条件を同時に満たさず | **FAIL** |

actual MAEはdivergence quantile q0の`3.386843` ftからq9の`15.940152` ftへ上昇した。
report-only PCA1もactual MAE `0.781531 [0.744880, 0.814806]`だったが、calibration biasは
`0.050812 [-0.024266, 0.126782]`であり、primaryの失敗を救済しない。

Kaggle kernelはversion 1 / id_no `127594096`、private CPU・internet offで約90秒。
全10 artifactsを`kaggle/output/train_v1/artifacts/`へ取得し、manifestのbyte SHAと一致した。
CV / LB / submissionはこのdiagnosticの対象外。

## 解釈

連続divergenceはcandidate-bank actual-error riskの強いrank軸だが、事前仮説に含めた
calibration低下方向はfold-stableではなかった。actual-error相関だけを見て別add-only候補へ
進めると事後的な成功条件変更になるため、`separate_add_only_candidate_supported=false`とする。

## 次

exp267 K=3 branchはclosedのまま維持する。PCA1 / candidate別結果、clip / subset / segment gridで
救済せず、学習・raw-test inference・submissionへ進めない。追加backlogも作らない。
