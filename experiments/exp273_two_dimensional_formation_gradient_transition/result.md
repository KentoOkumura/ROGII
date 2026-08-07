# exp273_two_dimensional_formation_gradient_transition 結果

## 状態

Kaggle CPU shard 0/1 version 1とaggregate version 1を完了した。coverage、input、SHA、
saved-control parityはPASSしたが、5 gradient direct candidatesはすべてscalar controlより悪化した。
仮説はdirect candidateとして不採用とし、inference / submissionへ進めない。

## 仮説

known prefixの`S=TVT_input+Z`から推定した2D formation gradientをexact HMMの弱いtransition候補にすると、
scalar surface-rate controlをturning wellで補完できる。

## 設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- plane: per-well known-prefix deterministic Huber IRLS
- candidates: gradient center + covariance 2軸の`+-1 sigma`、合計5
- transition: `gx*dX + gy*dY + residual_rate*dMD - dZ`
- control: 保存済みexp209 scalar HMM、再生成なし
- shards / LightGBM config / fold / booster: 2 / 0 / 0 / 0
- GPU / inference / submission: なし / なし / なし

## 親と変更点

exp209の保存済みscalar HMMを再生成しないcontrolとし、grid、residual-rate state/dynamics、
Gaussian GR emission、calibration、priorを固定した。変更はknown-prefix Huber planeから得る
5 gradient prototypesと、position transitionの2D surface moveだけである。

## 結果

| メトリック | 値 |
| --- | --- |
| scalar control RMSE | 11.938287 |
| best gradient direct RMSE | 12.169871（axis1-minus、`+0.231584 ft`） |
| 1000+ best gradient delta | `+0.263798 ft` |
| hidden-like spatial / typewell-purged best delta | `+0.318575 / +0.322224 ft` |
| geometry valid / fallback | 111 / 662 wells |
| valid-row best gradient delta | `+1.687249 ft` |
| turning-row best gradient delta | `+0.981277 ft` |
| worst-well regression | `+36.118726 ft`（`dd7d638e`） |
| row / block128 / block256 / block512 / well oracle delta | `-0.188841 / -0.188599 / -0.188204 / -0.187164 / -0.178637 ft` |
| Public / Private LB | 対象外 |

静的検証ではexp273固有9 testsとrepository全119 tests、Jupytext round-trip、py_compile、
Ruff F821、strict experiment/template validationを通過した。Kaggle aggregateは3,783,989 rows / 773 wells、
runtime `161.445`秒で完了し、生成された10 CSVのSHAはsummary記載値と一致した。

5 gradient候補のoverall deltaは`+0.231584`から`+0.242444 ft`で全滅した。geometry-valid 566,316 rowsでも
best axis1-minusが11.898526、scalarが10.211277で大幅悪化した。期待したturning群でもbest gradientは
11.799598、scalarは10.818321だった。valid 111 wellsでは各候補59--61 wellsが改善した一方、
50--52 wellsが悪化し、median deltaは約`-0.01`から`-0.03 ft`でもmeanは`+0.53`から`+0.56 ft`、
最大回帰は約`+36 ft`だった。少数wellの巨大な外挿失敗がglobalとworst-wellを壊している。

一方、scalar + gradient bankのoracleはrowで`-0.188841 ft`、whole-wellでも`-0.178637 ft`だった。
whole-well unique bestはscalar 50 wellsに対し、axis1-minus / axis1-plus / axis2-minus / axis2-plusが
`21 / 21 / 3 / 16` wells、centerは0 wellsだった。候補headroomはあるが、target-freeな選択根拠は本実験では得ていない。

## 再現性

- deterministic submission anchor: いいえ。no RNGだがNumba/LAPACK微小差を許容するtrain-side audit。
- seed policy: no RNG、well shardだけstable SHA256。
- input SHA: exp209 control decompressed
  `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`をhard guardした。
- shard 0/1 decompressed SHA:
  `347b87554261cb904bc7f98d6d1eb64ed5aaa46f15720011b94797d814aeac97` /
  `98939d080e4b5bfa3ab93631601d496946b0e9ddcc8aac4bb10a430eeda7d407`。
- aggregate prediction content SHA:
  `87e59647018cf69f187d293e462afa737334c1d90da6e56ed385a85ecae0b79d`。
- model/submission SHA: 対象外。

## 解釈

2D formation gradientを無条件のexact-HMM transition candidateとして使う仮説は棄却する。guard通過wellでも
turning、1000+、hidden-like、worst-wellを悪化させたため、condition / rank / azimuth thresholdの緩和や
gradient scale gridでは救済しない。covariance-axis 4 prototypesはcenterと近く、whole-well unique bestでcenterが0、
prototype間の追加多様性も小さい。

ただしvalid wellsの改善/悪化がほぼ二分されwhole-well oracleが残るため、局所面そのものより
known prefix内でgradientが安定して外挿可能かのtarget-free診断が不足している可能性がある。

## 次

inference、raw-test path、candidate平均、selector、submissionは実行しない。次は新規学習やHMM再実行を行わない
`formation_gradient_prefix_stability_risk_readout_on_exp273`で、full / last-256 / last-512 known-prefix planeの
gradient角度・大きさ・fit残差の安定性がcandidate悪化とfold横断で対応するかだけを0-booster監査する。
