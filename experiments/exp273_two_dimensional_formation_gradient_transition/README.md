# exp273 two-dimensional formation-gradient transition

## 状態

- Route: `pf_beam`
- Status: completed / direct gradient candidates rejected
- train-side scalar / best gradient RMSE: 11.938287 / 12.169871
- Public LB / Private LB: 対象外 / 対象外
- inference / submission: disabled
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 親と変更点

- 科学的な親はexp209。保存済みscalar HMMをcontrolとして読み、再生成しない。
- exp268からはself-containedな2-shard/aggregate notebook構成だけを参照した。
- 変更はknown-prefix Huber plane、5 fixed gradient prototypes、2D surface transition moveだけ。
- TVT grid、residual-rate grammar、GR emission、calibration、prior、score rowsは固定する。

## 仮説

各wellのknown prefixで `S=TVT_input+Z` の2D局所面をrobust fitし、
`gx*dX+gy*dY-dZ` をexact HMMの弱いposition-transition centerに加えると、
scalar `dS/dMD`だけでは表現しにくいazimuth変化を持つwellで補完候補を作れる。

## 検証方針

- per-well known prefixだけでdeterministic Huber planeをfitする。
- XY rank、condition number、axial azimuth coverageを事前固定guardに使う。
- gradient中心とcovariance 2軸の`+-1 sigma`を合わせた5候補を独立decodeする。
- HMMのGR emission、TVT grid、residual-rate grammar、scale、priorはexp209から固定する。
- invalid geometryはscalar HMMを1回生成して5候補へ複製する。
- 保存済みexp209 scalar HMMをcontrolとし、overall、1000+、geometry、hidden-like、worst-well、
  row/block/whole-well oracle、candidate重複を監査する。
- LightGBM 0、fold 0、booster 0、GPUなし。raw-test inferenceとsubmissionは禁止する。

## 所見

5 gradient direct candidatesはすべてscalarより`+0.231584`から`+0.242444 ft`悪化した。
valid geometryでもbest gradientは`+1.687249 ft`、turningは`+0.981277 ft`、hidden-like 2面と
worst-wellも悪化したため不採用とする。whole-well oracleは`-0.178637 ft`残るが、target-free gateの
根拠がない状態でhard switchやselectorへ進めない。

## 実行構成

- shard 0: `exp273_two_dimensional_formation_gradient_transition_train_variant0.ipynb`
- shard 1: `exp273_two_dimensional_formation_gradient_transition_train_variant1.ipynb`
- aggregate: `exp273_two_dimensional_formation_gradient_transition_train.ipynb`
- inference: disabled guard notebook
- logical HMM variants: 5
- target wells / maximum well-runs: 773 / 3,865
- invalid geometry wellは5 logical candidatesを1 scalar runへ縮約する。

## 参照ファイル

- `config.yaml`
- `SESSION_NOTES.md`
- `result.md`
- `docs/legacy/steering/20260717-exp273-two-dimensional-formation-gradient-transition/`

## 読み方

gradient candidateはdeployableではない。oracle headroomは診断値であり、予測、平均、selectorは保存していない。

## 次のアクション

低-中優先の0-booster prefix-stability readoutだけを次候補とし、inference / submissionは行わない。
