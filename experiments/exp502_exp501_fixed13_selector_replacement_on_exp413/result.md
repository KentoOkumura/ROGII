# exp502_exp501_fixed13_selector_replacement_on_exp413 結果

## 仮説

exp413 Stage Cのnested compact74を、保存済みexp501 fixed13 selector compact77へ
置換すると、他のexp413 feature/model条件を固定したままTVT OOFを改善できる。

## 設定

- 親 / control: `exp413_scale5_likpf_full_replacement_on_exp335`
- selector source: `exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264`
- feature: `clean273 + exp501 compact77 + signed23 = final373`
- 検証: `well_id` GroupKFold outer 5、exp413と同一fold manifest
- メトリック: RMSE
- シード: 42
- actual train: 1 treatment × 3 configs × 5 folds = 15 GPU boosters
- control / selector / signed selector再学習: 0 / 0 / 0

## 変更点

exp413のnested selector compact74だけを除去し、同じ位置へexp501 compact77を挿入する。
clean273とsigned23は固定し、add-onlyやdual selector surfaceにはしない。

## 結果

| メトリック | 値 |
| --- | --- |
| exp413 saved control CV | 7.884802794404715 |
| exp502 CV | 7.882143903310376 |
| gain | +0.002658891 ft（改善、必要値+0.03 ftを未達） |
| nonworse fold | 3 / 5 |
| 最大固定scope delta | +0.140943998 ft（上限+0.02 ftを超過） |
| primary gate | FAIL_CLOSE |
| Public LB | 未実行 |
| Private LB | - |

### Fold別

| outer fold | exp413 | exp502 | exp502 - exp413 |
| ---: | ---: | ---: | ---: |
| 0 | 7.919988324 | 7.863179015 | -0.056809309 |
| 1 | 8.377381333 | 8.315025697 | -0.062355636 |
| 2 | 7.539713352 | 7.280264955 | -0.259448397 |
| 3 | 7.574331167 | 7.690358020 | +0.116026853 |
| 4 | 7.982868393 | 8.217554230 | +0.234685837 |

### 固定scope

| scope | exp413 | exp502 | exp502 - exp413 |
| --- | ---: | ---: | ---: |
| MD 0--250 | 1.472437865 | 1.439411566 | -0.033026299 |
| MD 250--1000 | 3.890857670 | 3.844131286 | -0.046726384 |
| MD 1000+ | 8.663017156 | 8.664338336 | +0.001321181 |
| hidden-like spatial | 8.364712530 | 8.504299093 | +0.139586563 |
| hidden-like typewell-purged | 8.307714847 | 8.448658845 | +0.140943998 |

### Report-only tail

- by-well delta p95: `+1.293097772 ft`
- worst well: `a8ed028a`、delta `+8.159899027 ft`
- `+1 / +3 / +5 ft`以上悪化well: `60 / 14 / 8`

## 実行状態

- replacement-only final373をKaggle T4 version 1で完走した。
- exp501 compact77 / removed exp413 nested74 / retained signed23 / saved exp413 OOFを
  SHA・fold manifest・row role・key・anchorで検証し、technical checksは全PASS。
- 実行量は15 / 15 GPU models、control / selector / signed selector再学習0、
  HMM / PF / Beam再実行0で契約どおり。
- notebook metrics出力は`20702.431 sec`、最大log timestampは`20715.351 sec`。
- 固定primary gate FAILで停止し、inferenceとsubmissionは未実行。

## 再現性

- deterministic anchor: false（GPU bitwise一致は主張しない）
- seed policy: exp413固定seedとper-config seedを継承
- kernel: `kentookumura/exp502-exp501-fixed13-replace-exp413-train` version 1、
  id_no `129459588`、`NvidiaTeslaT4`、internet disabled
- 15 model grid / SHA: 3 configs × 5 foldsを一意に網羅、15 SHAすべて一意
- best iteration min / median / max: `522 / 1950 / 9832`
- exp501 compact manifest SHA:
  `32317a715997c7a7e145d7122a8ac37733adb30710e571ccbf11a81c2d79c257`
- fold manifest SHA:
  `fa41084c5fcb4adffb88d44211b4cc5d2d2f46b5bd4d65828b6af941184b2a6d`
- exp413 control OOF SHA:
  `9bd2d17778b3b27d771b12cbff72def8b87e6cdf14062e1c0ba192434cef4a9d`
- exp502 final373 schema SHA:
  `5599190a273cda6708053c8cbc046b49619d13db69fca29d0e20ff5d3c8ea865`
- exp502 feature manifest / model manifest / OOF SHA:
  `024754edc0fc8efd472a386dc13c8d5c275fc1c634a6d122ee7585b055be9457` /
  `ae221d2f88f282108d440419b4fd602330d5d2a2e7a0dc8809ac244c3eaea575` /
  `97230e2e421635db564e5988865e907536f7da19a09e11e8671e07e15777ae99`
- metrics / input contract SHA:
  `7ee56bff89d292df8d3153d7a59dce3024d3788a7251ac81b55017198dbfe07a` /
  `c59b862ec538622672629e103b4653396e448ceb8a78bada4c83787443b8edfc`
- sealed package config / notebook / metadata SHA:
  `2377570dc24c68ff6e9bf439e75a2869efeb5b92e87261da8ff4cd8b7eeafd4c` /
  `d7ff5c254d16ca2749c59a4d690f9c56eeb6b98bd672f45c5261e2076d91a94b` /
  `cd5e8957e196a9129b1ed24b2ffff4eb44e08da7b195c2aa05b067bbd3b9fbe7`
- submission SHA: 未生成

## 解釈

exp501 compact77への全面置換はpooled CVを`0.002659 ft`だけ改善したが、事前固定した
`0.03 ft`の実用差に届かなかった。fold 0--2は改善した一方、fold 3 / 4は
`+0.116027 / +0.234686 ft`悪化し、hidden-like 2面も約`+0.14 ft`悪化した。
selector単体での強いpooled改善は、exp413のclean/signed面と組み合わせたdownstream TVTへ
安定転移しない。technical failureではなくscientific failureである。

## 次

`FAIL_CLOSE_EXP501_FIXED13_SELECTOR_REPLACEMENT_ON_EXP413`として閉じる。feature subset、
blend、weight、threshold、gateのsame-OOF救済は行わず、inference / submissionへ進めない。
原因説明が必要なら、保存exp502 OOFだけを使うfold 3/4・hidden-like transfer attributionを
別実験・別承認の低優先readoutとして扱う。
