# exp450_dzdmd_conditioned_tvt_rate_likelihood_pf 結果

## 仮説

visible-prefixから`TVT-rate = beta*dZ/dMD + intercept`をwell別に推定し、
その平均との差だけを持続させれば、exp446で失われたknown-Z forcingを
TVT-rate transitionへ戻し、exp404 likelihood-PFを改善できる。

## 設定

- Route: `pf_beam`
- 親endpoint: exp417
- 保存control: exp404 temperature-5 / GR scale x1.0
- particles / seeds / temperature: `500 / 128 / 5.0`
- scientific variant: learned prefix-affine residual-AR 1本
- Stage 0A: sentinel12、parent+exact transform 24 PF well-runs
- Stage 0B: fixed32、candidate 32 PF well-runs
- 保存control PF rerun / model / booster / HMM / Beam / GPU: 全て0

fixed32はmechanism preflightでありCVではない。

## Kaggle実行

- kernel: `kentookumura/exp450-dzdmd-tvt-rate-likpf-train`
- id_no: `129167787`
- version 1: `COMPLETE`、元の内部exact parity gateでFAIL
- version 2: `ERROR`、改訂Stage 0A PASS後にexp404 source SHA方式不一致で停止
- version 3: `COMPLETE`、status `stage0b_mechanism_failed_closed`
- version 3 runtime: `779.670823 sec`
- Python / NumPy / pandas: `3.12.13 / 2.0.2 / 2.3.3`
- peak RSS: `1.977406 GB`

## Stage 0A

ユーザー承認により、数学的に等価な座標表現の内部粒子差ではなく、
最終temperature-5集約予測の最大差`<=1e-6 ft`をhard gateとした。

- `passed=true`
- 最大temperature-5差: `4.836693e-09 ft`
- artifact readback、finite、clip decision、実行量: 全PASS
- 内部resampling差57回、最大seed prediction差`21.176791 ft`などは
  diagnosticとして保存し、gateには使用していない
- parity report SHA:
  `84856a32a220719a1c4038841f00dfba02d149ba72339c6e8272c3d59e4303a1`

## Stage 0B

全16 gate中10 PASS、6 FAILで、総合`passed=false`。

良かった点:

- prefix tail20 backtest SSE ratio: `0.241989`、非悪化`5/5 folds`
- persistent scope pooled RMSE:
  `12.785573 -> 12.462589 ft`（`-0.322984 ft`）
- persistent改善well: `10/16`でgate PASS
- fixed32全体:
  `9.616741 -> 9.468335 ft`（`-0.148406 ft`）
- runtime全773投影: `11,906.624 sec`でPASS
- OLS係数・予測finite、実行量、role/fold、truth-late、SHA: PASS

失敗した点:

- zero-directed under-response share:
  `0.185047 -> 0.189697`、削減量`-0.004650`で悪化
- forward-cause episode SSE削減: `-5.7969%`
- persistent episode SSE削減: `+1.3603%`で必要な`+5%`未達
- persistent改善fold: `2/5`で必要な`4/5`未達
- matched control pooled RMSE:
  `4.392083 -> 4.684612 ft`（`+0.292528 ft`、上限`+0.02`）
- matched control by-well delta p95:
  `+1.678265 ft`（上限`+0.25`）

## 再現性

- scientific contract SHA:
  `e8a9f5abf42f654a925c002b3e7940f19c407c8fa2a379bbd1af5518605442fe`
- candidate prediction logical SHA:
  `327c190c17ebc23b8568076e5b4a56b9d26b49538ed24821d7eb370c0d72ab03`
- candidate prediction decompressed SHA:
  `97db5a745934ae8d98924676d91d8d8ca35f020762553fe68c92014caeb3ecad`
- prefix-fit logical SHA:
  `ed47678864d149a4016617a48fd75346024d5f631a35c343f54b059f45e57881`
- 保存exp404 source logical SHA:
  `5f4b6e715081b598b0a34607ad0c81339d0ecd5882ea3a45dd79f33123959a00`
- truth/fold/role/episode read before freeze: 0
- deterministic anchor: no

専用testは`19 passed`。Jupytext、`py_compile`、Ruff、strict experiment
validationもPASSした。全体`make test`はexp450外の既存5 test moduleの
collection errorがあるため、repository全体PASSは主張しない。

## 解釈

visible-prefix上ではaffine centerが`mu=-g`より明確に良いが、その関係は
unknown suffixで安全に一般化しなかった。persistent wellsの平均は改善した一方、
狙ったunder-responseとforward/persistent episode効果が弱く、matched controlを
大きく悪化させた。平均改善だけで採用できないという事前gateが機能した。

## 判断

`stage0b_mechanism_failed_closed`として終了する。Stage 1、再実行、
beta/intercept・window・support・momentum・noise・temperature・GR scaleの探索、
well/row gate、blend/selector、inference、submissionへ進まない。
