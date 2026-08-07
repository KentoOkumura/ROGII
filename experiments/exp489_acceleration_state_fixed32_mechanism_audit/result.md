# 結果

## 結論

Kaggle private CPU Stage 0B version 1は`COMPLETE`しました。technical gateは
`10/10 PASS`でしたが、mechanism gateは`2/8 PASS`でall-ANDを満たさず、
`stage0b_fail_closed`です。Stage 1、inference、submissionへは進みません。

## 実行

- kernel: `kentookumura/exp489-accel-state-fixed32-mechanism-audit-train`
- version / id_no: `1` / `129171668`
- fixed32: 32 wells / 156,088 rows
- 新規計算 / 再利用: 28 / 4 wells
- new28 decode wall: `794.097712 sec`
- notebook elapsed: `851.420677 sec`
- process-tree peak RSS: `13.971542 GiB`
- outer workers / worker threads: `4 / 1`

## Gate結果

| 項目 | 実測 | 閾値 | 判定 |
|---|---:|---:|---|
| nonzero acceleration mass | 0.664839 | 0.01–0.80 | PASS |
| future-curvature方向一致 | 0.500309 | >=0.60 | FAIL |
| 方向一致positive folds | 0/5 | >=4/5 | FAIL |
| forward-cause episode SSE改善 | 0.4355% | >=10% | FAIL |
| persistent episode SSE改善 | -3.6667% | >=5% | FAIL |
| persistent改善wells | 8/16 | >=10/16 | FAIL |
| persistent改善folds | 2/5 | >=4/5 | FAIL |
| control pooled RMSE delta | -0.162849 ft | <=+0.02 ft | PASS |
| control by-well delta p95 | +0.077808 ft | <=+0.25 ft | PASS |

方向一致のfold値は`0.498489 / 0.501190 / 0.500353 / 0.502353 /
0.499597`で、全foldがほぼ50%でした。固定persistent acceleration stateは
十分に使われている一方、未来のrate曲率方向を識別していません。

## 再現性

- scientific contract SHA:
  `f4a0bbbcc8b9cb44a55cff29e07f49ed251e11a896b3e877b4e2d6f9d08f4972`
- runtime engine contract SHA:
  `cb14a4f1dfc1d5de03a6e9329402fef476918b7ffb235552e9cd9d98d6c71451`
- executed bootstrap config SHA:
  `6c98d6c1eb45b1f8e9efbf1a770170e20918c72d90b29d8feff144a2a0007633`
- prediction decompressed SHA:
  `b1b21bb6d6007b78b49d226c09a6d3a1ec65bed7fabc6d6f65c683b08d96e960`
- acceleration posterior decompressed SHA:
  `c256258e0f89a98159a7cf830d23ccfed4f75172028a1afb0dc7ebadd4f76634`
- diagnostic logical SHA:
  `3baae5da89ac06fc7e87cd2c56a507bf1acce43a091f1d0de48e0cbca71d8cf0`
- summary SHA:
  `bbd3bb20250c4702ced2f9f8500fb31d07468bd00b029694a7d0aa610916439a`

## 考察

control wellsでは安全かつ改善しましたが、仮説対象のpersistent episodeでは
SSEが悪化しました。exp459のPF acceleration experimentでも方向一致約50%・
persistent悪化だったため、固定3状態persistent accelerationという表現自体に
negative evidenceが揃いました。span、transition、prior、engine、gateを
same-scopeで救済せず、このacceleration branchを閉じます。
