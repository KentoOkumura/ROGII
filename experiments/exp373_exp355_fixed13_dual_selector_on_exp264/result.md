# exp373_exp355_fixed13_dual_selector_on_exp264 結果

## 仮説

exp355 direct HMMを既存HMMの置換ではなく13番目の候補として追加すれば、
dual selectorが平均signalを有効区間だけに利用できる。

## 実行

- Kaggle kernel:
  `kentookumura/exp373-exp355-fixed13-selector-train`
- version / id_no: `1 / 128435229`
- status: `KernelWorkerStatus.COMPLETE`
- runtime: `6350.504 sec`
- route: `ensemble`
- 学習量:
  `1 variant / 2 objectives / outer 5 × inner 4 = 40 CPU boosters`
- control再学習 / GPU / downstream TVT / inference / submission:
  `0 / 0 / 0 / 0 / 0`

## 結果

| メトリック | exp373 fixed13 | 親exp264 fixed12 | 差 |
| --- | ---: | ---: | ---: |
| pooled hard OOF RMSE | 8.695438 | 8.652532 | +0.042906 |
| near 0--250 ft | 1.685224 | 1.663645 | +0.021579 |
| distance 1000+ ft | 9.547140 | 9.503799 | +0.043341 |
| hidden-like spatial | 9.398972 | 9.536496 | -0.137525 |
| hidden-like typewell-purged | 9.286938 | 9.412065 | -0.125127 |
| fixed fallback pooled RMSE | 8.238332 | - | - |

fold別ではfold 2が`-0.124587 ft`、fold 4が`-0.019557 ft`改善し、
fold 0 / 1 / 3はそれぞれ`+0.281268 / +0.020847 / +0.044885 ft`
悪化した。親改善は`2/5 folds`だった。

## selector・安全性

- exp355 top-1 usage:
  pooled `12.3192%`、全`5/5 folds`で正
- selector score guard:
  expected-error MAE、within10 logloss、within10 Brierのpooledと全foldが
  priorより改善しPASS
- leakage audit:
  `40 models / 25 partitions / 18,919,945 compact rows /
  49,191,857 outer-valid candidate-score rows`を満たしPASS
- fixed fallback error parity:
  最大絶対差`0.0 ft`
- by-well p95 delta:
  `+1.008261 ft`
- worst well:
  `b19b0395`、親RMSE `8.776489 → 37.839076`、
  delta `+29.062587 ft`

scientific integration gateは、selector score、exp355 usage、hidden-like 2面だけを
PASSし、pooled非劣化、4/5 fold改善、near、1000+、by-well p95、worst-wellを
FAILした。最終判断は`FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`とする。

## 技術契約

- exp355 OOF:
  `3,783,989 rows / 773 wells`
- 読み込みallowlist:
  `well_id,row_idx,fold,candidate_tvt`
- truth/error列の事前読み込み:
  `0`
- global key join後のmissing key:
  `0`
- source fold:
  provenance-only、model feature利用なし
- Stage A:
  65万行監査、153特徴から90特徴をfreeze、compact 77特徴
- feature schema SHA:
  `908043637ab6af5033d6ae95be0c4505f9e68a7de9f07d857a94e2665a477b8d`

## 再現性

- exp355 raw SHA:
  `28da6ffb17300f7757d51496f2dc56402d477fc5a79e24dec7514e855c960a41`
- exp355 decompressed SHA:
  `3c49f25e138f94c9e09fb551f199fa4f92b0d776899485e67e61e2fcdb83ede3`
- exp355 upstream logical prediction SHA:
  `634303f022bced6685367094304da6182fee42815302344469b5919a36cd5e21`
- selector model manifest SHA:
  `45876171d9d8de5697146abde3120c184466cb36fc8b9bcaa15b22e1f8bf8dce`
- compact manifest SHA:
  `877a731456f2b93ce59c15b39baf32c850a10f565634754eb8d2b3ebf5710a65`
- outer-valid candidate score SHA:
  `694a1800238b80333ea36ae9d8d098e5de28c0a5e0c63b08547ba44c57e8aeb6`
- 小型gate成果物4件はKaggle summary記録SHAとローカル取得SHAが一致した。
- rerun:
  未実施。version 1を正の科学runとする。

## 解釈

exp355候補には実際にselectorが反応し、hidden-like 2面では親より改善した。
しかしpooled・near・1000+では小幅悪化し、特に少数wellの大幅回帰を抑えられなかった。
候補の利用率が十分あることや内部selector指標の改善だけでは、TVTのwell-level safetyを
保証しない。

同一OOFでのweight、threshold、domain、gate救済は行わない。exp355固定13枝は閉じ、
独立候補であるexp375の結果は別仮説として扱う。

## 結論

scientific gate FAILのため、downstream TVT、current-test inference、
submissionへ進めない。Public / Private LBは対象外。
