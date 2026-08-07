# exp388 結果

## 結論

`student_t_exact_hmm`は既存bankに補完性を持ち、selectorから実際に利用されたが、
fixed13 hard selectorは親fixed12より悪化した。科学gateをFAILし、
`FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`として閉じる。

downstream TVT、raw-test inference、submission、same-OOF rescueは行わない。

## 実行

- kernel: `kentookumura/exp388-exp374-student-t-fixed13-selector-train`
- version / id_no: `1 / 128464582`
- status: `KernelWorkerStatus.COMPLETE`
- runtime: `7253.168438 sec`
- active variant / objectives / outer / inner: `1 / 2 / 5 / 4`
- CPU selector models: `40 / 40`
- parent/control再学習 / GPU / downstream TVT / inference / submission:
  `0 / 0 / 0 / 0 / 0`

## Technical / selector score

- 3,783,989 rows / 773 wells、key欠損0、truth/error読込0
- exp374 source foldは存在せず、selector特徴利用0
- Stage A: 153候補特徴から90特徴、compact 77特徴
- Stage C: 40 models / 25 partitions / 18,919,945 compact rows /
  49,191,857 outer-valid candidate-score rows
- leakage audit: PASS
- expected-error MAE: `5.844225 -> 3.838680`
- within10 logloss: `0.509949 -> 0.358495`
- within10 Brier: `0.164938 -> 0.111596`
- 3 score指標すべてpooled・5/5 folds改善、score guard PASS

## Hard selector

| 指標 | fixed13 | 親fixed12 | 差 |
|---|---:|---:|---:|
| pooled RMSE | 8.736104 | 8.652532 | +0.083572 |
| near 0--250 | 1.672694 | 1.663645 | +0.009049 |
| 1000+ | 9.593523 | 9.503799 | +0.089724 |
| hidden-like spatial | 9.624681 | 9.536496 | +0.088184 |
| hidden-like typewell-purged | 9.503317 | 9.412065 | +0.091252 |

- fold差: `-0.021005 / +0.200900 / +0.304599 / -0.127122 / +0.048978 ft`
- 改善fold: `2 / 5`
- Student-t top1: `692,647 rows / 18.304678%`
- Student-t利用fold: `5 / 5`
- improved / regressed wells: `366 / 407`
- by-well delta median / p95: `+0.006058 / +0.910123 ft`
- worst well `d2f3b1ab`: `+6.708956 ft`

利用率gateとnear gateだけはPASSしたが、pooled、fold数、1000+、hidden-like、
by-well p95、worst-wellをFAILした。

## 非gating oracle診断

- H512 oracle: `3.700320 -> 3.603021`、`0.097299 ft`改善
- whole-well oracle: `4.801786 -> 4.728379`、`0.073408 ft`改善
- Student-t strict unique-best: H512 `1127/7787`、whole-well `112/773`

候補自体の局所補完性は確認できる。一方、現行dual selectorはStudent-tを
18.3%まで選びながらtailを抑えられず、候補追加をdeployable gainへ変換できなかった。

## 再現性

- exp374 decompressed SHA:
  `668fe87da902955acee742c72d30724abb53f32050bb5d0a5c1b3dee0cbd626e`
- feature schema SHA:
  `66568a948768a8dd4953a404b6b88f8e7f58c5ecf6f5afe22ca8bcd0a4b881fe`
- model manifest SHA:
  `9acdc4a165b69f737ee4807e953653070b0f3db0647514554744b7790d573044`
- outer-valid score SHA:
  `7ad8419f299419824447ad2b500ebdb353af4a39773ef5aefc702692ef36ecd8`
- summary SHA:
  `bc5ce77913862c963d6a65ff4491a3f193405b8e6fdf83c9d648615b16b74b99`

## 次アクション

fixed13 hard selector枝は閉じる。同じOOF上のweight、usage threshold、
candidate exclusion、gate緩和は行わない。

再訪するなら、Student-t TVTをhard candidateとして選ぶのではなく、
Gaussian--Student-t disagreement、posterior std、log-likelihoodだけを
target-free continuous risk featureとして downstream MLへadd-onlyする別仮説に限定する。
exp371/373/375/388のfixed13 selector失敗を踏まえ、優先度は低・P4とする。
