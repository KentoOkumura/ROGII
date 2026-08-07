# exp485_stratified_initial_rate_bank_pf 結果

## 仮説

equal-strata initial-rate bankがPF初期mode lossを減らす。

## 設定

- centers: tail30 / 32 / 64 / 128 / 256
- particles per center: 100
- total particles / seeds / temperature: `500 / 128 / 5.0`

## 変更点

exp404のtail30単一centerだけを5 center equal-strataへ置換した。以後の
dynamics、Gaussian GR emission、resampling、roughening、seed aggregationは固定した。

## 結果

Kaggle private CPU version 1は`COMPLETE`。fixed32 32 wells / 156,088 rows、
4,096 seed-well、2,048,000 particle startsを完走した。

14 checks中13はPASSしたが、full runtime projectionが
`30,894.444062 sec`となり、固定上限`30,600 sec`を`294.444062 sec`
（`0.962%`）超過した。`runtime_projection_within_limit`だけがFAILし、
元のStage 0判定は`stage0_fail_closed`のまま保持する。

その後、ユーザーがこの程度の実行時間を許容し、Stage 1を明示承認した。
これは元のgate結果の改変ではなく、記録付きのruntime例外である。

## Stage 1結果

canonical Kaggle kernel version 2で全773 wells、3,783,989 rows、
98,944 seed-well、49,472,000 particle startsのtarget-free候補を生成・freezeした。
version 2はその後のtruth-late readout中、保存exp209 HMMのgzip-content
整合性確認で停止した。凍結成果物のSHA、行数、well数、実行量、
freeze前read 0を確認してprivate Datasetへ固定し、candidate PFを再実行せず
version 3で同じ評価だけを再開した。

version 3は`COMPLETE`。technical gateは19/19 PASSしたが、scientific gateは
FAILした。

- candidate RMSE: `11.092618091`
- saved exp404 control RMSE: `10.914522073`
- improvement: `-0.178096018 ft`
- improved folds: `1/5`（必要`4/5`）
- by-well delta p95: `+0.422388632 ft`（上限`0.0`）
- worst-well regression: `+33.053515117 ft`（上限`0.25`）
- fixed exp209 HMM + PF 50:50:
  `10.117590985` vs control `10.084909849`、`+0.032681136 ft`悪化

scope別ではhigh-missing wellsだけ`+0.018240364 ft`改善した。一方、
raw-GR observed `-0.250495864 ft`、raw-GR missing `-0.020821867 ft`、
MD-since 1000+ `-0.207084840 ft`、hidden-like spatial
`-0.108049509 ft`、hidden-like typewell-purged `-0.109059150 ft`で悪化した。

## 生成物

- 正規train Notebookとcompact Jupytext source
- 専用契約test
- run-on-pushで実行済みのKaggle train package versions 1–3
- version 2 target-free凍結成果物のprivate Dataset
- version 3のprimary/by-well/fixed-blend metrics、promotion gate、runtime ledger

inferenceとsubmissionは実行していない。

## 再現性

- deterministic anchor: no
- component assignment: particle index modulo 5
- seed: exp404 stable per-well SHA256
- scientific/input/rate-bank/component/prediction content SHA: Stage 0実行時にfreeze
- Stage 0 truth access: 0行
- scientific contract SHA:
  `c3cc258bafb9489d4ce02f06e9cc4a63f00805230a86982c1ae8140cae8ee86e`
- prediction decompressed SHA:
  `39941f00e25927611e84c10af0057f85cf81813020f45ed5351138e74e464c9c`
- rate-bank decompressed SHA:
  `369cf494018ec648a9bed8cabff21f2870d3bd3d3b3254164ac656acfafc9903`
- component-ledger decompressed SHA:
  `e9cca2189cf8513ba5b1cb116572f36d6ed94a1c852cc4de0ea89f791ea23538`
- Stage 1 scientific contract SHA:
  `599d39931c9e5f820469b531d1ed64b383f381918cdb0055700aa0beb7dd4233`
- Stage 1 prediction logical / decompressed SHA:
  `246e7473289bc19743fc3957b319b95a8b72543fb1f5748e0f34f390e980ea46` /
  `7cb11d339d92ca0ae3fef2de243e9754b83f2ec74211707b898f7ff7a2e77750`
- freeze前truth/control/fold/hidden-like read: すべて0
- aggregate Stage 1 runtime / peak RSS:
  `9,935.484823 sec` / `3.470413 GiB`

## 解釈

5×100 allocation、interleave、rate/fallback、duplicate保持、finite coverage、
posterior normalization、count conservation、stable seed、exp404 parity、
truth read 0、RSSはPASSした。rate bankは25/32 wellsで複数center、7/32で
単一centerとなりglobal degeneracyは回避したが、component extinction seed
fraction maxは`0.921875`だった。

実装とmechanismは成立し、actual aggregate runtimeも元の30,600秒上限内だった。
しかしequal-strataで複数の初期rate modeへ固定配分すると、high-missing wellsの
小さな改善と引き換えに、GR observed、long-tail、hidden-like、fold 4で大きく
悪化した。alternative centerへ常時100粒子ずつ割くため、観測が十分なwellでも
親tail30 modeの有効粒子数を減らしたことが主な失敗原因と解釈する。

## 次

事前登録どおりbranchを`stage1_gate_failed_terminal_close`で閉じる。
window、allocation、spread、particle/seed、temperature、gate、blend、
selectorによる同一OOF救済は行わない。inferenceとsubmissionも行わない。
次はこのinitial-rate bankを派生させず、既存の独立した高優先度候補を進める。
