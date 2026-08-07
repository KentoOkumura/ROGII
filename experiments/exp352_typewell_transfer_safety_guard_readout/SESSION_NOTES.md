# exp352_typewell_transfer_safety_guard_readout セッションノート

## 目的

exp313の着眼点を旧branchから分離し、exp311保存済みType Well群priorに対する
target-free availability/fallback guardの診断価値だけを評価できる設計へ固定する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU version 1完了、固定gate FAIL、branch closed
- CV: Stage 0 7/8 checks PASS、総合FAIL
- LB: まだなし

## コマンドログ

- 2026-07-23: ユーザー承認により、旧exp313をreopenせず独立後継のsteeringとscaffoldを作成した。
- 2026-07-23: 設計確定までを承認範囲とし、実装・Notebook・Kaggle実行は行っていない。
- 2026-07-23: ユーザーの`exp352を実装してください`をStage 0 compact候補と
  contract testsの実装承認として受領した。正規Notebook採用とKaggle実行は別境界のまま。
- 2026-07-23: ユーザーの`実行してください`をcompact trainの正規Notebook採用と
  Kaggle CPU Stage 0 package/push/runの承認として受領した。
- canonical kernel:
  `kentookumura/exp352-typewell-transfer-guard-readout-train`。
- title: `exp352 typewell transfer guard readout train`。
- `make prepare-kaggle-notebooks ... --run-on-push --strict`: PASS。
- execution config / loose package / bootstrap manifest SHA:
  `9309d0286fa0454843c7a7a50dce274514d858a9d16f08d8d8ede5735c0c1b92`で一致。
- `make push-kaggle-train EXP=exp352_typewell_transfer_safety_guard_readout`:
  version 1 push成功。
- `kaggle kernels pull ... -m`: id_no `128360039`、CPU、internet off、
  exp311 kernel sourceを確認。
- `kaggle kernels logs ...`: 正常完了とStage 0 gateを確認。
- SHA/score実ファイル確認が必要なためKaggle outputを
  `/tmp/exp352-kaggle-output.YvJ8QF`へ取得した。repositoryには大容量生成物を保存しない。

## 変更点

- exp311/312のpromotion FAILを上書きしない。
- exp311保存値、fold、groupを固定し、peer/support/fallbackだけを変更対象とする。
- 1 diagnostic / 3 surfaces / 5 folds / model・booster・decoder・HMM各0。
- 全gate PASSでも旧exp314--320や補正を自動解禁しない。

## 再現性メモ

- seed policy: RNGなし、保存済みfold/group/well順固定。
- stochastic components: なし。
- CPU/GPU runtime: 実装時はKaggle CPU、GPU/internet off、30分上限。
- Kaggle kernel id / version:
  `kentookumura/exp352-typewell-transfer-guard-readout-train` / `1`。
- parent exp311 summary raw SHA:
  `821499b895a8bdb6ed8202c714ddb95ea8739defe9b4aaa0e13c35b346a1ca29`。
- parent pair decompressed SHA:
  `14f506da542a0d6f460425ddb56ff7119e19699b7d6110956fde86e63311e335`。
- availability manifest freeze SHA:
  `2cbc04ebc4badbdd0d4d482f6bf9447ef085b8b98ab3dd77d7095f0ed93331a3`。
- availability table content SHA:
  `6648b769c2f2eb15a0166670ca82019c3141f5c9e65582d6cfd93c4b36303be0`。
- score table content SHA:
  `b56498246fb401ca18a1d6e94b09f8dd82f047497f279504c3c834c12243c4d3`。
- model manifest / model SHA: 非該当。
- prediction SHA: 非該当。
- submission SHA: 非該当。
- rerun check: 未実行のため非該当。

## Kaggle Stage 0結果

- diagnostic runtime: `12.973409 sec`。
- availability/fallback rows: 1,746。score rows: 1,746。surface metrics rows: 18。
- exact coverage: `0.9728331177`、PASS。
- identity parity: `0.0`、PASS。
- outer-valid truth before manifest freeze: `0`、PASS。
- same-group gain: `+0.3815401372` GR API、5/5 folds、PASS。
- leave-group-out negative transfer: `-0.1648618797` GR API、PASS。
- spatial+typewell-purged negative transfer: `-0.4967524590` GR API、PASS。
- worst-well regression: `+12.9147159970` GR API、FAIL。
- worst well `d07aed8f`はexact groupを通過し、identity `5.587119`からguarded
  `18.501835`へ悪化した。
- 8 checks中7 PASS。worst-well safetyだけFAILしたため総合FAIL。

## 生成物SHA確認

- 6 manifest対象生成物のraw SHAをsummaryと照合し、全件一致した。
- availability raw:
  `b853453c769b33fbdc61f23907258381d42da4b05f54d6154c5f2cd1728d81f7`
- score raw:
  `adf380689ac1eaeb8ccd92456490467bef9e455ece8328f0d0205d0774d629f2`
- surface metrics raw:
  `ef262c200f74f6b0f92538623097f50fa6ce861fcae11e51ea374d365352b8eb`
- gate raw:
  `84b304d7d95177830d9d892d5dd593e6236bd922809fb28faf023bcf9d769972`
- summary raw:
  `d4130c7e72b8a840b76f434d9b94e28c6acb7f04c238974a7801def4045739e8`

## 実装

- compact self-contained train候補を9章で実装した。
- exp311 version 1 summary raw SHA
  `821499b895a8bdb6ed8202c714ddb95ea8739defe9b4aaa0e13c35b346a1ca29`と、
  pair table decompressed SHA
  `14f506da542a0d6f460425ddb56ff7119e19699b7d6110956fde86e63311e335`
  をhard preflightする。
- exp311 fold、native-overlap membership、group prior、固定score populationだけで
  exact availabilityとfallbackを決める。
- exact利用条件はpeer wells 2以上かつsupport 64 rows以上。利用不能時のglobalは、
  同じfold/surfaceで条件を満たす他群priorのequal-group medianとし、対象群を除外する。
  他群がなければidentity/no-correctionへ落とす。
- availability/fallback tableのcontent SHAを凍結した後だけexp311 suffix pairを読み、
  horizontal GR API単位のwell-equal RMSEを評価する。
- exp311はTVT候補、予測器、decoderを生成していない。禁止事項を維持するため、
  gate数値は変えず、設計時の`ft`表記だけを`horizontal_gr_api`へ訂正した。
- fail-closed inference候補はinference/submission関連flagが1つでも有効なら停止する。
- model、booster、decoder、HMM、prediction、submissionは実装していない。

## Notebook比較

- 親exp311 compact trainは10章 / 1,388行、exp352候補は9章 / 約1,200行。
- exp311のraw pairing/Huber再推定は保存値固定契約に反するため持ち込まず、
  runtime/path/SHA、input preflight、late-truth、metrics、gate、output orchestrationを
  notebookセルへ展開した。
- exp352候補に`__file__`参照はない。
- 実装時点では正規train/inference notebookを上書きせず、compact候補を別名で生成した。
- 実行承認後、compact trainだけを正規train notebookへ採用する。inferenceはplaceholderのまま。

## 実行コスト契約

- diagnostic variant: 1
- audit surfaces: 3
- reporting folds: 5
- model config / trained fold / booster / decoder / HMM well-run:
  `0 / 0 / 0 / 0 / 0`
- 親control再実行: 0
- Kaggle CPU、GPU/internet off、30分上限。package/push/run承認済み。

## 静的検証

- compact train/inferenceの`py_compile`、ruff
  `F821,F401,F841,E722,E501`: PASS。
- exp352 contract tests: 5 passed。
- compact train/inferenceのJupytext変換と`--test`: PASS。
- `make validate-exp EXP=exp352_typewell_transfer_safety_guard_readout`: PASS。
- 親exp311 + exp352関連テスト: 12 passed。
- repository全体: 690 passed / 5 skipped / 4 failed。4失敗は既存の
  exp296（2件）、exp335（1件）、exp343（1件）で、完了後configと旧test期待値の不一致。
  exp352専用5件はすべてPASSし、本実装はこれら既存実験を変更していない。

## 実装SHA

- config:
  `84d1297559fc9ead8c6a7281fbc93028918014768b06df75c1ef5a53f577631a`
- compact train source:
  `8058956057ce61836c2489edb7cbd74d83144f067551a4364c2f8e9af03a8784`
- compact inference source:
  `0d76bc514412a710c22506c08aa31845cae9d2154a250fb5b180347a7009702e`
- contract test:
  `fb32f2a3c51bbe9c4fedb6969d793d7607b077e979451351da3982537296e1bd`
- canonical train notebook:
  `d3d7b265e20cf42c12efc129ca6c667549a734ea9859e0678cb6c0a1910bceb6`
- compact train notebook:
  `65bb74829f435a9e0fb6dbd51f9a9d008f88e2a2b54f465f1281ba1446806aac`
- compact inference notebook:
  `a71c5053caf2391a96d147ec4138263ee822f808ddefaabe75787429a967a895`

## 次のアクション

1. threshold、fallback順、global集約を同じscoreで救済調整せずbranchを閉じる。
2. exp311/312のFAILと旧exp314--320の閉鎖を維持する。
3. 既存exp353はdirect priorから独立したsoft quality feature preflightとしてのみ残す。
