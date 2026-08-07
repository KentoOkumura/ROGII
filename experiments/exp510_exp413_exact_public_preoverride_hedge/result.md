# exp510_exp413_exact_public_preoverride_hedge 結果

## 状態

hidden-safe修正版Kaggle version 4は14,151行・3 wellを`385.108 sec`で完走し、
`technical_gate_pass`。公開test固定exp413 sidecarを廃止し、dynamic sample上でexp413を再生成した。
submit-check FAIL/WARN 0を確認してref `55231514`として提出し、627分後にCOMPLETE、Public LB
`7.201`。Kaggle UIは`Your latest submission scored 7.201, matching your best.`と表示した。過去のversion 2
submission ref `55225634`はhidden rerun失敗の履歴として維持する。公開source titleや上流LBは
exp510の結果ではない。

## 実装結果

- archived source SHA `4d071298...623eb`とpre-override cell境界を固定した。
- 候補sourceはprojected-SP45、保存Pipeline-B 3 booster推論、`0.55/0.45` public component、
  exp413との`0.90/0.10` blend、truth-free readout、再現性manifestを一続きで実装した。
- Pipeline-A tabular stackはprojected-SP45欠損時のfallbackにしか使われないため削除し、fallback
  rowsが1件でもあれば停止する契約へ置換した。
- Pipeline-B dataset version 1の4ファイルを取得してSHA固定した。artifact欠落時の再学習経路はない。
- PFはwell/family/seed index由来のstable seedへ移植した。同一well IDはtest側空間priorからも除外する。
- 正規inference notebookは上書きせず、Jupytext候補`.py/.ipynb`を別名で作成した。
- exp413 v4 source SHA `0f6fc81e...f1388`をguardしたhidden-safe runtimeを同梱し、11 upstream
  kernel sourceと保存75 boosterからdynamic sample上のexp413成分を再生成する。
- dynamic exp413が書いたCSVを即時に読み戻してfixed blendへ渡し、従来のexact component boundaryを
  維持する。static public-test sidecarはruntime inputに含めない。

## 静的検証

| 項目 | 結果 |
| --- | --- |
| dedicated tests | 14 passed |
| source SHA / boundary | PASS |
| 禁止route AST scan | PASS |
| fixed formula / ID / finite / duplicate guard | PASS |
| model・exp413 parent source / dynamic artifact SHA fail-close | PASS |
| stable per-well seed | PASS |
| repository tests（既知のexp293/exp296 contract failure除外） | 1821 passed, 8 skipped |
| 既知の別実験failure | exp293 2件、exp296 2件 |
| CV / Public LB / Private LB | なし / 7.201 / なし |
| Kaggle visible technical gate | version 4 PASS、14,151 rows / 3 wells、385.11秒 |
| Kaggle code submission | version 4 ref 55231514 COMPLETE、627分、Public LB 7.201。過去ref 55225634はhidden rerun FAIL |

## Kaggle実行結果

- kernel: `kentookumura/exp510-exp413-exact-public-preoverride-inference` version 4。
- version 1はJupytext code-cell marker欠落により`run_particle_filter`未定義で失敗し、marker 1行と
  回帰testを追加してversion 2を再実行した。科学条件は変更していない。
- version 3はhidden-safe動的生成を完走したが、in-memory float32を直接blendしてCSV boundaryとの
  最大差`4.36e-4`を検出した。version 4で生成CSVを読み戻す契約へ修正した。
- version 4はfallback / duplicate / nonfinite `0 / 0 / 0`、public/final formula parity `0.0`。
- exp413 dynamic artifactのdecompressed SHAは公開reference `875a1334...dc4`と一致し、
  serialization roundtrip最大差`4.84375e-4`を記録した。
- final prediction content SHAは`ea61118d...e9d1`、Kaggle生成`submission.csv` SHAは
  `7209a4bd...4e52`でversion 2と完全一致。output archive取得後のsubmission形式検証もPASSした。
- parent/publicを含む新規学習、GPU、外部提出はいずれも0。保存boosterは75 + 3 = 78本を読んだ。

## scoring結果と再監査

- ref `55231514`は`2026-08-04 16:48:01 UTC`にCOMPLETEを観測した。提出から627分、
  CLI/APIの`publicScore`は`7.201`。
- exp413 ref `55080377`も公開値は`7.201`。Kaggle UIの`matching your best`は公開された精度での
  同値を示すが、CLI/APIはいずれも3桁しか返さないため、full-precision RMSEの完全一致は確認できない。
- 提出済みkernelをpullし、notebook内payloadのcompact source SHAとhidden-safe runtime SHAが
  ローカルversion 4と一致することを再確認した。notebookは26 cellsで、最終cellが同期的に
  `run_inference()`を呼び、後続cellはない。
- hidden-safe parentは途中でexp413単独の`submission.csv`を書くが、関数return後にexp510が
  `0.90 * exp413 + 0.10 * public_preoverride`を計算し、blend済み`final`で同じpathを上書きする。
  subprocess、background process、非同期write、例外時のexp413 fallbackはない。
- visible current-testではfinalとexp413の差分がMAE `555.138 ft`、RMSE `912.374 ft`、最大
  `3125.436 ft`で、14,151行中8,564行が100 ft超、3,020行が1,000 ft超変化した。これはhidden
  predictionの証拠ではないが、blend式が死んでいないことを示す回帰証拠である。
- 以上から技術実装はPASSを維持する。一方、Public LBではexp413比の測定可能な改善がなく、
  public hedge仮説は支持されない。honest OOFもないためexp413をanchorとして維持する。

## hidden rerun失敗

- raw API: scriptVersionId `340025138`、status `COMPLETE`、score空欄。
- Kaggleの`errorDescription`はhidden datasetでの未処理例外という汎用文言で、tracebackは非公開。
- 高確度の静的原因は、exp413成分だけを公開test固定の
  `exp413_current_test_predictions.csv.gz`から読むこと。hidden sampleのID集合と一致せず、
  `load_exp413_component()`の完全一致guardで停止する。
- 修正にはexp413 hidden-safe inferenceをexp510内で動的再生成する必要がある。単にexp413 kernel
  outputをmountし直すだけではhidden rerun時に上流notebookは再実行されない。
- version 4で上記修正を実装し、visible current-test上のexact output parityまで確認した。
  hidden code rerun自体は再提出していないため未検証である。

## 解釈

version 4はdynamic sample契約、公開current-test exact parity、hidden code rerun完走を満たした。
ただし対応するhonest OOFとPrivate安全性はなく、Public LBもexp413と公開3桁で同値だったため、
科学的promotionやdeterministic anchorとは判断しない。

## 次

scoring監視は完了した。正規inference notebookへの採用は別承認とし、追加提出やweight変更は行わない。
