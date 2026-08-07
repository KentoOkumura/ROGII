# exp333 結果

## 2026-07-23 current-test候補生成

exp361のadd-one novelty PASSを根拠に、同じexp333内で保存済みStage 1
outer-fold 5 modelをcurrent testへ適用した。Kaggle private CPU
`kentookumura/exp333-k16-segment-residual-candidate-inference` version 2
（`id_no=128368525`）は`65.258 sec`でCOMPLETEし、14,151 rows / 3 wells /
48 K16 segmentsのcandidate artifactを生成した。

- raw-test replayはexp072 v2と同じstable per-well seedで再生成し、205返却列から
  exp072正規allowlistの196列だけを使用した。
- trainと同一の129 row features / 136 model features、feature schema、
  5 model SHA、manifest SHA、saved train summary / OOF SHAを照合した。
- exp226 v1 base、K16境界、ID順、finite、5-fold ensemble、全row/well coverageをPASSした。
- segment offsetは`-4.249479～+2.592369 ft`、row平均`+0.289689 ft`。
- candidate file SHA / decompressed SHAは
  `d7a6bae97b7ea81aaa41f7b7850a1d56286ccb57db354ed62893c28d74ef49c1` /
  `7571c6281bd2ab484e7bf536a876b8072407b272a0ef0ec5112ca06897a717cd`。
- 学習model/booster、parent/control再学習は0。selector、blend、fixed12平均、
  `submission.csv`、competition submitも0。

version 1はraw replayの205列をexp072 train cacheの196列と直接比較して停止した。
科学計算・予測前のcontract errorであり、version 2ではexp072自身の
`feature_columns_for_variant`を使う最小修正だけを行った。

## 2026-07-23 downstream候補パス再評価

`exp361_exp333_candidate_path_addone_novelty_audit`で、exp333 Stage 1 OOFを
exp293 fixed12へadd-oneしたnovelty監査はPASSした。H512 oracle改善
`+0.133103876 ft`、whole-well改善`+0.102132339 ft`、H512 strict unique-best
`11.5064%`、5/5 folds改善だった。

以下の元direct gate FAILとbranch closeは履歴として維持する。一方、
exp228/exp263の単体置換ではなくcandidate pathとしてはcurrent-test生成を行う価値が
支持された。inference実装は別承認で同じexp333内へ追加し、submissionや固定bankへの
組み込みへ自動移行しない。

## 状態

Stage 0とStage 1 preflightはPASS。full Stage 1 Kaggle CPU v1は完走したが、
固定pooled・near 0--250・worst-well gateをFAILしたためdirect branchを閉じた。
exp361の別根拠・別承認によりcurrent-test candidate artifactだけは生成済み。
単独推論、selector/blend組み込み、提出は未実施。

## 固定した仮説

exp226 residualをK16 segment内で平均したoffset targetは、row-wise residualより低分散であり、target-freeな区間集約featureから予測しやすい。

## 親からの変更

親exp226のK16 predictionをbaseに固定し、exp228のrow-wise residual学習単位をK16 segmentへ変更する。targetはsegment mean residual、featureはtarget-free row featureのsegment finite mean、出力はsegment一定offsetだけとする。

## 固定した実験

- Stage 0: 保存済みexp226 OOFのK16 oracle mean-offset headroom、0 model/0 booster。
- Stage 1: strict nested exp226 target、K16 segment finite-mean feature、exp228 `lgb1` 1 config、5 CPU boosters。
- 補正: predicted offsetをsegment内へconstant broadcast。slope/clip/shrink/smoothingなし。
- control: exp226 `9.427109597`、exp228 `8.944085501`、推論候補追加基準exp263 `8.238331715`。

## 結果

| メトリック | 値 |
| --- | --- |
| Stage 0 decision | `PASS_STAGE0` |
| exp226 RMSE | `9.427109597` |
| K16 oracle mean-offset RMSE | `1.130602526` |
| exp226比改善 | `8.296507070 ft` |
| fold改善 | `8.359821 / 8.002651 / 9.211417 / 8.005800 / 7.879885 ft`、5/5 PASS |
| rows / wells / segments | `3,783,989 / 773 / 12,368` |
| Stage 1 preflight | `COMPLETE`、32 wells / 166,533 feature rows / 25 fits / 160 prediction well-runs |
| 親OOF parity最大差 | `1.818989404e-12 ft`、上限`1e-8 ft`をPASS |
| preflight実測 | `491.884531 sec`、model / booster `0 / 0` |
| full Stage 1実行時間外挿 | `6,434.436920 sec = 1.787344 h`、上限`8.5 h`をPASS |
| Stage 1 CV | `9.076676661` |
| exp226比 | `-0.350432936 ft`（改善） |
| exp228比 | `+0.132591160 ft`（悪化） |
| fold改善 vs exp226 | `0.554733 / 0.617955 / 0.338390 / 0.086735 / 0.163200 ft`、5/5 PASS |
| near 0--250 delta | `+0.057439 ft`、FAIL |
| 1000+ / hidden 2面 / boundary | `-0.380695 / -0.367928 / -0.355678 / -0.329524 ft`、PASS |
| by-well p95 / worst delta | `-0.352018 ft` PASS / `+8.099023 ft` FAIL |
| Stage 1 decision | `FAIL_CLOSE_BRANCH` |
| Candidate inference | version 2 `COMPLETE`、`14,151 rows / 3 wells / 48 segments` |
| Candidate feature / saved models | `129 row / 136 model / 5 models` |
| Candidate offset min / max / mean | `-4.249479 / +2.592369 / +0.289689 ft` |
| Candidate technical guards | 全PASS |
| Candidate training / submission | `0 / 0` |
| Public LB | - |
| Private LB | - |

## 再現性

- deterministic anchor: direct replacementとしてはいいえ。current-test candidate
  artifact version 2はsource/model/feature/prediction SHA付きanchorとして固定する。
- seed policy: saved outer fold + stable SHA256 inner fold + LightGBM random state 0。
- Kernel: `kentookumura/exp333-k16-segment-residual-stage0-train` version 1、`id_no=128109500`、COMPLETE。
- Stage 1 preflight Kernel: `kentookumura/exp333-k16-segment-residual-stage1-preflight` version 1、`id_no=128114252`、COMPLETE。
- Stage 1 train Kernel: `kentookumura/exp333-k16-segment-residual-stage1-train` version 1、`id_no=128116592`、COMPLETE。
- Candidate inference Kernel: `kentookumura/exp333-k16-segment-residual-candidate-inference` version 2、`id_no=128368525`、COMPLETE。
- Stage 1 train runtime: summary`1,781.997 sec`、final log`1,790.189 sec`。
- 実行時間: readout完了`104.649 sec`、final log`114.707 sec`。
- input decompressed SHA: `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`。
- segment assignment SHA: `6b833c0bcdbe2b82b2e16df23f5fd0dae1412a890a90cf53743610fbb01e07e3`。
- segment target SHA: `15f47cbb46b655c50b57718d6549d79d5e15101aa7ffde74022c069f1c1f6dfb`。
- oracle readout SHA: `616157b44fc769b111071771f959b37e3ab3f958b7657ee25b20252cf4753d38`。
- oracle offset、segment target、oracle predictionはdeployable生成物として保存していない。
- 実装検証: 専用pytest `10 passed`、py_compile / Ruff / Jupytext変換PASS。保存済みexp226 OOFのdecompressed SHAは固定値`709eb726...e4c609`と一致。
- Stage 1 compact self-contained trainを正規train Notebookへ採用。candidate-only
  compact self-contained inferenceも別承認後に正規inference Notebookへ採用し、
  実行完了後はauthorization consumedでfail-closedへ戻した。
- Stage 1は別名Jupytext source / Notebookへ実装し、strict nested exp226、許可済み3 feature群、K16集約、固定LightGBM 5 boosters、全固定gateとSHA artifactを含む。Stage 0正規Notebookは上書きしていない。
- Stage 1専用を含むpytestは`14 passed`。Stage 1 Kaggle preflight/full runは各1件、boosterは固定5本。
- feature freeze / model manifest / OOF prediction / segment target SHAは`b2c7bff4...a078 / 3e4c99b2...97a9 / dbb3f416...0784 / 65a73d74...0027`。
- current row feature / segment feature / segment prediction / candidate prediction
  content SHAは`94757211...2a79 / c6cbe6de...786c / fc54e72e...3923 /
  316e3b77...5d02`。
- 既知のexp296 testを除くrepository regressionは`494 passed, 2 skipped`。strict experiment / project validation、bootstrap依存解決もPASS。

## 解釈

target-freeなsegment offset学習はexp226を全5 folds・pooledで改善し、特に250行以降、hidden-like 2面、segment境界、by-well p95では有効だった。しかし改善量はexp228に届かず、near 0--250を悪化させ、特定wellでは最大`+8.099 ft`悪化した。大きなoracle headroomの多くは現在のtarget-free segment mean featureでは回収できず、constant offsetの過補正リスクも残る。固定gateに従い追加configやsame-OOF clip/shrink救済は行わない。

## 次

exp333のdirect branchは閉じたまま、current-test candidate生成は完了した。次は
「fixed bankへ13番目としてselectorを再学習する」か「target-free safety gateで
exp333だけを限定追加する」かを別設計として選ぶ必要がある。どちらも今回の承認には
含めず、候補重み、平均blend、selector変更、submissionへ自動移行しない。
