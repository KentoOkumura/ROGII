# exp253 セッションノート

## 目的

対象公開notebookから、prefixによる既存候補評価とbounded correctionだけを導入する。
Beam/PF本体、候補family、contact/heel/affine/bimodal、親モデルは変更しない。

## 現在の状態

- Route: `ensemble`
- 状態: Stage 1 aggregate完了・性能guard不通過・不採用
- active mode: `stage1_full_audit`
- CV / LB: なし
- inference / submission: Stage 1性能guard不通過のため禁止

## 実装契約

- prefix cuts: `0.50 / 0.65 / 0.75`
- candidate score: `median(cut RMSE) + 0.10 * std(cut RMSE)`
- default candidate: `likpf_mean`
- balanced gate: min gain 0.55、max best score 12、min consistency 0.50
- alpha: base 0.08、gain scale 0.20、margin scale 0.06、quality bonus 0.04、cap 0.40
- move: exponential fade-in、clip base 10、gain 4.5、最大30 ft
- delta guard: soft 30 ft、p95 hard 75 ft
- candidate: exp072既存9 familyのみ
- base: exp238保存済み`lgb_mean` OOF / current-test submission

## Kaggle実行コスト

- active variant: 1 (`balanced_prefix_verified_controller`)
- model / LightGBM config: 0
- training folds: 0
- boosters: 0
- parent/control再学習: なし
- Stage 0: 32 wells x 3 cuts、CPU、single process、internet off
- Stage 1: 773 wells x 3 cuts。Stage 0実測の単純外挿は約80,041秒（22.2時間）のため、stable SHA256 well modulo 4で各約5.6時間へ分割する。
- Stage 1は1 scientific variant / 4 execution shards / config 0 / fold 0 / booster 0。各shard内はCPU `n_jobs=1`、parent/control再学習なし。
- 2026-07-15: ユーザー判断によりworst-well回帰と補正well勝率を拒否条件から外し、監視指標として全well Stage 1を実行する。
- shard kernel: `kentookumura/exp253-prefix-bounded-controller-s0` ～ `s3`。
- aggregate kernel: `kentookumura/exp253-prefix-bounded-controller-aggregate`。4 shard完了後だけpushする。
- 2026-07-15: ユーザーからStage 0実行の明示承認を受領した。
- push対象: `kentookumura/exp253-prefix-bounded-controller-train` のみ。inferenceは対象外。
- push command: `make push-kaggle-train EXP=exp253_prefix_verified_bounded_candidate_controller`
- kernel: `kentookumura/exp253-prefix-bounded-controller-train` version 1
- URL: https://www.kaggle.com/code/kentookumura/exp253-prefix-bounded-controller-train
- push前の同slug pullは403。slugを変更せずcanonical IDへpushし、version 1作成を確認した。
- push後pullは成功し、Kaggle id_no `127306443`、CPU、internet off、exp072/exp238 kernel sourceを確認した。
- 監視中の最終確認では`KernelWorkerStatus.RUNNING`。CLI途中ログは空だった。
- 2026-07-15: ユーザー指示によりこちらの定期監視を停止。Kaggle kernel自体は停止していない。完了連絡後に同じversion 1のlogs/必要生成物を監査する。

### Stage 0 version 1失敗

- 最終状態: `KernelWorkerStatus.ERROR`
- runtime: 3,270秒。prefix replayは96/96 request、error 0で完走した。
- 最初の意味のあるtraceback: official candidate cache読込時の
  `missing columns: ['likpf_mean']`。
- 原因分類: data/cache schema contract。exp072 v2 cacheは絶対値`likpf_mean`ではなく、
  `likpf_mean_d`だけを保存していた。
- exp072 v2 feature schemaをKaggleから個別取得して確認した。既存候補は
  `likpf_mean = last_known_tvt + likpf_mean_d`で厳密に復元できる。
- 修正: train/inferenceのcandidate materializationを絶対値/差分値の両schema対応にし、
  重いprefix replayより前にcandidate cacheとexp238 OOFの必須列をfail-fast検証する。
- selector式、候補集合、cut、gate、alpha/move cap、seed、32-well対象は変更していない。

### Stage 0 version 2

- 同じcanonical kernel `kentookumura/exp253-prefix-bounded-controller-train`へpush成功。
- version: 2
- 変更は`likpf_mean_d`絶対値復元とpre-replay schema validationだけ。
- 実行契約: 32 wells x 3 cuts、1 variant、0 config、0 fold、0 booster、parent/control再学習なし。
- 最終状態: `KernelWorkerStatus.COMPLETE`
- runtime: 3,313.476秒
- 96 requests、error 0、scored wells 100%、全well 3 cuts、全request 9 candidates。
- Stage 0 technical checks: 7/7 pass。
- overall: exp238 base 6.769362623 -> controller 6.570761195（-0.198601427）。
- 1000_plus: 7.461871699 -> 7.235082448（-0.226789251）。
- hidden-like spatial / typewell-purged: いずれも -0.002448163。
- folds: 4/5改善。fold 3のみ +0.118235538。
- 補正適用9 wellsのうち改善6、悪化3、残り23はbase維持。
- alpha最大0.311241487、絶対move最大10.939005666 ftでcap内。
- worst well `052d64df`: 1.878220647 -> 3.003020246（+1.124799599）。
  prefixでは`tvt_dense`がdefaultよりgain 1.0320449、consistency 1.0だったがofficial tailで逆転した。
- 当初のadoption guardはworst-wellだけfailし、全体passはfalseだった。Stage 1ではユーザー判断によりworst-well判定値を保存しつつvetoから外す。
- Stage 1でも同じraw wells、imputer、request ID、per-request stable seed、candidate cacheを使うため、
  `052d64df`の+1.1248 ft回帰は残る見込みだが、Stage 1ではmonitor-onlyとして全体RMSEと残りのsurface guardを評価する。
- request manifest / cut scores / well reportはsummary記録SHAとローカル再計算SHAが一致した。
- rerun content SHA比較は行っていないためdeterministic anchorとは呼ばない。

## 変更点

- compact self-contained Jupytext train notebookを新規実装した。
- synthetic cut後の`TVT_input`と`TVT`をmaskしてからexp072候補を再生する。
- exp072のsource-well exclusionをsynthetic request wellから元wellへ変換する。
- candidate score、well report、move report、OOF、distance/hidden-like/fold/by-well metrics、SHAを保存する。
- inference notebookはStage 1採用guardをfail-fastし、current testでも同じmasked-prefix scoreとbounded moveを再生する。
- Stage 1 shard notebookは正規train notebookの科学コードを機械的に複製し、shard indexの環境変数だけを追加する。
- 全773 wellsは各shardのimputerへ渡し、評価wellだけをstable SHA256で分割する。単一Stage 1からper-well seed、候補、予測を変えない。
- aggregate notebookは4 shardのwell集合・入力SHA・config SHA・OOF IDを検証し、row-level OOFを結合してglobal RMSEを再計算する。shard RMSEの平均は使わない。

## 再現性メモ

- `docs/06_reproducibility.md`確認済み。
- seed policy: exp072 stable seedをsource well / cut fraction / familyから生成。
- stochastic components: PF ANCC、PF Z、likelihood-PF。
- parallel policy: `n_jobs=1`。thread scheduling依存のglobal Numba RNGを避ける。
- CPU/GPU: CPUのみ、GPU 0。
- gzipはdecompressed content SHAを主証拠にする。
- rerun SHA未取得のためdeterministic anchorではない。

## 静的検証ログ

- `py_compile`: train / inference pass。
- Ruff `F821`: train / inference pass。
- Jupytext変換: train / inference完了。
- Jupytext round-trip test: train / inference pass。
- `make validate-exp EXP=exp253_prefix_verified_bounded_candidate_controller`: pass。
- `make validate-template`: pass。
- canonical Kaggle train / inference package: prepare完了。
- package metadata: CPU、internet off、canonical ID/title、必要なkernel sourceを確認済み。
- 公開notebookとの式再照合で、scoreの標準偏差を母標準偏差へ、consistencyを各cutの局所best勝率へ修正した。
- canonical packageは短縮slug/titleと`run_on_push=true`を保つため、README記載のtrain/inference別prepareを正とする。

### package preflight修正

- 初回prepareでは補助sourceのbootstrap markerは存在したが、kernel sourceがmetadata上空だった。
- 原因はnested `train.kernel_sources` / `inference.kernel_sources`を使ったこと。
- template規約の`runtime.kaggle.train_kernel_sources` / `runtime.kaggle.inference_kernel_sources`へ修正し、同じcanonical packageを再prepareした。
- train packageはexp072 candidate cacheとexp238 OOF、inference packageはexp253 train outputとexp238 inference outputを参照する。

### Stage 1 shard preflight

- shard well数: `s0=203 / s1=208 / s2=179 / s3=183`、合計773、重複なし。
- Stage 0実測比例のruntime見込み: `5.84 / 5.98 / 5.15 / 5.26`時間。
- scientific variant 1、execution shard 4、LightGBM config 0、fold 0、booster 0、parent/control再学習なし。
- canonical Jupytext train、4 shard、aggregateの`py_compile`、Ruff F821、round-trip testを通過した。
- strict experiment validation / template validationを再通過した。
- 4 shard metadataはCPU、internet off、run-on-push true、exp072/exp238の2 kernel sources。
- aggregate metadataはCPU、internet off、4 shard sources、run-on-push false。shard完了前には実行しない。
- source / loose package / bootstrap `config.yaml` SHAは5 packageですべて一致した。
- package config SHA: `c169ce3c40f7a671481e7b39b34c1b2b3028be60c43392c72c2cd1992f4405be`。
- pre-push pullは4 IDとも403で、既存kernelなしとしてcanonical IDを維持した。
- 4 shardはいずれもversion 1としてpush成功した。
- `s0`: id_no `127330449`、https://www.kaggle.com/code/kentookumura/exp253-prefix-bounded-controller-s0
- `s1`: id_no `127330452`、https://www.kaggle.com/code/kentookumura/exp253-prefix-bounded-controller-s1
- `s2`: id_no `127330448`、https://www.kaggle.com/code/kentookumura/exp253-prefix-bounded-controller-s2
- `s3`: id_no `127330451`、https://www.kaggle.com/code/kentookumura/exp253-prefix-bounded-controller-s3
- post-push pullで4 ID、CPU、internet off、exp072/exp238 kernel sourcesを確認した。
- ユーザーの既指示どおり定期監視は行わない。4本の完了連絡後にlogsを監査してaggregateをpushする。

### Stage 1 shard完了

- 2026-07-16のユーザー完了連絡後に、4 shardのversion 1 logsを取得した。
- 4 shardすべて`stage1_shard_complete`、technical checks pass、request error 0、scored well 100%、3 cuts、9 candidates。
- well / request: `s0 203/609`、`s1 208/624`、`s2 179/537`、`s3 183/549`。合計773 wells / 2,319 requests。
- runtime秒: `20,843.716 / 21,961.972 / 30,765.423 / 31,860.519`。
- shard overall delta RMSE: `+0.499754 / -0.096850 / +0.328914 / +0.356943`。shard scoreは採用判定に使わず、row-level aggregateで確定する。
- shard単独は設計どおり`partial_stage1=true`、`aggregate_required=true`、`inference_allowed=false`。

### Stage 1 aggregate完了

- canonical kernel: `kentookumura/exp253-prefix-bounded-controller-aggregate` version 1 / id_no `127430343`。
- Kaggle CPU、internet off、4 shard kernel source、runtime 73.396秒で`KernelWorkerStatus.COMPLETE`。
- 4 shardのwell集合、入力SHA、config SHA、OOF ID重複を検証し、773 wells / 3,783,989 rows / 2,319 requestsをrow-level結合した。
- technical Stage 1 checksは9/9通過。request error 0、scored well 100%、全well 3 cuts、各request 9 candidates、alpha最大0.38、move最大20.629570 ft。
- overall: exp238 base 7.936700845 -> controller 8.205455485（+0.268754640）。
- 000-050 ftは-0.002746371、050-100は-0.005645043、100-250は-0.012375419、250-500は-0.024179132とnear側は改善した。
- 500-1000は+0.004171953、1000_plusは+0.307983195、hidden-like spatialは+0.282873255、typewell-purgedは+0.267543159と悪化した。
- fold deltaは`+0.233741 / +0.195464 / +0.498533 / +0.074539 / +0.303746`で、改善0/5。
- 361 wellsへ補正が適用され、150改善 / 211悪化、412 wellsはbase維持。worstは`fcfcc902`の+10.310640703。
- worst-wellはユーザー判断どおりmonitor-onlyで拒否条件に含めていない。それでもoverall、1000_plus、hidden-like 2面、fold stabilityが必須guard不通過となった。
- 結論: `adoption_supported=false`、`inference_allowed=false`。この固定prefix評価 + bounded correctionは全well適用に採用せず、inference / submissionへ進めない。
- aggregate OOF decompressed SHA: `a66296559152eed8b3b9a753c0965ad5db2f693a28576d51b05861487dd03b22`。
- 小生成物だけを個別取得し、metrics / summary / by-wellのSHAを`metrics.json`へ記録した。大きなKaggle output archiveは取得していない。

## 次のアクション

1. exp253 branchは不採用として終了する。parameter、gate、alpha、clipの事後gridは行わない。
2. inference / submissionは行わない。
3. 新規backlogは追加しない。再訪する場合は同じbounded correctionの調整ではなく、prefix scoreがlong-tailへ転移しない原因を別仮説として切り出す。
4. 親exp238と関連backlogを確認した。`prefix_near_continuity_far_only_revisit`は異なるfeature仮説だが、exp253でfar側が悪化したため優先度を上げる根拠には使わず、既存の低優先・hard correction禁止を維持する。
