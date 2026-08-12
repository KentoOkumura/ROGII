# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- 親を exp209 に固定し、exp223 は参照実験に留めた。
- top-K=5、TVT grid-index sequence dedup、oracle block 128/256/512 を確定した。
- route、leakage boundary、再現性、実行量、artifact 契約を設計した。
- self-contained Jupytext train notebook と fail-closed inference notebook を実装した。
- exact joint-state top-5 decoder、TVT sequence dedup、direct/oracle/path診断を実装した。
- exp209 FB parity、小規模全列挙、dedup、oracle、leakage、Kaggle input resolver、実行量 contract の targeted tests 7件を追加した。
- py_compile、ruff、Jupytext test、strict experiment validation、Kaggle package準備を完了した。
- version 3 time limitと生成物なしを記録し、2 deterministic shards + aggregateでの再実行についてユーザー承認を得た。
- stable SHA256 2-shard生成とSHA固定aggregate notebookを3つのself-contained sourceとして実装した。
- targeted tests 10件を通常環境・Numba JIT環境で通し、Jupytext、py_compile、Ruff、strict validationを完了した。
- 2 shard version 1のsession timeout不足と、version 2のresource/time不足を監査し、いずれも生成物なしと記録した。
- per-well gzip stream、線形parity、binary-part SHA、stage/RSS logを実装し、13 targeted tests、Jupytext、Ruff、py_compile、strict validationを通した。
- 同じ2 shard kernelへ`--timeout 43200`でversion 3をpushし、両方の初期status `RUNNING`を確認した。
- version 3の両shardが全HMMを4.30--4.57時間、peak RSS 1.56 GB未満で完了後、exp209のfloat32保存列とexp270のfloat64再計算列のparityでfail-closeしたことを監査した。
- posterior meanだけをexp209保存dtypeへ正規化し、HMM・新mode候補・許容値を変えないversion 4最小修正を実装し、targeted tests 14件を通常Numba/JIT無効の両環境で通した。
- 同じ2 shard kernelへ`--timeout 43200`でversion 4をpushし、両方の初期status `RUNNING`を確認した。
- version 4の両shardが`COMPLETE`となり、合計773 wells / 3,783,989 rows、exp209 parity max 0.0 ft、peak RSS 1.56 GB未満を確認した。
- Kaggle outputを取得し、candidate gzipのraw/decompressed SHA、全sidecar SHA、prediction content SHA、ID/row順、stable shard割当を全照合した。
- 監査済みshard SHAをaggregateのfail-closed入力契約としてconfigへ固定した。
- canonical aggregate kernelへ0 HMM / 0 boosterのversion 4を`--timeout 43200`でpushし、初期status `RUNNING`を確認した。
- aggregate version 4の`COMPLETE`、3,783,989 rows / 773 wells、runtime 156.241秒、peak RSS 3,097.277 MBを確認した。
- exp209 parity max / mean 0.0 / 0.0 ft、ID/order/finite/禁止列、13 artifact SHA、candidate raw/decompressed SHA、prediction content SHAを再照合した。
- direct readoutでposterior mean 11.938287が最良、marginal MAP 12.592479、global Viterbi 15.551665、top-2からtop-5も全悪化と確認した。
- hidden-like 2面でもposterior meanが最良と確認した。oracleは診断専用に分離し、top-2からtop-5の3候補bankへの追加改善が最大0.000342 ftだけと記録した。
- `result.md`、`metrics.json`、`README.md`、`SESSION_NOTES.md`、`experiment_summary.md`、`KAGGLE_DIRECTION.md`へ最終結果を反映し、候補追加、selector、inference、submissionなしでbranchを閉じた。
