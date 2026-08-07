# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし

## 完了

- steering docs作成。
- 再現性設計を `design.md` に記入。
- exp239実験ディレクトリ作成。
- raw train well metadata readout実装。
- quantile / holdout rows / GR change / missing block / curvature cutoff候補生成実装。
- deterministic distribution-matched selectorとcap実装。
- source-well GroupKFold manifestとleakage assertion実装。
- sampling前後のdistribution report、cutoff manifest、prefix replay request、SHA summary保存実装。
- Jupytext self-contained train/inference notebook作成とipynb変換。
- static check、F821、strict experiment validation pass。
- `SESSION_NOTES.md`、`result.md`、`metrics.json`、`KAGGLE_DIRECTION.md`を実装状態へ更新。
- Kaggle metadataとbootstrap内config、exp115 input同梱の整合を確認。
- Kaggle CPU audit v1実行、output取得、SHA/leakage/distribution readout完了。
- v2 global quota selector、事前guard、v1 artifact preflightを実装。
- v2 Jupytext、ruff、py_compile、strict validation、Kaggle package prepare pass。
- Kaggle CPU audit v2実行、distribution/leakage guard pass、output/SHA記録完了。
- v3 materialization範囲、target分離、row sampling/capを設計。
- v3 anchor/prefix statistics materialization実装、Jupytext/ruff/py_compile/F821/strict validation pass。
- Kaggle CPU v3実行、800 requests / 799,961 rowsのmaterialization guardとdownload後SHA一致を確認。
- v4 CPU residual probe実行。global/bucket改善を確認したがmax well regression +63.415661でguard failed、direct route停止。
- v5全380特徴再生成、official-only validation、pseudo weight 0.5、15 boosters実装・静的検証。
- v7 single-job本評価を実行し、特徴生成中のDeadKernel/OOM推定を確認（学習0 booster）。
- v8 CPU cache preflightとfull生成を実行。32 shards / 800 requests / 799,961 rows / 380 features guard pass。
- v9 GPU cached-training v1はpseudo 32 shards検証pass後、official feature assemblyでOOM（学習0 booster）。
- v10 CPU official exp218 feature-cache生成完了。16 shards / 3,783,989 rows / 380 features、pseudo cacheとのschema一致を確認。
- v11 GPU dual-cache streaming本評価完了。15/15 boosters、OOF 8.697380066でexp218から+0.221586314悪化。不採用、inference/submitなし。
- v12 trial inference v1完了。15 model SHA、380-feature schema、14,151行、fallback 0、submit-check PASS。
- trial submission ref `54720769` を提出。記録時点PENDING。
- ref `54720769` scoring完了。Public LB 7.944でexp218/exp238より悪化し、不採用を確定。
