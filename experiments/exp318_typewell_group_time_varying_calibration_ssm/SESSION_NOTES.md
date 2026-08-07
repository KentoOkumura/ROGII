# exp318 セッションノート

## 現在の状態

- 2026-07-21: 2-stage設計確定。exp311/313待ち。実装・実行なし。
- Route: `pf_beam`
- 実行契約: Stage 0/1各1 variant、0 model / 0 booster / 0 TVT decoder。

## 設計契約

- stateはintercept/log-scale、local-level random walk、causal Kalman filter。
- process noiseはouter-train empirical Bayesで1回だけ固定する。
- Stage 0はlast640 mask、fixed 16-well runtime microbenchmark、8.5h hard gate。
- prior/state/boundary prediction/runtime profileのSHAを保存する。

## 次

先行gate PASS後にStage 0だけを実装し、Stage 1は再承認制とする。
