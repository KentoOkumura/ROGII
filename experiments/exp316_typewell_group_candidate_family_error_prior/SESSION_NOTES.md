# exp316 セッションノート

## 現在の状態

- 2026-07-21: 2-stage設計確定。exp313/exp315待ち。実装・実行なし。
- Route: `ml_model`
- Stage A: 1 prior readout / 0 model。
- Stage B: 1 variant / 2 objectives / outer 5 × inner 4 / 40 selector models。

## 設計契約

- well等重みのgroup×family MAE/RMSE/best率、support k=10 wellsを固定する。
- fallbackはgroup family→global family→neutral。
- outer-valid errorはprior凍結後だけ結合し、well ID/hard routerを禁止する。
- family manifest/prior/feature/model/OOFのSHAを段階別に記録する。

## 次

先行gate PASS後にStage Aだけを実装し、Stage Bは別承認とする。
