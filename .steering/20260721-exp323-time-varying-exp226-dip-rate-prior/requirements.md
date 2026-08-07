# 要件

> **閉鎖済み（2026-07-22）**: 旧exp309 parent chain不成立のため未実装・未実行でterminal closeした。reparentしない。exp338 PASS時だけ新番号のexp323相当を別設計する。

## 依頼

exp226を結果blendではなく、exact HMMの時間変化するdip-rate prior平均として取り込む設計を確定する。実装は行わない。

## 制約

- Route: `pf_beam`。
- exp309全gate PASSとdependency SHA固定を先行条件とする。
- exp226 final prediction、absolute TVT unary、GR correction、U projection、blendを使わない。
- validation wellと同fold validation wellsをdonor fieldから除外する。
- suffix truthはrate scheduleとSHA凍結後にだけ評価へ結合する。
- Stage 0 FAIL後のparameter rescueを禁止する。

## 受け入れ基準

- Stage 0/1、式、fallback、依存、実行量、停止条件がconfigと文書で一致する。
- 実装・Kaggle push・inference・submissionがfail-closedである。
- RNG、fold、input/rate-schedule/prediction SHA方針が明記される。
