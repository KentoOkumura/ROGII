# exp318_typewell_group_time_varying_calibration_ssm

## 状態

- ルート: `pf_beam`
- 状態: 設計確定・exp311/313とruntime gate待ち・未実装
- 親: `exp311_typewell_group_prefix_suffix_gr_calibration_readout`
- CV / LB: 未実行 / 未提出

## 仮説と変更点

GR offset/scale driftはstatic group priorより2-state causal Kalman modelで追跡できる。exp295のjoint position-rate stateを再利用せず、last640 known-prefix maskで可同定性と8.5h runtimeを先に判定する。

## 検証方針

Stage 0 mask backtestとmicrobenchmarkを全PASSした場合だけStage 1 suffix extrapolationを検討する。boundary越えsmoothing、joint TVT decoder、gridは禁止。

## 所見

可同定性とruntimeを先に判定し、exp295型の巨大state計算を避ける。

## 実行入口

notebookは未実装placeholder。`implementation.enabled=false`でKaggle package/push/runは禁止。
