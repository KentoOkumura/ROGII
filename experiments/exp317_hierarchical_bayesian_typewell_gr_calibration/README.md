# exp317_hierarchical_bayesian_typewell_gr_calibration

## 状態

- ルート: `pf_beam`
- 状態: 設計確定・exp311/313待ち・未実装
- 親: `exp311_typewell_group_prefix_suffix_gr_calibration_readout`
- CV / LB: 未実行 / 未提出

## 仮説と変更点

group noise/reliabilityはplug-in中央値よりglobal→group→wellのpartial poolingで安定する。primaryはidentity affineのままsigmaだけを階層化し、deterministic MAP/Laplace posterior predictiveを監査する。

## 検証方針

5 folds + leave-group-outでNLL/RMSE/fold/worstを判定する。MCMCとdecoder統合は範囲外。本exp PASS後もHMM統合は別設計が必要。

## 所見

経験的に共通性が強いnoiseをprimaryとし、affine全体の階層化はdiagnosticへ下げる。

## 実行入口

notebookは未実装placeholder。`implementation.enabled=false`でKaggle package/push/runは禁止。
